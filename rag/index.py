"""Local SentenceTransformers embeddings and independent cosine collections."""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from rag import RUNTIME_ENVIRONMENT
from rag.chunking import (
    FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, ROOT, SENTENCES_PER_CHUNK,
    SENTENCE_SPLIT_PATTERN, STRATEGIES, chunk_documents, load_documents,
)


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
COLLECTION_NAMES = {
    "fixed_overlap": "practo_fixed_overlap",
    "sentence": "practo_sentence",
}


def _fingerprint(value: dict) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _load_model():
    """Share the immutable CPU embedding model across index instances."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    try:
        return SentenceTransformer(
            MODEL_ID, revision=MODEL_REVISION, device="cpu",
            local_files_only=True, token=False,
        )
    except OSError as error:
        raise RuntimeError(
            "The pinned embedding model is not fully cached locally; "
            "download it during setup before running the offline demo."
        ) from error


class RAGIndex:
    """Load only a cached model and persist regenerable indexes outside Git."""

    def __init__(
        self,
        store_path: Path = ROOT / ".runtime" / "chroma",
        kb_dir: Path = ROOT / "kb",
    ):
        self.store_path = Path(store_path)
        self.kb_dir = Path(kb_dir)
        self._client = None
        self._model = None

    @property
    def documents(self) -> dict[str, str]:
        # Reload so edits are detected even by a long-lived index instance.
        return load_documents(self.kb_dir)

    @staticmethod
    def _corpus_fingerprint(documents: dict[str, str]) -> str:
        return _fingerprint({
            "documents": documents,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "normalized_embeddings": True,
            "distance": "cosine",
            "fixed_chunk_size": FIXED_CHUNK_SIZE,
            "fixed_chunk_overlap": FIXED_CHUNK_OVERLAP,
            "sentences_per_chunk": SENTENCES_PER_CHUNK,
            "sentence_split_pattern": SENTENCE_SPLIT_PATTERN,
            "normalization": "strip-markdown-headings-collapse-whitespace",
            "sentence_boundary_metadata": True,
        })

    @property
    def corpus_fingerprint(self) -> str:
        return self._corpus_fingerprint(self.documents)

    @property
    def config_fingerprint(self) -> str:
        """Calibration identity includes the corpus, model, and chunk settings."""
        return self.corpus_fingerprint

    @staticmethod
    def _strategy_fingerprint(corpus_fingerprint: str, strategy: str) -> str:
        return _fingerprint({"corpus_fingerprint": corpus_fingerprint,
                             "strategy": strategy})

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.store_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.store_path),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    @staticmethod
    def _check_strategy(strategy: str) -> None:
        if strategy not in COLLECTION_NAMES:
            raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")

    @staticmethod
    def _check_cosine(collection) -> None:
        if (collection.configuration.get("hnsw") or {}).get("space") != "cosine":
            raise RuntimeError(f"Collection {collection.name} must use cosine distance")

    def get_collection(self, strategy: str):
        """Return an existing named collection without creating one silently."""
        self._check_strategy(strategy)
        from chromadb.errors import NotFoundError

        try:
            return self.client.get_collection(
                name=COLLECTION_NAMES[strategy], embedding_function=None,
            )
        except NotFoundError as error:
            raise RuntimeError(
                f"Missing {strategy} index; run python -m rag.index first."
            ) from error

    def build(self) -> dict[str, int]:
        """Upsert both strategies and remove only chunk IDs no longer present."""
        documents = self.documents
        corpus_fingerprint = self._corpus_fingerprint(documents)
        counts = {}
        for strategy in STRATEGIES:
            chunks = chunk_documents(documents, strategy)
            collection = self.client.get_or_create_collection(
                name=COLLECTION_NAMES[strategy],
                configuration={"hnsw": {"space": "cosine", "num_threads": 1}},
                embedding_function=None,
            )
            self._check_cosine(collection)
            existing_ids = set(collection.get(include=[])["ids"])
            current_ids = {chunk.id for chunk in chunks}
            collection.upsert(
                ids=[chunk.id for chunk in chunks],
                embeddings=self._embed([chunk.text for chunk in chunks]),
                documents=[chunk.text for chunk in chunks],
                metadatas=[{
                    "parent_id": chunk.parent_id, "strategy": chunk.strategy,
                    "index": chunk.index, "start": chunk.start, "end": chunk.end,
                    "starts_sentence": chunk.start == 0 or bool(re.search(
                        r"[.!?]\s+$", documents[chunk.parent_id][:chunk.start])),
                } for chunk in chunks],
            )
            stale_ids = sorted(existing_ids - current_ids)
            if stale_ids:
                collection.delete(ids=stale_ids)
            # Mark fresh only after all upserts and removals have succeeded.
            collection.modify(metadata={
                "corpus_fingerprint": corpus_fingerprint,
                "strategy_fingerprint": self._strategy_fingerprint(
                    corpus_fingerprint, strategy),
                "strategy": strategy, "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            })
            counts[strategy] = collection.count()
            if counts[strategy] != len(chunks):
                raise RuntimeError(f"Unexpected chunk count for {strategy}")
        return counts

    def retrieve(
        self, query: str, strategy: str = "sentence", top_k: int = 3,
    ) -> list[dict]:
        """Retrieve chunks; cosine similarity is explicitly one minus distance."""
        if not query.strip():
            raise ValueError("Query must not be blank")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        collection = self.get_collection(strategy)
        self._check_cosine(collection)
        documents = self.documents
        fingerprint = self._corpus_fingerprint(documents)
        expected = self._strategy_fingerprint(fingerprint, strategy)
        metadata = collection.metadata or {}
        if (metadata.get("corpus_fingerprint") != fingerprint
                or metadata.get("strategy_fingerprint") != expected
                or collection.count() != len(chunk_documents(documents, strategy))):
            raise RuntimeError(
                f"Stale or incomplete {strategy} index; run python -m rag.index."
            )
        result = collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append({
                "chunk_id": chunk_id, "parent_id": metadata["parent_id"],
                "text": text, "distance": float(distance),
                "similarity": 1.0 - float(distance), "metadata": metadata,
            })
        return hits


def main() -> None:
    index = RAGIndex()
    print(f"Model: {MODEL_ID}")
    print(f"Model revision: {MODEL_REVISION}")
    print("Runtime environment: " + json.dumps(
        {name: os.environ[name] for name in RUNTIME_ENVIRONMENT}, sort_keys=True))
    print(f"Fixed chunk size: {FIXED_CHUNK_SIZE} characters")
    print(f"Fixed overlap: {FIXED_CHUNK_OVERLAP} characters")
    print(f"Sentence chunk size: {SENTENCES_PER_CHUNK} sentences")
    print(f"Parent document count: {len(index.documents)}")
    print(f"Corpus fingerprint: {index.corpus_fingerprint}")
    first_counts = index.build()
    print("First build counts: " + json.dumps(first_counts, sort_keys=True))
    repeat_counts = index.build()
    print("Repeated upsert counts: " + json.dumps(repeat_counts, sort_keys=True))
    if first_counts != repeat_counts:
        raise RuntimeError("Repeated upsert changed collection sizes")
    print("Repeated upsert preserved collection sizes: True")
    print("Cosine similarity = 1 - cosine distance")
    query = "What is the cancellation and rescheduling window?"
    print(f"Sample query: {query}")
    for strategy in STRATEGIES:
        print(f"Collection: {COLLECTION_NAMES[strategy]}")
        for rank, hit in enumerate(index.retrieve(query, strategy), start=1):
            print(f"  {rank}: {hit['parent_id']} "
                  f"distance={hit['distance']:.6f} "
                  f"similarity={hit['similarity']:.6f}")
            print(f"     {hit['text']}")


if __name__ == "__main__":
    main()
