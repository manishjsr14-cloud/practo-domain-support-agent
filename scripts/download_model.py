"""One-time online setup; runtime code never imports this module.

Only the required PyTorch checkpoint and tokenizer files are downloaded.
The download uses public access, with no Hugging Face token or API key.
"""

import os


def main() -> None:
    # This explicit setup entry point permits network access before imports.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        token=False,
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
    )
    print(f"Public embedding checkpoint cached at: {path}")
    print("Setup complete; all RAG runtime commands enforce offline mode")


if __name__ == "__main__":
    main()
