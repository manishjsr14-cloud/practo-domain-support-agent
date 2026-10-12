# Practo (Healthcare) — Part 1 in progress

This repository implements the Dataset Design and RAG Core portion of the Practo capstone. Later Parts have not been implemented. All appointment records and clinic policies are fabricated educational examples; the policies are not statements of Practo's actual services.

## Dataset design

The private random generator uses seed **42** and produces **50** records. The first reserved rows cover every category and status; remaining rows use weighted draws. Each row independently draws a continuous uniform fee rounded to the nearest **50 INR**, an integer age uniformly from **0–30** days, and a Bernoulli follow-up flag with probability **0.20**.

| Category | Weight | Fee range INR | Observed count |
|---|---:|---:|---:|
| General Medicine | 0.30 | 300–600 | 12 |
| Cardiology | 0.15 | 800–1500 | 8 |
| Dermatology | 0.20 | 500–1000 | 11 |
| Pediatrics | 0.20 | 400–800 | 8 |
| Orthopedics | 0.15 | 600–1200 | 11 |

| Status | Weight | Observed count |
|---|---:|---:|
| Completed | 0.45 | 20 |
| Scheduled | 0.25 | 15 |
| Cancelled | 0.12 | 8 |
| Rescheduled | 0.10 | 4 |
| No-Show | 0.08 | 3 |

The overall designed fee range is **300–1500 INR**. These ranges reflect typical Indian outpatient consultation fees, with general practice at the low end and cardiology at the high end because of specialist training and diagnostic overhead.

The measured follow-up count is **13/50 = 26.00%**, within the required **10–30%** band. No individual records were edited. All design and count values above appear in [the dataset transcript](transcripts/task1_dataset.txt).

```powershell
python dataset.py --export data/appointments.json
python -m unittest tests.test_part1 -v
```

[Generated dataset](data/appointments.json) · [Dataset tests](transcripts/task1_tests.txt)

## Knowledge base

[The knowledge base](kb/) contains **12** original draft documents covering all required topics. Each has **4** sentences, except consultation fees with **5**; [the validator output](transcripts/task2_knowledge_base.txt) records these counts and verifies fee consistency. The author should review and rewrite these educational policy drafts in their own voice before submitting the capstone.

```powershell
python -m scripts.validate_kb
```

[Dataset and knowledge-base tests](transcripts/task2_tests.txt)

## Local setup and indexing

The tested Python version and complete dependency pins are saved in [dependencies.txt](transcripts/dependencies.txt); [pip check passed](transcripts/dependency_check.txt). Install dependencies and cache the public embedding checkpoint once while online:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
```

For subsequent commands, activate the environment or use `.\.venv\Scripts\python.exe` in place of `python`. The runtime uses `sentence-transformers/all-MiniLM-L6-v2`, pinned to the exact checkpoint recorded in [the indexing transcript](transcripts/task3_indexing.txt), on CPU with normalized embeddings and `local_files_only=True`. The online setup command was not needed on this machine because that exact checkpoint was already cached; offline loading was verified.

```powershell
python -m rag.index
python -m unittest tests.test_part1 -v
```

Fixed chunks use **320 characters** with **80 characters** of overlap; sentence chunks group **2 sentences**. The two independent cosine collections, `practo_fixed_overlap` and `practo_sentence`, contain **24** and **25** chunks respectively. Both use explicit local embeddings and `collection.upsert()`; repeated upserts preserve these counts. Every chunk retains its parent ID and source offsets, and changed source content causes retrieval to fail until reindexing.

Chroma cosine distance is converted as **similarity = 1 − distance**. Similarity is not a probability. [The sample retrieval and indexing evidence](transcripts/task3_indexing.txt) shows both strategies retrieving the cancellation policy, and [integration tests](transcripts/task3_tests.txt) check cosine arithmetic, parent metadata, idempotence, and stale-index rejection.

`rag/__init__.py` enforces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `ANONYMIZED_TELEMETRY=False`, `CREWAI_DISABLE_TELEMETRY=true`, and `OTEL_SDK_DISABLED=true` before framework imports. No crew or kickoff call exists in this Part; telemetry is already disabled for later integration. Model weights, the virtual environment, and regenerated binary Chroma files are ignored under `.venv/` and `.runtime/`, keeping repository deliverables code and text.

API references: [local SentenceTransformer loading and encoding](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html), [Chroma persistent client](https://docs.trychroma.com/reference/python).

## Measured calibration and grounded generation

Run calibration after indexing, then demonstrate the deterministic `MOCK_LLM` generator:

```powershell
python scripts/calibrate_threshold.py
python scripts/run_part1_demo.py
python -m unittest tests.test_generation -v
```

The calibration queries below were declared separately from the demonstration queries before measurement. The values are top-one cosine similarities from [the real calibration run](transcripts/task4_calibration.txt).

| Scope | Calibration query | Fixed overlap | Sentence |
|---|---|---:|---:|
| In | How can I reserve a clinic appointment? | 0.756113172 | 0.732594550 |
| In | How much does a Cardiology consultation cost? | 0.744205773 | 0.904522061 |
| In | What documents are needed for an insurance claim? | 0.434606612 | 0.496595979 |
| In | Who is allowed to authorize a prescription refill? | 0.689840138 | 0.698915839 |
| Out | How do I repair a punctured bicycle tyre? | 0.174819827 | 0.193964124 |
| Out | What is the orbital period of Neptune? | 0.108673513 | 0.125301600 |
| Out | Write a sorting algorithm in Rust. | 0.047496557 | 0.050853133 |

| Strategy | Maximum out-of-scope | Minimum in-scope | Midpoint threshold |
|---|---:|---:|---:|
| Fixed overlap | 0.174819827 | 0.434606612 | 0.304713219 |
| Sentence | 0.193964124 | 0.496595979 | 0.345280051 |

The threshold is computed as `(max_out + min_in) / 2` from full-precision observations and saved in [calibration.json](rag/calibration.json). Calibration fails if the observed clusters overlap; there is no untested default. Corpus/model/chunk fingerprints prevent silently reusing stale calibration.

Retrieval takes the **3** most similar chunks. Under `MOCK_LLM`, top-one similarity is the only groundedness signal: below the strategy's calibrated threshold, the generator returns “I don't know based on the available clinic-policy context.” Otherwise it extracts only retrieved text from the most similar parent, merges overlapping spans, and cites the chunk IDs. It calls no remote LLM and uses no API keys or paid accounts.

[The demonstration](transcripts/task4_demo.txt) answers **6** in-scope questions about booking, cancellation, specialty fees, laboratory reports, follow-up discounts, and insurance claims. Its separate brownie-recipe query scores **0.023517489** with sentence chunks and triggers the fallback. [Structured answers and original retrieved context](transcripts/task4_demo.json) allow the extraction to be audited; [generation tests](transcripts/task4_tests.txt) check both strategies, the fallback, extraction boundaries, and stale calibration.

This small calibration set contains unrelated negative queries. Similarity alone does not establish medical safety, detect prompt injection, or distinguish every unsupported question that shares clinic terminology; those later guardrails are outside this session's scope.
