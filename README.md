# Practo (Healthcare) — Part 1 completed

The Practo (Healthcare) track's Part 1, Dataset Design and RAG Core, is completed. Later Parts have not been implemented. All appointment records and clinic policies are fabricated educational examples; the policies are not statements of Practo's actual services.

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

## Parent-document evaluation

```powershell
python -m rag.evaluate
```

The evaluator uses the exact same **6** in-scope queries as the demo, with hand-labelled relevant parents in [rag/queries.py](rag/queries.py). It retrieves **3 chunks** from each collection, maps each chunk to its parent document, deduplicates parents, and computes precision as relevant retrieved parents divided by all retrieved parents, and recall as relevant retrieved parents divided by all labelled relevant parents. Evaluation applies no similarity filter; averages are macro averages across queries.

| Query | Policy question |
|---|---|
| q01 | How do I book an appointment and know that my slot is confirmed? |
| q02 | What is the cancellation and rescheduling window? |
| q03 | What are the consultation fees for each specialty? |
| q04 | How long do routine and specialized lab test reports take? |
| q05 | When does a follow-up visit qualify for a discount? |
| q06 | How do I submit an insurance reimbursement claim? |

### Fixed overlap

| Query | Labelled parent documents | Retrieved parent documents (deduped) | Precision arithmetic | Recall arithmetic |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling, 01_appointment_booking | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround, 03_consultation_fees | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| Macro average | | | 0.500000 | 1.000000 |

### Sentence chunks

| Query | Labelled parent documents | Retrieved parent documents (deduped) | Precision arithmetic | Recall arithmetic |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling, 01_appointment_booking | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 12_home_visits | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| Macro average | | | 0.500000 | 1.000000 |

Across 6 hand-labelled queries, fixed_overlap achieved macro precision 0.500000 and macro recall 1.000000, while sentence achieved macro precision 0.500000 and macro recall 1.000000. Deploy sentence because it has equal macro recall and precision, with complete-sentence boundaries. This small policy sample supports a local choice and does not establish universal retrieval quality.

[The full evaluation transcript](transcripts/task5_evaluation.txt) includes both strategies side by side, retrieved chunk IDs, deduplicated parent IDs, labels, and arithmetic. [The machine-readable report](transcripts/task5_evaluation.json) and [deployment choice](rag/deployment.json) were generated by the evaluator. Grounded generation now defaults to `practo_sentence` and its measured threshold **0.345280051**; changing the corpus invalidates the saved deployment choice.

Precision is **0.500000** because each query retrieves its relevant parent plus another parent. The generation step's extraction from the strongest parent does not change these reported retrieval metrics.

## Reproduce and verify

After the one-time dependency/model setup, all commands below run offline with zero API keys and zero paid accounts:

```powershell
.\.venv\Scripts\Activate.ps1
python dataset.py --export data/appointments.json
python -m scripts.validate_kb
python -m rag.index
python scripts/calibrate_threshold.py
python -m rag.evaluate
python scripts/run_part1_demo.py
python -m unittest discover -s tests -v
python scripts/verify_offline.py
```

If script activation is disabled, run each command with `.\.venv\Scripts\python.exe` directly. Run commands from the repository root. Rerun indexing, calibration, and evaluation after editing policies; the demo then uses the resulting deployment choice.

[The full suite](transcripts/part1_tests.txt) passed **21 tests**. [The offline verification](transcripts/part1_offline.txt) clears API-key environment variables, blocks Python sockets and DNS before importing frameworks, rebuilds both collections in a fresh temporary store, recalibrates, evaluates, and demonstrates the answers and fallback. It recorded **0** outbound Python network attempts. Those runtime commands also ran successfully under the execution environment's network restrictions.

| Task | Saved evidence |
|---|---|
| Dataset | [Generator output](transcripts/task1_dataset.txt), [tests](transcripts/task1_tests.txt), [generated records](data/appointments.json) |
| Knowledge base | [Validator output](transcripts/task2_knowledge_base.txt), [tests](transcripts/task2_tests.txt) |
| Chunking and indexing | [Indexing and retrieval](transcripts/task3_indexing.txt), [tests](transcripts/task3_tests.txt) |
| Grounded generation | [Calibration](transcripts/task4_calibration.txt), [demo](transcripts/task4_demo.txt), [retrieved contexts](transcripts/task4_demo.json), [tests](transcripts/task4_tests.txt) |
| Strategy evaluation | [Arithmetic and recommendation](transcripts/task5_evaluation.txt), [structured metrics](transcripts/task5_evaluation.json), [tests including deployment freshness](transcripts/task5_tests.txt) |
| Environment | [Exact dependencies](transcripts/dependencies.txt), [dependency check](transcripts/dependency_check.txt), [offline run](transcripts/part1_offline.txt), [complete tests](transcripts/part1_tests.txt) |

## Scope and remaining verification

Only Part 1 is implemented. CrewAI orchestration, appointment lookup, memory, structured crew output, guardrails, API deployment, judge evaluation, Autogen review, governance, and caching remain for their requested sessions. The escalation formula and later-Part evidence therefore are not claimed here.

No dataset design choices changed. The dependency installation required initial network access; the exact embedding checkpoint was already cached, so a fresh download through `scripts/download_model.py` was not executed. Loading that checkpoint and the complete RAG workflow were verified offline. Chroma emits a non-failing legacy embedding-configuration deprecation warning in the saved unit-test output; explicit embeddings and disabled automatic embedding remain verified.

Public GitHub repository: [Practo Domain Support Agent](https://github.com/manishjsr14-cloud/practo-domain-support-agent). The KB prose is an original assisted draft that the student should review and rewrite before submission. Results establish the requested small Part 1 demonstration, with the similarity-only limitations described above.
