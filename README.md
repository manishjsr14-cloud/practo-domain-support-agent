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

[The knowledge base](kb/) contains the student's **12** handwritten policy documents, copied verbatim from the supplied policy files. Each has **5** sentences and covers its declared topic; [the validator output](transcripts/task2_knowledge_base.txt) records these counts. [The import transcript](transcripts/kb_import.txt) and [file-hash manifest](transcripts/kb_import.json) verify byte-for-byte equality with the supplied originals.

The handwritten fee policy sets these standard prices for new bookings:

| Specialty | Standard consultation fee INR |
|---|---:|
| General Medicine | 500 |
| Cardiology | 1200 |
| Dermatology | 800 |
| Pediatrics | 600 |
| Orthopedics | 1000 |

These prices are within the dataset's unchanged per-specialty ranges, as verified in the validator transcript. For an existing appointment, the recorded booking fee remains authoritative and may differ from the standard price because of a documented adjustment. This explicitly replaces the earlier KB's range-based fee wording; the generated dataset has not been changed.

The previous persisted RAG database was deleted, then rebuilt from the handwritten files. [The deletion record](transcripts/kb_database_rebuild.txt) and [database verification](transcripts/handwritten_verification.txt) confirm that both active collections contain exactly the current source chunks and no previous-policy or unexpected chunks. Historical transcripts retain earlier measured results for audit and are excluded from indexing.

```powershell
python -m scripts.validate_kb
```

[Knowledge-base tests](transcripts/task2_tests.txt)

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

Fixed chunks use **320 characters** with **80 characters** of overlap; sentence chunks group **2 sentences**. The two independent cosine collections, `practo_fixed_overlap` and `practo_sentence`, contain **39** and **36** chunks respectively. Both use explicit local embeddings and `collection.upsert()`; repeated upserts preserve these counts. Every chunk retains its parent ID and source offsets, and changed source content causes retrieval to fail until reindexing.

Chroma cosine distance is converted as **similarity = 1 − distance**. Similarity is not a probability. [The sample retrieval and indexing evidence](transcripts/task3_indexing.txt) shows both strategies retrieving the cancellation policy, and [integration tests](transcripts/task3_tests.txt) check cosine arithmetic, parent metadata, idempotence, and stale-index rejection.

Retrieval retains the original whole-query cosine winner as its first chunk. It scans each collection's candidates, ranks supporting evidence using both parent-document and chunk semantics, and gives additional weight to coverage of grammatical content phrases that are not already covered. For a question preceded by a narrative clause, it also embeds the explicit question clause. Supporting primary-policy chunks prefer a complete sentence boundary, and still-uncovered multiword question components can reserve their strongest semantic match before repeated chunks fill the context. A supporting match must contain a least-common observed content word from the component, so a generic shared word cannot stand in for its distinguishing term. Isolated words retain soft coverage scoring. No query IDs, labelled parents, or clinic-topic routing are used. The final context remains **3 chunks**, and ranking scores are recorded separately from cosine similarities. A separate retrieval-policy fingerprint invalidates the deployment recommendation when evidence selection changes, without altering calibration identity.

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
| In | How can I reserve a clinic appointment? | 0.734329104 | 0.746177197 |
| In | How much does a Cardiology consultation cost? | 0.674873888 | 0.708590269 |
| In | What documents are needed for an insurance claim? | 0.440953672 | 0.486387312 |
| In | Who is allowed to authorize a prescription refill? | 0.605015993 | 0.600208402 |
| Out | How do I repair a punctured bicycle tyre? | 0.145102620 | 0.152278662 |
| Out | What is the orbital period of Neptune? | 0.143519402 | 0.147324979 |
| Out | Write a sorting algorithm in Rust. | 0.075274765 | 0.082972288 |

| Strategy | Maximum out-of-scope | Minimum in-scope | Midpoint threshold |
|---|---:|---:|---:|
| Fixed overlap | 0.145102620 | 0.440953672 | 0.293028146 |
| Sentence | 0.152278662 | 0.486387312 | 0.319332987 |

The threshold is computed as `(max_out + min_in) / 2` from full-precision observations and saved in [calibration.json](rag/calibration.json). Calibration fails if the observed clusters overlap; there is no untested default. Corpus/model/chunk fingerprints prevent silently reusing stale calibration. Replacing the KB changed its fingerprint, so this import recalibrated the new corpus with the original declared queries and midpoint rule.

Under `MOCK_LLM`, the original whole-query top-one cosine similarity remains the only acceptance signal: below the strategy's calibrated threshold, the generator returns “I don't know based on the available clinic-policy context.” Once accepted, it selects a primary parent semantically and includes a supporting parent when that parent best matches a question component and has an anchor on a least-common observed content word from that component. This generic coverage heuristic selects excerpts; it is not a groundedness guardrail. The generator merges overlaps within each selected parent, extracts only retrieved text, and cites the contributing chunk IDs. Supporting excerpts are not individually rejected against the whole-query threshold, which was calibrated for the initial acceptance decision. It calls no remote LLM and uses no API keys or paid accounts.

[The demonstration](transcripts/task4_demo.txt) produces responses for **9** in-scope questions about booking, cancellation, specialty fees, laboratory reports, follow-up discounts, insurance claims, a follow-up paraphrase, cardiologist home visits, and video second opinions. Its separate brownie-recipe query scores **0.054437876** with the deployed sentence strategy and triggers the fallback. Producing a response establishes that the similarity gate accepts it; it does not establish that the response answers the question correctly. [Structured answers and original retrieved context](transcripts/task4_demo.json) record each selected parent excerpt and its contributing sources, and the answer is exactly the joined excerpts. [The original generation tests](transcripts/task4_tests.txt) and [current coverage and generation tests](transcripts/answer_coverage_tests.txt) check both strategies, the fallback, extraction boundaries, source spans, and stale calibration.

The current demo follows the handwritten policies. q07 cites the **20 percent** discount within **seven calendar days**, with the same clinician, specialty, and concern required. q08 quotes the restriction that home visits are available only to General Medicine patients aged **18 years or older** within **five kilometres**, together with that service's **INR 800** total and **INR 300** travel surcharge; it does not offer a cardiologist home visit. q09 includes the second-opinion process and telemedicine's General Medicine/Dermatology, adult-age, clinician-approval, identity-check, connection, and consent conditions. Quoting these policies does not approve an individual's eligibility.

This small calibration set contains unrelated negative queries. Similarity alone does not establish medical safety, detect prompt injection, or distinguish every unsupported question that shares clinic terminology; those later guardrails are outside this session's scope.

## Near-topic threshold stress test

```powershell
python scripts/stress_threshold.py
```

These fixed unsupported questions are a separate check and were never added to calibration. The table reports top-one cosine similarity from both collections and the actual deployed generator decision. Deployment uses `sentence`; the transcript also records each collection's decision against its own existing calibrated threshold.

| Unsupported question | Fixed overlap cosine | Sentence cosine | Deployed threshold | Deployed decision |
|---|---:|---:|---:|---|
| Can you diagnose my chest pain? | 0.256390333 | 0.263177156 | 0.319332987 | FALLBACK |
| Which medicine should I take for a fever? | 0.272614956 | 0.202158451 | 0.319332987 | FALLBACK |
| What are the clinic's opening hours on Sundays? | 0.486783445 | 0.524604023 | 0.319332987 | ANSWER |
| Which hospital in Mumbai has the shortest waiting time? | 0.421420097 | 0.439948082 | 0.319332987 | ANSWER |

The deployed generator accepts **2** unsupported queries and falls back on **2**. Both collections accept the Sunday-opening-hours and Mumbai-hospital-waiting-time questions above their respective thresholds. The resulting answers quote booking and lab-delay policies, which do not supply Sunday opening hours or comparative hospital waiting times. The home-visit slot times in the KB also do not establish Sunday clinic opening hours. These are measured false acceptances of similarity-only groundedness, preserved in [the complete stress transcript](transcripts/task4_near_topic.txt) and [structured answers with retrieved context](transcripts/task4_near_topic.json).

The chest-pain diagnosis and fever-medicine questions fall back in this run. This observation does not establish a medical-safety guardrail. Part 2's groundedness guardrail will need to verify that context supports the actual question; passing a similarity threshold is insufficient. The new corpus was recalibrated using the unchanged calibration queries and midpoint rule; the stress check itself does not change either threshold. No keyword exception to the acceptance gate or Part 2 implementation was introduced.

The stress script verifies that calibration bytes remain unchanged before and after execution; the matching hashes are saved in its transcript. The CLI integration test checks every fixed question, both similarities, actual decisions and extracts, and calibration-file preservation.

## Parent-document evaluation

```powershell
python -m rag.evaluate
```

The evaluator uses the exact same **9** in-scope queries as the demo, with hand-labelled relevant parents in [rag/queries.py](rag/queries.py). It retrieves **3 chunks** from each collection, maps each chunk to its parent document, deduplicates parents, and computes precision as relevant retrieved parents divided by all retrieved parents, and recall as relevant retrieved parents divided by all labelled relevant parents. Evaluation applies no similarity filter; averages are macro averages across queries.

| Query | Policy question |
|---|---|
| q01 | How do I book an appointment and know that my slot is confirmed? |
| q02 | What is the cancellation and rescheduling window? |
| q03 | What are the consultation fees for each specialty? |
| q04 | How long do routine and specialized lab test reports take? |
| q05 | When does a follow-up visit qualify for a discount? |
| q06 | How do I submit an insurance reimbursement claim? |
| q07 | My doctor asked me to come back, do I have to pay the full price again? |
| q08 | What will I pay for a home visit by a cardiologist? |
| q09 | Can I get a second opinion over a video call? |

### Fixed overlap

| Query | Labelled parent documents | Retrieved parent documents (deduped) | Precision arithmetic | Recall arithmetic |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 10_follow_up_discount | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 10_follow_up_discount, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 03_consultation_fees, 12_home_visits | 03_consultation_fees, 12_home_visits | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 07_telemedicine, 11_second_opinion | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.888889 | 1.000000 |

### Sentence chunks

| Query | Labelled parent documents | Retrieved parent documents (deduped) | Precision arithmetic | Recall arithmetic |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 03_consultation_fees, 10_follow_up_discount | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 03_consultation_fees, 12_home_visits | 03_consultation_fees, 12_home_visits | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 07_telemedicine, 11_second_opinion | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.944444 | 1.000000 |

Across 9 hand-labelled queries, fixed_overlap achieved macro precision 0.888889 and macro recall 1.000000, while sentence achieved macro precision 0.944444 and macro recall 1.000000. Deploy sentence because it has equal macro recall and higher macro precision. This small policy sample supports a local choice and does not establish universal retrieval quality.

[The full evaluation transcript](transcripts/task5_evaluation.txt) includes both strategies side by side, retrieved chunk IDs, deduplicated parent IDs, labels, and arithmetic. [The machine-readable report](transcripts/task5_evaluation.json) and [deployment choice](rag/deployment.json) were generated by the evaluator. Grounded generation defaults to `practo_sentence` and its newly measured threshold **0.319332987**; changing the corpus or retrieval policy invalidates the saved deployment choice.

q07 has precision **0.500000** and recall **1.000000** for both strategies: the follow-up parent is retrieved alongside another parent, and the answer cites the follow-up parent only. q08 and q09 each give precision and recall **1.000000** for both strategies. Sentence q03 retrieves only its labelled fee parent, giving precision **1.000000**, while fixed overlap also retrieves a follow-up parent, giving precision **0.500000**. The other original queries retrieve only their labelled parent.

The fixed labels retain **7** single-parent queries and **2** multi-parent queries. Query wording and labels, chunk sizes, top-k, calibration questions, and the recall-then-precision-then-sentence selection rule remain unchanged. Calibration values were regenerated because the KB changed. Regression tests check q07's discount conditions, q08's General Medicine-only home-visit restriction and charges, and q09's second-opinion process and telemedicine limits under both strategies. q08's answer can cite the home-visit policy alone to explain why the requested cardiologist visit is unavailable; its retrieval labels still include the unchanged fee and home-visit parents.

[The original pre-fix evaluation](transcripts/before_answer_fix/task5_evaluation.txt), [pre-fix demo](transcripts/before_answer_fix/task4_demo.txt), and [failing regression run](transcripts/before_answer_fix/regressions.txt) retain the earlier miss and incomplete answers. [The pre-import evaluation](transcripts/before_handwritten_kb/task5_evaluation.txt) and [pre-import demo](transcripts/before_handwritten_kb/task4_demo.txt) record the previous KB. The current fixes also pass additional paraphrases of repeat-visit pricing, cardiologist home-visit restrictions, and video second opinions; they do not establish general answer completeness.

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
python scripts/stress_threshold.py
python -m unittest discover -s tests -v
python scripts/verify_offline.py
```

If script activation is disabled, run each command with `.\.venv\Scripts\python.exe` directly. Run commands from the repository root. Rerun indexing, calibration, and evaluation after editing policies; the demo then uses the resulting deployment choice.

[The full suite](transcripts/part1_tests.txt) passed **38 tests**. The earlier [evidence-refresh baseline](transcripts/refresh_tests.txt) passed **21** tests against the previous KB. [Initial import failures](transcripts/handwritten_initial_tests.txt) exposed obsolete wording and policy-dependent test assumptions, plus missing paraphrase context. The validator, fixtures, and evidence selection were updated while preserving the supplied policy bytes. [The offline verification](transcripts/part1_offline.txt) clears API-key environment variables, blocks Python sockets and DNS before importing frameworks, rebuilds both collections in a fresh temporary store, recalibrates, evaluates, demonstrates the answers and fallback, and runs the near-topic stress check using that fresh temporary calibration. It recorded **0** outbound Python network attempts. Those runtime commands also ran successfully under the execution environment's network restrictions. A [parallel fresh-store attempt](transcripts/handwritten_offline_initial.txt) encountered Chroma's “Nothing found on disk” segment-reader error; the subsequent serial fresh-store verification passed. Concurrent rebuild reliability is not established by this run.

| Task | Saved evidence |
|---|---|
| Dataset | [Generator output](transcripts/task1_dataset.txt), [tests](transcripts/task1_tests.txt), [generated records](data/appointments.json) |
| Knowledge base | [Validator output](transcripts/task2_knowledge_base.txt), [tests](transcripts/task2_tests.txt) |
| Chunking and indexing | [Indexing and retrieval](transcripts/task3_indexing.txt), [tests](transcripts/task3_tests.txt) |
| Grounded generation | [Calibration](transcripts/task4_calibration.txt), [demo](transcripts/task4_demo.txt), [retrieved contexts](transcripts/task4_demo.json), [tests](transcripts/task4_tests.txt) |
| Strategy evaluation | [Arithmetic and recommendation](transcripts/task5_evaluation.txt), [structured metrics](transcripts/task5_evaluation.json), [tests including deployment freshness](transcripts/task5_tests.txt) |
| Evidence refresh | [Indexing](transcripts/refresh_indexing.txt), [calibration](transcripts/refresh_calibration.txt), [evaluation](transcripts/refresh_evaluation.txt), [demo](transcripts/refresh_demo.txt), [existing tests](transcripts/refresh_tests.txt) |
| Expanded comparison | [Evaluation and generation tests](transcripts/fix_a_tests.txt), [latest complete tests](transcripts/part1_tests.txt) |
| Retrieval and answer fixes | [Pre-fix regression failures](transcripts/before_answer_fix/regressions.txt), [pre-fix evaluation](transcripts/before_answer_fix/task5_evaluation.txt), [current coverage and generation tests](transcripts/answer_coverage_tests.txt), [semantic diagnostics](transcripts/retrieval_diagnostics.txt), [evidence consistency checks](transcripts/answer_fix_checks.txt) |
| Handwritten KB import | [Exact-copy hashes](transcripts/kb_import.txt), [manifest](transcripts/kb_import.json), [old database removal](transcripts/kb_database_rebuild.txt), [current chunk verification](transcripts/handwritten_verification.txt), [documentation checks](transcripts/handwritten_docs_checks.txt), [pre-import evidence](transcripts/before_handwritten_kb/) |
| Near-topic stress | [Scores and actual decisions](transcripts/task4_near_topic.txt), [structured context](transcripts/task4_near_topic.json), [historical invariant checks](transcripts/part1_invariants.txt) |
| Environment | [Exact dependencies](transcripts/dependencies.txt), [dependency check](transcripts/dependency_check.txt), [offline run](transcripts/part1_offline.txt), [complete tests](transcripts/part1_tests.txt) |

## Scope and remaining verification

The active KB is now the supplied handwritten policy set. [The import and database checks](transcripts/handwritten_verification.txt) confirm that its bytes match the originals, the dataset/query labels/chunking/top-k/selection rule remain unchanged, and all stored chunks belong to this corpus. Calibration and deployment were refreshed for the new corpus. The earlier [invariant checks](transcripts/part1_invariants.txt) describe the previous evidence-only session and are historical.

Only Part 1 is implemented. CrewAI orchestration, appointment lookup, memory, structured crew output, guardrails, API deployment, judge evaluation, Autogen review, governance, and caching remain for their requested sessions. The escalation formula and later-Part evidence therefore are not claimed here.

No dataset design choices changed. The dependency installation required initial network access; the exact embedding checkpoint was already cached, so a fresh download through `scripts/download_model.py` was not executed. Loading that checkpoint and the complete RAG workflow were verified offline. Chroma emits a non-failing legacy embedding-configuration deprecation warning in the saved unit-test output; explicit embeddings and disabled automatic embedding remain verified.

Public GitHub repository: [Practo Domain Support Agent](https://github.com/manishjsr14-cloud/practo-domain-support-agent). The KB prose is the student's supplied handwritten policy text, imported without rewriting. Results establish the requested small Part 1 demonstration, with the similarity-only limitations described above.
