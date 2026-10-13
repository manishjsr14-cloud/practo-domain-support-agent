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

Retrieval retains the original whole-query cosine winner as its first chunk. It scans each collection's candidates, ranks supporting evidence using both parent-document and chunk semantics, and gives additional weight to coverage of grammatical content phrases that are not already covered. For a question preceded by a narrative clause, it also embeds the explicit question clause; no query IDs, labelled parents, or clinic-topic routing are used. The final context remains **3 chunks**, and ranking scores are recorded separately from cosine similarities. A separate retrieval-policy fingerprint invalidates the deployment recommendation when evidence selection changes, without altering calibration identity.

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

Under `MOCK_LLM`, the original whole-query top-one cosine similarity remains the only acceptance signal: below the strategy's calibrated threshold, the generator returns “I don't know based on the available clinic-policy context.” Once accepted, it selects a primary parent semantically and includes a supporting parent when that parent best matches a question component and has a textual anchor for that component. This generic coverage heuristic selects excerpts; it is not a groundedness guardrail. The generator merges overlaps within each selected parent, extracts only retrieved text, and cites the contributing chunk IDs. Supporting excerpts are not individually rejected against the whole-query threshold, which was calibrated for the initial acceptance decision. It calls no remote LLM and uses no API keys or paid accounts.

[The demonstration](transcripts/task4_demo.txt) produces responses for **9** in-scope questions about booking, cancellation, specialty fees, laboratory reports, follow-up discounts, insurance claims, a follow-up paraphrase, cardiologist home visits, and video second opinions. Its separate brownie-recipe query scores **0.023517489** with the deployed sentence strategy and triggers the fallback. Producing a response establishes that the similarity gate accepts it; it does not establish that the response answers the question correctly. [Structured answers and original retrieved context](transcripts/task4_demo.json) record each selected parent excerpt and its contributing sources, and the answer is exactly the joined excerpts. [The original generation tests](transcripts/task4_tests.txt) and [current coverage and generation tests](transcripts/answer_coverage_tests.txt) check both strategies, the fallback, extraction boundaries, source spans, and stale calibration.

The reported failures are fixed in the current demo. q07 now cites the follow-up policy, including the **20 percent** discount within **7 calendar days** and the same-specialty/same-issue conditions. q08 includes the **INR 800–1500** cardiology range and the home-visit policy's separate travel quote before confirmation, without inventing a total price. q09 includes both the second-opinion process and telemedicine's routine-follow-up, clinician-confirmation, consent, and secure-video requirements; these excerpts do not promise that every second opinion is video-eligible.

This small calibration set contains unrelated negative queries. Similarity alone does not establish medical safety, detect prompt injection, or distinguish every unsupported question that shares clinic terminology; those later guardrails are outside this session's scope.

## Near-topic threshold stress test

```powershell
python scripts/stress_threshold.py
```

These fixed unsupported questions are a separate check and were never added to calibration. The table reports top-one cosine similarity from both collections and the actual deployed generator decision. Deployment uses `sentence`; the transcript also records each collection's decision against its own existing calibrated threshold.

| Unsupported question | Fixed overlap cosine | Sentence cosine | Deployed threshold | Deployed decision |
|---|---:|---:|---:|---|
| Can you diagnose my chest pain? | 0.245835125 | 0.316201031 | 0.345280051 | FALLBACK |
| Which medicine should I take for a fever? | 0.228224874 | 0.218418896 | 0.345280051 | FALLBACK |
| What are the clinic's opening hours on Sundays? | 0.508999348 | 0.529146969 | 0.345280051 | ANSWER |
| Which hospital in Mumbai has the shortest waiting time? | 0.403571725 | 0.416320801 | 0.345280051 | ANSWER |

The deployed generator accepts **2** unsupported queries and falls back on **2**. Both collections accept the Sunday-opening-hours and Mumbai-hospital-waiting-time questions above their respective thresholds. The resulting answers quote booking, emergency, and cancellation policies, which do not supply opening hours or comparative hospital waiting times. These are measured false acceptances of similarity-only groundedness, preserved in [the complete stress transcript](transcripts/task4_near_topic.txt) and [structured answers with retrieved context](transcripts/task4_near_topic.json).

The chest-pain diagnosis and fever-medicine questions fall back in this run. This observation does not establish a medical-safety guardrail. Part 2's groundedness guardrail will need to verify that context supports the actual question; passing a similarity threshold is insufficient. No threshold adjustments or keyword exceptions were introduced, and no Part 2 implementation was started.

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
| q01 | 01_appointment_booking | 01_appointment_booking, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling, 01_appointment_booking | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround, 03_consultation_fees | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 03_consultation_fees, 10_follow_up_discount | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 03_consultation_fees, 12_home_visits | 03_consultation_fees, 12_home_visits | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 07_telemedicine, 11_second_opinion | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.611111 | 1.000000 |

### Sentence chunks

| Query | Labelled parent documents | Retrieved parent documents (deduped) | Precision arithmetic | Recall arithmetic |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling, 01_appointment_booking | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround, 10_follow_up_discount | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims, 11_second_opinion | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 11_second_opinion, 10_follow_up_discount | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 03_consultation_fees, 12_home_visits | 03_consultation_fees, 12_home_visits | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 07_telemedicine, 11_second_opinion | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.666667 | 1.000000 |

Across 9 hand-labelled queries, fixed_overlap achieved macro precision 0.611111 and macro recall 1.000000, while sentence achieved macro precision 0.666667 and macro recall 1.000000. Deploy sentence because it has equal macro recall and higher macro precision. This small policy sample supports a local choice and does not establish universal retrieval quality.

[The full evaluation transcript](transcripts/task5_evaluation.txt) includes both strategies side by side, retrieved chunk IDs, deduplicated parent IDs, labels, and arithmetic. [The machine-readable report](transcripts/task5_evaluation.json) and [deployment choice](rag/deployment.json) were generated by the evaluator. Grounded generation now defaults to `practo_sentence` and its existing measured threshold **0.345280051**; changing the corpus or retrieval policy invalidates the saved deployment choice.

The unchanged labels now give q07 precision **0.500000** and recall **1.000000** for both strategies: the follow-up parent is retrieved alongside another parent, and the answer cites the follow-up parent only. q08 and q09 each give both precision and recall **1.000000** for both strategies. Sentence q03 now retrieves only its labelled fee parent, giving precision **1.000000**; the other original queries retain precision **0.500000** because another parent is also retrieved.

The labels were fixed before measurement and retain **7** single-parent queries and **2** multi-parent queries. Query wording and labels, chunk sizes, top-k, calibration questions and file, and the recall-then-precision-then-sentence selection rule remain unchanged. The fixes change evidence selection and extraction, rather than those evaluation controls. Retrieval metrics measure parent coverage, while regression tests separately check that q07 cites the discount, q08 includes fees and travel-quote terms, and q09 includes the second-opinion process and telemedicine limits under both strategies.

[The preserved pre-fix evaluation](transcripts/before_answer_fix/task5_evaluation.txt), [pre-fix demo](transcripts/before_answer_fix/task4_demo.txt), and [failing regression run](transcripts/before_answer_fix/regressions.txt) retain the original miss and incomplete answers for comparison. The fixes also pass additional paraphrases of repeat-visit pricing, cardiologist home-visit pricing including travel, and video second opinions; they do not establish general answer completeness.

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

[The full suite](transcripts/part1_tests.txt) passed **31 tests**. Before source changes, [the refreshed baseline](transcripts/refresh_tests.txt) passed all **21** existing tests with no failures. [The offline verification](transcripts/part1_offline.txt) clears API-key environment variables, blocks Python sockets and DNS before importing frameworks, rebuilds both collections in a fresh temporary store, recalibrates, evaluates, demonstrates the answers and fallback, and runs the near-topic stress check using that fresh temporary calibration. It recorded **0** outbound Python network attempts. Those runtime commands also ran successfully under the execution environment's network restrictions.

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
| Near-topic stress | [Scores and actual decisions](transcripts/task4_near_topic.txt), [structured context](transcripts/task4_near_topic.json), [unchanged KB and calibration checks](transcripts/part1_invariants.txt) |
| Environment | [Exact dependencies](transcripts/dependencies.txt), [dependency check](transcripts/dependency_check.txt), [offline run](transcripts/part1_offline.txt), [complete tests](transcripts/part1_tests.txt) |

## Scope and remaining verification

The evidence refresh used the saved KB files currently on disk. Their corpus fingerprint matches the preceding run; unsaved or externally stored rewrites were unavailable for verification. [The invariant checks](transcripts/part1_invariants.txt) confirm the KB files, original query labels, chunking, top-k, selection rule, calibration questions, and calibration file remained unchanged during this evidence update and the retrieval/answer fixes.

Only Part 1 is implemented. CrewAI orchestration, appointment lookup, memory, structured crew output, guardrails, API deployment, judge evaluation, Autogen review, governance, and caching remain for their requested sessions. The escalation formula and later-Part evidence therefore are not claimed here.

No dataset design choices changed. The dependency installation required initial network access; the exact embedding checkpoint was already cached, so a fresh download through `scripts/download_model.py` was not executed. Loading that checkpoint and the complete RAG workflow were verified offline. Chroma emits a non-failing legacy embedding-configuration deprecation warning in the saved unit-test output; explicit embeddings and disabled automatic embedding remain verified.

Public GitHub repository: [Practo Domain Support Agent](https://github.com/manishjsr14-cloud/practo-domain-support-agent). The KB prose is an original assisted draft that the student should review and rewrite before submission. Results establish the requested small Part 1 demonstration, with the similarity-only limitations described above.
