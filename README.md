# Practo (Healthcare) — Part 1 completed

I built the dataset and local RAG core for my Practo (Healthcare) capstone. My dataset has **50** fabricated appointments, with **13 marked as requiring follow-up (26.00%)**, and my handwritten KB has **12** fictional clinic policies. Everything in Part 1 runs under deterministic `MOCK_LLM` with zero API keys after setup.

On the **9** development questions, plain cosine retrieval gives precision **0.703704** for fixed chunks and **0.759259** for sentence chunks; custom ranking gives **0.888889** and **0.944444**. Recall is **1.000000** throughout. I developed the custom ranking using these questions, so its scores are in-sample. On **4** newly frozen questions, custom ranking helps sentence precision but reduces fixed-chunk precision; I kept that regression instead of retuning.

What surprised me most was that an unsupported Sunday-hours question scores higher than a legitimate insurance question. A higher similarity threshold cannot separate them. My answers quote retrieved policy text, but similarity does not prove that the text answers the question or establish medical safety. Part 2 is not implemented.

## Dataset and handwritten policies

I use seed **42**, generate **50** records, and reserve coverage for every category and status before drawing the remaining rows. Fees are continuous uniform draws rounded to the nearest **50 INR**; `days_since_created` is a uniform integer from **0–30**, and each follow-up flag is a Bernoulli draw with **p = 0.20**.

| Category | Weight | Dataset fee range INR | Count | KB standard new-booking fee INR |
|---|---:|---:|---:|---:|
| General Medicine | 0.30 | 300–600 | 12 | 500 |
| Cardiology | 0.15 | 800–1500 | 8 | 1200 |
| Dermatology | 0.20 | 500–1000 | 11 | 800 |
| Pediatrics | 0.20 | 400–800 | 8 | 600 |
| Orthopedics | 0.15 | 600–1200 | 11 | 1000 |

| Status | Weight | Count |
|---|---:|---:|
| Completed | 0.45 | 20 |
| Scheduled | 0.25 | 15 |
| Cancelled | 0.12 | 8 |
| Rescheduled | 0.10 | 4 |
| No-Show | 0.08 | 3 |

The overall designed fee range is **300–1500 INR**. These ranges reflect typical Indian outpatient consultation fees, with general practice at the low end and cardiology at the high end because of specialist training and diagnostic overhead.

The measured **26.00%** follow-up rate is within the required **10–30%** band; I did not edit individual records. The KB's fixed standard prices apply to new bookings; an existing appointment's recorded fee remains authoritative. My **12** policies each have **5** sentences and cover booking, cancellation/rescheduling, fees, insurance, refills, labs, telemedicine, emergencies, privacy, follow-ups, second opinions, and home visits. They were imported verbatim, and the old database was deleted and rebuilt exclusively from them. [Dataset, KB validation, and provenance evidence](transcripts/EVIDENCE.md).

## Run locally

The tested environment is **Python 3.12.10** on Windows. Dependency/model setup needs network access once; subsequent Part 1 commands use the cached pinned `sentence-transformers/all-MiniLM-L6-v2` checkpoint. Model revision and exact dependency versions are recorded in the evidence index.

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\Activate.ps1
```

Bash setup (Linux/macOS):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_model.py
source .venv/bin/activate
```

After activation, the same commands work in PowerShell or Bash:

```bash
python dataset.py --export data/appointments.json
python -m scripts.validate_kb
python -m rag.index
python scripts/calibrate_threshold.py
python -m rag.evaluate
python scripts/run_part1_demo.py
python scripts/stress_threshold.py
python scripts/compare_retrieval.py
python -m unittest discover -s tests -v
python scripts/verify_offline.py
```

Run from the repository root. If PowerShell activation is blocked, replace `python` with `.\.venv\Scripts\python.exe`; Bash can use `.venv/bin/python`. Rerun indexing, calibration, and evaluation after changing policies. The frozen comparison belongs to this exact KB and retrieval revision; changed inputs require a new experimental protocol, not silent replacement of its labels or freeze manifest. Bash instructions have not been executed on Linux/macOS.

The runtime loads only local model files and sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `ANONYMIZED_TELEMETRY=False`, `CREWAI_DISABLE_TELEMETRY=true`, and `OTEL_SDK_DISABLED=true` before framework imports. Binary model, index, and environment files stay outside Git in ignored directories.

## Retrieval and calibration

Fixed chunks use **320 characters** with **80 characters** of overlap; sentence chunks group **2 sentences**. The separate cosine collections contain **39** fixed chunks and **36** sentence chunks. Both use explicit MiniLM embeddings and `collection.upsert()`, with parent IDs and source offsets. Similarity is **1 − Chroma cosine distance**, not a probability.

My deployed retriever keeps the closest cosine match first. It chooses supporting chunks using the whole policy and the question, favors complete sentences, and reserves space for separately worded parts of a question. Supporting policies need a matching distinctive word; the final context stays at **3 chunks**. The generator quotes this context, while the original top-one cosine threshold alone controls fallback.

These coverage heuristics are not an independent support check. The calibration queries were declared separately from the demo queries; I choose `(max_out + min_in) / 2` and reject overlapping clusters rather than use a preset. [Full calibration output](transcripts/task4_calibration.txt).

| Scope | Calibration query | Fixed cosine | Sentence cosine |
|---|---|---:|---:|
| In | How can I reserve a clinic appointment? | 0.734329104 | 0.746177197 |
| In | How much does a Cardiology consultation cost? | 0.674873888 | 0.708590269 |
| In | What documents are needed for an insurance claim? | 0.440953672 | 0.486387312 |
| In | Who is allowed to authorize a prescription refill? | 0.605015993 | 0.600208402 |
| Out | How do I repair a punctured bicycle tyre? | 0.145102620 | 0.152278662 |
| Out | What is the orbital period of Neptune? | 0.143519402 | 0.147324979 |
| Out | Write a sorting algorithm in Rust. | 0.075274765 | 0.082972288 |

| Collection | Maximum out-of-scope | Minimum in-scope | Threshold |
|---|---:|---:|---:|
| fixed_overlap | 0.145102620 | 0.440953672 | 0.293028146 |
| sentence | 0.152278662 | 0.486387312 | 0.319332987 |

The demo and deployment evaluation use exactly the same **9** labelled questions. q07 quotes the same-clinician follow-up discount, q08 quotes the General Medicine-only home-visit restriction rather than offering a cardiologist visit, and q09 includes second-opinion arrangements plus telemedicine restrictions. The separate brownie-recipe query scores **0.054437876** under sentence retrieval and falls back. [Demo answers, citations, and retrieved context](transcripts/task4_demo.json).

## Plain baseline and custom ranking

The plain baseline returns the closest **3** chunks directly from each Chroma collection, without parent, question-focus, or component reranking. Both methods use the same current KB, embeddings, chunk sizes, queries, labels, and parent-level metric arithmetic: deduplicate parents, then compute precision as relevant retrieved/all retrieved and recall as relevant retrieved/all labelled relevant. Averages are macro averages across questions, without similarity filtering.

These are the **9 development questions used during ranking changes**, not an unbiased test set. The plain baseline is the chunking comparison requested in Task 5; the custom rows describe chunking under my deployed retriever. [Reproducible baseline/custom arithmetic and context](transcripts/retrieval_comparison.txt).

| Retrieval on development questions | Fixed precision | Fixed recall | Sentence precision | Sentence recall |
|---|---:|---:|---:|---:|
| Plain cosine | 0.703704 | 1.000000 | 0.759259 | 1.000000 |
| Custom ranking | 0.888889 | 1.000000 | 0.944444 | 1.000000 |

### Fixed chunks — plain baseline

| Query | Relevant parents | Retrieved parents, deduped | Precision | Recall |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling, 12_home_visits | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 10_follow_up_discount, 07_telemedicine | 1 relevant of 3 retrieved = 0.333333 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround, 04_insurance_claims | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 03_consultation_fees | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 10_follow_up_discount, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 03_consultation_fees, 12_home_visits | 03_consultation_fees, 12_home_visits | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 07_telemedicine, 11_second_opinion | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.703704 | 1.000000 |

### Sentence chunks — plain baseline

| Query | Relevant parents | Retrieved parents, deduped | Precision | Recall |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 07_telemedicine, 10_follow_up_discount | 1 relevant of 3 retrieved = 0.333333 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 03_consultation_fees | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 03_consultation_fees, 10_follow_up_discount, 12_home_visits | 1 relevant of 3 retrieved = 0.333333 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 03_consultation_fees, 12_home_visits | 03_consultation_fees, 12_home_visits, 04_insurance_claims | 2 relevant of 3 retrieved = 0.666667 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 07_telemedicine, 11_second_opinion | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.759259 | 1.000000 |

Sentence chunks have higher development precision under both methods, with equal recall, so the existing recall-then-precision-then-sentence rule selects sentence retrieval. The deployed custom strategy uses threshold **0.319332987**. These current-KB raw results already retrieve q07's labelled parent; the complete miss in older evidence belongs to the earlier KB. I did not mix those older measurements into this baseline.

## Frozen held-out check

I fixed these **4** new questions and their labels before running them, then saved hashes of the query file, KB, and production retrieval sources. I made no retrieval or label changes after seeing their results. The questions are separate from the demo, calibration, and deployment selection. [Frozen inputs and labels](transcripts/retrieval_comparison_freeze.json).

| ID | Question | Relevant parents |
|---|---|---|
| ho01 | My refill request has been submitted; does that mean I am guaranteed the same medicine? | 05_prescription_refills |
| ho02 | The clinic cancelled my appointment with hardly any notice. What refund can I expect, and when? | 02_cancellation_rescheduling |
| ho03 | Can a child use the video service from home instead of visiting the clinic? | 07_telemedicine |
| ho04 | Can I arrange a home appointment and use my discounted return visit for its travel charge? | 10_follow_up_discount, 12_home_visits |

| Retrieval on held-out questions | Fixed precision | Fixed recall | Sentence precision | Sentence recall |
|---|---:|---:|---:|---:|
| Plain cosine | 0.916667 | 1.000000 | 0.750000 | 1.000000 |
| Custom ranking | 0.875000 | 1.000000 | 0.875000 | 1.000000 |

Custom ranking **reduces fixed-chunk precision from 0.916667 to 0.875000**: the child/video question also retrieves an irrelevant home-visit parent. Sentence precision improves from **0.750000 to 0.875000**. This is a small hand-labelled sanity check; it measures parent retrieval, not answer completeness or independent clinical safety, and is no longer unseen after this run.

## Why raising the threshold cannot fix nearby unsupported questions

The sentence Sunday-hours score is **0.524604023**, above the legitimate insurance calibration score of **0.486387312**. A threshold high enough to reject Sunday-hours must also reject that weaker legitimate question. The same ordering occurs for fixed chunks. [Measured threshold-separation check](transcripts/threshold_limit.txt).

Deployment uses sentence retrieval and threshold **0.319332987**:

| Unsupported question | Fixed cosine | Sentence cosine | Deployed decision |
|---|---:|---:|---|
| Can you diagnose my chest pain? | 0.256390333 | 0.263177156 | FALLBACK |
| Which medicine should I take for a fever? | 0.272614956 | 0.202158451 | FALLBACK |
| What are the clinic's opening hours on Sundays? | 0.486783445 | 0.524604023 | ANSWER |
| Which hospital in Mumbai has the shortest waiting time? | 0.421420097 | 0.439948082 | ANSWER |

The generator falsely accepts **2** unsupported questions: Sunday opening hours and comparative Mumbai hospital waiting times. Their booking/lab excerpts do not answer those questions. These queries were not added to calibration, and the stress script leaves calibration unchanged.

Part 2 therefore needs another question-support signal beyond similarity. Its revised specification requires support for distinctive qualifiers such as “Sundays” and “Mumbai”, along with the question's relationships and eligibility restrictions; word overlap alone is insufficient. That guardrail is specified in `AGENTS.md` and is not implemented in Part 1.

## CrewAI dependency preflight

A fresh standalone environment installed **CrewAI 1.15.28**; `pip check` and imports of `Agent`, `Crew`, and `BaseLLM` passed with telemetry disabled. Installing the same release together with the exact Part 1 requirements failed: pip reported the ChromaDB conflict. Release metadata also conflicts with the Pydantic and regex pins:

| Package | Part 1 pin | CrewAI requirement |
|---|---|---|
| chromadb | 1.1.1 | ~=1.5.9 |
| pydantic | 2.14.1 | <2.13,>=2.11.9 |
| regex | 2026.9.29 | ~=2026.1.15 |

The standalone pass does not establish compatibility with the Part 1 stack. Before Part 2 implementation, I still need a tested joint dependency set; Part 1's pins and environment remain unchanged. No crew kickoff or LLM request was made. [Preflight results](transcripts/crewai_preflight_summary.json).

To repeat the online dependency check in newly created environments, choose an unused run name.

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/check_crewai_compatibility.py --version 1.15.28 --run-name crewai-check-new
```

Bash:

```bash
.venv/bin/python scripts/check_crewai_compatibility.py --version 1.15.28 --run-name crewai-check-new
```

## Verification and evidence

The current suite passed **45 tests**. The offline verifier blocks sockets and DNS before framework imports, rebuilds a fresh index, and runs calibration, evaluation, demo, stress, and the baseline/held-out comparison. It recorded **0** outbound Python network attempts. [Current tests](transcripts/part1_tests.txt) · [Offline run](transcripts/part1_offline.txt).

[Current evidence by task](transcripts/EVIDENCE.md) links the detailed outputs; [historical evidence](transcripts/history/README.md) contains earlier checkpoints, diagnostics, and failed runs. Historical results are excluded from indexing. A prior parallel rebuild hit a Chroma segment-reader error; concurrent rebuild reliability remains unverified. The model was already cached, so a fresh model download has not been exercised.

Only Part 1 is implemented. The Part 2 scope must be requested explicitly; its revised groundedness requirement and dependency preflight do not claim orchestration, tools, memory, guardrails, or later-Part functionality.

Public repository: [Practo Domain Support Agent](https://github.com/manishjsr14-cloud/practo-domain-support-agent).
