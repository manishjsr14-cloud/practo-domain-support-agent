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
