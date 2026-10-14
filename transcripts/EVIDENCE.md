# Current Part 1 evidence

This index links the current measured outputs for the supplied handwritten policy corpus. Earlier checkpoints and unsuccessful runs are preserved in [history](history/README.md). Historical transcripts are excluded from indexing.

| Area | Current evidence |
|---|---|
| Task 1: dataset | [Generator output](task1_dataset.txt), [generated records](../data/appointments.json) |
| Task 2: knowledge base | [Topic, sentence-count, and fee validation](task2_knowledge_base.txt) |
| Task 3: chunking and indexing | [Model, collections, upserts, and sample retrieval](task3_indexing.txt) |
| Task 4: calibration and generation | [Calibration observations](task4_calibration.txt), [demo](task4_demo.txt), [answers and retrieved context](task4_demo.json) |
| Task 4: unsupported nearby topics | [Stress scores and actual decisions](task4_near_topic.txt), [structured stress report](task4_near_topic.json) |
| Task 5: custom development retrieval | [Document-level arithmetic and recommendation](task5_evaluation.txt), [structured development metrics](task5_evaluation.json) |
| Plain-cosine baseline and frozen held-out comparison | [Raw/custom comparison](retrieval_comparison.txt), [structured results](retrieval_comparison.json), [premeasurement input manifest](retrieval_comparison_freeze.json), [freeze hashes](retrieval_comparison_freeze.txt), [comparison checks](retrieval_comparison_tests.txt) |
| Threshold limitation | [Current-corpus threshold separation check](threshold_limit.txt) |
| Complete tests | [Current full suite](part1_tests.txt) |
| Offline operation | [Fresh-store offline verification](part1_offline.txt) |
| Dependencies | [Exact tested environment](dependencies.txt), [dependency consistency](dependency_check.txt) |
| Handwritten policy provenance | [Import hashes](kb_import.txt), [import manifest](kb_import.json), [Git content verification](kb_git_verification.txt) |
| Active database replacement | [Old database removal and rebuild](kb_database_rebuild.txt), [source and stored-chunk verification](handwritten_verification.txt) |
| CrewAI compatibility preflight | [Summary](crewai_preflight_summary.json), [combined resolver failure](crewai_preflight_combined_install.txt), [standalone dependency check](crewai_preflight_pip_check.txt), [standalone imports](crewai_preflight_import.txt), [pin comparison](crewai_preflight_pin_comparison.json), [reproduction script](../scripts/check_crewai_compatibility.py) |

The stress report preserves unsupported queries that the similarity gate accepts. Test and offline passes do not establish general answer completeness, medical safety, or concurrent rebuild reliability.

The CrewAI preflight is a separate compatibility check. Combining CrewAI with the exact Part 1 dependency pins failed; dependency and import checks passed in an isolated standalone environment. No Part 2 implementation, model request, or crew kickoff was performed.
