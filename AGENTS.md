# AGENTS.md — Practo Domain Support Agent (CrewAI Capstone)

Track: **Practo (Healthcare)**. Total marks: 100. Duration: 14 days.
Deliverable: ONE public GitHub repository containing the dataset, RAG core, CrewAI crew, Autogen review stage, and FastAPI deployment.

This file is the single source of truth for coding agents working in this repo. Read it fully before writing any code.

---

## 0. Rules for the coding agent (read first)

1. **Scope control.** Work only on the Part/Task the user names in the current session. Do not start later Parts unprompted. There is no default Part: Part 1 is complete, and implementation of Part 2 requires an explicit request. Dependency preflight and changes to the future task specification do not authorize Part 2 implementation.
2. **Plan first.** Before coding, show the planned files and any open choices, then proceed.
3. **No invented numbers.** Every number in the README, transcripts, or comments (similarity scores, precision/recall, percentiles, thresholds, eval scores) must come from code you actually ran. Save real output to `transcripts/`. If something could not be run, say so; never write plausible-looking values.
4. **Offline and keyless.** Everything must run with zero API keys, zero paid accounts, and zero network access at run time (after dependencies and the embedding model are downloaded). The graded mode is `MOCK_LLM`. A real LLM may optionally be wired behind an env flag, but every criterion must pass under `MOCK_LLM` alone.
5. **Fabricated data only.** Never use real patient names, diagnoses, insurance IDs, or phone numbers. All examples are fake.
6. **No binary deliverables.** No images, screenshots, PDFs, slide decks, video, or audio. Everything is code or text in the repo.
7. **Never hand-edit generated data** to hit a target (e.g. the follow-up percentage). Change seed or weights and regenerate.
8. **Commit after each task** with a clear message. Add or update tests with each task.
9. **Pin versions** in `requirements.txt`. CrewAI and Autogen APIs change often; check the installed version's docs before using an API from memory.
10. **Set `CREWAI_DISABLE_TELEMETRY=true`** (or `OTEL_SDK_DISABLED=true`) before any `crew.kickoff()` call, in code, scripts, and docs, and confirm it in the README.
11. **Report deviations.** At the end of each session, list anything unverified, skipped, or different from this brief.
12. **Originality.** Dataset, knowledge base, code, and analysis must be the student's own work. Keep code readable and commented so the student can explain every design choice (escalation formula, thresholds, risk classification).

---

## 1. Scenario

Practo's patient-experience team wants an agent that answers clinic-policy questions and checks a specific appointment's status, giving patients instant, accurate answers about scheduling, fees, and follow-ups. Build it to a standard that could pass a production governance review, not just a happy-path demo.

The agent:
- answers policy questions from a self-written knowledge base,
- looks up appointment records from a self-designed dataset,
- remembers a conversation,
- is guarded against misuse,
- has its answers reviewed by a second independent agent team (Autogen) before reaching the user,
- operates under an explicit governance policy,
- is orchestrated with CrewAI, deployed with FastAPI, and evaluated end to end.

### Vocabularies

Categories (use every value at least once): `General Medicine`, `Cardiology`, `Dermatology`, `Pediatrics`, `Orthopedics`.

Statuses (use every value at least once): `Scheduled`, `Completed`, `Cancelled`, `No-Show`, `Rescheduled`.

Required knowledge-base topics (>= 12 documents, 2-5 sentences each):
1. appointment-booking policy
2. cancellation/rescheduling window
3. consultation-fee structure by specialty
4. insurance-claim process
5. prescription-refill policy
6. lab-test turnaround times
7. telemedicine eligibility
8. emergency-visit protocol
9. patient-data privacy policy
10. follow-up-visit discount policy
11. second-opinion process
12. home-visit eligibility

PII handling:
- **Contact number** is fixed-format and MUST be masked by the input-side guardrail (Task 10) and masked in logs (Task 12).
- Patient name, diagnosis/condition, and insurance ID are free text or have no universal format; masking them is out of scope under a keyless `MOCK_LLM` masker. Use only fabricated examples for these.

---

## 2. Fixed dataset design choices (student's choices; use exactly these unless told otherwise)

- Seed: `42`
- Number of records: `50`
- Category weights: General Medicine 0.30, Cardiology 0.15, Dermatology 0.20, Pediatrics 0.20, Orthopedics 0.15
- Status weights: Completed 0.45, Scheduled 0.25, Cancelled 0.12, Rescheduled 0.10, No-Show 0.08
- `consultation_fee_inr`: uniform draw rounded to the nearest 50 from a per-category range:
  - General Medicine 300-600
  - Pediatrics 400-800
  - Dermatology 500-1000
  - Orthopedics 600-1200
  - Cardiology 800-1500
  - Overall range: 300-1500 INR
- Fee justification (use this sentence in the README): "These ranges reflect typical Indian outpatient consultation fees, with general practice at the low end and cardiology at the high end because of specialist training and diagnostic overhead."
- `days_since_created`: uniform integer 0-30
- `follow_up_required`: Bernoulli with p = 0.20 per record
- Coverage guarantee: first assign one record to every category and every status, then fill the remainder from the weighted draws.
- If the follow-up percentage is not within 10-30%, change the seed or weights and regenerate. If the seed changes, update this section and the README.
- The handwritten consultation-fee KB supersedes the earlier range-based policy: standard new-booking fees are General Medicine 500, Cardiology 1200, Dermatology 800, Pediatrics 600, and Orthopedics 1000 INR. These lie within the unchanged dataset ranges; an existing appointment's recorded fee remains authoritative.

---

## 3. Part 1 — Dataset Design and RAG Core (30 marks)

**Task 1: Dataset.** Write `dataset.py`, a seeded deterministic generator producing an `APPOINTMENTS` list of >= 40 records. Fields: `record_id`, `category`, `status`, `consultation_fee_inr`, `days_since_created` (int 0-30), `follow_up_required` (bool). Running `python dataset.py` prints: count per category (every category >= 3), count per status (every status >= 1), and the percentage with `follow_up_required=True` (must be between 10% and 30%). Save output to `transcripts/`.

**Task 2: Knowledge base.** Author >= 12 documents (2-5 sentences each, own words) covering every required topic. One file per document under `kb/`.

**Task 3: Chunking and indexing.**
- Strategy A: fixed-size with overlap (size and overlap as named constants).
- Strategy B: sentence-based chunks.
- Embed every chunk with a free local SentenceTransformers model (use `all-MiniLM-L6-v2`).
- Index each strategy in its OWN ChromaDB collection using `collection.upsert()`.
- Store the parent document id in every chunk's metadata (needed for Task 5).
- Use cosine distance and be explicit about converting distance to similarity.

**Task 4: Grounded generation.**
- Retrieve top-k chunks (from either collection) and generate an answer using ONLY the retrieved context.
- Under `MOCK_LLM`, retrieval similarity is the only groundedness signal.
- **Calibrate the "I don't know" threshold empirically** in `scripts/calibrate_threshold.py`: measure top-1 cosine similarity for >= 3 in-scope and >= 2 deliberately out-of-scope queries, print all values, and set the threshold between the two observed clusters. Do NOT use untested presets (0.5 / 0.6 / 0.7).
- State measured values and the chosen threshold in the README.
- Demonstrate >= 5 in-scope queries plus 1 out-of-scope query that must trigger the fallback.

**Task 5: Evaluate both chunking strategies.**
- Use the same >= 5 queries as Task 4, each with a hand-labelled set of relevant parent documents.
- For EACH collection: map retrieved chunks to parent documents, dedupe, then compute precision and recall per query with visible arithmetic (e.g. "2 relevant of 3 retrieved = 0.667").
- Print per-query results and averages for both collections side by side.
- Write 2-3 sentences recommending which strategy to deploy, citing the actual numbers.
- Preserve a plain cosine top-k baseline alongside any custom retrieval comparison. Label results on queries used during retrieval development as in-sample, and report a separately frozen held-out check without retuning on its results. Keep the demo/evaluation query set unchanged; held-out queries are a separate supplement and must not select deployment or calibration settings.

---

## 4. Part 2 — CrewAI Orchestration, Tools, Memory, Guardrails (30 marks)

Use Part 1's RAG core (the recommended chunking strategy's collection) as a fixed input.

**Task 6: Appointment tool with designed escalation score.**
Implement `check_appointment_status(record_id: str) -> dict` returning `status`, `consultation_fee_inr`, and a **designed** `escalation_score` in `[0, 1]` combining `follow_up_required` with a normalized recency signal from `days_since_created`. State the exact formula and an escalation threshold justified by the dataset's own distribution (e.g. a percentile of `days_since_created`). Not a bare boolean OR.

**Task 7: CrewAI crew.** >= 3 agents: Retrieval Agent (RAG tool), Lookup Agent (`check_appointment_status` tool), Response Composer (combines outputs into one draft). Run via `.kickoff()`. Demonstrate the RAG tool and the lookup tool each actually invoked on different sample queries.

**Task 8: Session memory.** LangChain session memory (e.g. `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`). The `LangChainDeprecationWarning` is expected and need not be silenced. Provide one multi-turn transcript showing state carried across turns, and a SEPARATE fresh-conversation transcript showing state absent. In-process only.

**Task 9: Structured output.** A Pydantic `BaseModel` that every crew response must conform to (`response_format`); validate every response in code.

**Task 10: Guardrails.**
- Input side: PII masking (contact number) and prompt-injection detection.
- Output side: groundedness check that refuses to answer when retrieved context does not support the question. Use an independent question-support signal in addition to calibrated similarity: distinctive location, date/day, age, service, and other qualifiers (for example, "Sundays" or "Mumbai") must be supported by the retrieved passages. Missing qualifiers require refusal or clarification. Word overlap alone is insufficient: preserve negation, eligibility restrictions, and the relationship between the question and the cited answer.
- Demonstrate each guardrail firing on one deliberate test case.
- Include the unsupported Sunday-hours and Mumbai-waiting-time cases, and demonstrate refusal even when their similarity exceeds the calibrated threshold. This is a Part 2 requirement, not a claim about the Part 1 similarity-only generator.

---

## 5. Part 3 — Evaluation, Observability, FastAPI (20 marks)

Use Part 2's crew as a fixed input.

**Task 11: FastAPI.** >= 2 HTTP endpoints (e.g. `POST /ask`, `POST /add-document`) with Pydantic request/response models, plus one WebSocket endpoint (`@app.websocket`) for multi-turn chat. Catch `WebSocketDisconnect` and keep the server running for other clients.

**Task 12: Structured logging.** One JSON-Lines entry per request with a trace ID and timing. The logged request text MUST NOT contain the raw contact number; apply the same masking to logs as to what the model sees.

**Task 13: Evaluation at scale.** LLM-as-judge prompt (under `MOCK_LLM`). Test set of **15 queries**: >= 1 per required KB topic plus >= 2 deliberately out-of-scope or edge-case queries. Score Accuracy, Grounding, Completeness, and Safety for every query; report all four per query and the four averages across all 15.

---

## 6. Part 4 — Resilience and Governance (20 marks)

Use Part 2's crew as a fixed input.

**Task 14: Autogen review stage.**
- 2-agent `RoundRobinGroupChat` (e.g. Policy-Compliance-Reviewer and Final-Editor), bounded with `max_turns=2` (the parameter is `max_turns`, not `max_iterations`). If using `MaxMessageTermination` instead, use `MaxMessageTermination(3)`, since the initiating task message counts as message 1.
- Final-Editor has Pydantic structured output via `output_content_type=<VerdictModel>`. The Team must ALSO be built with `custom_message_types=[StructuredMessage[VerdictModel]]`, or the run crashes with `ValueError: Message type ... is not registered`.
- Verdict model fields: `approved: bool`, `final_answer: str`, `reason: str`.
- Input: the CrewAI Composer's draft plus the original retrieved context.
- Demonstrate >= 2 queries: one where the draft is approved unchanged, one where it is revised (e.g. catches a deliberately injected ungrounded claim).

**Task 15: Four-layer governance.**
- Application layer, least autonomy: only the Lookup Agent may call `check_appointment_status`. Demonstrate the restriction (wiring to another agent is blocked or never wired) with a one-paragraph explanation of the guard.
- Risk classification (Low/Medium/High): classify the system with a one-paragraph justification. (Medical data falls under High.)
- Runtime layer: per-request token/cost budget cap; demonstrate a deliberately oversized request being rejected rather than silently exceeding the budget.

**Task 16: Response caching.** In-memory cache keyed by normalized query text for the grounded-generation step. Demonstrate a repeated identical query producing a cache hit that avoids a redundant LLM/tool call, with before/after evidence (call counter or timing).

---

## 7. MOCK_LLM implementation notes (silent-failure pitfalls)

Implement `MOCK_LLM` for CrewAI by extending `crewai.llms.base_llm.BaseLLM` (CrewAI's documented extension point for a non-`litellm` LLM). Do not try to intercept calls externally.

Two known silent bugs (no crash, just wrong output). Write a regression test for each:

1. **Template "Observation:" trap.** CrewAI's built-in ReAct system-prompt template contains the example text `"Observation: the result of the action"`. A parser that searches the whole conversation for `"Observation:"` will match the template on the first call, before any tool has run, and return placeholder text as the final answer. Parse the model's own generated text, not the system-prompt template.
2. **Tool dispatch by name.** Do not dispatch tool arguments by matching the tool's name (e.g. checking whether `"lookup"` is a substring); a tool named `rag_lookup` would be misclassified. Dispatch off the tool's own declared argument schema.

Telemetry: set `CREWAI_DISABLE_TELEMETRY=true` before `crew.kickoff()`.

The judge for evaluation, grounded generation, the CrewAI crew, and the Autogen review team must all work under `MOCK_LLM` with deterministic output.

---

## 8. Repository layout

```
AGENTS.md
README.md
requirements.txt
dataset.py
kb/                       one file per knowledge-base document
rag/
  chunking.py             fixed-overlap and sentence chunkers
  index.py                embeddings + two ChromaDB collections
  generate.py             retrieval, threshold, grounded generation
  evaluate.py             document-level precision/recall
crew/                     (Part 2) tools, agents, crew, schema, guardrails, memory
review/                   (Part 4) Autogen review stage
governance/               (Part 4) least autonomy, risk, budget cap, cache
api/                      (Part 3) FastAPI app, logging
eval/                     (Part 3) 15-query evaluation
scripts/
  calibrate_threshold.py
  run_part1_demo.py
transcripts/              real saved outputs for every task
tests/
```

---

## 9. README.md requirements

The README must:
- State at the top that the **Practo (Healthcare)** track was completed.
- List the exact dataset design choices (seed, category/status weights, amount range) so the grader can reproduce the dataset.
- Include the one-sentence fee-range justification.
- Report measured calibration values and the chosen similarity threshold.
- Show both precision/recall tables and the chunking recommendation.
- Confirm everything runs under `MOCK_LLM` with zero API keys.
- Confirm `CREWAI_DISABLE_TELEMETRY=true` was set.
- State the escalation-score formula and threshold justification (Part 2).
- Link to transcripts demonstrating every task.
- Quote only numbers that appear in saved transcripts.

---

## 10. Acceptance checklist

**Part 1**
- [x] `dataset.py` generates >= 40 records; category counts >= 3 each; all statuses present; follow-up % within 10-30%; choices stated in README
- [x] >= 12 KB documents covering every required topic
- [x] Both chunking strategies indexed into two separate ChromaDB collections; both retrieve sensibly
- [x] Grounded generation demonstrated on >= 5 in-scope queries + 1 out-of-scope fallback
- [x] Precision/recall for BOTH collections on the same queries with visible arithmetic + numbers-cited recommendation

**Part 2**
- [ ] `check_appointment_status` works; designed, justified `escalation_score`
- [ ] Crew has >= 3 agents; RAG tool and lookup tool each invoked via `.kickoff()` on different queries
- [ ] Multi-turn memory transcript + separate fresh-conversation transcript
- [ ] Every response validates against a Pydantic schema
- [ ] PII masking, injection detection, and groundedness guardrails each demonstrated firing

**Part 3**
- [ ] >= 2 HTTP endpoints + 1 WebSocket that survives a disconnect, with Pydantic models
- [ ] One JSON-Lines log entry per request with trace ID; no raw PII on disk
- [ ] 4 scores per query for all 15 queries + 4 averages under `MOCK_LLM`

**Part 4**
- [ ] Autogen review approves one draft unchanged and revises another, with structured Pydantic verdicts
- [ ] Least autonomy demonstrated; risk classification + justification; oversized request rejected by budget cap
- [ ] Cache hit demonstrated with before/after evidence

---

## 11. Session log (update as you go)

| Part | Status | Notes |
|---|---|---|
| Part 1 | complete; baseline and held-out evidence added | Current-corpus plain cosine baseline and unchanged custom retriever compared on the same development queries; separate held-out labels and source hashes frozen before measurement. Fixed-chunk held-out precision regression retained without retuning. First-person README, history archive, and Bash commands added. All 45 tests and fresh socket-blocked offline workflow passed, including comparison. KB, dataset, production retrieval, calibration and deployment unchanged. Similarity-only false acceptances remain; no later-Part implementation. |
| Part 2 | not implemented; dependency preflight complete | CrewAI 1.15.28 installed in a fresh standalone environment; pip check and imports passed. Combined installation with exact Part 1 pins failed on ChromaDB; metadata also conflicts with Pydantic and regex. Joint integration remains unverified; no kickoff or LLM request. Groundedness specification now requires an independent question-support signal. |
| Part 3 | not started | |
| Part 4 | not started | |
