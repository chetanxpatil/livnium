# Shared Working-State Handoff Between AI Agents

Date: 2026-11-01  
Idea proposed by: Chetan Patil  
Status: Proposed LIVNIUM research direction; no implementation or experimental result claimed.

## Original question

Chetan asked why an agent should convert its understanding into text for another agent to reconstruct, rather than transfer the relevant state directly. He initially described this as moving the agent's "weights."

## Clarified hypothesis

Weights are learned model parameters, normally fixed during inference. They are distinct from the context, activations, and task memory that change while an agent works. The intended handoff concerns a compatible internal task representation or shared working state, rather than copying weights or retraining the receiving agent.

Hypothesis: when agents have a defined compatibility contract, transferring selected task state may preserve useful information and reduce repeated encoding and reconstruction through prose. A future compatible latent-state interface could test a stronger version of this idea. Compatibility and usefulness must be demonstrated; neither follows from calling something a shared representation.

No hidden model internals are accessible in this session. Direct activation transfer has not been implemented or tested here. This note records a research proposal, makes no priority claim, and offers no evidence about consciousness.

## Potential benefits

- Less repeated text encoding and reconstruction of an already established task context.
- Less loss of constraints, evidence relationships, intermediate artifacts, and unresolved questions across handoffs.
- Lower coordination latency, token use, or compute overhead if state preparation, transport, and ingestion are cheaper than rebuilding context.

These are predictions, not results. Text remains useful for human inspection, explanation, interoperability, and recovery. Latent transfer may cost more or perform worse on some tasks.

## Shared task-state alternative for present tools

A practical first experiment can use an implementation-independent, versioned task-state record through a file, database, or API. Suggested fields are task ID, schema version, revision and parent revision, goal, constraints, evidence references, artifacts and hashes, decisions with provenance, uncertainties, pending work, owner, and permitted readers.

An illustrative contract is `read(task_id, revision)` followed by `propose_update(task_id, expected_revision, changes, provenance)`. A coordinator validates the schema, permissions, and revision before accepting an update and recording a receipt. The receiver checks sources and tool authorization independently; a state record does not grant authority. Keep a readable summary and a recoverable prior revision.

This transfers explicit task memory using available tools. JSON may itself be serialized as text; this approach is not direct activation transfer and does not imply access to a model's hidden reasoning or internal understanding.

## Open problems

- **Representation compatibility and cross-model alignment:** define state meaning, model/version identity, ordering, dimensions, and any adapters; test whether the receiver can use the information correctly.
- **Safety and trust boundaries:** imported state may contain malicious instructions or mistaken assumptions. Preserve source trust and enforce the receiver's permissions.
- **Privacy:** identify which task information may cross sessions, users, or machines; minimize payloads and support access revocation and deletion.
- **Auditability:** retain evidence links, readable explanations, integrity checks, and handoff receipts even when a payload is opaque.
- **Context ownership:** define who may read or mutate state, resolve concurrent edits, and prevent one task's state from contaminating another.
- **Hardware and memory cost:** account for payload size, device transfers, storage, network bandwidth, adapter compute, and working memory; large state transfers may erase any savings.

## Falsifiable experiment plan

1. Select tasks that require a handoff, such as continuing a constrained code change or completing an evidence-backed analysis. Separate development tasks from held-out evaluation tasks. Fix source information, producer/receiver models, tool permissions, hardware, task budgets, and stopping rules; preregister seeds, repeats, quality tolerances, and a minimum useful cost or latency improvement before evaluation.
2. Compare **A: text-only handoff**, using a strong readable summary with evidence and artifact references, against **B: structured shared state**, using the explicit contract above. Match the underlying information and artifact access; vary the handoff interface. Count the full cost of preparing and consuming either representation.
3. Define **C: hypothetical compatible latent-state interface** as a future arm only. Run it only if an explicitly accessible, validated interface exists for the chosen models. Document its representation and adapters, count adapter preparation/training separately and in amortized totals, and hold other conditions fixed. Otherwise report C as unavailable, with no inferred performance.
4. Measure end-to-end and handoff latency; input/output tokens; total compute cost and memory/transfer overhead; task accuracy against independent checks; error propagation over successive handoffs; and provenance completeness and recovery from a missing, corrupt, or stale state. Inject controlled incorrect facts and untrusted instructions to measure detection and propagation without exposing private data.
5. The prediction is supported on a tested setup only if a state arm improves the preregistered latency or total-cost target against A while meeting accuracy, provenance, recovery, and safety tolerances. Failure to meet those criteria, or gains disappearing after preparation/transport costs, rejects that prediction for the setup. Report regressions, variation across repeats, and task-specific tradeoffs; do not generalize to all models or tasks.

## Current evidence

Concept and experiment plan only. No experiments were run, no measurements are reported, and no direct latent or activation handoff is claimed. The next concrete step would be the A-versus-B comparison; C remains conditional on a real compatible interface.
