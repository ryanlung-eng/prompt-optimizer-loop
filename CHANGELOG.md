# Changelog

177 commits, 2026-06-30 → 2026-08-06. Grouped by what was being learned rather
than by release, because there were no releases — this was a measurement problem
being chased down.

The recurring theme is worth stating up front: **most apparent generation
failures turned out to be measurement failures.** A judge confidently reporting a
bug that isn't there and a generator producing a real one look identical in a
score. A large share of the work below is telling those two apart.

Full detail is in `git log`. Deleted files remain in history.

---

## Phase 1 — Get a number at all (Jun 30 – Jul 2)

Prompt-optimization loop against the Knowledge Assistant endpoint: synthetic
inputs → evaluate → judge → improve → repeat.

- n8n is unreachable from Databricks (network ACL), so the loop never reads or
  writes n8n. Prompts are pasted in and out by hand. This constraint shaped
  everything after it.
- Dropped the Anthropic dependency for Databricks-native endpoints.
- **Single-shot JSON replaced with simulated multi-turn conversations.** The real
  assistant is allowed to ask clarifying questions; forcing structured output
  measured something the product doesn't do.
- Native Databricks assessments removed — confirmed non-functional for this
  endpoint rather than assumed working.
- Approval routing corrected from optional to mandatory; the exact sub-workflow
  pattern taught directly in the prompt.

Most commits here are wire-format and crash fixes: request shape, null-valued
keys, `RetryError` unwrapping, rate limiting, JSON extraction from prose.

## Phase 2 — Stop trusting the judge (Jul 6 – Jul 20)

The judge was flagging correct workflows. Each fix replaces an opinion with
something checkable.

- **Deterministic structural validator, separate from the LLM judge** — the first
  real split between "does it parse" and "is it right".
- **Self-repair loop**: feed validation errors back instead of accepting broken
  JSON.
- **Schema-based check for hallucinated node parameters** (`check_params.js`),
  then a long tail of false positives against it: `resourceMapper` fields,
  `pollTimes`, filter fields, `set`.
- Judge given the full transcript, not just the opening message.
- Whole conversations cached on `(prompt, input)`, with a **logic-version hash**
  so changing the validator invalidates stale results automatically. Extended
  when it emerged that `check_params.js` was invoked by subprocess and therefore
  never hashed — three consecutive fixes to it had changed nothing in the scores.
- Judge dimensions retired, folded, and re-weighted; `workflow_accuracy` removed
  as unmeasurable.
- Knowledge base corrected: wrong Slack channel examples, outdated node syntax
  swept out exhaustively.

## Phase 3 — Retrieval (Jul 28 – Jul 30)

Replaced the Knowledge Assistant with a custom RAG pipeline, so the retrieval
step could be inspected and tuned rather than treated as a black box.

- Chunker, Databricks setup notebook, retriever; index over the existing UC
  volume.
- **Structure-aware chunking** — split on each file's own `## ` headers. Several
  files cover 15–20 node types each; embedding one as a single vector diluted it
  enough that the needed node missed top-K.
- `top_k` raised to 10, matching Ibotta's own HR Bot's validated production
  config rather than a guess.
- **`custom_rag_v2`**: post-retrieval relevance filter plus a grounding note.
- Fixed a chunk-merge bug gluing small sections onto unrelated large ones
  ("Credential Types" onto "Google Sheets Trigger").
- Fixed the custom-RAG arm silently generating on Haiku — every prior run had
  been benchmarking the wrong model.
- Index recreated rather than synced when the chunk schema changes; the queryable
  column set is fixed at creation, so a sync silently keeps serving the old
  schema.
- **Layer 4 hard scenarios**: 18 hand-crafted difficult cases replacing the
  easier 200-item synthetic set.
- Graph-soundness layers: reachability, sub-node wiring, self-loop detection.

## Phase 4 — Deterministic beats persuasive (Jul 31 – Aug 3)

- **Hand-maintained `NODE_TYPE_MAP` replaced with the package's own
  `nodes.json`.** The map was a second source of truth that could only ever
  drift from the first.
- Operation values validated against the resource they're actually gated to.
- Deterministic checks written for the specific things the LLM reviewer kept
  getting wrong — each one converts a recurring argument into a fact.
- Infrastructure chunks always injected; `protected_sources` alone wasn't enough
  to keep them past the relevance filter.
- Soundness review severity-tiered, with blocker-free as the headline number.
- MLflow tracing for hard-scenario runs, which is what made the later
  trace-parsing audits possible.

## Phase 5 — Verify against a live instance (Aug 4 – Aug 5)

Disputed runtime facts settled by **executing workflows on a real n8n instance**
instead of reasoning about them. Several long-held beliefs were wrong.

- `SplitInBatches` "done" output accumulates all fed-back items.
- `Merge` does not deadlock on mutually exclusive branches.
- Jira transitions returns one item per transition.
- Sheets read with a non-matching filter emits **zero items**, and downstream
  nodes are skipped entirely.
- Google Docs `get` returns a flat `content` string, not `body.content[]`.
- Gmail field casing is mixed (`Subject`, `From`, `To`).

Also: the **approval gate became a default with an explicit opt-out** rather than
an absolute. Re-gating a send the user explicitly asked to be automatic is an
intent violation, not extra safety.

Arms added and dropped on measurement, not intuition: query rewriter (dropped),
execution-trace checker (dropped — burned every turn without converging), state
simulation (dropped — its apparent gain was an artifact of prose leaking into
replies), Opus 5 (kept as a comparison arm).

## Phase 6 — Ship it as one thing (Aug 6)

- **The whole pipeline packaged as a single Databricks Model Serving endpoint.**
  Retrieval, filtering, generation, validation and repair were separate n8n
  nodes; they are now one call. The repair loop only works if the thing that
  finds a defect can hand it back with that defect's docs still in hand.
- Side effect that matters more: benchmark and production now run the same code
  path, so a benchmark number predicts production behaviour.
- **`check_params.js` ported to pure Python**, asserted equivalent on 16
  workflows including the 65-node production builder. The subprocess version
  needed Node in the serving image, and silently reported "unavailable" when
  `node_modules` was missing.
- Content-block responses handled centrally — fixes `'list' object has no
  attribute 'strip'` across 8 call sites.
- n8n rewired: `CUSTOM.databricks` (not the identically-named built-in node,
  which needs a credential type this instance doesn't have), plus a small unwrap
  node so everything downstream reads the response unchanged.
- **Every hardcoded credential ID removed.** Resolved per request from the live
  instance instead. Shared credentials (Slack, Jira, Databricks) are looked up by
  type, since they're bot accounts and aren't named after the requesting person —
  which is why they'd been hardcoded.
- Approval sub-workflow ID corrected: the value in config was the **staging**
  instance's ID, so the deterministic gate check was satisfied by the broken case
  and would have rejected the working one.
- Endpoint auth switched from a secret-backed PAT to `resources=` passthrough.
- Index readiness redefined as "a query succeeds" rather than "status says
  ONLINE" — fixing a first-run failure reported repeatedly.

---

## Known open items

- Two dangling n8n expression references (`$('Validator Parser')`,
  `$('Workflow Validator')`) left by the node consolidation, both in an orphaned
  branch.
- The Opus 5 arm never completed a full benchmark run — rate limits. Sonnet 4.6
  is what ships; the comparison is unresolved rather than decided.
- The judge's verified-facts list covers only nodes actually tested live.
