# Production Agent Workflow, Swarm Harness & Model Routing

Multi-agent (swarm) development protocol for VocalForge, adopted as the project scales up
beyond solo sequential-phase work. See "Decisions" at the end for the resolved design
decisions behind Sections 2 and 4.

## 1. Model Strategy & Role Separation

* **Planner & Final Sign-Off (Opus 4.8):** Architecture, high-level planning, writing `plan.md`, assigning task dependencies, conducting human checkpoint reviews, and final workspace validation.
* **Reviewer (Opus 4.8 / Designated Subagent):** Intermediate code review, catching style/API inconsistencies, verifying test suites, and reviewing low-confidence tasks *before* integration.
* **Execution Specialists (Sonnet 5):** Parallel implementation assigned to specific roles (Backend, Frontend, Database, Testing, DevOps, Documentation).

---

## 2. File-Based State & Task Management (`plans/`)

All tasks must maintain state inside `plans/` using a numbered folder naming convention (e.g., `plans/01-feature-name/`). Each folder contains:

1. **`plan.md`:** The architectural roadmap created by Opus 4.8.
2. **`tasks.md`:** The task table tracking ownership, statuses, and confidence levels.
3. **`handoff.log`:** Structured activity records following a strict format.

**Ownership:** `tasks.md` has exactly one writer — the orchestrating session/`Workflow` run (Opus). Workers never edit it and never self-claim; they report progress into their own `handoff.log` entry and the orchestrator updates status on their behalf. This is a deliberate choice (see "Lease atomicity" below) to eliminate concurrent-write races by construction rather than by locking.

### Task Lease Status Legend

* `[ ]` **Pending:** Unclaimed task.
* `[~]` **Claimed:** Assigned to a worker by the orchestrator.
* `[>]` **In Progress:** Actively being worked on.
* `[R]` **In Review:** Submitted to the worker review queue.
* `[H]` **Human Review Required:** Milestone reached (e.g., 10% batch complete); paused for human operator sign-off.
* `[x]` **Complete:** Verified, merged, and signed off.
* `[!]` **Blocked:** Escalated due to error circuit breaking.

```markdown
| ID | Task | Owner / Role | Status | Depends On | Confidence |
|----|------|--------------|--------|------------|------------|
| T01 | JWT middleware | Sonnet-Backend | [>] In Progress | — | 95% |
| T02 | RBAC service | Sonnet-Backend | [~] Claimed | T01 | — |
```

### Structured Handoff Log Format (`handoff.log`)

* **Timestamp:**
* **Agent:**
* **Task ID:**
* **Files:**
* **Decision:**
* **Problem / Remaining Risks:**
* **Next Action:**

---

## 3. Strict Branch & Merge Policy

Code must move through a deterministic, unidirectional pipeline to guarantee stability:

```
Worker Branch --> Review Branch --> Integration Branch --> Main
```

1. **Worker Branch:** Subagents work exclusively in isolated Git worktrees.
2. **Review Branch:** Staged code that has passed worker self-tests for code review.
3. **Integration Branch:** Aggregated and merged reviewed code for workspace-wide test execution.
4. **Main Branch:** Protected branch; only updated after Opus 4.8 and human operator sign-off.

---

## 4. Execution Lifecycle & Protocol

### Step 1: Planning & Initialization (Opus 4.8)

* Create `plans/[task-folder]/`, write `plan.md`, and generate the initial `tasks.md` breakdown with clear dependencies and milestone intervals.

### Step 2: Assignment & Execution (Sonnet 5 Specialists)

* **Task Assignment:** The orchestrator sets a task's status in `tasks.md` from `[ ]` (Pending) to `[~]` (Claimed) and dispatches it to a worker. Workers never edit `tasks.md` themselves — there is nothing to race.
* **Worktree Isolation:** Workers operate in isolated Git worktrees/branches (`Workflow`'s `agent(..., {isolation: 'worktree'})` gives this directly).
* **Worker Self-Test:** Before submitting, workers must independently run linting, unit tests, and compilation locally.
* **Confidence Scoring & Handoff:** Workers report a confidence score and remaining risks in `handoff.log`, update status to `[R]` (In Review), and push to the **Review Branch**.

### Step 3: Review Queue, Circuit Breaking & Human Checkpoints (HITL)

1. **Reviewer Gate:** Code on the Review Branch is inspected for style, duplicate logic, and API consistency (low-confidence tasks prioritized).
2. **Circuit Breaking / Retry Limit:** If a worker fails twice on a task, it must log failure details in `handoff.log`, set status to `[!]` (Blocked), and escalate to Opus.
3. **Integration Branch Merge:** Reviewed and approved code is merged into the Integration Branch.
4. **Milestone Checkpoints & Human Gate (e.g., Every 10% Progress):**
   * Upon hitting a 10% task completion milestone (or major architectural milestone), the swarm **must halt execution**.
   * Opus 4.8 marks completed items with `[H]`, tags a Git checkpoint (e.g., `checkpoint-10pct`), and presents a summary to the user:
     > *"10% milestone reached (Tasks T01-T03 complete). Please review `plans/[folder]/` and reply 'Continue', 'Pivot: ', or 'Rollback'."*
   * **`Continue`**: Swarm resumes execution for the next batch.
   * **`Pivot`**: Opus updates `plan.md` and `tasks.md` according to human feedback before resuming.
   * **`Rollback`**: Reverts the integration branch to the last tagged checkpoint.

### Step 4: Final Sign-Off (Opus 4.8 & Human)

* Opus runs full workspace validation tests on the Integration Branch and presents the final build to the human operator. Upon final human sign-off, Opus merges into **Main**, sets status to `[x]` (Complete), and finalizes the task.

---

## Decisions

1. **Scale fit — adopted in full.** The 4-tier branch pipeline (Section 3) and the 10%-milestone human gates (Section 4, Step 3) apply now, not gated behind some future task-count threshold — this protocol is meant for real use as the project scales up.

2. **Token-threshold mechanics — dropped.** Claude Code already auto-compacts conversation context as it approaches its limit; there's no tool exposing an exact running token count for a model to self-police against 150k/180k/200k, so the old Section 5 would have been followed inconsistently at best. Removed rather than kept as dead weight.

3. **Lease atomicity — resolved: Option A (single-writer orchestrator).** Workers are subagents spawned from one orchestrating session/`Workflow` run, not independent processes with no shared parent — so the orchestrator can simply be the sole writer of `tasks.md` and the race condition class doesn't exist. See below.

4. **Mechanism vs. protocol — resolved: both, layered.** The mechanical engine (worker fan-out, worktree isolation, structured handoffs, review, integration) is a named `Workflow` script; a thin `/swarm-feature` skill is the human-facing entry point that invokes it. See below.

### 3. Lease atomicity — resolved

Workers run as subagents of one orchestrating session/`Workflow` run, so the orchestrator can be `tasks.md`'s sole writer (Section 2) instead of needing a lock:

* **Adopted — single-writer orchestrator.** Only the orchestrator (Opus / the driving `Workflow` script) ever edits `tasks.md`; it assigns tasks to workers explicitly rather than workers self-claiming from a shared file. Workers report progress into their own `handoff.log` entry, and the orchestrator (which is what's actually driving `pipeline()`/`parallel()` in the workflow script) updates status on their behalf as each result comes back.
* Not needed given this topology, but worth remembering if the topology ever changes to independent sessions/processes with no shared parent (e.g. someone runs a worker from a separate terminal against the same `plans/` folder): **per-task file + git-push-as-compare-and-swap** (each task gets its own status file under `plans/<folder>/tasks/<id>.status`; a rejected non-fast-forward push means someone else already claimed it) or **GitHub Issues/PRs as the lock** (self-assign via `gh issue edit --add-assignee`, offloading atomicity to GitHub's API). Revisit if that happens.

### 4. Mechanism — resolved: both, layered

* The mechanical parts (Sonnet worker fan-out, per-worker Git **worktree isolation** — matching Section 3's "Worker Branch" concept exactly, since `Workflow`'s `agent(..., {isolation: 'worktree'})` already gives each worker an isolated worktree, auto-cleaned if unchanged — structured handoff capture via a JSON schema instead of trusting free-text formatting, a reviewer pass, and integration-branch aggregation) map directly onto the `Workflow` tool's `pipeline()`/`parallel()`/`phase()` primitives. That's a deterministic engine with real state tracking, not markdown-file self-reporting — it should be authored as a named workflow script (e.g. `swarm-feature`) implementing Steps 1–4.
* A **skill** (e.g. `/swarm-feature`) is the thin, human-facing entry point: explains the protocol, collects the task-folder name and scope, and invokes the workflow. Skills are just loaded instructions with no enforcement power of their own, which makes them the wrong place for the actual concurrency/state-tracking logic but the right place to document reply conventions (`Continue` / `Pivot: …` / `Rollback`) and when to invoke this protocol at all versus doing a normal single-session change.
* One governance detail: the `Workflow` tool requires explicit opt-in per call (an "ultracode" session, or the user directly asking for a workflow/naming a saved one). A saved named workflow invoked via the skill satisfies that — the skill's own invocation *is* the explicit ask — so this doesn't need a special exception.
* `docs/agent_workflow.md` (this file) stays the spec/design doc either way — not the enforcement mechanism itself.

## Implementation

- `.claude/workflows/swarm-feature.js` — the Workflow script. One invocation = one
  batch (plan/resume → assign → execute in isolated worktrees → record → review →
  integrate), or the Step 4 final sign-off once every task is complete/blocked.
- `.claude/skills/swarm-feature/SKILL.md` — the `/swarm-feature` entry point. Loops
  across workflow invocations, tags a Git checkpoint per batch, and asks the human
  Continue/Pivot/Rollback between batches (the workflow itself has no way to pause
  mid-script for a human reply — it just returns after each batch).
