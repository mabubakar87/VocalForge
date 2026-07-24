---
name: swarm-feature
description: Run the multi-agent swarm feature-build protocol from docs/agent_workflow.md — plans a feature into plans/<NN-name>/, then executes it in ~10% batches (assign → isolated-worktree execution → review → integrate), pausing between batches for a human Continue/Pivot/Rollback decision. Use when the user wants to build a non-trivial feature with the swarm protocol rather than a normal single-session change.
user-invocable: true
allowed-tools:
  - Read
  - Glob
  - Bash(git *)
  - Bash(ls *)
  - AskUserQuestion
  - Workflow
---

# /swarm-feature — swarm feature build

Runs the protocol documented in `docs/agent_workflow.md`. Read that file first if you
haven't — this skill is the human-facing entry point into the `swarm-feature` Workflow
script (`.claude/workflows/swarm-feature.js`), which is the actual mechanism. This skill
does the parts a Workflow script cannot: talking to the human between batches and
deciding whether to invoke the workflow again.

Arguments passed: `$ARGUMENTS` — either a new feature description (starting a new
`plans/` folder), or an existing folder name/number to resume (e.g. `02` or
`02-realtime-transcription`).

**This is heavier process than a normal change.** If the task is small enough for one
session to just implement directly, say so and suggest that instead of invoking the
swarm — this protocol exists for scaling up to multi-agent parallel work, not as the
default way to make any change in this repo.

---

## 1. New feature vs. resume

- List `plans/` (`ls plans`) to find existing numbered folders (`NN-name`).
- If `$ARGUMENTS` names or clearly matches an existing folder → **resume** that folder.
- Otherwise → **new feature**. Pick the next zero-padded two-digit number (`01`, `02`,
  ...) and a short kebab-case slug from the description. Confirm the folder name with
  the user before creating anything if it's not obvious from their request.

## 2. Run one batch

Invoke the workflow:

```
Workflow({
  name: 'swarm-feature',
  args: {
    taskFolder: '<NN-name>',
    mode: 'init' | 'resume',
    featureBrief: '<only for mode=init: the feature description>',
    pivotFeedback: '<only when resuming after the user chose Pivot last time>',
  },
})
```

Wait for the result. It returns one of:

- `status: 'batch_complete'` — a batch ran; `message` has the human-facing summary.
- `status: 'stuck'` — no eligible tasks (dependency deadlock or 2 permanently-blocked
  tasks blocking everything downstream). Show `stillOpen` and `permanentlyBlocked` to
  the user and ask them how to unblock it (re-scope a task, drop a dependency) — do not
  re-invoke the workflow until they've responded, since it will return the same stuck
  state.
- `status: 'complete'` — all tasks done or permanently blocked, final sign-off
  validation already ran (`signOff`). Present it as ready (or not) for merge to `main`;
  do not merge yourself without an explicit human go-ahead (Step 4 requires human
  sign-off, not just Opus's).

## 3. On `batch_complete`: checkpoint, then ask

1. Tag the integration branch so `Rollback` has something to revert to:
   `git tag swarm/<folder>/complete-<completedSoFar>` (use the `completedSoFar` value
   from the result — it's monotonically increasing, so the tag name is unique).
2. Show the user the `message` field plus which task IDs were approved / rejected by
   review / failed self-test this batch.
3. Ask via AskUserQuestion (header "Swarm batch", options: Continue / Pivot / Rollback —
   Continue recommended) rather than waiting for free text, since this is exactly the
   kind of discrete decision that tool is for. If they pick "Other" to type free-form
   pivot feedback, that's expected — fold it into `pivotFeedback` on the next call.

Then:

- **Continue** → invoke the workflow again with `mode: 'resume'`, no `pivotFeedback`.
- **Pivot: <feedback>** → invoke again with `mode: 'resume'` and `pivotFeedback` set to
  what they said. The workflow's planning step updates `plan.md`/`tasks.md` before
  running the next batch.
- **Rollback** → find the previous checkpoint tag: `git tag --sort=-creatordate --list
  'swarm/<folder>/*'`, take the second one (the one before the tag you just created),
  and reset the **integration branch only** to it (`git checkout <integration-branch> &&
  git reset --hard <tag>` — never touch `main`, and confirm with the user before running
  a hard reset even though they just asked for Rollback, per the repo's standing rule on
  destructive git operations). Then invoke the workflow again with `mode: 'resume'` so it
  re-reads the now-reverted state.

Loop steps 2–3 until the workflow returns `complete` or `stuck`, or the user stops you.

## 4. Final sign-off (`status: 'complete'`)

Present `signOff.summary` and `signOff.testsPassed` plus any `permanentlyBlocked` tasks.
Only merge the integration branch into `main` after the human explicitly confirms —
this mirrors Step 4 of the protocol ("Opus 4.8 & Human"), and merging to `main` is
exactly the kind of shared/hard-to-reverse action that needs a real go-ahead, not an
inferred one.
