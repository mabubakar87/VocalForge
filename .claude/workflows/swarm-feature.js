export const meta = {
  name: 'swarm-feature',
  description: 'Multi-agent feature swarm per docs/agent_workflow.md: plan once, then run one ~10% batch of tasks (assign, execute in isolated worktrees, record, review, integrate) per invocation.',
  whenToUse: 'Invoked by the /swarm-feature skill, not called directly. One call = one batch. The calling session loops across invocations and talks to the human between batches (see docs/agent_workflow.md Step 3.4).',
  phases: [
    { title: 'Plan' },
    { title: 'Assign' },
    { title: 'Execute' },
    { title: 'Record' },
    { title: 'Review' },
    { title: 'Integrate' },
    { title: 'Sign-off' },
  ],
}

// Implements docs/agent_workflow.md. Read that file before editing this script —
// it is the spec; this is the mechanism.
//
// Key design constraint this script works around: the script body has NO filesystem
// access (per the Workflow tool contract) — only agent() calls can Read/Write/Bash.
// So "tasks.md has exactly one writer" (agent_workflow.md Section 2) is enforced by
// *timing*, not by file locks: every write to tasks.md/handoff.log happens via a
// single sequential agent() call awaited by the top-level script, never from inside
// a parallel()/pipeline() fan-out. Two such calls never overlap because this script
// is single-threaded JS awaiting each step in turn.
//
// Each invocation processes exactly one batch (~batchFraction of remaining eligible
// tasks, default 10%) and returns. There is no in-script blocking wait for a human
// reply — the milestone gate (agent_workflow.md Step 3.4) is realized by the workflow
// simply stopping here; the /swarm-feature skill (running in the calling session) is
// what shows the batch summary to the human and decides whether to invoke this
// workflow again (Continue), first update plan.md/tasks.md via a small edit (Pivot),
// or revert the integration branch (Rollback) before invoking again.

const STATUS_LABEL = {
  pending: '[ ] Pending',
  claimed: '[~] Claimed',
  in_progress: '[>] In Progress',
  in_review: '[R] In Review',
  human_review: '[H] Human Review Required',
  complete: '[x] Complete',
  blocked: '[!] Blocked',
}

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    tasks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Short stable id, e.g. T01' },
          title: { type: 'string' },
          role: { type: 'string', description: 'e.g. Backend, Frontend, Testing, Docs' },
          dependsOn: { type: 'array', items: { type: 'string' } },
          status: { type: 'string', enum: Object.keys(STATUS_LABEL) },
          priorFailures: { type: 'number', description: 'Count of prior circuit-breaker failures for this task, from handoff.log history. 0 if none/unknown.' },
        },
        required: ['id', 'title', 'role', 'status'],
      },
    },
  },
  required: ['tasks'],
}

const HANDOFF_SCHEMA = {
  type: 'object',
  properties: {
    confidence: { type: 'number', description: '0-100' },
    filesTouched: { type: 'array', items: { type: 'string' } },
    decision: { type: 'string', description: 'What was implemented and how' },
    risks: { type: 'string', description: 'Remaining risks / problems, empty string if none' },
    nextAction: { type: 'string' },
    selfTestsPassed: { type: 'boolean' },
    branch: { type: 'string', description: 'Name of the worker branch/worktree this was committed on' },
  },
  required: ['confidence', 'filesTouched', 'decision', 'risks', 'nextAction', 'selfTestsPassed', 'branch'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    approved: { type: 'boolean' },
    reasons: { type: 'string' },
  },
  required: ['approved', 'reasons'],
}

const INTEGRATION_SCHEMA = {
  type: 'object',
  properties: {
    testsPassed: { type: 'boolean' },
    conflicts: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['testsPassed', 'summary'],
}

const FOLDER = args.taskFolder
const MODE = args.mode // 'init' | 'resume'
const BATCH_FRACTION = args.batchFraction || 0.1

if (!FOLDER || !MODE) {
  throw new Error("swarm-feature requires args: { taskFolder, mode: 'init'|'resume', featureBrief? , pivotFeedback? }")
}

phase('Plan')

let planResult
if (MODE === 'init') {
  log(`Planning plans/${FOLDER} from scratch.`)
  planResult = await agent(
    `You are the planning role from docs/agent_workflow.md (Opus, Section 1 "Planner & Final Sign-Off").
Create the folder plans/${FOLDER}/ in this repo with:
- plan.md: the architectural roadmap for this feature. Feature brief: ${JSON.stringify(args.featureBrief || '')}
- tasks.md: a markdown table matching the format in docs/agent_workflow.md Section 2 (ID, Task, Owner/Role, Status, Depends On, Confidence). Every task starts as "${STATUS_LABEL.pending}". Break the feature into small, independently reviewable tasks with explicit dependsOn chains where real ordering constraints exist.
- handoff.log: create it empty (it will be appended to later).
Commit these three files with message "plan(swarm): ${FOLDER} initial plan".
Return the full task list as structured data (ids, titles, roles, dependsOn, status — all "pending", priorFailures 0).`,
    { phase: 'Plan', schema: PLAN_SCHEMA, label: 'plan:init' }
  )
} else {
  log(`Resuming plans/${FOLDER} — reading current task state.`)
  planResult = await agent(
    `Read plans/${FOLDER}/plan.md, plans/${FOLDER}/tasks.md, and plans/${FOLDER}/handoff.log.
${args.pivotFeedback ? `The human gave this pivot feedback before this run — first update plan.md and tasks.md to reflect it (add/remove/reorder tasks as needed), commit that change with message "plan(swarm): ${FOLDER} pivot", then continue: ${JSON.stringify(args.pivotFeedback)}` : ''}
For each task row in tasks.md, also scan handoff.log for prior "[!] Blocked" entries for that task id and count them (priorFailures).
Return the full current task list as structured data.`,
    { phase: 'Plan', schema: PLAN_SCHEMA, label: 'plan:resume' }
  )
}

const allTasks = planResult.tasks
const totalTasks = allTasks.length
const doneIds = new Set(allTasks.filter((t) => t.status === 'complete').map((t) => t.id))
const permanentlyBlocked = allTasks.filter((t) => t.status === 'blocked' && (t.priorFailures || 0) >= 2)
const blockedIds = new Set(permanentlyBlocked.map((t) => t.id))

const eligible = allTasks.filter(
  (t) => t.status === 'pending' && (t.dependsOn || []).every((d) => doneIds.has(d))
)
const batchSize = Math.max(1, Math.ceil(totalTasks * BATCH_FRACTION))
const batch = eligible.slice(0, batchSize)

const stillOpen = allTasks.filter(
  (t) => !['complete'].includes(t.status) && !blockedIds.has(t.id)
)

if (batch.length === 0) {
  if (stillOpen.length === 0) {
    // Step 4: Final Sign-Off
    phase('Sign-off')
    const signOff = await agent(
      `You are the Opus "Final Sign-Off" role from docs/agent_workflow.md Section 4 Step 4.
All tasks in plans/${FOLDER}/tasks.md are complete or permanently blocked. Run this repo's full workspace validation (see CLAUDE.md for the test command) on the integration branch for plans/${FOLDER}. Report whether it is ready to merge into main, and summarize what's blocked (if anything) for the human operator.`,
      { phase: 'Sign-off', schema: INTEGRATION_SCHEMA, label: 'sign-off' }
    )
    return {
      status: 'complete',
      totalTasks,
      permanentlyBlocked: permanentlyBlocked.map((t) => t.id),
      signOff,
      message: signOff.testsPassed
        ? 'All tasks done, integration branch validated. Ready for human sign-off and merge to main.'
        : 'All tasks done or blocked, but integration validation failed — see signOff.summary before merging.',
    }
  }
  return {
    status: 'stuck',
    totalTasks,
    stillOpen: stillOpen.map((t) => ({ id: t.id, status: t.status, dependsOn: t.dependsOn })),
    permanentlyBlocked: permanentlyBlocked.map((t) => t.id),
    message: 'No eligible pending tasks this round — remaining tasks are blocked on dependencies that never completed, or permanently blocked (2 circuit-breaker failures). Human input needed: unblock a dependency or re-scope the stuck task(s) before the next run.',
  }
}

log(`Batch: ${batch.map((t) => t.id).join(', ')} (${batch.length}/${eligible.length} eligible, ${totalTasks} total)`)

phase('Assign')
await agent(
  `Update plans/${FOLDER}/tasks.md: set status to "${STATUS_LABEL.claimed}" for task IDs: ${batch.map((t) => t.id).join(', ')}. Do not modify any other row. Commit with message "chore(swarm): claim ${batch.map((t) => t.id).join(', ')}".`,
  { phase: 'Assign', label: 'claim-batch' }
)

phase('Execute')
const workerResults = await parallel(
  batch.map((task) => () =>
    agent(
      `You are a Sonnet execution specialist (docs/agent_workflow.md Section 1, role: ${task.role}) implementing task ${task.id}: ${task.title}.
Read plans/${FOLDER}/plan.md and plans/${FOLDER}/tasks.md for full context and acceptance criteria.
You are in an isolated worktree — this is your "Worker Branch" (Section 3). Implement the task fully, run this repo's linting/tests locally per CLAUDE.md before finishing, and commit your work on this branch.
Report your confidence (0-100), files touched, your decision/approach, remaining risks, next action, whether self-tests passed, and the branch name you committed to.`,
      { phase: 'Execute', isolation: 'worktree', schema: HANDOFF_SCHEMA, label: `work:${task.id}` }
    ).then((result) => ({ task, result }))
  )
)

const liveWork = workerResults.filter(Boolean)

phase('Record')
await agent(
  `Append one entry per item below to plans/${FOLDER}/handoff.log, in the "Timestamp / Agent / Task ID / Files / Decision / Problem-Remaining Risks / Next Action" format from docs/agent_workflow.md Section 2 (use a placeholder timestamp marker "<now>" — do not fabricate a clock time).
For each item, set that task's tasks.md status: "${STATUS_LABEL.in_review}" if selfTestsPassed is true and confidence >= 50, otherwise "${STATUS_LABEL.blocked}" (circuit breaker — self-test failure or low confidence).
Items: ${JSON.stringify(liveWork.map((w) => ({ id: w.task.id, ...w.result })))}
Commit with message "chore(swarm): record batch ${batch.map((t) => t.id).join(', ')}".`,
  { phase: 'Record', label: 'record-execute' }
)

const forReview = liveWork.filter((w) => w.result.selfTestsPassed && w.result.confidence >= 50)
const selfTestFailed = liveWork.filter((w) => !(w.result.selfTestsPassed && w.result.confidence >= 50))

phase('Review')
const reviewResults = await parallel(
  forReview.map((w) => () =>
    agent(
      `You are the Reviewer role from docs/agent_workflow.md Section 1 and Section 4 Step 3.1 — Opus or a designated subagent, deliberately adversarial.
Review task ${w.task.id} ("${w.task.title}") on branch/worktree "${w.result.branch}" against the rest of vocalforge/: check style, duplicate logic, and API consistency. The worker reported ${w.result.confidence}% confidence — scrutinize harder the lower that number is. Decision reported by the worker: ${w.result.decision}
Return approved (true/false) and reasons.`,
      { phase: 'Review', schema: REVIEW_SCHEMA, label: `review:${w.task.id}` }
    ).then((review) => ({ ...w, review }))
  )
)

const liveReviews = reviewResults.filter(Boolean)
const approved = liveReviews.filter((r) => r.review.approved)
const rejected = liveReviews.filter((r) => !r.review.approved)

phase('Integrate')
let integration = null
if (approved.length > 0) {
  integration = await agent(
    `You are executing docs/agent_workflow.md Section 4 Step 3.3 (Integration Branch Merge) for plans/${FOLDER}.
Merge these approved worker branches into the integration branch: ${approved.map((a) => `${a.task.id} (${a.result.branch})`).join(', ')}.
Then run this repo's full workspace test suite (see CLAUDE.md) on the integration branch. Report testsPassed, any merge conflicts (list them), and a short summary.`,
    { phase: 'Integrate', schema: INTEGRATION_SCHEMA, label: 'integrate-batch' }
  )
}

phase('Record')
const finalizeParts = []
if (approved.length > 0) {
  finalizeParts.push(
    `Mark these task IDs "${integration && integration.testsPassed ? STATUS_LABEL.complete : STATUS_LABEL.blocked}" and append the reviewer's approval + this integration result to handoff.log: ${JSON.stringify(
      approved.map((a) => ({ id: a.task.id, review: a.review, integration }))
    )}`
  )
}
if (rejected.length > 0) {
  finalizeParts.push(
    `Mark these task IDs "${STATUS_LABEL.blocked}" (reviewer rejected) and append the reviewer's reasons to handoff.log: ${JSON.stringify(
      rejected.map((r) => ({ id: r.task.id, review: r.review }))
    )}`
  )
}
if (selfTestFailed.length > 0) {
  finalizeParts.push(
    `These task IDs already recorded as blocked (self-test/confidence circuit-breaker) earlier this batch — no further action needed: ${selfTestFailed.map((w) => w.task.id).join(', ')}`
  )
}
await agent(
  `Update plans/${FOLDER}/tasks.md and plans/${FOLDER}/handoff.log per docs/agent_workflow.md's circuit-breaking rule (Section 4 Step 3.2): a task blocked for the SECOND time (check handoff.log history) stays "${STATUS_LABEL.blocked}" permanently and is escalated (note "ESCALATED to Opus" in handoff.log); a task blocked for the first time should instead be reset to "${STATUS_LABEL.pending}" so it is retried in a future batch.
${finalizeParts.join('\n')}
Commit with message "chore(swarm): finalize batch ${batch.map((t) => t.id).join(', ')}".`,
  { phase: 'Record', label: 'finalize-batch' }
)

const completedThisBatch = approved.length && integration && integration.testsPassed ? approved.length : 0
const completedSoFar = doneIds.size + completedThisBatch
const milestoneFraction = totalTasks > 0 ? completedSoFar / totalTasks : 0

return {
  status: 'batch_complete',
  folder: FOLDER,
  batch: batch.map((t) => t.id),
  approved: approved.map((a) => a.task.id),
  rejectedByReview: rejected.map((r) => r.task.id),
  failedSelfTest: selfTestFailed.map((w) => w.task.id),
  integration,
  totalTasks,
  completedSoFar,
  milestoneFraction,
  message: `Batch done: ${completedThisBatch}/${batch.length} tasks integrated (${completedSoFar}/${totalTasks} total, ${Math.round(milestoneFraction * 100)}%). Per docs/agent_workflow.md Step 3.4: reply Continue to run the next batch, "Pivot: <feedback>" to adjust the plan first, or Rollback to revert this batch's integration merge.`,
}
