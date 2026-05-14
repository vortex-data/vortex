---
name: epic-issues
description: Work on Vortex GitHub epic issues, especially updating epic plans, preserving checklist state, linking PRs inline, and managing draft PR stacks for epic work.
---

# Vortex Epic Issues Skill

Use this when the user asks to plan, update, or implement work tracked by a GitHub issue labeled
`epic`, or when PR stacks need to be linked back to an epic issue.

## Core Rules

1. Always read the latest epic issue body before proposing or editing the plan. GitHub state may
   have changed since the conversation started.
2. Preserve existing checklist state exactly. Never turn `- [x]` back into `- [ ]` when updating
   an epic body.
3. Put PR references inline on the relevant checklist items as `#1234`. Do not add a separate
   "stack", "PRs", or "implementation history" section unless the user explicitly asks for one.
4. Checklist boxes are only for work items. Put rationale, examples, constraints, and notes in
   non-checkbox sub-bullets.
5. Use plain GitHub issue/PR shorthand (`#1234`) in epic bodies, not full PR URLs, unless the
   surrounding text already uses full links and the user wants that format preserved.
6. Avoid rewriting the whole epic from memory. Patch the current body narrowly and preserve labels,
   assignees, milestone, title, and checked boxes unless the user asks to change them.
7. Treat user corrections during epic work as possible durable workflow guidance. Update this skill
   only when the correction will be relevant and helpful for future epic invocations.
8. For an active stacked epic, "what's next" means implementation movement: finish the current
   phase if it is incomplete, or start the next phase in a new branch/PR. Do not stop at "review
   the current PR" as the next step.

## Planning Workflow

When asked to revise an epic plan:

1. Fetch/read the actual epic body.
2. Inspect relevant existing PRs/issues before assuming what is current.
3. Propose a coherent phase plan before editing the issue if the user asks for approval first.
4. Keep phases as executable work, not documentation of rationale.
5. Prefer small, reviewable steps that map naturally to PRs or sub-issues.
6. If the user approves, update the epic body from the latest fetched body and preserve every
   existing checked box.

## PR Stack Workflow

When implementing an epic via stacked PRs:

1. Use the `gh-stack` skill for local stacked-branch mechanics.
2. Keep branches locally until the user asks to push.
3. Use signed commits with `Signed-off-by: Name <email>`.
4. Create draft PRs unless the user asks for ready PRs.
5. Add a `changelog/*` label to every PR.
6. Do not use ordinal prefixes in PR titles. They pollute changelog entries; rely on PR bases and
   epic links to communicate stack order.
7. Write PR bodies as valid Markdown with real newlines. Do not include escaped newline text.
8. After PRs exist, update the epic by adding each PR number next to the relevant todo item.
9. When asked for the next step in an active stack, identify the next implementation branch or PR
   to create/update. Treat review status as context, not as the next work item.
10. If the next phase exposes a missing prerequisite, split that prerequisite into a lower stacked
    branch/PR instead of bundling it into the phase branch.

## When Stack State Changes

1. Before rebasing or retargeting, check whether lower PRs have merged or closed.
2. If a lower PR has merged, fetch `develop`, rebase dependent branches onto current
   `origin/develop`, and retarget open PRs accordingly.
3. If a PR was merged accidentally and its work still needs review separately, revert the merge if
   needed, create a replacement PR, then update the epic to point at the replacement PR number.
4. Closed PRs generally cannot be retargeted. Create a replacement PR when the review unit still
   needs to exist.
5. After any retargeting, verify PR metadata and mergeability from GitHub, not only local git
   ancestry.

## Epic Body Update Checklist

Before saving an epic body:

- Confirm no checked boxes were unintentionally changed.
- Confirm PR numbers appear inline on the relevant todo items.
- Confirm no separate PR stack section was added.
- Confirm every checkbox is a work item.
- Confirm any merged/replacement PR numbers are current.
