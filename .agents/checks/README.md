# Agent checks

Agent checks are weekly code reviews by an agent. Each check is a rubric plus a little
configuration in `.agents/checks/<id>/CHECK.md`. Every Monday, CI gives each crate with new or
changed files to a Claude Code session, which reviews them against the rubric. For each crate it
then opens:

- a draft PR with the fixes it could make,
- an issue with the findings it could not fix.

| Check | What it asks |
| --- | --- |
| [`benchmark-usefulness`](benchmark-usefulness/CHECK.md) | Do the benchmarks measure what they claim, and are they worth their CI time? |
| [`test-focus`](test-focus/CHECK.md) | Does each test check one behavior precisely, and could it pass by accident? |

## Got an agent PR or issue?

Both carry the `agent-checks` label, and are titled "<Check> fixes in <crate>" and "<Check>
findings in <crate>". Whoever owns the crate triages them.

| You have | To accept | To decline for now | To reject for good |
| --- | --- | --- | --- |
| A draft PR (fixes) | Review, mark ready, and merge. | Close it. The files are reviewed again when they next change. | Add `agent-checks:rejected`, then close it. |
| An issue (unfixed findings) | Fix what you agree with, then close it as completed. | Leave it open. Later runs add new findings as comments, never repeats. | Close it as not planned. |

Rejected findings are never raised again for that file and item. To undo a rejection, use
`agent_checks.py unsuppress` (see [Resetting](#resetting)).

## Adding a check

Create `.agents/checks/<id>/CHECK.md`, where `<id>` matches the `id` key, and add a row to the
table above:

```markdown
+++
id = "my-check"
title = "My check"
description = "One sentence: what this check looks for."
include = ["**/src/**/*.rs"]
# CI cannot build these crates on a plain runner; see the existing checks for the full list.
exclude = ["vortex-cuda/**", "vortex-duckdb/**", "vortex-ffi/**"]
verify = ["cargo clippy -p {crate} --all-targets --all-features -- -D warnings"]
labels = ["changelog/chore"]
+++

You are reviewing ... (the rubric: categories, how to tell a real problem from an acceptable
one, what to fix vs. only report, and how to name the `item` of a finding)
```

Then preview it locally (see [Running it](#running-it)). `test_repository_checks_parse` in
`scripts/tests/test_agent_checks.py` checks that the file parses.

The runner appends the rest to every prompt: the results format, the scope rules, the `verify`
commands, the assigned files, and the findings maintainers rejected. So the rubric only describes
the check.

| Key | Default | Meaning |
| --- | --- | --- |
| `id`, `title`, `description` | required | `id` is kebab-case and names the branches and the storage prefix. `title` starts the PR and issue titles. |
| `include` | required | Repo-relative globs (`*`, `?`, `**`, `{a,b}`) over tracked files. |
| `exclude` | `[]` | Globs removed from `include`. |
| `content_match` | none | Regex that a file must contain, such as `#\[test\]`. |
| `verify` | `[]` | Commands run on the edits. `{crate}` becomes the package name. Each runs without a shell, so no pipes, `&&` or `VAR=value` prefixes, and must work on a plain runner with only Rust set up. Any failure reverts all edits. The agent may run exactly these. |
| `fixes` | `true` | `false` makes a report-only check: edits are reverted, and findings only go to issues. |
| `labels` | `[]` | Extra PR labels, created if missing. Include a `changelog/*` label. |
| `max_files_per_shard` | `12` | Files reviewed per crate per run. The rest wait for later runs. |
| `max_lines_per_shard` | none | Also caps a crate's total lines per run. A crate always gets at least one file. |
| `max_shards_per_run` | `8` | Crates reviewed per run, one after another. |
| `max_open_prs` | `10` | No new crates are reviewed while this many of the check's PRs are open. |
| `max_findings` | `20` | Findings kept per crate, most severe first. |
| `version` | `1` | Bump it when the rubric changes enough that reviewed files should be reviewed again. |
| `revisit_after_days` | none | Re-review unchanged files after this many days. |
| `allowed_tools` | `[]` | Extra Claude Code tools, such as `"Bash(cargo test:*)"`. Always allowed: Read, Edit, Write, Grep, Glob, `git diff/log/show/blame/status`, `git checkout --`, `cargo check`, `cargo clippy`. |
| `model`, `max_turns` | `claude-opus-5-5`, `120` | Agent settings. |
| `enabled` | `true` | Disabled checks run only when named. |

Writing a good rubric:

- Give each category a stable kebab-case slug, what to look for, why it matters, and how to tell a
  real problem from acceptable code.
- Say which categories the agent may fix and which it may only report. Only allow fixes that are
  mechanical, local, and verifiable.
- Say what a finding's `item` is, such as the benchmark function or the test's module path. The
  runner identifies a finding as `<category>:<file>:<item>`, so dedup and rejection depend on
  `item` staying the same from run to run.
- Tell the agent that zero findings is a good result.
- Make the rubric's build instructions agree with `verify`.

A `Bash(<prefix>:*)` tool allows any command that starts with the prefix, so `env X=:*` allows any
command at all. The agent's runner holds no secrets, but keep prefixes specific.

## Running it

Locally:

```bash
python3 scripts/agent_checks.py list
# What the next run would review, and why the other crates wait:
python3 scripts/agent_checks.py plan --check benchmark-usefulness > /dev/null
# The prompt for one crate:
python3 scripts/agent_checks.py run benchmark-usefulness --crate vortex-buffer --dry-run
# Review it with the local `claude` CLI. Edits stay in your working tree; the PR body is written to
# target/agent-checks/<check>/<crate>.md. Add --record to mark the files reviewed in local state.
python3 scripts/agent_checks.py run benchmark-usefulness --crate vortex-buffer \
  --format-cmd "rustup run nightly-2026-11-13 rustfmt --edition 2024"
```

In CI, use **Actions → Agent Checks → Run workflow**, optionally with a `check`, a `crate`,
`max_shards`, and `dry_run`. A dry run shows the findings, decisions and diffs in the job
summaries, and publishes and records nothing. Any run reviews only what changed since the last
review, so a manual run never repeats the weekly one.

Each run's summary page shows:

- the plan: every crate to review, and every crate that waits and why,
- one summary per crate, with the decision, the findings and the diff,
- what was recorded.

Every run is also archived under the check's `runs/` prefix in S3 (see [Storage](#storage)).

## Setup

**GitHub.** The `claude-automation` environment, as for `fuzzer-fix-automation.yml`, with:

- the App secrets `APP_ID` and `APP_PRIVATE_KEY`,
- the variable `CLAUDE_APP_LOGIN`, the App's bot login. Only its markers are trusted, and
  non-dry runs refuse to start without it.
- the Anthropic variables `ANTHROPIC_ORGANIZATION_ID`, `ANTHROPIC_SERVICE_ACCOUNT_ID` and
  `ANTHROPIC_FEDERATION_RULE_WRITE`, the id of the `vortex-claude-write` rule.

**Anthropic.** The `vortex-claude-write` federation rule must allow
`vortex-data/vortex/.github/workflows/agent-checks-shard.yml@refs/heads/develop` as
`job_workflow_ref`. For a dry run from a branch or PR, also allow `@refs/heads/<branch>` or
`@refs/pull/<n>/merge`. Without it, every agent fails its token exchange with a 401, and nothing
is recorded.

**AWS.** A bucket, with versioning on, and these repository variables:

| Variable | Value |
| --- | --- |
| `AGENT_CHECKS_STORE_URI` | `s3://<bucket>/<prefix>`, for example `s3://vortex-ci/agent-checks` |
| `AGENT_CHECKS_AWS_ROLE_ARN` | A role for GitHub's OIDC provider. Its trust policy must require `token.actions.githubusercontent.com:sub` = `repo:vortex-data/vortex:ref:refs/heads/develop` and `aud` = `sts.amazonaws.com`. |
| `AGENT_CHECKS_AWS_REGION` | The bucket's region (default `us-east-1`) |

The role needs these permissions:

- `s3:GetObject` and `s3:PutObject` on `<prefix>/*`,
- `s3:ListBucket` on the bucket for that prefix. Without it, S3 answers a missing key with 403
  instead of 404, and the runner refuses to treat 403 as empty state.

To use S3 locally, you need an AWS CLI v2 recent enough for `put-object --if-match` (late 2024 or
later).

**Dry runs** may start from any branch. Without `AGENT_CHECKS_STORE_URI` they plan from empty
state. The role's subject condition rejects other branches, so leave the variable unset for branch
dry runs.

## Reference

### How a run works

```text
plan ──► per check, in parallel: crate ──► crate ──► … (one at a time) ──► record
                                   │
                                   └─ agent job ──► publish job (fresh runner)
```

| Workflow | Runs |
| --- | --- |
| `agent-checks.yml` | The weekly trigger: plan, then one review per check, then record. |
| `agent-checks-review.yml` | One check's crates, one after another (`max-parallel: 1`). |
| `agent-checks-shard.yml` | One crate: the `agent` job, then the `publish` job. |

1. **Plan.** The plan job reads each check's `state.json`, all open PRs, and the closed PRs and
   issues labelled `agent-checks`. For each check:
   - It collects the files that need review:
     - `changed`: the blob differs from every blob recorded at the file's last review,
     - `new`: never reviewed,
     - `rubric`: the check's `version` was bumped,
     - `stale`: older than `revisit_after_days`.
   - It picks crates (see [Sharding](#sharding)).
   - It writes the plan. The plan carries the check's definition, so every later job uses the
     same rubric, and the agent cannot change it.
2. **Agent job.** The agent reviews the crate's assigned files and edits the working tree. It
   rewrites its results file after each file. The job hands over a patch and the results file.
3. **Publish job.** On a fresh runner:
   1. `apply` applies the patch, but only for the crate's assigned files.
   2. `finalize` drops findings that are malformed, about other files, or rejected before. It
      reverts edits to files the agent did not finish, formats the rest with the pinned nightly
      `rustfmt`, and runs `verify`. If either fails, every edit is reverted and its findings are
      reported unfixed.
   3. `decide` works out what to publish from live GitHub state (see [Publishing](#publishing)).
      Then the job mints the App token, posts the issue, and opens the PR.
   4. The job hands over its state update, but only if everything decide asked for was published.
4. **Record.** The record job merges the handed-over reviews into each check's `state.json` with
   compare-and-swap. Each file stores its blob before and after the agent's fix, so merging the
   fix does not trigger a re-review. The job also adds newly rejected findings, and archives the
   run under `runs/`.

### Publishing

Each crate decides on its own, from the PRs and issues it fetches right before publishing:

| Situation | Published |
| --- | --- |
| Edits survived finalize | A draft PR on `agent-checks/<check>/<crate>` with the fixes. |
| New unfixed findings | A comment on the crate's open issue for this check, or a new issue. Posted before the PR, so the PR links it. |
| Nothing new | Nothing. The review is recorded as is. |

PRs carry only fixes, and unfixed findings always go to the issue, so they outlive a merged PR.
A finding that an open agent PR or issue of the same check already lists is not repeated.

If a PR is somehow already open on the branch, which the planner prevents, the crate publishes
and records nothing, and is retried later.

Every PR and issue body the runner writes ends with a hidden `<!-- agent-check: … -->` marker. It
lists the files the PR edits and the findings. The runner trusts a marker only if all of these
hold:

- the App's login (`CLAUDE_APP_LOGIN`) wrote that body or comment,
- the item is an issue, or a PR from this repository rather than a fork,
- the marker is the last one in its text, and is well formed.

A stranger's PR or comment therefore cannot claim files, hide findings, or reject them. Each
rejected PR or issue is turned into suppressions once.

### Sharding

The weekly schedule is the cadence. Each run reviews only what needs review.

A shard is exactly one crate, the nearest `Cargo.toml` package of each file. A crate is never
split, so at most one agent per check works on a crate at a time, on the stable branch
`agent-checks/<check>/<crate>`.

- Files outside any Cargo package are not reviewed.
- Shard ids are package names, or the package's path when two packages share a name.

A crate with more pending files than its budget reviews the most urgent ones and defers the rest:

- **Within a crate,** changed files come first, but take at most half the budget while other
  files wait. Then never-reviewed files, then the rest, oldest review first.
- **Across crates,** slots alternate between crates with changed files and a rotation of the
  others, oldest last review (or never reviewed) first. Neither churn nor one big crate can starve
  the rest.

A crate waits when any of these holds; the plan summary says which:

- its PR is still open,
- another planned crate or an open agent PR may edit the same files,
- the check has `max_open_prs` PRs open,
- the run's `max_shards_per_run` is used up.

A run reviews at most `max_shards_per_run × max_files_per_shard` files per check: 96 with the
defaults.

### Fault tolerance

| Failure | Effect |
| --- | --- |
| The agent errors, times out, or hits `max_turns`. | The files it finished are recorded and their edits kept. Edits to unfinished files are reverted. |
| No results file, or an invalid one. | The crate records nothing and is offered again next week. |
| One finding is malformed, or agent text is too long. | Only that finding is dropped, and long text is clipped. The body's notes say what happened. |
| An edit does not format or compile. | Every edit in the crate is reverted, and its findings are reported unfixed. |
| Opening the PR or posting to the issue fails. | Nothing is recorded, so the review is redone rather than lost. |
| One crate fails. | The check's other crates still run, and record still records them. |
| One check's state cannot be read (access denied, corrupt, unknown schema). | That check is not planned, and the plan summary shows the error. The other checks run. Only `NoSuchKey` counts as "no state yet". |
| One check's state cannot be written. | The other checks are still recorded, then the record job fails. |
| Two writers race on a `state.json`. | S3 rejects the stale write (412, 409, or `NoSuchKey` if the object was deleted). `record` re-reads, merges again, and retries, up to 8 times. |
| The run is cancelled, or the record job fails. | The reviews are redone next time. Open PRs still claim their files, and dedup prevents repeated findings. |
| An old run is re-run. | Re-run attempts refuse plans older than 20 hours. State merges keep each file's newest review. |
| A maintainer removes the label from an open agent PR. | Its branch still marks the crate as busy. |

Scheduled and manual runs share a concurrency group, so they never overlap.

### Trust boundary

The agent can run cargo, including build scripts and tests, and can write anywhere on its runner.
So its runner holds nothing worth stealing:

- **The agent job** holds the read-only `GITHUB_TOKEN`, and the OIDC permission that the Anthropic
  token exchange needs. It hands over only a patch and a results file.
- **The publish job** runs on a fresh runner, with the runner script from its own checkout and the
  plan from the plan job. It applies the patch only for assigned files, never `.git`,
  `Cargo.lock`, symlinks or submodules, and with git hooks disabled. It compiles the edits, which
  does not run them, and only then mints the App token.
- **The AWS role** is assumed only by the plan and record jobs, which run without an environment.
  Their OIDC subject is `repo:vortex-data/vortex:ref:refs/heads/develop`. The agent's job has the
  subject `repo:vortex-data/vortex:environment:claude-automation`, so it can never assume the
  role.
- **Agent-written text** is neutralized and clipped in PR and issue bodies, so it cannot forge the
  runner's marker.

What remains: a misbehaving agent can spend its Anthropic budget, and can propose bad edits, which
still need a human to review and merge.

### Storage

Each check owns `s3://<bucket>/<prefix>/<check-id>/`. Locally, `--store` defaults to
`target/agent-checks-state`, with the same layout:

```text
<check-id>/
  state.json                                  what has been reviewed; written with compare-and-swap
  runs/<plan time>_<run id>-<attempt>/        one per run; write-once
    plan.json                                 this check's part of the plan
    <crate>/
      results.json  patch.diff  outcome       what the agent handed over
      final.json  body.md                     what finalize kept, and why
      decision.json  pr.md  issue.md          what decide published
```

**`state.json`** holds, for each reviewed file:

- the reviewed blob hashes,
- the commit and time of the review,
- the rubric version,
- its findings.

It also holds the suppressed findings, and the rejected PRs and issues already processed.

Every write reads the object and its ETag, merges, and writes with `If-Match: <etag>`, or
`If-None-Match: *` for a new object. If another writer got there first, the write is retried.
Merges are idempotent, and each file keeps its newest review, so a retry never rolls the state
back.

**`runs/`** is the history: what each agent saw, did, and published. The record job writes it
with `If-None-Match: *` only, so a re-run never rewrites history. It keeps only the files above,
each up to 5 MB. An S3 lifecycle rule on `*/runs/` can expire old runs.

Only the plan and record jobs touch the store: plan reads, record writes.

### Resetting

- **Review a check's files again:** bump its `version`.
- **Allow a rejected finding again:**

  ```bash
  python3 scripts/agent_checks.py unsuppress --store s3://<bucket>/<prefix> --check <id> <category>:<file>:<item>
  ```

  It stays allowed, even though its PR or issue stays closed.
- **Start a check over:** delete its `state.json`. The `runs/` history stays.
