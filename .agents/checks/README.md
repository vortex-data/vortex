# Agent checks

Agent checks are periodic code reviews that an agent runs on a schedule. Each check is a rubric
plus a little configuration. A runner (`scripts/agent_checks.py`) splits the repository into
per-crate shards and gives each shard to a headless Claude Code session. It validates what
comes back, then opens one draft pull request per shard for the fixes and keeps one issue per
shard for the findings it could not fix. CI runs it weekly.

| Check | What it asks |
| --- | --- |
| [`benchmark-usefulness`](benchmark-usefulness/CHECK.md) | Do the benchmarks measure what they claim, and are they worth their CI time? |
| [`test-focus`](test-focus/CHECK.md) | Does each test check one behavior precisely, and could it pass by accident? |

Three workflows run it:

| Workflow | Runs |
| --- | --- |
| `.github/workflows/agent-checks.yml` | The weekly trigger: plan, then one `review` per check in parallel, then record. |
| `.github/workflows/agent-checks-review.yml` | One check's shards, one after another (`max-parallel: 1`). |
| `.github/workflows/agent-checks-shard.yml` | One shard: an `agent` job, then a `publish` job on a fresh runner. |

## How a run works

```text
plan ──► per check, in parallel: shard ──► shard ──► … (one at a time) ──► record
                                   │
                                   └─ agent job ──► publish job (fresh runner)
```

1. **Plan.** The plan job reads each check's state from S3, all open PRs, and the closed PRs and
   issues labelled `agent-checks`. It plans the checks that are due (see [Cadence](#cadence)).
   For each one:
   - It collects the in-scope files that need review. A file needs review if it is:
     - `changed`: its blob differs from every blob recorded at its last review,
     - `new`: it has never been reviewed,
     - `rubric`: the check's `version` was bumped since its last review,
     - `stale`: its last review is older than `revisit_after_days`, when that key is set.
   - It groups those files by crate, and picks up to `max_shards_per_run` shards (see
     [Sharding](#sharding)).
   - It writes the check's definition into the plan. Later jobs read the check from the plan, so
     every job of the run sees the same rubric, and the agent cannot change it.
2. **Agent job,** one per shard. The runner builds the prompt from the rubric, the results format,
   and the assigned files. Each changed file gets a hint to run `git diff` against the commit it
   was last reviewed at. Claude reviews the files, edits the working tree, and rewrites a results
   JSON after each file. The job uploads the agent's edits as a patch, plus the results file.
3. **Publish job,** one per shard, on a fresh runner with a fresh checkout. It:
   1. applies the patch, but only for paths the shard may edit;
   2. runs `finalize`, which:
      - drops findings that are malformed, outside the shard, or previously rejected;
      - reverts edits outside the scope, edits to files the agent did not finish, and any
        `Cargo.lock` change;
      - formats the edited Rust files with the pinned nightly `rustfmt`, and runs the check's
        `verify` commands. If either fails, it reverts every edit and reports the findings
        unfixed;
   3. runs `decide` against live GitHub state (see [Publishing](#publishing));
   4. mints the App token, and opens the PR and/or posts to the issue;
   5. hands the shard's state update to the record job, but only if everything it had to
      publish was published.
4. **Record.** The record job merges the handed-over reviews into each check's state document in
   S3 with compare-and-swap. For each file it stores the blob before and after the agent's fix,
   so merging the fix does not trigger a re-review. It also adds the findings that maintainers
   rejected, and moves the weekly clock.

### Publishing

Each shard decides on its own, from the open PRs and issues it fetches right before publishing:

| Situation | PR | Issue |
| --- | --- | --- |
| Edits survived finalize, and no PR is open on `agent-checks/<check>/<crate>` | `open`: a draft PR with the fixes | — |
| Edits survived, but a PR is already open on that branch | `blocked`: nothing is published or recorded | — |
| Unfixed findings that no open agent PR or issue of this check lists yet | — | `comment` on the shard's open issue, or `create` one |
| Nothing new | `none` | `none`: the review is recorded as is |

PRs carry only fixes. Unfixed findings always go to the shard's issue, so they outlive a merged
PR. Findings are deduplicated by fingerprint, so a run never repeats what an open PR or issue
already says. That relies on the agent building the same fingerprint for the same problem, which
is why rubrics define fingerprints from names, not line numbers.

Only the App's own markers count. Every PR and issue body the runner writes ends with a hidden
`<!-- agent-check: … -->` marker listing the shard, the files the PR may change, and the
fingerprints. The runner trusts a marker only if all of these hold:

- the App's login (`CLAUDE_APP_LOGIN`) wrote that body or comment,
- the item is an issue, or a PR from this repository rather than a fork,
- the marker is the last one in its text, and is well formed.

So a stranger's PR or comment cannot claim files, hide findings, or suppress them.

Maintainers reject findings in two ways: by closing an agent PR without merging it, or by closing
an issue as "not planned". Either one suppresses every fingerprint the App listed in it, and
future agents are told not to raise them again. Each rejected PR or issue is processed once. To
bring a finding back, delete it from `suppressed` in the state; it stays deleted.

### Cadence

The workflow fires weekly, on Mondays. A check is due when its last scheduled run started at
least `cadence_days` ago, with 12 hours of slack. A scheduled run of a due check stamps
`last_run` with the plan's start time, under either condition:

- at least one of its shards is recorded,
- it had nothing to review.

So a week in which every shard fails leaves the clock alone, and the next weekly run retries.

Manual runs never move the clock, whether forced, filtered to some crates, or plain. A Wednesday
re-run therefore does not push the next scheduled run back a week.

### Sharding

A shard is exactly one crate, the nearest `Cargo.toml` package of each file. A crate is never
split, so at most one agent per check works on a crate at a time, on the stable branch
`agent-checks/<check>/<crate>`. Files outside any Cargo package are not reviewed. Shard ids are
package names, or the package's path when two packages share a name.

When a crate has more pending files than `max_files_per_shard` or `max_lines_per_shard` allow,
the most urgent files are reviewed and the rest are deferred:

- **Within a crate,** changed files come first but take at most half the file budget while other
  files are waiting. The rest are never-reviewed files, then the remaining ones, oldest review
  first.
- **Across crates,** slots alternate between crates with changed files and the rotation of all
  the other crates, oldest last review (or never reviewed) first.

So neither churn nor one crate's backlog can starve the others.

Each shard has an edit scope: its assigned files (`edit_scope = "files"`, the default), or its
crate directory (`"crate"`). Scopes never overlap:

- The planner skips files that an open agent PR of any check may still change, and a crate whose
  branch already has an open PR.
- It skips a shard whose scope overlaps one already planned in this run, from any check. A
  `"crate"`-scoped check therefore holds whole crates for the run, and while its PR is open.

Throughput is at most `max_shards_per_run × max_files_per_shard` files per check per week. With
the current settings, that is 8 crates and 96 files. With no code changing in the meantime, the
planner simulated over consecutive weeks gives these first full passes:

- `benchmark-usefulness`: 138 files in 24 crates, 67 of them in `vortex-array`. About 8 weeks.
- `test-focus`: 800 files in 40 crates, 371 of them in `vortex-array`. About 39 weeks.

Churn and crates blocked by open PRs make this longer. To finish sooner, raise
`max_files_per_shard` or `max_shards_per_run`. After the first pass, only changed files are
reviewed.

### Fault tolerance

| Failure | Effect |
| --- | --- |
| The agent errors, times out, or hits `max_turns`. | The results file is rewritten after each file, so the files it finished are recorded and their edits kept. Edits to files it did not finish are reverted, whatever the edit scope. |
| No results file, or an invalid one. | The shard records nothing. Its files are offered again on the next run. |
| One finding is malformed, or agent text is too long. | Only that finding is dropped; long text is clipped. Notes in the body say what happened. |
| An edit does not format or compile. | Every edit in the shard is reverted, and its findings are reported unfixed. |
| Opening the PR or posting to the issue fails, or the PR is `blocked`. | No state is handed over, so the review is redone rather than the fix or findings lost. |
| One shard fails. | The check's other shards still run (`fail-fast: false`), and `record` still records them. |
| One check's state cannot be read (access denied, corrupt, unknown schema). | That check is not planned and the run shows an error. The other checks run. Only `NoSuchKey` counts as "no state yet". |
| One check's state cannot be written. | The other checks are still recorded, then the record job fails. |
| Another writer changes a state document between `record`'s read and write. | S3 rejects the conditional write (412, 409, or `NoSuchKey` if it was deleted), and `record` re-reads, merges again, and retries, up to 8 times with backoff. |
| The run is cancelled, or the record job fails. | The run's reviews are redone next time. Open PRs still claim their files, and `decide` dedups findings, so nothing is published twice. |
| An old workflow run is re-run. | Re-run attempts refuse plans older than 20 hours. A file keeps its newest review when state merges. First attempts are never refused: a check's last shards may start many hours after planning. |
| A maintainer removes the label from an open agent PR. | Its branch name still marks the crate as busy, and the PR step refuses to push onto a branch with an open PR. |

Runs never overlap: scheduled and manual runs share a concurrency group.

### Trust boundary

The agent can run cargo, which runs build scripts and tests, and can write anywhere on its runner.
So the agent's runner holds nothing worth stealing:

- **The agent job** holds only the read-only `GITHUB_TOKEN` and the OIDC permission that the
  Anthropic token exchange needs. It references none of the `claude-automation` secrets, and
  hands over only a patch and a results file.
- **The publish job** runs on a fresh runner with a fresh checkout. It runs the runner script from
  that checkout and the plan from the plan job, never anything the agent wrote. It applies the
  patch only for paths inside the shard's scope, never `.git`, `Cargo.lock`, symlinks or
  submodules, and with git hooks disabled. Then it formats and compiles the edits, which does not
  run them, and only then mints the App token.
- **The state store's role** is assumed only by the plan and record jobs of `agent-checks.yml`,
  which run without an environment. Their OIDC subject is
  `repo:vortex-data/vortex:ref:refs/heads/develop`. Trust the role for that subject only: the
  agent's job has the subject `repo:vortex-data/vortex:environment:claude-automation`, so it can
  never assume the role.
- **Agent-written text** in PR and issue bodies has its HTML comments neutralized and its length
  clipped, so it cannot forge the runner's marker.

What remains: a misbehaving agent can spend its own Anthropic budget, and can propose bad edits,
which still need a human to review and merge the draft PR.

### Storage

State is one JSON document per check, at `s3://<bucket>/<prefix>/<check-id>/state.json`. Each
check has its own prefix, so writers to different checks never contend. Locally, `--state`
defaults to the directory `target/agent-checks-state`, with the same layout.

Every write is compare-and-swap:

1. `record` reads the document and its ETag.
2. It merges the update.
3. It writes with `If-Match: <etag>`, or `If-None-Match: *` when the document does not exist yet.
4. If another writer got there first, S3 rejects the write, and `record` re-reads and tries again.

Merges are idempotent, and a file keeps its newest review, so a retry can never roll the state
back. Only the record job writes; the plan job reads.

Each file's entry holds:

- the reviewed blob hashes,
- the commit and time of the review,
- the rubric version,
- the fingerprints of its findings.

The document also holds:

- the suppressed fingerprints,
- the numbers of the rejected PRs and issues already processed (`suppression_refs`),
- the check's `last_run`.

Turn on bucket versioning to keep the history and to roll back a bad write.

## Setup

**GitHub.** The workflows use these, the same as `fuzzer-fix-automation.yml`:

- the `claude-automation` environment, with the GitHub App secrets,
- the `CLAUDE_APP_LOGIN` variable,
- the Anthropic variables.

Only `CLAUDE_APP_LOGIN`'s markers are trusted, and non-dry runs refuse to start without it.

**Anthropic.** The `vortex-claude-write` federation rule must allow the shard workflow as
`job_workflow_ref`: `vortex-data/vortex/.github/workflows/agent-checks-shard.yml@refs/heads/develop`.
For a dry run from another branch, also allow `@refs/heads/<branch>`, or `@refs/pull/<n>/merge`
for a PR. Without it, every agent fails its token exchange with a 401, records nothing, and is
retried the next week.

**AWS.** The state store needs these repository variables:

| Variable | Value |
| --- | --- |
| `AGENT_CHECKS_STATE_URI` | `s3://<bucket>/<prefix>`, for example `s3://vortex-ci/agent-checks` |
| `AGENT_CHECKS_AWS_ROLE_ARN` | A role for GitHub's OIDC provider whose trust policy requires `token.actions.githubusercontent.com:sub` = `repo:vortex-data/vortex:ref:refs/heads/develop` and `aud` = `sts.amazonaws.com` |
| `AGENT_CHECKS_AWS_REGION` | The bucket's region (default `us-east-1`) |

The role needs `s3:GetObject` and `s3:PutObject` on `<prefix>/*`, and `s3:ListBucket` on the
bucket for that prefix. Without `ListBucket`, S3 answers a missing key with 403 instead of 404,
and the runner refuses to treat 403 as empty state. Using S3 locally needs an AWS CLI v2 recent
enough to support `put-object --if-match` (late 2024 or newer).

**Dry runs** may start from any branch. Without `AGENT_CHECKS_STATE_URI` they plan from empty
state. With it, the role's subject condition rejects other branches, so leave it unset for branch
dry runs, or widen the trust deliberately.

**Before merging the PR that adds this,** remove the temporary `pull_request` trigger and the
trial-only lines in `agent-checks.yml`. They are marked `TEMPORARY`.

## Adding a check

Create `.agents/checks/<id>/CHECK.md`, where `<id>` matches the `id` key, and add a row to the
table at the top of this file. `CHECK.md` is TOML frontmatter between `+++` lines, then the rubric
in Markdown:

```markdown
+++
id = "my-check"
title = "My check"
description = "One sentence: what this check looks for."
include = ["**/src/**/*.rs"]
# CI cannot build these crates on a plain runner; see the existing checks for the full list.
exclude = ["vortex-cuda/**", "vortex-duckdb/**", "vortex-ffi/cmake/**"]
verify = ["cargo clippy -p {crate} --all-targets --all-features -- -D warnings"]
labels = ["changelog/chore"]
+++

You are reviewing ... (the rubric: categories, how to tell a real problem from an acceptable
one, what to fix vs. only report, and how to build fingerprints)
```

`test_repository_checks_parse` in `scripts/tests/test_agent_checks.py` checks that the file
parses. Do not restate the results format in the rubric. The runner appends it to every prompt,
along with the scope rules, the `verify` commands, and the file assignment.

| Key | Default | Meaning |
| --- | --- | --- |
| `id`, `title`, `description` | required | `id` is kebab-case. It names the branches and the state document. |
| `include` | required | Repo-relative globs (`*`, `?`, `**`, `{a,b}`) over tracked files. |
| `exclude` | `[]` | Globs removed from `include`. |
| `content_match` | none | Regex that a file must contain, such as `#\[test\]`. |
| `edit_scope` | `"files"` | What the agent may edit: only its assigned `files`, or anything in its `crate`. A `crate` check holds whole crates, so other checks skip them meanwhile. |
| `max_files_per_shard` | `10` | Files reviewed per crate per run. The rest are deferred. |
| `max_lines_per_shard` | none | Also caps the total lines per shard. A shard always gets at least one file. |
| `max_shards_per_run` | `4` | Crates reviewed per run, one after another, each on its own runners. |
| `mode` | `"incremental"` | `incremental` reviews only files that need review; `full` reviews everything. |
| `cadence_days` | `7` | Minimum days between scheduled runs. The workflow fires weekly, so use multiples of 7. |
| `revisit_after_days` | none | Re-review unchanged files after this many days. |
| `version` | `1` | Bump this when the rubric changes enough that reviewed files should be reviewed again. |
| `action` | `"pr"` | `pr` publishes the agent's fixes. `report` reverts all edits and only posts findings to the issue. |
| `verify` | `[]` | Commands run on the edits. `{crate}` becomes the package name. Each runs without a shell, so no pipes, `&&`, or `VAR=value` prefixes. They must work on a plain runner with only Rust set up. Any failure reverts all edits. The agent is allowed to run exactly these commands. |
| `labels` | `[]` | Extra PR labels, created if missing. `agent-checks` is always added. Include a `changelog/*` label. |
| `skills` | `[]` | Skills under `.agents/skills/` that the agent should load. |
| `allowed_tools` | `[]` | Extra Claude Code tools, such as `"Bash(cargo check:*)"`. Always allowed: Read, Edit, Write, Grep, Glob, Skill, `git diff/log/show/blame`, and `git checkout --` (so the agent can undo its own edits). |
| `max_findings` | `20` | Findings kept per shard, most severe first. |
| `model`, `max_turns` | workflow default, `80` | Agent settings. |
| `enabled` | `true` | Disabled checks run only when named explicitly. |

A `Bash(<prefix>:*)` tool allows any command that starts with that prefix. A prefix such as
`env X=:*` therefore allows any shell command. The agent's runner holds no secrets, but keep
prefixes specific anyway.

Writing a good rubric:

- Give each category a stable kebab-case slug, what to look for, why it matters, and how to tell a
  real problem from acceptable code.
- Say which categories the agent may fix and which it may only report. Only allow fixes that are
  mechanical, local, and verifiable.
- Define the fingerprint, usually `<category>:<file>:<item name>`. Keep it free of line numbers
  so it stays stable across unrelated edits. Dedup and suppression both depend on it.
- Tell the agent that zero findings is a good result.
- Make the rubric's own build instructions agree with `verify`.

## Running it

```bash
# List checks and their configuration.
python3 scripts/agent_checks.py list

# Show what the next run would review, without running an agent.
python3 scripts/agent_checks.py plan --check benchmark-usefulness --force \
  | jq '.checks[].shards[] | {id, deferred, files: [.files[].path]}'

# Print the prompt for one crate.
python3 scripts/agent_checks.py run benchmark-usefulness --shard vortex-buffer --dry-run

# Run it with the local `claude` CLI. Edits stay in your working tree, and the PR body is written
# to target/agent-checks/<id>/<crate>.md. Local state lives in target/agent-checks-state.
python3 scripts/agent_checks.py run benchmark-usefulness --shard vortex-buffer
```

In CI, run one check or one crate with **Actions → Agent Checks → Run workflow**:

- `force` ignores the cadence.
- `dry_run` shows the findings, decisions and diffs in the job summaries, without opening PRs or
  issues or updating the state.

## Resetting state

- **Review a check's files again:** bump the check's `version`.
- **Forget a suppression:** delete its entry from `suppressed` in
  `s3://<bucket>/<prefix>/<check-id>/state.json`. It stays deleted.
- **Start a check over:** delete that object.
