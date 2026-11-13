# Agent checks

Agent checks are periodic code reviews that an agent runs on a schedule. Each check is a rubric
plus a little configuration. A runner (`scripts/agent_checks.py`) splits the repository into
per-crate shards, gives each shard to a headless Claude Code session on its own CI runner,
validates what comes back, and opens one draft pull request per shard when there is something to
change, or one issue per shard for findings it could not fix. CI runs it daily from
`.github/workflows/agent-checks.yml`.

| Check | What it asks |
| --- | --- |
| [`benchmark-usefulness`](benchmark-usefulness/CHECK.md) | Do the benchmarks measure what they claim, and are they worth their CI time? |
| [`test-focus`](test-focus/CHECK.md) | Does each test check one behavior precisely, and could it pass by accident? |

## How a run works

```text
plan ──► review (one runner per shard, in parallel) ──► record
```

1. **Plan.** The plan job reads the state branch `agent-checks-state`, all open PRs, and the
   closed PRs and issues labelled `agent-checks`. It skips checks that are not due (see
   [Cadence](#cadence)). For each remaining check:
   - It collects the in-scope files that need review. A file needs review if it is:
     - `changed`: its blob differs from every blob recorded at its last review,
     - `new`: it has never been reviewed,
     - `rubric`: the check's `version` was bumped since its last review,
     - `stale`: its last review is older than `revisit_after_days`, when that key is set.
   - It groups those files by crate.
   - It emits up to `max_shards_per_run` shards.
   - It writes the check's definition into the plan. The review jobs read the check from the plan,
     so all jobs see the same rubric and the agent cannot change it.
2. **Review.** Each shard runs on its own runner:
   - The runner builds the prompt from the rubric, the results format, and the assigned files.
     Each changed file gets a hint to run `git diff` against the commit it was last reviewed at.
   - Claude reviews the files, edits the working tree, and keeps a results JSON up to date.
   - The job formats the edited Rust files with the pinned nightly `rustfmt`.
   - `finalize` validates the results and drops findings that are malformed, outside the shard,
     or previously rejected.
   - It reverts edits outside the shard's edit scope, edits to files the agent did not finish,
     and any `Cargo.lock` change.
   - It runs the check's `verify` commands, and if one fails, reverts every edit.
   - If edits remain, the job opens a draft PR on the shard's branch; its body also lists the
     unfixed findings. With no edits but unfixed findings, it opens the shard's issue, or comments
     on it if it is already open. The job summary shows the findings and the diff either way.
3. **Record.** The record job merges each shard's reviewed files into the state branch, but only
   for shards whose PR or issue was published. For each file it stores the blob before and after
   the agent's fix (after formatting), so merging the fix does not trigger a re-review.

Maintainers reject findings in two ways: by closing an agent PR without merging it, or by
closing a findings issue as "not planned". Either one suppresses every fingerprint listed in it,
including the unfixed ones and those in the issue's comments, and future agents are told not to
raise them again. This also applies to a PR closed only because it went stale. To bring a finding
back, delete its entry from `suppressed` in the state.

### Cadence

The workflow fires daily. A check is due when its last run started at least `cadence_days` ago,
with 12 hours of slack, so a run that starts a little earlier in the day than last week's still
counts. A due run stamps `last_run` with the time the plan was made, under either condition:

- at least one of its shards is recorded,
- it had nothing to review.

So a day on which every shard fails leaves the clock alone, and the next daily run retries. Runs
with `force` or a `shard` filter review immediately but do not move the clock.

### Sharding

A shard is exactly one crate, the nearest `Cargo.toml` package of each file. A crate is never
split, so at most one agent per check works on a crate at a time. Its branch is the stable
`agent-checks/<check>/<crate>`.

Files outside any Cargo package are not reviewed. Shard ids are package names, or the package's
path when two packages share a name.

A crate can have more pending files than `max_files_per_shard` or `max_lines_per_shard` allow.
Then the most urgent files are reviewed and the rest are deferred to later runs. Within a crate,
files are ordered: changed first, then never reviewed, then the rest, oldest review first.

Crates are ordered across the repo this way:

1. crates with changed files,
2. then the remaining crates in rotation, the one reviewed longest ago (or never) first.

A big backlog in one crate therefore cannot starve the others.

Throughput is at most `max_shards_per_run × max_files_per_shard` files per check per cadence
period. With the current settings, both checks review up to 8 crates and 96 files a week:

- The first full pass of `benchmark-usefulness` (138 files in 24 crates, 67 of them in
  `vortex-array`) takes about 8 weeks.
- The first full pass of `test-focus` (800 files in 40 crates, 371 of them in `vortex-array`)
  takes about 35 weeks.

After the first pass, only changed files are reviewed. To finish the first pass sooner, raise
`max_files_per_shard` or `max_shards_per_run`, or shorten `cadence_days`.

Each shard also has an edit scope. With `edit_scope = "files"`, the default, it is the assigned
files. With `"crate"`, it is the crate directory. Scopes never overlap:

- The planner skips files that an open agent PR of any check may still change.
- It skips a crate whose branch already has an open PR.
- It skips a shard whose scope overlaps a shard already planned in this run, which can happen
  between checks.

### Fault tolerance

| Failure | Effect |
| --- | --- |
| The agent errors, times out, or hits `max_turns`. | The results file is rewritten after each file, so the files it finished are recorded and their edits kept. Edits to unfinished files are reverted. |
| No results file, or an invalid one. | The shard fails and records nothing. Its files are offered again on the next run. |
| One finding is malformed. | Only that finding is dropped, with a note in the PR body. |
| An edit does not compile. | `verify` fails, every edit in the shard is reverted, and its findings are reported unfixed. |
| Formatting, opening the PR, or opening the issue fails. | No state is recorded, so the review is redone rather than the fix or findings lost. |
| The App token would expire during a long agent run. | It is minted only after finalize, just before publishing. |
| One shard fails. | The other shards are unaffected (`fail-fast: false`), and `record` still records them. |
| Another update to the state branch wins the push race. | `record` starts again from the remote state, up to five times. |
| The state push fails for good. | The run's reviews are redone next time. Open PRs still claim their files, so no duplicate PRs are opened. |
| The state branch cannot be read. | The plan job fails instead of re-planning everything as new. |
| An old workflow run is re-run. | Plans older than 20 hours are refused, and a file keeps its newest review when state merges. |
| A maintainer removes the label from an open agent PR. | Its branch name still marks the crate as busy, and the PR step refuses to push onto a branch with an open PR. |

Runs never overlap: they share the `agent-checks` concurrency group. Only dry runs may start from a
branch other than the default branch.

### Trust boundary

The agent can run cargo, which runs build scripts, and can write anywhere in the checkout. So:

- It holds only the read-only `GITHUB_TOKEN`. The write-capable App token is minted after
  finalize, for the publish steps.
- The runner, the plan, and every output used to build PRs and state are copied to
  `$RUNNER_TEMP` before the agent starts.
- PR and issue bodies neutralize HTML comments in agent-written text. Planning reads only the
  last marker in each body, the one the runner appends, and only from PRs on `agent-checks/`
  branches.

### Storage

State is one JSON file per check (`<id>.json`) on the orphan branch `agent-checks-state`. Only
the record job writes it. Each file's entry holds:

- the reviewed blob hashes,
- the commit and time of the review,
- the rubric version,
- the fingerprints of its findings.

The state also holds the suppressed fingerprints. **TODO:** replace the branch with a store that
supports atomic per-shard writes, so a run does not need serializing.

## Setup

The workflow uses the `claude-automation` environment, the GitHub App secrets, and the Anthropic
variables that `fuzzer-fix-automation.yml` uses. The `vortex-claude-write` federation rule must
list this workflow as an allowed `job_workflow_ref`,
`vortex-data/vortex/.github/workflows/agent-checks.yml@refs/heads/develop`. Without it, every
agent fails its token exchange with a 401, records nothing, and is retried the next day.

## Adding a check

Create `.agents/checks/<id>/CHECK.md`, where `<id>` matches the `id` key. The file is TOML
frontmatter between `+++` lines, then the rubric in Markdown:

```markdown
+++
id = "my-check"
title = "My check"
description = "One sentence: what this check looks for."
include = ["**/src/**/*.rs"]
verify = ["cargo clippy -p {crate} --all-targets --all-features -- -D warnings"]
+++

You are reviewing ... (the rubric: categories, how to tell a real problem from an acceptable
one, what to fix vs. only report, and how to build fingerprints)
```

Do not restate the results format in the rubric. The runner appends it to every prompt, along
with the scope rules and the file assignment.

| Key | Default | Meaning |
| --- | --- | --- |
| `id`, `title`, `description` | required | `id` is kebab-case. It names the branches and the state file. |
| `include` | required | Repo-relative globs (`*`, `?`, `**`, `{a,b}`) over tracked files. |
| `exclude` | `[]` | Globs removed from `include`. |
| `content_match` | none | Regex that a file must contain, such as `#\[test\]`. |
| `edit_scope` | `"files"` | What the agent may edit: only its assigned `files`, or anything in its `crate`. |
| `max_files_per_shard` | `10` | Files reviewed per crate per run. The rest are deferred. |
| `max_lines_per_shard` | none | Also caps the total lines per shard. A shard always gets at least one file. |
| `max_shards_per_run` | `4` | Crates reviewed per run for this check, each on its own runner. |
| `mode` | `"incremental"` | `incremental` reviews only files that need review; `full` reviews everything. |
| `cadence_days` | `7` | Minimum days between runs of this check. The workflow fires daily. |
| `revisit_after_days` | none | Re-review unchanged files after this many days. |
| `version` | `1` | Bump this when the rubric changes enough that reviewed files should be reviewed again. |
| `action` | `"pr"` | `pr` commits the agent's edits. `report` reverts all edits and only opens the shard's findings issue. |
| `verify` | `[]` | Commands run after the agent finishes. `{crate}` is replaced by the package name. Any failure reverts all edits. |
| `labels` | `[]` | Extra PR labels, created if missing. `agent-checks` is always added. Include a `changelog/*` label. |
| `skills` | `[]` | Skills under `.agents/skills/` that the agent should load. |
| `allowed_tools` | `[]` | Extra Claude Code tools, such as `"Bash(cargo check:*)"`. Always allowed: Read, Edit, Write, Grep, Glob, Skill, `git diff/log/show/blame`, and `git checkout --` (so the agent can undo its own edits). |
| `max_findings` | `20` | Findings kept per shard, most severe first. |
| `model`, `max_turns` | workflow default, `80` | Agent settings. |
| `enabled` | `true` | Disabled checks run only when named explicitly. |

Writing a good rubric:

- Give each category a stable kebab-case slug, what to look for, why it matters, and how to tell a
  real problem from acceptable code.
- Say which categories the agent may fix and which it may only report. Only allow fixes that are
  mechanical, local, and verifiable, and have the agent compile what it touched.
- Define the fingerprint, usually `<category>:<file>:<item name>`. Keep it free of line numbers
  so it stays stable across unrelated edits.
- Tell the agent that zero findings is a good result.
- Make the rubric's own build instructions agree with `verify`. The runner tells the agent its
  `verify` commands and that any failure discards every edit in the shard.

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

In CI, run one check or one crate with **Actions → Agent Checks → Run workflow**. Set `force`
to ignore the cadence. Set `dry_run` to see the findings and diffs in the job summaries without
opening PRs or issues or updating the state. Dry runs may start from any branch.

## Resetting state

To make a check review everything again, bump its `version`. To forget a suppression, delete its
entry from the check's `suppressed` map on the `agent-checks-state` branch.
