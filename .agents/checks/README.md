# Agent checks

Agent checks are periodic code reviews that an agent runs on a schedule. Each check is a rubric
plus a little configuration. A runner (`scripts/agent_checks.py`) shards the repository, gives
each shard to a headless Claude Code session, validates what comes back, and opens a draft pull
request with the fixes. When there is nothing to commit, it opens an issue with the findings.
CI runs it daily from `.github/workflows/agent-checks.yml`.

| Check | What it asks |
| --- | --- |
| [`benchmark-usefulness`](benchmark-usefulness/CHECK.md) | Do the benchmarks measure what they claim, and are they worth their CI time? |

## How a run works

```text
plan ──► review (one job per shard, in parallel) ──► record
```

1. **plan** reads the state branch `agent-checks-state` and the list of `agent-checks` PRs. It
   skips checks whose `cadence_days` has not passed. For each remaining check it collects the
   in-scope files that need review, groups them into shards, ranks the shards, and emits a job
   matrix. A file needs review if it is:
   - `changed`: its blob differs from every blob recorded when it was last reviewed,
   - `new`: it has never been reviewed,
   - `rubric`: the check's `version` was bumped since the file's last review,
   - `stale`: its last review is older than `revisit_after_days`, when that key is set.

   Shards that already have an open PR are skipped, so PRs never pile up on one shard.
2. **review** renders the prompt: the rubric, the output contract, and the assigned files, each
   with a `git diff <last-reviewed-commit>` hint when it changed. Claude reviews the files, edits
   the working tree, and writes a results JSON. `finalize` then validates the results, drops
   findings outside the shard or previously rejected, reverts edits outside the shard's crate
   directories (and `Cargo.lock`), and builds the PR body.
3. **record** merges each shard's reviewed files into the state branch. It records the blob
   before and after the agent's fix, so merging the fix does not trigger a re-review. When a PR
   is closed without merging, its fingerprints are suppressed, and the agent is told not to
   raise those findings again.

A shard that fails, times out, or writes no results records nothing. Its files are offered again
on the next run.

## Adding a check

Create `.agents/checks/<id>/CHECK.md`, where `<id>` matches the `id` key. The file is TOML
frontmatter between `+++` lines, then the rubric in Markdown:

```markdown
+++
id = "my-check"
title = "My check"
description = "One sentence: what this check looks for."
include = ["**/src/**/*.rs"]
+++

You are reviewing ... (the rubric: categories, how to tell a real problem from an acceptable
one, what to fix vs. only report, and how to build fingerprints)
```

Do not restate the output contract in the rubric. The runner appends it to every prompt, along
with the scope rules and the file assignment.

| Key | Default | Meaning |
| --- | --- | --- |
| `id`, `title`, `description` | required | `id` is kebab-case. It names the branch (`agent-checks/<id>/<shard>`) and the state file. |
| `include` | required | Repo-relative globs (`*`, `?`, `**`, `{a,b}`) over tracked files. |
| `exclude` | `[]` | Globs removed from `include`. |
| `content_match` | none | Regex that a file must contain, such as `#\[cfg\(test\)\]`. |
| `shard_by` | `"crate"` | `crate` (nearest `Cargo.toml` package), `file`, or `none`. |
| `max_files_per_shard` | `10` | Larger crates are split into `<crate>-1`, `<crate>-2`, and so on. |
| `max_lines_per_shard` | none | Also split by total line count to bound the context per agent. |
| `max_shards_per_run` | `4` | Upper bound on agent sessions per run for this check. |
| `mode` | `"incremental"` | `incremental` reviews only files that need review; `full` reviews everything. |
| `cadence_days` | `7` | Minimum days between runs of this check. The workflow fires daily. |
| `revisit_after_days` | none | Re-review unchanged files after this many days. |
| `version` | `1` | Bump this when the rubric changes enough that reviewed files should be reviewed again. |
| `action` | `"pr"` | `pr` commits the agent's edits; `report` discards edits and only files issues. |
| `labels` | `[]` | Extra PR labels. `agent-checks` is always added. Include a `changelog/*` label. |
| `skills` | `[]` | Skills under `.agents/skills/` that the agent should load. |
| `allowed_tools` | `[]` | Extra Claude Code tools, such as `"Bash(cargo check:*)"`. Read, Edit, Write, Grep, Glob, and read-only git are always allowed. |
| `max_findings` | `20` | Findings kept per shard, most severe first. |
| `model`, `max_turns` | workflow default, `80` | Agent settings. |
| `enabled` | `true` | Disabled checks run only when named explicitly. |

Writing a good rubric:

- Give each category a stable kebab-case slug, what to look for, why it matters, and how to tell a
  real problem from acceptable code.
- Say which categories the agent may fix and which it may only report. Only allow fixes that are
  mechanical, local, and verifiable, and have the agent compile what it touched.
- Define the fingerprint, usually `<id>:<category>:<file>:<item name>`. Keep it free of line
  numbers so it stays stable across unrelated edits.
- Tell the agent that zero findings is a good result.

## Running locally

```bash
# List checks and their configuration.
python3 scripts/agent_checks.py list

# Show what the next run would review, without running an agent.
python3 scripts/agent_checks.py plan --check benchmark-usefulness --force | jq '.checks[].shards[] | {id, files: [.files[].path]}'

# Print the prompt for one shard.
python3 scripts/agent_checks.py run benchmark-usefulness --shard vortex-buffer --dry-run

# Run it with the local `claude` CLI. Edits stay in your working tree, and the PR body is written
# to target/agent-checks/<id>/<shard>.md. Local state lives in target/agent-checks-state.
python3 scripts/agent_checks.py run benchmark-usefulness --shard vortex-buffer
```

In CI, you can run one check or one shard with **Actions → Agent Checks → Run workflow**. Set
`force` to skip the cadence.

## Resetting state

State is one JSON file per check on the `agent-checks-state` branch. To make a check review
everything again, bump its `version`. To forget a suppression, delete its entry from the check's
`suppressed` map on that branch.
