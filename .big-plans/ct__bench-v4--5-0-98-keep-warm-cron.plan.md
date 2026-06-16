# Benchmarks Web Keep-Warm Cron (PR-5.0.98) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one scheduled GitHub Actions workflow that periodically fetches the benchmarks site's hot read endpoints so the Vercel Data Cache + edge CDN never go cold on this low-traffic site.

**Architecture:** A single new workflow file `.github/workflows/web-keep-warm.yml` runs every 5 minutes (plus manual `workflow_dispatch`). One `ubuntu-latest` job uses preinstalled `curl` + `jq` to GET the landing page, GET `/api/groups`, parse `.groups[].slug`, then GET each `/api/group/{slug}?n=100`. The production base URL is hardcoded; no secret or repo variable is involved (read-only public traffic). `curl --fail` makes a genuinely broken endpoint fail the run, so the keep-warm doubles as a lightweight uptime check.

**Tech Stack:** GitHub Actions (YAML), bash, `curl`, `jq` (both preinstalled on `ubuntu-latest`). No third-party actions. No checkout. Linted by `yamllint --strict -c .yamllint.yaml`.

---

## Scope notes / why this is one file

This is sub-PR-5.0.98 of the `ct/bench-v4` big-plans migration, inserted ahead of PR-5.1 via the Amend flow. It is intentionally a single low-risk file:

- The file lives under `.github/workflows/`, **not** under `benchmarks-website/web/**`, so it does **not** (and must not be made to) trigger `web-deploy.yml`.
- The only project lint that applies is `yamllint --strict -c .yamllint.yaml` (per `.github/AGENTS.md`: "All files under `.github/` are linted by yamllint --strict"). **Do NOT run Rust/`cargo`/web checks for this change** — it touches no Rust and no TypeScript.
- Match the existing workflow header convention from `.github/workflows/web-deploy.yml`: the two leading SPDX comment lines.

### yamllint `--strict` constraints already baked into the content below

From `.yamllint.yaml`:

- `braces`: empty braces must be `{ }` (exactly one space inside) — so `workflow_dispatch: { }`, never `{}`.
- `quoted-strings`: `quote-type: double` (use `"..."` when quoting; cron + URL strings are quoted).
- `empty-values`: enabled — no bare `key:` with an empty value (that is why `workflow_dispatch` uses `{ }`).
- `comments` + `comments-indentation`: `# ` with a space after the hash; comments indented to their block.
- `new-line-at-end-of-file`: file must end with exactly one trailing newline.
- `truthy: check-keys: false`: the `on:` key is allowed as-is.
- `line-length: disable`: no column cap (but keep run-block lines reasonable for readability).

---

## File Structure

- Create: `.github/workflows/web-keep-warm.yml` — the entire change. One workflow, one job, one primary run step (plus the summary write folded into the same step).

There are no other files. No tests directory (a workflow has no unit-test harness); verification is `yamllint --strict` plus an optional live network sanity check of the bash logic.

---

## Task 1: Create the keep-warm workflow

**Files:**
- Create: `.github/workflows/web-keep-warm.yml`

- [ ] **Step 1: Write the workflow file with this exact content**

```yaml
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Scheduled "keep-warm" pings for the benchmarks-website v4 read service
# (https://benchmarks-web.vercel.app). The site wraps its default ?n=100 query
# window in Vercel's Data Cache (1h backstop) and serves read-API responses
# with a 5-minute edge CDN s-maxage. On this low-traffic site real users would
# otherwise repeatedly hit cold caches and pay the ~7.8s cold-RDS path, so this
# workflow periodically fetches the hot endpoints (landing page, /api/groups,
# and each group's default last-100 bundle) to keep both cache layers warm.
#
# No secret or repo variable is needed: every request is read-only public
# traffic against the hardcoded production URL.
#
# Failure policy: requests use `curl --fail`, so a genuinely broken endpoint
# (a non-200 response or an unreachable/ unparseable /api/groups) fails the
# run. That is intentional: it doubles as a lightweight uptime signal for a
# site that otherwise sees little organic traffic. The benchmark data is
# trusted and regenerable, so this is a freshness optimization, not
# user-facing reliability hardening.

name: Benchmarks Web Keep-Warm

concurrency:
  # Never let two keep-warm runs overlap; a delayed run is simply superseded
  # by the next schedule tick.
  group: ${{ github.workflow }}
  cancel-in-progress: true

on:
  # GitHub's scheduler has a 5-minute minimum granularity and is best-effort
  # (scheduled runs are frequently delayed under load). That is acceptable
  # for a keep-warm whose only job is to prevent long cold-cache gaps.
  schedule:
    - cron: "*/5 * * * *"
  workflow_dispatch: { }

permissions:
  contents: read

env:
  BENCH_SITE_BASE_URL: "https://benchmarks-web.vercel.app"

jobs:
  keep-warm:
    name: Keep Vercel cache warm
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Warm landing page, group index, and per-group bundles
        run: |
          set -Eeuo pipefail

          base="${BENCH_SITE_BASE_URL}"

          # 1. Landing page (force-dynamic HTML; warms the route + CDN edge).
          curl --fail --silent --show-error --output /dev/null "${base}/"

          # 2. Group index. Capture the JSON so we can both warm it and read
          #    the group slugs out of it.
          groups_json="$(curl --fail --silent --show-error "${base}/api/groups")"

          # Parse slugs in a pipeline so a malformed payload trips `jq` and,
          # via `pipefail`, fails the run (the uptime-signal behavior above).
          slugs="$(printf '%s' "${groups_json}" | jq -r '.groups[].slug')"

          # 3. Each group's default last-100 bundle (the Expand-All hot path).
          count=0
          while IFS= read -r slug; do
            [ -n "${slug}" ] || continue
            encoded="$(printf '%s' "${slug}" | jq -rR '@uri')"
            curl --fail --silent --show-error --output /dev/null \
              "${base}/api/group/${encoded}?n=100"
            count=$((count + 1))
          done <<< "${slugs}"

          printf 'Warmed landing page + /api/groups + %d group bundles.\n' \
            "${count}" >> "${GITHUB_STEP_SUMMARY}"
```

Notes for the implementer (do not paste these into the file):
- `workflow_dispatch: { }` MUST keep the single inner space (yamllint `braces`).
- The `while ... done <<< "${slugs}"` loop runs in the current shell (here-string, not a pipe), so `count` survives the loop — keep it as a here-string, do not convert it to `slugs | while ...` (that subshell would lose `count`).
- `jq -rR '@uri'` URL-encodes each slug defensively; slugs are simple kebab strings today but this stays correct if a slug ever contains a reserved character.
- End the file with exactly one trailing newline.

- [ ] **Step 2: Lint the workflow with the project linter**

Run: `yamllint --strict -c .yamllint.yaml .github/workflows/web-keep-warm.yml`
Expected: no output, exit code 0.

If `yamllint` is not installed locally, install it first (`pipx install yamllint` or `uv tool install yamllint` or `pip install --user yamllint`), then re-run. Do not skip this check — it is the gating lint for `.github/` files.

- [ ] **Step 3: Confirm no whitespace errors**

Run: `git add .github/workflows/web-keep-warm.yml && git diff --check --cached`
Expected: no output (no trailing-whitespace / conflict markers).

- [ ] **Step 4: Commit**

```bash
git commit -F - <<'EOF'
ci: add benchmarks-web keep-warm cron (PR-5.0.98)

Scheduled (*/5) workflow that GETs the landing page, /api/groups, and each
/api/group/{slug}?n=100 against the hardcoded production URL so the Vercel
Data Cache + edge CDN never go cold on the low-traffic benchmarks site. No
secret or repo variable needed (read-only public traffic); curl --fail makes
a broken endpoint fail the run as a lightweight uptime signal.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

## Task 2: Live sanity check of the warm logic (best-effort, network)

This task verifies the bash/jq pipeline against the real site. It hits the network, so it is best-effort: if the runner has no outbound network, note that and skip — the workflow itself still runs on GitHub.

**Files:** none (read-only verification).

- [ ] **Step 1: Confirm `/api/groups` parses and slugs resolve**

Run:
```bash
base="https://benchmarks-web.vercel.app"
groups_json="$(curl --fail --silent --show-error "${base}/api/groups")"
slugs="$(printf '%s' "${groups_json}" | jq -r '.groups[].slug')"
printf '%s\n' "${slugs}"
test -n "${slugs}"
```
Expected: a non-empty list of kebab-case group slugs (one per line), exit code 0.

- [ ] **Step 2: Confirm one bundle URL resolves**

Run:
```bash
base="https://benchmarks-web.vercel.app"
first="$(printf '%s' "${slugs}" | head -n1)"
encoded="$(printf '%s' "${first}" | jq -rR '@uri')"
curl --fail --silent --show-error --output /dev/null -w '%{http_code}\n' \
  "${base}/api/group/${encoded}?n=100"
```
Expected: `200`.

If either step fails because of sandbox network restrictions (not a site problem), record that the check was network-blocked and rely on the workflow's own first scheduled/dispatched run for live confirmation.

---

## Self-Review checklist (completed by plan author)

- **Spec coverage:** schedule `*/5` + `workflow_dispatch` ✓; `permissions: contents: read` ✓; `ubuntu-latest` + `timeout-minutes: 10` ✓; hardcoded base URL, no secret/var ✓; GET `/`, GET `/api/groups`, GET each `/api/group/{slug}?n=100` ✓; `jq` slug parse ✓; defensive URL-encode ✓; `set -Eeuo pipefail` ✓; `curl --fail` failure policy documented ✓; `$GITHUB_STEP_SUMMARY` count ✓; SPDX header lines ✓; yamllint verification step ✓; `git diff --check` ✓; no third-party actions ✓; does not touch `benchmarks-website/web/**` ✓.
- **Placeholder scan:** none — the full file content is inline.
- **Type/identifier consistency:** `BENCH_SITE_BASE_URL`, `base`, `groups_json`, `slugs`, `encoded`, `count` are used consistently across the run block.
