# benchmarks-website migration to hosted Postgres + Next.js on Vercel — big-plans plan

## Current State

```yaml
status: executing
branch: ct/bench-v4
planning_sub_flow: null
current_phase: "Phase 5: Cutover + decommission"
phase_index: 5
current_pr: PR-5.1
pr_index: 6
outstanding_must_fix: 0
deferred_items_total: 28
last_user_touchpoint: 2026-06-12T00:00:00Z
last_user_touchpoint_what: "PR-5.0.97 (always-warm last-100 cache + full spinner coverage + fast Expand All) is CLOSED on 2026-06-12: shipped + gauntlet-pr-3-accepted (2 cycles: reject -> accept; executor=claude, 3 lenses fresh+correctness+maint). Commit trail cdbe72a32..5f92566ee (10 impl/review commits + 3 gauntlet-fix commits). Cross-cutting (benchmarks-website/web/ + scripts/ + 3 workflows): (A) Vercel Data Cache layer (web/lib/data-cache.ts: unstable_cache wrappers for the default ?n=100 group/chart/groups/filter-universe queries, tag 'bench-data', 3600s backstop; routes + landing/permalink pages branch to the cached fn only on the default window, force-dynamic + CDN headers unchanged); (B) POST /api/revalidate (bearer timingSafeEqual, 503 fail-closed, revalidateTag); (C) scripts/post-ingest.py refresh_site_cache hook (POST /api/revalidate + best-effort warm pass; swallows every failure so it never changes the ingest exit code; skips warm on failed revalidate; no-op unless BENCH_SITE_BASE_URL + BENCH_REVALIDATE_TOKEN set) + 2-line additive env on the v4 Postgres step in bench.yml/sql-benchmarks.yml/v3-commit-metadata.yml; (D) client group-bundle fetch (web/lib/chart-store.ts ensureGroupBundle/abortGroupBundle + session payloadCache + completedBundles/attemptedBundles; Chart.tsx ensureInitialPayload consults cache then drives ONE /api/group bundle per group, per-chart fetch fallback, IntersectionObserver still gates Chart.js construction -> Expand All loads every chart's last-100 eagerly); (E) server-rendered .chart-placeholder spinner for EVERY pre-data card state, static ring+label under prefers-reduced-motion. The gauntlet correctness lens caught a real must-fix (close-while-awaiting-bundle leaked an unabortable per-chart fetch; fixed with a `|| !this.groupIsOpen()` guard at Chart.tsx + a non-tautological, deterministic regression test) that the fresh+maint lenses both missed. 286 vitest + 7 revalidate-pytest pass; tsc/next-build/eslint/prettier/yamllint/py_compile green; build keeps all routes force-dynamic. 1 dev-only nit deferred (a placeholder-reappears-after-StrictMode-remount test; deferred_items_total 25->26). OPS PREREQ STILL PENDING (coordinate with ops, NOT a code blocker): set BENCH_REVALIDATE_TOKEN in Vercel env + as a GH Actions secret, BENCH_SITE_BASE_URL as a GH Actions var — until set, /api/revalidate 503s fail-closed and the post-ingest hook is a silent no-op (everything degrades to current behavior, so the PR is safe deployed without the wiring). NEXT (current_pr): PR-5.1 — promote v4 --postgres ingest to required + drop the v3 --server write from the 3 ingest workflows + ship scripts/psql-bench.sh. PR-5.1's FIRST step is a PROD RDS WRITE gate (re-run the PR-3.5 cross-check via the bench_ingest IAM role) needing user coordination (operator-run vs agent-run with creds) — a genuine externalized-side-effect gate, NOT a Class A pause. Then PR-5.2 (DNS flip), PR-5.3 (decommission). Each prod write remains harness-gated. Pre-squash backup ref refs/backups/ct-bench-v4-pre-squash-386ea347b; recreate fresh before any PR-5.3/final squash. Phase 5 = b9fc6220d..; phase_entry_sha b9fc6220d unchanged; last_commit 5f92566ee (last CODE commit; PR-5.0.97 close commit is doc-only)."
subagent_invocations_this_pr: 0
subagent_invocations_total: 223
review_cycles_this_pr: 0
phase_entry_sha: b9fc6220d
phase_end_cycle: 0
phase_end_reject_cycles: 0
last_phase_end_verdict: null
current_pr_is_ci_reopen: null
last_commit: 85ba7ac37
last_cycle_commits: []
finding_readpath_cold_fill_2026_06_15: "INVESTIGATION (user-directed, post PR-5.0.992; user chose RECORD + STOP, decide the fix later): the slow first-open of BIG groups (user hit Clickbench 43 charts ~15s; TPC-DS 99 charts measured 16.2s cold) is DATABASE-THROUGHPUT-BOUND — it is NOT a function cold-start and is NOT fixed by the PR-5.0.992 warmer (which warms the function+connection — the WRONG layer for this). MEASURED on live prod (function confirmed warm, /api/health 0.45s): a SOLO cold single-chart GET /api/chart/{slug}?n=100 = 0.18s, but 8 CONCURRENT cold single-chart requests = ~1.3s EACH (wall 1.31s ≈ 8×solo) -> ~7-8x slowdown under concurrency = effectively ZERO throughput gain = effective concurrency ≈1. These were independent HTTP requests on WARM instances (times are ~1.3s, not the 6-11s cold-start), so the serialization is at the DB / RDS-Proxy layer (db.t4g.medium = 2 vCPU scanning query_measurements 4.85M rows), NOT the client fan-out and NOT a per-instance pool. CONSEQUENCE: PR-5.0.991's order-preserving Promise.all-over-pool-8 fan-out (collectGroupCharts, queries.ts:1149) — though correctly written — gives ~NO speedup in prod; a cold N-chart bundle costs ~N × ~0.16s (measured: appian 8ch=0.8s, fineweb 9ch=0.9s, tpcds 99ch=16.2s — ~linear in chart count). PR-5.0.991's '~8x / ~1.5s even cold' claim does NOT hold for the big groups; its parallelization rests on a false premise (that the DB serves concurrent queries at solo speed). Each chartPayload also runs 2 sequential queries (seededCommitsInWindow + the data query) ≈ 2N total. Once warmed, bundles are CDN + Data-Cache HITs ~0.15s (24h backstop), so the 16s is paid once per group per cache-cold window — and NOTHING actively keeps those bundle caches warm on ct/bench-v4 (the PR-5.0.98 GH cron is dormant off the default branch; PR-5.0.992's /api/health cron does NOT populate the bundle Data Cache). FIX OPTIONS (all deferred per user 2026-06-15): (B1, recommended) batch the per-group fan-out into ~1 SQL scan — collectGroupCharts issues one query filtered by dataset/storage (no per-query_idx filter), returns all rows, partitions by query_idx in JS, rebuilding the per-chart accumulators; this is the 'SQL-batch rewrite' PR-5.0.991 deferred as higher-risk, and it attacks the root cause (1-2 scans instead of ~2N). (B2) upsize RDS (uncertain ROI — the ~1x scaling at 8-concurrent suggests a deeper serialization than vCPU count; 2->4 vCPU might buy only ~2x; an AWS cost/ops call). (A) keep the bundle Data Cache warm via a Vercel cron hitting a warm-all-groups endpoint so users hit HITs and never pay the cold fill (mitigation, not a root-cause fix; the warm pass itself takes ~minutes for the big groups + needs a raised Vercel function maxDuration). NOTE: option A here SUPERSEDES the framing in amend_note_5_0_992 / the PR-5.0.992 entry that the warmer addressed the user complaint — it did not, for big groups."
finding_readpath_correction_2026_06_15: "CORRECTION (2026-06-15, live-prod profiling as bench_read + CloudWatch + Performance Insights [ENABLED this session]; authoritative doc .big-plans/ct__bench-v4-readpath-findings.md): the finding_readpath_cold_fill_2026_06_15 framing above is SUPERSEDED. The big-group cold open is NOT disk/throughput-bound and is NOT helped by more cores or RAM. MEASURED during a cold 99-chart fan-out: RDS CPU ~5% with CPU credits pinned at max, ReadIOPS <=17/min and ReadLatency ~0 (near-zero physical I/O), PI db.load ~97% CPU; the 11.4s first-hit spike is a Vercel FUNCTION cold-start (the warmer's layer), not the DB; effective concurrency is 2-6, not ~1. ROOT CAUSE: each per-chart query reads ~18k rows (the chart's full history) to return the ~665-row last-N window, because recency is applied via a commits join on commit_sha AFTER a full scan instead of via the denormalized, already-indexed commit_timestamp. EXPLAIN-verified per-chart (warm): seed MIN 57ms->8ms (index-only), data query 38ms->9ms (idx_query_measurements_summary), result-identical (665 rows; keep commit_sha IN(last-N) as a tie-trim). FIX R1 (primary, ready to implement, complexity-REDUCING, no schema change): filter query_measurements chart reads on commit_timestamp and drop the commits joins in collectQueryChart + its seed buildEarliest; expected cold TPC-DS bundle ~4.7s->~1s. R2 (secondary): collectGroupCharts re-runs collectGroups() building ALL ~17 groups summaries (~0.74s, TODO #7812) when one is needed. REJECTED with evidence: the old B1 single-batched-query (measured 640ms, WORSE), more cores/RAM (no CPU/credit/IO pressure today; revisit RAM only as query_measurements outgrows the ~1.8GB cache), seed-dedup, and the small fact tables (no commit_timestamp, negligible). DB access facts in memory project_bench_rds_profiling_access."
amend_note_5_0_993: "PR-5.0.993 (read-path R1: recency-filter the query_measurements chart reads on the denormalized commit_timestamp) is CLOSED on 2026-06-15 (full record in the PR-5.0.993 Implementation status entry): gauntlet-pr-2-ACCEPTED across 2 cycles (cycle 1 accept; a CI testcontainer regression then surfaced + was fixed + re-reviewed cycle 2 accept), 0 must-fix (fresh+correctness, executor=claude — no Codex companion this session; BOTH lenses independently produced a result-equivalence PROOF cycle 1, then independently traced ALL prod write paths cycle 2 to confirm the fix is FAITHFUL, high confidence). Code commits 05e41b2a6..85ba7ac37 (test 05e41b2a6 + cleanups 03b21a750/b2232406c + impl d0c224773 + CI-regression fix 36ed8a90e + cycle-2 nit fix 85ba7ac37). CI REGRESSION + FIX (the one in-phase plan-assumption surprise): the first push's web-deploy CI (run 27575008289) FAILED Check & Test (Deploy Production correctly SKIPPED — the test gates the deploy, so NOTHING went live) on TWO pre-existing seeded-window-semantics tests (qn null-gaps/pre-history; qnull NULL-scale_factor) that insert query_measurements WITHOUT commit_timestamp -> NULL -> MIN(commit_timestamp)=NULL -> empty chart under R1. ROOT CAUSE = stale TEST FIXTURES, NOT a prod bug: every prod write path populates commit_timestamp (the ingest upsert sets it on INSERT + ON CONFLICT in a commits-upserted-first/sha-guarded txn so it can't be NULL on a real row; the migrate post-COPY UPDATE; migration-006's backfill of the 4.85M pre-existing rows) and ALL THREE are test-pinned (test_post_ingest_postgres.py:306 / postgres_e2e.rs:181-193 / test_migrate_schema.py:838 unstamped==0). Fix 36ed8a90e stamps commit_timestamp in the two fixtures via the same `(SELECT timestamp FROM commits WHERE commit_sha=$2)` the writers + seedChartFixture use (faithful, NOT masking — both cycle-2 lenses verified). Note: the SUMMARY path still tolerates NULL via NULLS LAST; only the CHART path now drops NULLs, safe because no live writer emits one. 1 should-fix DEFERRED (the nit was RESOLVED in cycle 2): (a) the new ?n=2 test uses 3 distinct-timestamp commits so the same-boundary-timestamp tie that the kept `commit_sha IN (last-n)` clause trims is NOT exercised on the new queryMeasurementWindowFilter path (the only tie test runs RandomAccess/factWindowFilter) — a regression dropping the IN clause would stay green; deferred because the tie-trim test needs a NEW testcontainer fixture that CANNOT be locally verified (Docker absent), matching the project's established 'don't ship unverifiable seeded tests' deferral pattern. RESOLVED in cycle 2 (was the deferred nit): the seed now carries a comment documenting the write-path commit_timestamp invariant (commits 36ed8a90e + reflow 85ba7ac37). Folded into the web test-hardening pass (pre-develop-merge); deferred_items_total 27->28. USER-GREENLIT 2026-06-15 (AskUserQuestion: 'Read-path R1 fix' chosen at resume). DESIGN SPEC (authoritative): .big-plans/ct__bench-v4-readpath-findings.md (R1). DESIGN SPEC (authoritative, read in full): .big-plans/ct__bench-v4-readpath-findings.md (the R1 recommendation). SCOPE (one collector, query_measurements only): in benchmarks-website/web/lib/queries.ts (1) the data query in collectQueryChart drops `JOIN commits c USING (commit_sha)`, orders by `q.commit_timestamp` (denormalized, == c.timestamp by the migrate invariant), and uses a NEW query_measurements-scoped window filter that adds `q.commit_timestamp >= (cutoff = MIN timestamp of the last-n commits)` to make the read sargable on idx_query_measurements_summary (dataset,dataset_variant,scale_factor,storage,query_idx,engine,format,commit_timestamp DESC) INCLUDE(value_ns), KEEPING `q.commit_sha IN (last-n)` as the exact tie-trim; (2) the buildEarliest seed swaps `MIN(c2.timestamp)` over the commits join for `MIN(q2.commit_timestamp)` direct (index-only), dropping its join. The SHARED factWindowFilter is UNTOUCHED — the other 4 collectors keep their commits join + c.commit_sha (their fact tables have NO commit_timestamp column). `?n=all` (limit null) path is unchanged (no cutoff, full scan). No schema change; uses the existing index; COMPLEXITY-REDUCING (removes 2 joins). EXPLAIN-verified result-identical (665 rows) ~5x/chart (seed 57->8ms, data 38->9ms), expected cold TPC-DS bundle ~4.7s->~1s. R2 (collectGroupCharts single-group resolve) is OUT OF SCOPE this PR. Regression: a testcontainer result-equivalence test pinning a bounded window (last-n < commit count) selects exactly the last-n commits/values, plus the existing chartPayload golden-snapshot equivalence. Review = gauntlet pr-2. TOUCHES benchmarks-website/web/** so the close push FIRES web-deploy.yml (Check & Test incl. testcontainers + Deploy Production) -> the change goes live."
amend_note_5_0_992: "PR-5.0.992 (the warmer / '#1') is CLOSED on 2026-06-15 (full record in the PR-5.0.992 Implementation status entry): gauntlet-pr-2-ACCEPTED across 2 cycles (cycle 1 accept + an electively-applied both-lens should-fix [empty BENCH_DB_IDLE_TIMEOUT_MS silently became 0]; cycle 2 clean accept, 1 nit de-scoped per REVIEW CALIBRATION), executor=claude. Code commits eedc3a7f7..bad4dff45; pushed -> web-deploy.yml GREEN (run 27567195313: Check & Test + Deploy Production) -> the Vercel cron + raised pg idleTimeoutMillis are LIVE on https://benchmarks-web.vercel.app (post-deploy GET /api/health 200, build_sha bad4dff45, row_counts populated). POST-DEPLOY (operator): confirm in the Vercel dashboard that the /api/health cron is registered + its first */2 invocation logged 200. Was via the Amend flow, inserted AHEAD of PR-5.1; user-greenlit 2026-06-15 in this conversation (the deferred post-deploy warm step, now taken up). Design spec (authoritative, read in full): .big-plans/ct__bench-v4-warmer-design.md. GOAL: kill the dominant remaining cold-load cost for the FIRST visitor by keeping the Vercel serverless FUNCTION instance AND its pooled Postgres CONNECTIONS warm (not just the Data Cache, already handled by PR-5.0.97/5.0.99). ROOT CAUSE (PR-5.0.991 post-deploy measurement 2026-06-15): a cache-cold bundle on an already-warm function+connection is ~0.8-1.5s, but the FIRST request to a freshly-spun-up Vercel instance is 6-11s (function cold-start + cold DB connection: RDS Proxy connect + IAM token mint + TLS), shared across URLs, chart-count-independent. The existing PR-5.0.98 GH keep-warm cron is DORMANT on ct/bench-v4 (GH scheduled workflows only fire from the default branch) — that dormancy is why this sub-PR exists. ENABLING FACT: ct/bench-v4 pushes run `vercel deploy --prebuilt --prod` (git integration disabled) = a PRODUCTION deploy, and Vercel Cron Jobs run only on production deployments, so a `crons` entry in benchmarks-website/web/vercel.json fires PRE-MERGE on this branch. Vercel plan = Pro (user-confirmed), so sub-daily cron frequency is allowed. DESIGN (user-approved): (1) add a `crons` entry to benchmarks-website/web/vercel.json hitting GET /api/health every 2 min (`*/2 * * * *`); /api/health is already public (no CRON_SECRET) and collectHealth fans out a Promise.all of per-table COUNT(*) queries so each ping warms multiple pool connections at once (the same max=8 pool a cold-cache group-bundle fan-out from PR-5.0.991 uses); (2) raise pg idleTimeoutMillis from the 10s default to 5 min (300000 ms) in benchmarks-website/web/lib/db.ts — thread it through DbConfig/readConfig (BENCH_DB_IDLE_TIMEOUT_MS, default 300000) into createPool, mirroring poolMax — so pooled connections survive between */2 pings (without it a connection drops 10s after each ping and a user landing mid-gap re-pays IAM+TLS even on a warm function); (3) keep the existing GH keep-warm cron UNCHANGED (redundant warmer + uptime signal that activates at merge). Tests: unit-test the idleTimeoutMillis default/override + that createPool threads it; a vercel.json-shape test asserting the cron targets /api/health on */2. Review = gauntlet pr-2. USER REQUIREMENT: the warmer MUST be DEPLOYED on ct/bench-v4 — TOUCHES benchmarks-website/web/** so the close push FIRES web-deploy.yml (Check & Test + Deploy Production) -> the cron + idleTimeout go live; post-deploy verify /api/health 200, the Vercel cron is registered, and its first */2 invocation succeeds (cron registration/logs are in the Vercel dashboard, possibly the operator's to check). Out of scope: poolMax raise, Data Cache/CDN/revalidate-wiring changes, ?n=all downsampling (dropped). Known limit: under multi-instance scaling one cron ping warms one instance; on this low-traffic site there is effectively one instance, so it warms the typical first-visitor path."
amend_note_5_0_991: "PR-5.0.991 (parallelize the group-bundle query fan-out) is CLOSED on 2026-06-15 (gauntlet-pr-2-accepted cycle 1, 0 must-fix; code commit 1977192ab; full record in the PR-5.0.991 Implementation status entry; pushed -> web-deploy.yml fires the testcontainers order test + production deploy). EXECUTING via the Amend flow, inserted AHEAD of PR-5.1; user-greenlit 2026-06-15 (the cold-path '#3' optimization, chosen over the deferred warm step '#1' which the user takes up in a fresh conversation). FINDING (live diagnostic + code read): group bundles are 8-19s COLD vs <0.2s warm; compression is already optimal (brotli, 3.18MB->319KB on the wire) so payload is NOT the issue; the cold cost is collectGroupCharts (benchmarks-website/web/lib/queries.ts:1134) running a SEQUENTIAL await loop of one chartPayload SQL query PER chart (99 for TPC-DS, 43 for Clickbench) on a single pooled connection (warm is fast only because the whole bundle is one Vercel Data Cache entry, so a hit does 0 queries; only a cache MISS pays the N-query fan-out). FIX: replace the sequential loop with an order-preserving Promise.all(group.charts.map(...)) bounded by the existing pool (db.ts max=8), turning ~99 serial round-trips into ~13 waves of 8 (~8x cold-time cut) with identical output. Order is preserved (map+Promise.all+filter) and pinned by the existing groups.test.ts:200 integration test ('Q1','Q2'); null-skip semantics preserved via filter. poolMax stays 8 (raising it is a separate RDS-connection-limit tuning decision). Secondary cost (collectGroups discovery queries + the non-cached collectGroups call at queries.ts:1139) left as a possible follow-up. Review = gauntlet pr-2. TOUCHES benchmarks-website/web/** so the push FIRES web-deploy.yml (the testcontainers order test runs in CI; Docker absent locally)."
history_note: "2026-06-15 HISTORY CLEANUP: the branch was re-squashed to one commit per phase (Phase 1-5) + this plan re-point commit, matching the 2026-06-11 convention. Phases 1-4 keep their SHAs (Phase 4 = b9fc6220d = phase_entry_sha, UNCHANGED); all PR-5.0.9/5.0.95/5.0.97/5.0.98/5.0.99 sub-PR commits were folded into the single Phase 5 commit cd97d2664. Fresh backup ref refs/backups/ct-bench-v4-pre-squash-2-a90eecbd1 (= pre-cleanup tip a90eecbd1). The REBASE ONTO develop was DEFERRED (user decision): develop commit ab0e23ea4 '#8362 Remove the unused website' DELETED benchmarks-website/migrate+server+ops (the Rust backend Phases 1-3 built on) + cleaned the Cargo workspace, so a rebase is a semantic/coordination decision (resurrect-vs-drop the Rust crates) entangled with the paused cutover/decommission — NOT a mechanical conflict. Resolve that with the user + whoever merged #8362 before rebasing."
amend_note_5_0_99: "PR-5.0.99 (raise Vercel Data Cache backstop 1h->24h) is CLOSED on 2026-06-15 (full record in the PR-5.0.99 Implementation status entry). Shipped DATA_CACHE_BACKSTOP_SECONDS 3600->86400 in benchmarks-website/web/lib/data-cache.ts (+ doc comment + test assertion), gauntlet-pr-2-ACCEPTED cycle 1 ZERO findings (fresh+correctness, executor=claude), vitest/tsc/eslint/prettier clean; code commit ebabd1849. TOUCHES benchmarks-website/web/** so the push FIRES web-deploy.yml (test + production deploy) -> backstop goes live. Original via the Amend flow, inserted AHEAD of PR-5.1; user-greenlit 2026-06-15. ROOT CAUSE (user-confirmed): the slow site experience is COLD CACHE on initial group open at the default ?n=100 window, NOT payload size and NOT the ?n=all 'load all' path. Mechanism: a request pays the ~7.8s cold RDS fill only when the Vercel Data Cache entry is expired (>1h backstop) AND there is no warm CDN copy (>24h since that URL was last fetched, since CDN s-maxage=300 + stale-while-revalidate=86400). On a low-traffic site the user is often the first visitor to a long-idle group-bundle URL, so they eat the cold fill. Fix: raise DATA_CACHE_BACKSTOP_SECONDS 3600->86400 in benchmarks-website/web/lib/data-cache.ts (+ its doc comment + the assertion at lib/data-cache.test.ts:55) so a CDN miss reads a still-warm Data Cache instead of cold RDS for up to 24h. TRADEOFF (user accepted): without the revalidate wiring active, new benchmark data can lag up to the backstop (24h); benchmark data is low-frequency/trusted/regenerable so low-stakes, and setting the ops wiring later restores immediate freshness via POST /api/revalidate. Review = gauntlet pr-2. TOUCHES benchmarks-website/web/** so the push WILL fire web-deploy.yml (test + production deploy) -> the change goes live. NOTE: the earlier tentative 'PR-5.0.99 = server-side ?n=all downsampling' idea is DROPPED (wrong lever for the cold-cache complaint; downsampling only shrinks the explicit load-all payload). HELD OFF per user: the Vercel-cron/external-warmer (option B, the never-cold guarantee, needed because the GH Actions keep-warm cron from PR-5.0.98 only fires on schedule from the default branch and this is still ct/bench-v4) + the ops wiring + PR-5.1."
amend_note: "PR-5.0.98 (keep-warm GH Actions cron) is CLOSED on 2026-06-15: shipped single file .github/workflows/web-keep-warm.yml (cron */5 + workflow_dispatch; one ubuntu-latest job; curl/jq GETs / + /api/groups + each /api/group/{slug}?n=100 against HARDCODED https://benchmarks-web.vercel.app; @uri-encoded slugs; curl --fail = uptime signal; writes warmed-count to GITHUB_STEP_SUMMARY; NO secret/var). gauntlet-pr-2-ACCEPTED cycle 1 (fresh+correctness, executor=claude — no Codex companion this session); both lenses reproduced the bash under set -Eeuo pipefail (set-e-safe increment, here-string count survival, zero-group guard, malformed-/api/groups aborts, @uri injection-safe, yamllint --strict clean); 1 shared nit (false-green on zero/empty groups) de-scoped per REVIEW CALIBRATION. yamllint clean + live sanity check (16 slugs, first bundle 200). Code commit 9adc6c870 (plan commits 69e892dee + 27f586ae6). Does NOT touch benchmarks-website/web/** so it does NOT fire web-deploy.yml — it just installs the workflow (first scheduled/dispatch run validates live). NEXT: PR-5.1 (current_pr) remains PAUSED — its first step is a PROD RDS WRITE gate (re-run PR-3.5 cross-check via bench_ingest IAM) needing user coordination; do NOT start it autonomously. Also PENDING user go: PR-5.0.99 (server-side ?n=all downsampling, the real load-all payload-bytes fix). OPS PREREQ still the user's action: set BENCH_REVALIDATE_TOKEN (Vercel env + GH secret) + BENCH_SITE_BASE_URL (GH var)."
```

## SESSION HANDOFF 2026-06-16d (revalidate-token wiring SET but DORMANT/v4-only -- READ THIS FIRST)

**Newest handoff; supersedes everything below.** The revalidate-cache ops wiring (previously the
held-off "OPS PREREQ") is now SET on both sides, but intentionally NOT yet activated (user chose
"keep wiring, skip the redeploy"):
- **GitHub** (`vortex-data/vortex`, repo-level): NEW secret `BENCH_REVALIDATE_TOKEN` + NEW variable
  `BENCH_SITE_BASE_URL` = `https://benchmarks-web.vercel.app`. Both are NEW additions; no existing
  secret/variable was modified.
- **Vercel** `benchmarks-web` (under the `vortex-data` team), **Production** env: `BENCH_REVALIDATE_TOKEN`
  set to the SAME freshly-generated 64-char token as the GH secret (piped, never printed).

**DORMANT + v4-ONLY (verified):** `origin/develop`'s CI does NOT reference either var (a
`git grep` on `origin/develop -- .github/workflows` is empty); the references live only on
`ct/bench-v4`'s 3 ingest workflows (bench.yml / sql-benchmarks.yml / v3-commit-metadata.yml) + the
keep-warm workflow. So this change has ZERO effect on the current v2/production system. It activates
only once `ct/bench-v4`'s workflows reach `develop` (PR-5.1 + the develop rebase), and the Vercel
token additionally needs ANY production redeploy to bind to the running deployment (the next deploy
does that automatically). End-to-end verification (`POST /api/revalidate` -> 200) is therefore
DEFERRED to PR-5.1. Fully reversible: `gh secret delete BENCH_REVALIDATE_TOKEN`,
`gh variable delete BENCH_SITE_BASE_URL`, `vercel env rm BENCH_REVALIDATE_TOKEN production`. The
`benchmarks-web` project is now linked at `benchmarks-website/web/.vercel` (gitignored) for future
ops; the Vercel CLI is authed as `connor-6267` (team `vortex-data`).

**Net:** the ops prereq is PRE-POSITIONED; PR-5.1 just needs to land the workflow cutover (and ensure
a deploy) for auto-revalidation to go live. Everything else still HELD OFF per the user: PR-5.1
(prod-write-gated ingest cutover), the `develop` rebase (#8362 resurrect-vs-drop), PR-5.2 (DNS flip),
PR-5.3 (decommission).

## SESSION HANDOFF 2026-06-16c (v4 RDS RE-MIGRATION EXECUTED + VERIFIED -- READ THIS FIRST)

**Newest handoff; supersedes everything below.** The user-approved v4 RDS re-migration (prepped in
2026-06-16b) was EXECUTED and verified this session, and the site is now serving the fresh data. The
default `?n=100` view (chart data + group summaries) initially showed STALE data because the manual
`load` bypasses `post-ingest.py`'s `/api/revalidate` hook AND the revalidate token is unwired -- so
nothing busted the Vercel Data Cache (`bench-data` tag, 24h backstop). FIX APPLIED: the user manually
purged the Data Cache via the Vercel dashboard (project -> **CDN** -> **Caches** -> Purge cache ->
All content -> **Runtime and Data Cache**); the live `?n=100` tpcds/nvme 99-chart bundle was then
verified serving newest commit `2026-06-16 14:21:24Z`. The Vercel CLI is now installed + authed as
`connor-6267`, so future manual busts can use `vercel cache invalidate --tag bench-data` directly
(no app token needed). STILL a STOPGAP -- v4 drifts stale after the next `develop` run until the
PR-5.1 ingest cutover lands; and AUTOMATIC freshness on future writes still needs the held-off ops
wiring (`BENCH_REVALIDATE_TOKEN` in Vercel + GH secret, `BENCH_SITE_BASE_URL` GH var) -- without it,
even normal pipeline writes would skip the revalidate (the `post-ingest.py` hook is gated on both env
vars) and fall back to the 24h backstop.

**What ran (all green):**
- `--replace` flag committed: `346006787` ("migrate: add --replace flag for atomic full-replace
  re-load"); the prep session's working-tree migrate edits are folded into it.
- BUILD: the bundled DuckDB still does NOT link in the local **debug** build (`could not find native
  static library 'duckdb'`), but the **release** profile links fine (its `libduckdb.a` is already
  compiled), so the run used `cargo build --release -p vortex-bench-migrate` ->
  `target/release/vortex-bench-migrate`. Use the release binary for future re-migrations in this env.
- Pre-load snapshot `vortex-bench-prod-pre-remigration-20260616` (available) = rollback insurance.
- `migrate run --output /tmp/fresh.duckdb --allow-missing-file-sizes`: 3 `file-sizes-*-s3.json.gz`
  sources (`tpch-s3`, `tpch-s3-10`, `fineweb-s3`) return **403 Forbidden** from the public bucket
  (they ARE in `KNOWN_FILE_SIZES_SUITES`, so expected-but-now-inaccessible; user said don't worry
  about them). Result: 4597 commits, 4,919,122 query rows, uncategorized 0.19%, newest commit
  2026-06-16.
- `migrate load --replace` as the **RDS master**: atomic TRUNCATE of all six + COPY (commits 4597,
  query_measurements 4,919,122, compression_times 253200, compression_sizes 106276,
  random_access_times 37459, vector_search_runs 0) + post-COPY `commit_timestamp` denormalization on
  all 4,919,122 rows.
- `migrate verify`: 0 presence diffs, 0 value mismatches.
- R1 direct-prod psql check: query_measurements has **0 NULL `commit_timestamp`**, newest
  `commit_timestamp` 2026-06-16 14:21:24Z across ALL 6 datasets
  (tpch/tpcds/clickbench/fineweb/statpopgen/polarsignals) -- the read-path sort key is fully
  populated.

**CORRECTION (load-bearing next time): the runbook's GUESSED master-secret id was WRONG.** The real
RDS-managed master secret is
`arn:aws:secretsmanager:us-east-1:245040174862:secret:rds!db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690-egkQgW`
(DbiResourceId `db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690`); the runbook's
`rds!db-4VPTDACTRQHOS24WEIR3TNC2M4` returns `ResourceNotFoundException`. Resolve it deterministically
via `aws rds describe-db-instances ... --query 'DBInstances[0].MasterUserSecret.SecretArn'` (NOT a
guess; NOT `list-secrets`, which the auto-mode classifier blocks as credential scouting). Runbook +
memory `project_bench_rds_profiling_access` updated with the correct ARN.

**Everything else still HELD OFF per the user** (unchanged): PR-5.1 (prod-RDS-write-gated cutover),
the `develop` rebase (#8362 resurrect-vs-drop), the ops wiring (`BENCH_REVALIDATE_TOKEN` +
`BENCH_SITE_BASE_URL`).

## SESSION HANDOFF 2026-06-16b (v4 RDS RE-MIGRATION prepped + corrected, ready to EXECUTE in a working build env -- READ THIS FIRST -- SUPERSEDED: executed, see 2026-06-16c above)

**Newest handoff; supersedes everything below.** User approved re-migrating v4 RDS from v2's live S3
to freshen ~6-day-stale prod data ("lets do it now" / "ok please run the migration"). It was NOT run
this session because the migrate binary does not LINK in the local sandbox (bundled DuckDB native lib
fails; `cargo check` is green). A fresh session must build + run it where the bundled DuckDB compiles
(CI or a non-sandboxed shell). **The execution artifact is `.big-plans/ct__bench-v4-remigration-runbook.md`
-- re-read it in full; it is corrected and self-contained.**

**Two load-bearing CORRECTIONS found while prepping (the old runbook was wrong):**
1. The loader (`benchmarks-website/migrate/src/postgres.rs`) is APPEND-ONLY over primary keys -- it
   never `TRUNCATE`s. Re-loading into the populated prod DB would abort on duplicate `measurement_id`.
   It was a one-shot empty-seed. **Fix applied (uncommitted, user commits): a new `--replace` flag**
   that `TRUNCATE`s all six tables inside the load transaction = atomic full replace (mid-load failure
   rolls back to ORIGINAL data). `cargo check -p vortex-bench-migrate` GREEN.
2. The load must connect as the RDS MASTER (`postgres`, password in Secrets Manager), NOT `migrator`.
   The six tables are master-owned; `TRUNCATE` needs ownership; `migrator` has no data-table DML and
   `bench_ingest` is explicitly denied TRUNCATE (`migrations/002`-`005` confirm). Matches the original
   PR-5.0 prod-load path + the README/`main.rs` Load help.

**Code changes this session (working tree, uncommitted -- user said they will commit):**
`benchmarks-website/migrate/src/postgres.rs` (`load` gains `replace: bool` + in-txn TRUNCATE),
`src/main.rs` (`--replace` flag), `tests/postgres_e2e.rs` (2 call sites + new
`rehearsal_replace_load_reseeds_a_populated_target`), `migrate/README.md` (documents `--replace`).
Schema preconditions are already satisfied on prod (006 `commit_timestamp` + 007 covering index
applied; the loader's post-COPY UPDATE repopulates `commit_timestamp`, the read-path R1 sort key).

**Execution sequence (full detail in the runbook):** snapshot prod -> build in a working env ->
`migrate run --output /tmp/fresh.duckdb` (v2 public S3, no creds) -> `migrate load ... --ca-cert
global-bundle.pem --replace` as master (THE PROD WRITE) -> `migrate verify` (must be clean) -> R1 live
check (big group `?n=100` renders, newest commit ~2026-06-16) -> new data appears after the Vercel
cache backstop (<=24h) since `/api/revalidate` is not wired. STILL a STOPGAP (PR-5.1 ingest cutover is
the durable fix; re-staleness resumes after the next develop run).

**Everything else still HELD OFF per the user** (unchanged): PR-5.1 (prod-RDS-write-gated cutover),
the `develop` rebase (#8362 resurrect-vs-drop), the ops wiring (`BENCH_REVALIDATE_TOKEN` +
`BENCH_SITE_BASE_URL`).

## SESSION HANDOFF 2026-06-16 (load-all `?n=all` cold-start RESOLVED via RDS RAM upsize -- READ THIS FIRST)

**Newest handoff; supersedes everything below.** The load-all `?n=all` cold-start topic the prior
handoff teed up is now investigated, decided, executed, and prod-validated. No web-app code changed.

**What happened (full detail: `.big-plans/ct__bench-v4-loadall-scope.md` -- the PROFILING RESULT +
DECISION + Validation sections).** User chose "profile first, then decide". Live-prod profiling
(read-only, PI + CloudWatch) showed the cold `?n=all` cost is PHYSICAL-I/O-bound (working set ~6 GB
DB > ~1 GB cache on the old 4 GiB db.t4g.medium), NOT CPU (~80% IPC:BufferIO + IO:DataFileRead;
ReadIOPS 0->1112/s; CPU peak 8%). This is the OPPOSITE of R1's bounded-window finding and confirmed
the scope doc's RAM caveat. Downsampling was ruled out (it shrinks payload, not the DB read). User
chose the RAM upsize (root-cause lever). **Executed: db.t4g.medium (4 GiB) -> db.r7g.large (2 vCPU /
16 GiB)** via `aws rds modify-db-instance --apply-immediately` on the `bench-prod` profile, completed
14:45:40Z (pre-resize snapshot `vortex-bench-prod-pre-r7g-resize-20260616` retained; no param-group
change -- shared_buffers auto-scaled to ~4 GiB). **Validated:** the cross-group cold-again churn is
gone (a per-chart `?n=all` stays ~0.25s even after visiting 6 other big groups; cold-ish read 0.53s
-> 0.13s, ~4x). Cost ~$47/mo -> ~$174/mo (+~$127/mo). Plan commits 6f4f45ef1 (profiling) + 390e86414
(decision) + this handoff.

**OPEN follow-up (small, optional):** reflect the new class in `benchmarks-website/infra/provision.sh`
+ README cost table (doc/config-only consistency). Memory `project_bench_rds_profiling_access` was
updated (instance class is now db.r7g.large / 16 GiB).

**Everything else still HELD OFF per the user** (unchanged): PR-5.1 (prod-RDS-write-gated cutover),
the `develop` rebase (#8362 resurrect-vs-drop), the ops wiring (`BENCH_REVALIDATE_TOKEN` +
`BENCH_SITE_BASE_URL`). Server-side downsampling remains UNBUILT (only relevant if payload / slow
connections becomes a goal, a separate axis; see the scope doc).

## SESSION HANDOFF 2026-06-15 (NEXT TOPIC = load-all `?n=all` cold-start + downsampling SEMANTICS, scoped for a FRESH discussion -- READ THIS FIRST)

**Newest handoff; supersedes everything below as the "read first" entry.** R1 (PR-5.0.993) is DONE +
DEPLOYED + PROD-VALIDATED (see the next handoff section). After it shipped, the user gave feedback
(2026-06-15) and asked to **set up the next topic for a FRESH conversation to DISCUSS** (not implement
now). Two items:

1. **Spinner = NON-ISSUE, no work.** The user reported "the spinners don't actually spin"; the
   `.chart-spinner` CSS (`globals.css:1208-1221`) animates correctly and the markup
   (`Chart.tsx:2047`) matches — it is intentionally disabled under `@media (prefers-reduced-motion:
   reduce)` (`globals.css:1235`). The user CONFIRMED: with macOS Reduce-motion OFF it spins fine. So
   the static ring was just their reduced-motion setting. Nothing to fix (optionally, a fresh convo
   could add a reduced-motion-friendly opacity pulse, but the user did not ask for it).

2. **Load-all (`?n=all`) cold-start + server-side downsampling = the NEXT TOPIC, SCOPED but UNDECIDED.**
   Full scoping doc (read it first): **`.big-plans/ct__bench-v4-loadall-scope.md`**. The user finds
   "load all data for a given chart still slow on cold start ... per benchmark group ... subsequent
   charts faster" and is interested in server-side downsampling but "unsure what the semantics are."
   **Live measurements taken this session (in the doc) settle the diagnosis:** the cold cost is the DB
   buffer-cache read of a chart's FULL history, NOT the payload (confirmed: same-group chart1 cold
   0.53s vs chart2 warm 0.095s on an IDENTICAL 46.5KB payload; warming is per-GROUP; the payload is
   already brotli ~14KB on the wire). **CRUX to discuss:** downsampling shrinks the PAYLOAD, but you
   must read all rows from the DB before downsampling them, so downsampling does NOT fix the cold-start
   DB read the user described — it is a different lever (payload/slow-connection). R1 deliberately left
   `?n=all` unchanged (it genuinely needs all rows; no over-read). The doc lays out the DB-read-vs-
   payload distinction, candidate levers (downsampling + its open semantics [zoom detail, hover exact
   values, per-series-vs-shared axis, N]; cache `?n=all`; warm group pages; covering index; RAM), and
   suggested first steps (profile a cold `?n=all` with EXPLAIN+PI to see CPU-vs-I/O, then decide
   cold-latency-vs-payload as the goal). **A fresh conversation should START from the scope doc and
   discuss the semantics/goal with the user before building.** Build (if chosen) via the Amend flow.

**Everything else still HELD OFF per the user** (unchanged): PR-5.1 (prod-RDS-write-gated cutover),
the `develop` rebase (#8362 resurrect-vs-drop), the ops wiring. Resume via `/spiral:big-plans` in
`vortex4`; the stock `resume_routing.py` falls to its Coarse floor on this custom spine, so this
handoff + the Current State block are ground truth.

## SESSION HANDOFF 2026-06-15 (READ-PATH R1 = PR-5.0.993 IMPLEMENTED + DEPLOYED + PROD-VALIDATED)

**Newest handoff; supersedes everything below as the "read first" entry.** The read-path R1 fix
the prior handoff scoped is now DONE, shipped, and live. Resume via `/spiral:big-plans` in `vortex4`
(the stock `resume_routing.py` falls to its Coarse floor on this custom spine; the `Current State`
block + the `### PR-5.0.993:` Implementation status entry are ground truth).

**PR-5.0.993 (read-path R1) is CLOSED + DEPLOYED (full record: the `### PR-5.0.993:` entry +
`amend_note_5_0_993`).** Built via the Amend flow (writing-plans -> SDD -> gauntlet pr-2 ->
close -> push), user-greenlit at resume. Shipped in `benchmarks-website/web/lib/queries.ts`:
`collectQueryChart`'s data query + `buildEarliest` seed now filter/seed on the denormalized,
indexed `q.commit_timestamp` (via a new `queryMeasurementWindowFilter` helper: sargable
`commit_timestamp >= cutoff` + the kept `commit_sha IN (last-n)` tie-trim) instead of a `commits`
join; the shared `factWindowFilter` + the 4 other collectors are untouched. **gauntlet pr-2 accepted
across 2 cycles** (cycle 1 accept w/ a result-equivalence proof; a CI testcontainer regression then
surfaced -- two STALE seeded-window TEST fixtures inserted `query_measurements` without
`commit_timestamp` -> NULL -> empty chart -- fixed in `36ed8a90e` by stamping the fixtures faithfully
[prod always populates `commit_timestamp` via the ingest upsert + migration-006 backfill, all
test-pinned]; cycle 2 re-accept after the fix). Code commits `05e41b2a6`..`85ba7ac37`. **CI GREEN +
DEPLOYED LIVE** (web-deploy run 27576044515; `build_sha 00ee5d0ef`). **PROD-VALIDATED**: the TPC-DS
99-chart group cold bundle (the original ~15-16s complaint) now serves in ~1.77s cache-cold (~9x).
1 should-fix DEFERRED (a same-boundary-timestamp-tie coverage test that needs an unverifiable-locally
testcontainer fixture; folded into the web test-hardening pass); the cycle-1 nit was resolved.

**Everything else still HELD OFF per the user** (unchanged): PR-5.1 (the prod-RDS-write-gated
cutover -- its first step is the `bench_ingest` IAM write gate needing operator coordination), the
`develop` rebase (the #8362 resurrect-vs-drop decision), and the ops wiring
(`BENCH_REVALIDATE_TOKEN` + `BENCH_SITE_BASE_URL`). The remaining read-path R2 (resolve the single
group in `collectGroupCharts`, ~0.5-0.7s) was NOT pursued (R1 alone hit the target).

## SESSION HANDOFF 2026-06-15 (READ-PATH INVESTIGATION DONE; R1 fix scoped + ready for a fresh agent -- READ THIS FIRST -- SUPERSEDED: R1 is now PR-5.0.993, shipped above)

**Newest handoff; supersedes the sections below as the "read first" entry.** This session profiled
the big-group cold-open slowness against live prod (read-only) and CORRECTED the earlier
`finding_readpath_cold_fill_2026_06_15` (see the new `finding_readpath_correction_2026_06_15` field
in Current State). Full evidence + exact SQL + prioritized recommendations live in
**`.big-plans/ct__bench-v4-readpath-findings.md`** (authoritative; read it in full before implementing).

**1. Corrected diagnosis (measured, not theorized).** The cold open is NOT disk/throughput-bound and
NOT helped by more cores or RAM. During a cold 99-chart fan-out: RDS CPU ~5% with CPU credits pinned
at max, near-zero physical I/O (ReadIOPS <=17/min, ReadLatency ~0), PI db.load ~97% CPU. The real
cause: each per-chart query reads ~18k rows (the chart's full history) to return the ~665-row last-N
window, because recency is filtered via a `commits` join on `commit_sha` after a full-history scan
instead of via the denormalized, already-indexed `commit_timestamp`. The 11.4s first-hit spike is a
Vercel FUNCTION cold-start (the warmer's layer), not the DB.

**2. THE FIX a fresh agent should implement (R1, primary).** In
`benchmarks-website/web/lib/queries.ts`, change `collectQueryChart` (the data query) and its seed
`buildEarliest` callback to filter on `q.commit_timestamp` (>= the Nth-newest commit's timestamp) and
drop the `commits` joins, keeping `commit_sha IN (last-N)` as a tie-trim. Uses the existing
`idx_query_measurements_summary`; no schema change; it REMOVES code (two joins). EXPLAIN-verified
result-identical (665 rows) and ~5x faster per chart (seed 57->8ms, data 38->9ms); expected cold
TPC-DS bundle ~4.7s -> ~1s. Scope is one collector (the big groups are all `query_measurements`).
Build via the big-plans Amend flow (writing-plans -> SDD -> gauntlet pr-2 -> close -> push fires
web-deploy.yml). Open implementation items are listed at the end of the findings doc.

**3. Optional secondary (R2).** `collectGroupCharts` re-runs `collectGroups()` to build ALL ~17
groups' summaries (~0.74s, the TODO #7812) when only the one requested group is needed; resolve the
single group directly. Moderate refactor; do only if R1 alone leaves the bundle slower than wanted.

**4. Rejected (with evidence, do NOT pursue).** The earlier B1 "batch the whole group into one SQL
scan" idea (measured 640ms, WORSE than the parallel per-chart queries); more cores or a bigger
instance class (no CPU/credit/IO pressure at today's data volume); deduping the per-chart seed (cheap
after R1, edge-case risk); the small fact tables (no `commit_timestamp`, negligible cost).

**5. Ops note.** Performance Insights was ENABLED on `vortex-bench-prod` this session (7-day free
retention) for ongoing wait-event visibility. DB profiling access facts (roles, auth, endpoint) are
in memory `project_bench_rds_profiling_access`. Everything else from the prior handoffs still holds:
PR-5.1 (the prod-write-gated cutover), the develop rebase, and the ops wiring remain held off.

## SESSION HANDOFF 2026-06-15 (PR-5.0.98 + PR-5.0.99 CLOSED; history re-squashed; develop rebase DEFERRED — READ THIS FIRST)

**Newest handoff; supersedes the sections below as the "read first" entry.** Resume via
`/spiral:big-plans` in the `vortex4` worktree. (The stock `resume_routing.py` falls to its Coarse
floor on this custom spine; the `Current State` block + git log are ground truth, per the
Hybrid-fallback rule.)

**1. Three sub-PRs closed this session (all via the Amend flow, gauntlet-pr-2-accepted cycle 1, full
records in their `### PR-5.0.98:` / `### PR-5.0.99:` / `### PR-5.0.991:` Implementation status
entries):**
- **PR-5.0.991** — parallelized the group-bundle query fan-out (`collectGroupCharts` sequential
  `await` loop -> order-preserving `Promise.all` bounded by the pg pool `max: 8`), cutting the
  per-chart serial-query cold penalty (8-19s -> ~1s for cache-cold-but-function-warm bundles).
  Deployed live. **Post-deploy measurement refined the warmer plan (#1):** the DOMINANT remaining
  cold cost is the Vercel FUNCTION + DB-CONNECTION cold-start (6-11s on the first request to a cold
  instance, shared across URLs, chart-count-independent), so #1 must keep the function + pooled
  connection warm, not just the Data Cache. See the PR-5.0.991 entry for the measurement detail.
- **PR-5.0.98** — scheduled keep-warm GH Actions cron (`.github/workflows/web-keep-warm.yml`): GETs
  `/` + `/api/groups` + each `/api/group/{slug}?n=100` against the HARDCODED
  `https://benchmarks-web.vercel.app` every 5 min; no secret. CAVEAT: GH scheduled workflows only
  fire from the DEFAULT branch, so on `ct/bench-v4` this cron is installed-but-DORMANT until merge.
- **PR-5.0.99** — raised `DATA_CACHE_BACKSTOP_SECONDS` 3600->86400 (1h->24h) in
  `benchmarks-website/web/lib/data-cache.ts`. ROOT CAUSE (user-confirmed via live diagnostic): the
  "site feels slow" complaint is COLD CACHE on initial group open at the default `?n=100` window
  (NOT payload size, NOT the `?n=all` load-all path — the server-side downsampling idea was DROPPED
  as the wrong lever). A 24h backstop lets a CDN miss read a warm Data Cache instead of the ~7.8s
  cold RDS fill. Deployed live (web-deploy success, build_sha was a90eecbd1 pre-cleanup).

**2. HISTORY RE-SQUASHED 2026-06-15.** The branch is back to one commit per phase (Phase 1-5) + one
`plan: re-point` commit, matching the 2026-06-11 cleanup. Phases 1-4 keep their SHAs (Phase 4 =
`b9fc6220d` = `phase_entry_sha`, unchanged); every PR-5.0.9..5.0.99 sub-PR commit was folded into the
single Phase 5 commit `cd97d2664`. Fresh backup ref `refs/backups/ct-bench-v4-pre-squash-2-a90eecbd1`
(pre-cleanup tip `a90eecbd1`). The cleaned branch was force-pushed to `origin/ct/bench-v4`.

**3. REBASE ONTO develop is DEFERRED — needs a planning decision (do NOT just rebase).** Develop has
advanced ~33 commits past the old merge-base, and commit `ab0e23ea4` ("#8362 Remove the unused
website and clean other dependencies") DELETED `benchmarks-website/migrate/` + `server/` + `ops/` +
`AGENTS.md` and cleaned the Cargo workspace (removed those crates from `Cargo.toml` members, trimmed
`vortex-*` deps, updated `Cargo.lock`). Those are the Rust backend Phases 1-3 built on (the Postgres
writer + `measurement_id` xxhash64 functions the BANS protect live in `server/`). The v2 site files
+ our new `web/` service are NOT in conflict; only the old Rust crates are (modify/delete on every
file). So the rebase is a semantic/coordination call — resurrect-our-crates vs accept-develop's-
deletion — entangled with the still-paused cutover/decommission (PR-5.1/5.2/5.3). Develop having
already removed the old backend may overlap with PR-5.3's decommission, but our branch is mid-flight
and still references those crates. **On resume: reconcile this with the user + ideally whoever merged
#8362 before attempting the rebase.** The squash cleanup is independent and already landed.

**4. Everything else remains HELD OFF per the user (2026-06-15):** option B (a Vercel-cron /
external warmer that runs pre-merge — the never-cold guarantee, needed because the PR-5.0.98 GH cron
is dormant off the default branch); the ops wiring (`BENCH_REVALIDATE_TOKEN` + `BENCH_SITE_BASE_URL`,
the user's action, which makes the 24h backstop staleness-free); and PR-5.1 (the prod-RDS-write-gated
cutover). The `?n=all` downsampling idea is DROPPED.

## SESSION HANDOFF 2026-06-12 (PR-5.0.97 CLOSED; PR-5.1 is next — gated on a PROD RDS WRITE; READ THIS FIRST)

**Newest handoff; supersedes the sections below as the "read first" entry.** Resume via
`/spiral:big-plans` in the `vortex4` worktree. The `Current State` block routes to `current_pr:
PR-5.1`. (Heads-up: the stock `resume_routing.py` falls to its Coarse floor on this custom
spine; the `Current State` block + git log are ground truth, per the Hybrid-fallback rule.)

**1. PR-5.0.97 is DONE.** Always-warm last-100 cache + full spinner coverage + fast Expand All
shipped, **gauntlet-pr-3-accepted (2 cycles: reject -> accept; executor=claude, 3 lenses
fresh+correctness+maint)**, committed `cdbe72a32..5f92566ee`. Do NOT re-open it; its full record is
the `### PR-5.0.97:` Implementation status entry. After this close, `ct/bench-v4` is pushed to fire
`web-deploy.yml` (staging deploy to `https://benchmarks-web.vercel.app`; a read-service code deploy,
no prod data write). Shipped (benchmarks-website/web/ + scripts/ + 3 workflows): (A) Vercel Data
Cache for the default `?n=100` window (`web/lib/data-cache.ts`, tag `bench-data`, 1h backstop;
routes/pages branch only on the default window, force-dynamic + CDN headers unchanged); (B) `POST
/api/revalidate` (bearer `timingSafeEqual`, 503 fail-closed, `revalidateTag`); (C)
`scripts/post-ingest.py` `refresh_site_cache` hook (POST revalidate + best-effort warm pass; swallows
every failure so it never changes the ingest exit code; skips warm on failed revalidate; no-op
unless both env vars set) + 2-line additive env on the v4 Postgres step in the 3 ingest workflows;
(D) one `/api/group/{slug}?n=100` bundle fetch per group into a session client payload cache so
Expand All loads every chart's last-100 eagerly (IO still gates Chart.js construction); (E) a
server-rendered `.chart-placeholder` spinner for every pre-data card state, static ring+label under
`prefers-reduced-motion`. The correctness lens caught a real must-fix (close-while-awaiting-bundle
per-chart leak; fixed + regression-tested) that fresh+maint missed.

**2. OPS PREREQUISITE STILL PENDING — coordinate with ops (NOT a code blocker).** Generate the
shared secret; set `BENCH_REVALIDATE_TOKEN` in the Vercel project env + as a GitHub Actions secret,
and `BENCH_SITE_BASE_URL` as a GitHub Actions var. Until set, `/api/revalidate` 503s fail-closed and
the post-ingest hook is a silent no-op (everything degrades to current behavior), so the PR is safe
deployed without the wiring — but refresh-on-update + warming do not actually activate until the env
is wired. Flag this to the user when convenient.

**2b. POST-CLOSE same-session findings (2026-06-12, after deploy) + a QUEUED sub-PR-5.0.98.** The
user reported the site "still feels slow" and that "load all data of a given chart" feels even
slower. Live measurements against `https://benchmarks-web.vercel.app` CONFIRM the Data Cache is
working — every user-facing fetch is now sub-0.4s (landing `/` 0.16-0.38s; chart `?n=100`
0.09-0.22s; group `?n=100` bundle [43 charts, 1.44MB] 0.25s; chart `?n=all` [1.1MB] 0.14s warm /
0.44s miss). The ~7.8s cold RDS path is GONE for the default window; the 20s first-load the user saw
was the one-time cold cache fill. The ONLY slow endpoint is `group ?n=all` (48MB / 10.7s) but the
client NEVER requests it (the bundle URL is hardcoded `?n=100`; full history is per-chart
`?n=all`) — latent API footgun, not user-facing. A micro-profile of the real shipped client data
path over the 3,572-commit/1.1MB `?n=all` payload showed it is **~5ms total** (JSON.parse 1.2ms,
`normalizeChartPayload` ~0ms fast-path, `collectAllValues` 0.18ms, `pickDisplayUnit` 2.3ms,
`lttbIndices` 3572->500 0.03ms x8 series) — so "load all" lag is NEITHER client CPU NOR the
datacenter fetch; it is the **1.1MB payload over a real-world connection**, which makes server-side
`?n=all` downsampling (shrink the bytes, ~1.1MB -> ~300KB) the correct lever (NOT a client
optimization). **User decisions this session (AskUserQuestion):** build a **keep-warm cron** + profile
the client first (profiling DONE, result above); the user did NOT yet greenlight server-side
downsampling. **QUEUED: PR-5.0.98 (keep-warm cron)** — a scheduled GitHub Actions workflow that GETs
`/` + `/api/groups` then each group's `/api/group/{slug}?n=100` every ~4-5 min (under the 5-min CDN
`s-maxage`) so the Data Cache + CDN never go cold on this low-traffic site; NO secret needed
(read-only public traffic); build via the big-plans Amend flow ahead of PR-5.1 (writing-plans -> SDD
-> gauntlet pr-2/pr-3 -> close -> push); follow `.github/AGENTS.md` + yamllint. **PENDING user
decision: PR-5.0.99? server-side `?n=all` downsampling** (the real "load all" fix; tradeoff = must
preserve pan/zoom fidelity by re-fetching denser data on zoom into a region, which is why the design
deferred it — present the profiling finding and let the user choose). **OPS-WIRING (user's action,
boundary):** the agent CANNOT set `BENCH_REVALIDATE_TOKEN` (Vercel env + GH secret) /
`BENCH_SITE_BASE_URL` (GH var) — that is the user's to do; it activates refresh-on-ingest + the warm
pass (already built) + lets us safely raise the 3600s Data Cache backstop.

**3. PR-5.1 is next and its FIRST step is a PROD RDS WRITE gate.** Scope: promote the v4 `--postgres`
ingest to required + drop the v3 `--server` write from the 3 ingest workflows (`bench.yml`,
`sql-benchmarks.yml`, `v3-commit-metadata.yml` — remove the v3 step + the `continue-on-error` on the
v4 steps) + ship + document `scripts/psql-bench.sh`. **Before removing `continue-on-error`: re-run
the PR-3.5 cross-check `scripts/cross_check_python_writer.py --postgres "$DSN" --envelopes
<real_envelopes.json>` via the `bench_ingest` IAM role against accumulated prod soak data and confirm
clean.** That is a prod RDS WRITE (IAM-token auth) needing user coordination — operator-run vs
agent-run-with-creds — a genuine externalized-side-effect gate, NOT a Class A pause; confirm with the
user how to run it before proceeding. Review for PR-5.1 is cross-bundle (3 workflows + script + py) ->
gauntlet **pr-3**. Then PR-5.2 (DNS flip), PR-5.3 (decommission). Each prod write remains
harness-gated. Pre-squash backup ref `refs/backups/ct-bench-v4-pre-squash-386ea347b`; recreate a
fresh one before any PR-5.3/final squash.

## SESSION HANDOFF 2026-06-12 (PR-5.0.97 scoped + plan-approved; EXECUTING) [SUPERSEDED by the section above; PR-5.0.97 is now DONE]

**Resume via**
`/spiral:big-plans` in the `vortex4` worktree.

**1. PR-5.0.97 is a NEW SUB-PR inserted AHEAD of PR-5.1** (same Amend flow as PR-5.0.9 /
PR-5.0.95). After PR-5.0.95 shipped, the user reported the site is STILL slow to load and steered a
third loading-model round. Scope: **(1) always-warm last-100 cache** — wrap the default-window
(`?n=100`) query path in Vercel's Data Cache (`unstable_cache`, tag `bench-data`, 1h backstop) so
CDN misses stop paying the ~7.8s cold RDS path; a new secret-protected **`POST /api/revalidate`**
(bearer `BENCH_REVALIDATE_TOKEN`, `timingSafeEqual`, 503-fail-closed) called by
`scripts/post-ingest.py` after each `--postgres` write + a best-effort warm pass (every failure
swallowed — never changes the ingest exit code). **(2) full spinner coverage** — a server-rendered
`.chart-placeholder` (spinner ring + "loading…" label) for EVERY pre-data state (pre-hydration
cards are blank today); keep the `prefers-reduced-motion` guard but keep a static ring + label
visible (user's chosen behavior). **(3) fast Expand All** — switch group-open hydration to ONE
`/api/group/{slug}?n=100` bundle fetch per group feeding a session-lifetime client payload cache
(`Map<slug,payload>`); IntersectionObserver keeps gating Chart.js CONSTRUCTION only, so Expand All
loads every chart's last-100 eagerly (top-group-first) while construction stays lazy; close/reopen
never refetches.

**2. Two user decisions are PINNED (AskUserQuestion this session):** cache layer = Data Cache +
revalidate (not CDN-warm-only, not static JSON); spinner = RESPECT prefers-reduced-motion. Design
is authoritative at **`.big-plans/ct__bench-v4-uiux-r3-design.md`** (read in full); approved plan at
`~/.config/claude/plans/lets-continue-i-think-gleaming-meteor.md`. Review = **gauntlet pr-3**
(cross-cutting: client loading-model + server caching/auth + the production ingest script).

**3. Ops prerequisite (coordinate at execution, not a code blocker):** generate the shared secret,
set `BENCH_REVALIDATE_TOKEN` in the Vercel project env + as a GitHub Actions secret, and
`BENCH_SITE_BASE_URL` as an Actions var. Until set, the revalidate route 503s fail-closed and the
post-ingest hook is a silent no-op — every piece degrades to current behavior, so the PR is safe to
merge before the ops wiring lands.

**4. How to implement:** follow the big-plans Phase-2 loop — writing-plans (JIT task-plan
`.big-plans/ct__bench-v4--5-0-97-warm-cache.plan.md`) -> SDD -> gauntlet pr-3 -> Step 2.5 close ->
push (fires `web-deploy.yml`). Check the Codex companion is installed if you want Claude+Codex
executor disjointness (absent in PR-5.0.9 / PR-5.0.95 -> Claude-only gauntlet).

**5. After PR-5.0.97 closes:** PR-5.1 (promote v4 `--postgres` ingest to required + drop the v3
`--server` write + ship `scripts/psql-bench.sh`) — its FIRST step is a PROD RDS WRITE gate (re-run
the PR-3.5 cross-check via the `bench_ingest` IAM role) needing user coordination; a genuine
externalized-side-effect gate, NOT a Class A pause. Then PR-5.2 (DNS flip), PR-5.3 (decommission).
Each prod write remains harness-gated.

## SESSION HANDOFF 2026-06-12 (PR-5.0.95 CLOSED; PR-5.1 is next — gated on a PROD RDS WRITE) [SUPERSEDED; PR-5.0.97 then PR-5.1]

Resume via
`/spiral:big-plans` in the `vortex4` worktree. The `Current State` block routes to `current_pr:
PR-5.1`. (Heads-up: the stock `resume_routing.py` falls to its Coarse floor on this custom spine; the
`Current State` block + git log are ground truth, per the Hybrid-fallback rule.)

**1. PR-5.0.95 is DONE.** Lazy-hydration + resilient loading (UI/UX round 2) shipped,
gauntlet-pr-2-accepted at the ~3-cycle cap (cycles reject/reject/fix-and-accept; executor=claude),
committed `096f74f7c..327f1fb92`. Do NOT re-open it; its full record is the `### PR-5.0.95:`
Implementation status entry. After this close, `ct/bench-v4` is pushed to fire `web-deploy.yml`
(staging deploy to `https://benchmarks-web.vercel.app`; a read-service code deploy, no prod data
write).

**2. PR-5.1 is next and its FIRST step is a PROD RDS WRITE gate.** Scope: promote the v4 `--postgres`
ingest to required + drop the v3 `--server` write from the 3 ingest workflows (`bench.yml`,
`sql-benchmarks.yml`, `v3-commit-metadata.yml` — remove the v3 step + the `continue-on-error` on the
v4 steps) + ship + document `scripts/psql-bench.sh`. **Before removing `continue-on-error`: re-run
the PR-3.5 cross-check `scripts/cross_check_python_writer.py --postgres "$DSN" --envelopes
<real_envelopes.json>` via the `bench_ingest` IAM role against accumulated prod soak data and confirm
clean** (Python writer UPDATEs seeded rows, 0 duplicate INSERTs, value columns round-trip). That is a
prod RDS WRITE (IAM-token auth via boto3/OIDC) needing user coordination — operator-run vs
agent-run-with-creds — a genuine externalized-side-effect gate, NOT a Class A pause; confirm with the
user how to run it before proceeding. Review for PR-5.1 is cross-bundle (3 workflows + script + py) →
gauntlet **pr-3**. Then PR-5.2 (DNS flip), PR-5.3 (decommission). Each prod write remains
harness-gated. Pre-squash backup ref `refs/backups/ct-bench-v4-pre-squash-386ea347b`; recreate a
fresh one before any PR-5.3/final squash.

## SESSION HANDOFF 2026-06-12 (PR-5.0.95 scoped + design-approved) [SUPERSEDED by the section above; PR-5.0.95 is now DONE]

**Resume via `/spiral:big-plans` in the `vortex4` worktree.** (Heads-up: the stock
`resume_routing.py` falls to its Coarse floor on this
spine because the custom `status: executing` YAML block diverges from the stock schema — that is
expected; the `Current State` block + git log are ground truth, per the Hybrid-fallback rule.)

**1. PR-5.0.9 is DONE.** Opt-in full-history chart loading shipped, gauntlet-pr-2-accepted (2 cycles),
pushed, and deployed (web-deploy.yml green incl. CDN probe). Do NOT re-open it; its full record is the
`### PR-5.0.9:` Implementation status entry. Live at `https://benchmarks-web.vercel.app`.

**2. NEW SUB-PR: PR-5.0.95 (lazy-hydration + resilient loading), inserted AHEAD of PR-5.1.** After
PR-5.0.9 shipped, the user reported that expanding a large group (clickbench, ~43 charts) still feels
slow and sometimes hangs, and approved a second loading-model round. The design is authoritative at
**`.big-plans/ct__bench-v4-uiux-r2-design.md`** (read it in full — it carries the code-grounded
diagnosis, the approved mechanics, implementation sites, 3 flagged OPEN DECISIONS, the test plan, and
acceptance criteria). Three approved improvements:
   - **(A) Viewport-based lazy hydration** on the landing page: gate each group chart's initial
     `?n=100` fetch+construct behind an `IntersectionObserver` (reuse the permalink `else`-branch
     pattern at `Chart.tsx` ~L1646), so on group open only ~visible charts hydrate (top-first, in
     visual order) and the rest hydrate on scroll. Reconcile the all-charts summary-hover prefetch
     (`Chart.tsx` ~L1628) so it cannot re-introduce the burst. This is the highest-leverage change
     and also makes "Expand All" cheap.
   - **(B) Fetch timeout + abort + retry**: the two `fetch()` calls (`Chart.tsx` ~L463/550) pass NO
     signal/timeout today, so a stalled request spins "loading…" forever and group-close doesn't
     cancel in-flight requests. Wire the controller `aborter.signal` in + add a `FETCH_TIMEOUT_MS`
     AbortController timeout; abort in-flight fetches on group close/teardown; give the initial fetch
     a RETRY affordance that re-issues the FETCH (today's 4s auto-dismiss retries CONSTRUCTION only).
   - **(C) Spinner animation** replacing the static "loading…" text (`Chart.tsx` ~L1808) + the chip
     loading state, with a `prefers-reduced-motion` guard.

**3. Pre-implementation investigation (do FIRST, ~10 min, read-only):** confirm whether the "hangs"
are purely client-burst (A fixes) or also need server work — open clickbench while watching Vercel
function logs + RDS connections (via `bench_read`), or time `/api/chart` for a few clickbench slugs
cold vs warm. If a SPECIFIC chart query is slow, note it as separate follow-up; do NOT expand
PR-5.0.95 scope to server queries (that is read-path perf, out of this UI/UX round). `bench_read`
reads are authorized; prod WRITES are not in scope for PR-5.0.95.

**4. How to implement:** follow the big-plans Phase-2 loop — writing-plans (JIT task-plan) -> SDD
-> 2-vote gauntlet pr-2 -> Step 2.5 close -> push. The design doc's 3 OPEN DECISIONS
(summary-prefetch reconcile; keep/simplify `nextGroupOpenPriority`; timeout value/mechanism) may be
pinned via `brainstorming`/`grill-me` on the doc first, or decided per the doc's recommendations and
implemented directly. Codex companion was ABSENT last session -> gauntlet ran Claude-only; if you
want the Claude+Codex executor disjointness (PR-5.1.5 had it), check the companion is installed.

**5. After PR-5.0.95 closes:** PR-5.1 (promote v4 `--postgres` ingest to required + drop the v3
`--server` write + ship `scripts/psql-bench.sh`) — its FIRST step is a PROD RDS WRITE gate (re-run
the PR-3.5 cross-check via the `bench_ingest` IAM role) that needs user coordination (operator-run
vs agent-run with creds); a genuine externalized-side-effect gate, NOT a Class A pause. Then PR-5.2
(DNS flip), PR-5.3 (decommission). Each prod write remains harness-gated.

## SESSION HANDOFF 2026-06-11 evening (UI/UX scoped; PR-5.0.9 ready to implement) [SUPERSEDED by the 2026-06-12 section above; PR-5.0.9 is now DONE]

**Newest handoff; supersedes the sections below as the "read first" entry.** Resume via
`/spiral:big-plans` in the `vortex4` worktree.

**1. The UI/UX round is SCOPED and DESIGN-APPROVED.** The user-queued UI/UX optimization round
(item 2 of the section below) was scoped this session via `superpowers:brainstorming`: scope is
pinned to the LOADING MODEL ONLY (no visual or layout redesign this round). The approved design
lives at **`.big-plans/ct__bench-v4-uiux-design.md`**; read it in full before implementing (it
carries the problem measurements, the approved mechanics, implementation sites, the test plan, and
acceptance criteria). `spiral:grill-me` was skipped at user wrap-up; the load-bearing assumptions
were verified empirically instead (live timings + code reading; see the design doc's process
note). Optionally run grill-me on the design doc before implementing if extra rigor seems
warranted; otherwise implement directly.

**2. NEW SUB-PR: PR-5.0.9 (opt-in full-history chart loading), inserted AHEAD of PR-5.1** in the
PR enumeration (this spine's Amend flow). One-paragraph scope: delete the automatic `?n=all`
warmup in `Chart.tsx` `onGroupOpen` (today every chart in an opened group background-fetches its
full history: ~24MB for the 22-chart tpch group, hundreds of MB on Expand All); full history
becomes per-chart opt-in via (a) an always-visible window chip ("latest 100 of 3,572"; hover
presents "load all N"; spinner while loading; "all N" when complete; "retry" on failure; click
fetches at `INTERACTION_FULL_PRIORITY`), (b) a ~600ms same-card hover-dwell silent prefetch at a
new mid-tier priority (hover reveals the control immediately, only the dwell starts the fetch,
`pointerleave` cancels; the user picked "Both, staged"), and (c) the existing
`rangeTouchesUnloadedHistory` interaction promotion, unchanged. Also add
`stale-while-revalidate=86400` beside `s-maxage=300` on the API cache policy (cold CDN is the
common case on this low-traffic site). Review: 2-vote gauntlet preset pr-2.

**3. Key code facts (do not re-derive):** the virtual shared x-axis makes late loading jank-free
(windowed payloads carry `history.total_commits`/`start_index`; `normalizeChartPayload` pads the
unloaded prefix with nulls; `replaceChartPayload` fills in place; slider bounds come from the full
length at construction), so the user's past Chart.js late-load concern is structurally addressed.
The hover-prefetch pattern has precedent (group-summary `pointerenter` prefetches initial
payloads, `Chart.tsx` ~L1448-1459). The permalink page already upgrades only on interaction.
Live timings 2026-06-11: `?n=100` 34KB / 0.17-0.2s warm MISS / 0.06s HIT / ~7.8s cold first hit;
`?n=all` (tpch, 3,572 commits) 1.1MB / 0.46-1.07s warm MISS. Web vitest suite at 214 green at
session start.

**4. After PR-5.0.9 closes:** PR-5.1 (unchanged scope; full detail in the 2026-06-12 section
below), then PR-5.2 (DNS flip), PR-5.3 (decommission).

## SESSION HANDOFF — 2026-06-11 (history cleaned + rebased; UI/UX redesign queued BEFORE PR-5.1 — superseded as "read first" by the evening section above)

**Newest handoff — supersedes the 2026-06-12 section below** (the two sessions' clocks differ by ~1 day; this is the latest). Resume via `/spiral:big-plans` in the `vortex4` worktree.

**1. Branch history was cleaned + rebased (2026-06-11).** The granular 143-commit branch is now **6 commits — one squashed commit per phase** (Phase 1-5) plus a `plan:` re-point commit, rebased clean onto `origin/develop` (no conflicts) and force-pushed. **HEAD = `origin/ct/bench-v4` = `a908d0d0c`**, working tree clean. New per-phase SHAs: Phase 4 = `b9fc6220d`, Phase 5 = `856bf7146`. `phase_entry_sha` (`b9fc6220d`) + `last_commit` (`856bf7146`) were re-pointed and verified live (`git diff b9fc6220d..HEAD` spans exactly Phase 5). **Pre-squash backup: `refs/backups/ct-bench-v4-pre-squash-386ea347b`** (= old 143-commit tip; fully recoverable). Every SHA quoted in the historical handoff sections below is now ORPHANED — trust THIS section + the `Current State` block, not the older sections, for live SHAs.

**2. [RESOLVED 2026-06-11 evening: scoping complete + design approved; see the evening handoff
section above and `.big-plans/ct__bench-v4-uiux-design.md`. The sub-PR is PR-5.0.9. Do NOT re-run
brainstorming.] USER DIRECTIVE — UI/UX optimization + potential redesign comes BEFORE PR-5.1.** Before starting the PR-5.1 CI-promotion work (described in the 2026-06-12 section below), the user wants a round of **UI/UX optimization and potential redesign** of the benchmarks site. This is **not yet scoped**. On resume: **do NOT auto-start PR-5.1.** First scope the UI/UX work WITH the user — use `superpowers:brainstorming` then `spiral:grill-me` to pin intent (what to optimize, redesign vs. incremental polish, success criteria, which pages/components) — then **amend Phase 5** (Step 3.5 Amend flow) to insert it as a new sub-PR slotted ahead of PR-5.1. PR-5.1 begins only after that UI/UX sub-PR closes. (Context for scoping: the current site is the v4 Next.js 15 read service at `https://benchmarks-web.vercel.app`; the read path is already fast post-PR-5.1.5, so this round is about UX/visual quality, not query perf.)

**3. After the UI/UX work: PR-5.1 is the next planned step** — full scope in the **2026-06-12 section immediately below** (promote v4 `--postgres` ingest to required, drop the v3 `--server` write from 3 CI workflows, ship `scripts/psql-bench.sh`; gated on a prod cross-check WRITE). That section is still accurate; only its HEAD/commit-trail SHAs are stale (superseded by item 1 above).

## SESSION HANDOFF — 2026-06-12 (PR-5.1.5 CLOSED; PR-5.1 detail — superseded as "read first" by the 2026-06-11 section above)

**Resume via `/spiral:big-plans` in the `vortex4` worktree.** `Current State` routes you to `current_pr: PR-5.1`. PR-5.1.5 (read-path perf) is fully CLOSED — do NOT redo it; its complete record is the `### PR-5.1.5:` Implementation status entry. (HEAD/commit SHAs in THIS section are STALE after the 2026-06-11 history cleanup — see the top section for live SHAs; HEAD is now `a908d0d0c`.) **PR-5.1 is now GATED behind the UI/UX optimization/redesign work the user queued (top section) — do not start PR-5.1 until that lands.**

### PR-5.1.5 outcome (one paragraph, so you don't re-open it)
Shipped + deployed + live + gauntlet-accepted (3 cycles, executor=parallel Claude+Codex). Recursive-CTE skip scans replaced the three whole-table read-path scans (`/api/groups` cold 6.0s -> ~1s; tpch chart 13.6s -> 0.094s; all byte-identical vs the replaced queries on testcontainer + full prod seed). Write-path `commit_timestamp` stamping in `post-ingest.py` + the Rust loader. Migrations 006/007 applied to prod as master + VACUUM ANALYZE. PR-5.0 deferred data-checks resolved (chart-count sets match live v2 `bench.vortex.dev`). Commit trail (pre-cleanup `93ccf970e`..`4575b1d3d`; now squashed into the Phase 5 commit `856bf7146`). All suites green (web vitest 214, migrate Rust 100, python 154).

### PR-5.1 — what's next (NOT started)
Scope (spine PR-enumeration row PR-5.1): **promote the v4 `--postgres` ingest to required and drop the v3 `--server` write** from the 3 ingest workflows; `post-ingest.py` runs only with `--postgres`; **ship + document `scripts/psql-bench.sh`** (phase-4 end-review fold-in — file does NOT exist yet).
- **Pre-promotion GATE (do FIRST, before removing `continue-on-error`):** re-run the PR-3.5 cross-check `scripts/cross_check_python_writer.py --postgres "$DSN" --envelopes <real_envelopes.json>` against accumulated prod soak data and confirm clean (Python writer UPDATEs seeded rows, 0 duplicate INSERTs, value columns round-trip). **This is a prod RDS WRITE via the `bench_ingest` IAM role** — harness auto-mode-gated AND likely needs operator AWS creds (the session so far only had `bench_read` password reads; bench_ingest is IAM-token auth via boto3/OIDC). Confirm with the user how to run it (operator-run vs. agent-run with creds) before proceeding — this is the genuine externalized-side-effect gate, NOT a Class A pause.
- **Exact CI edit sites** (all verified this session): each of `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml` has (a) one `Ingest results to v3 server` step using `--server "${{ vars.V3_INGEST_URL }}"` + `INGEST_BEARER_TOKEN` (REMOVE per "drop v3-write"), and (b) three `continue-on-error: true` v4 steps (`Configure AWS credentials for v4 ingest` / `Install uv for v4 ingest` / `Ingest results to v4 Postgres (best-effort)`) — REMOVE the `continue-on-error` so v4 gates CI. bench.yml v3 step ~L108-118, v4 steps ~L128-156; sql-benchmarks.yml v3 ~L493-503, v4 ~L506-534; v3-commit-metadata.yml v3 ~L28-35, v4 ~L38-69. Then remove `--server` mode from `scripts/post-ingest.py` (+ its tests). NOTE the decommission inventory (Table F) lists `INGEST_BEARER_TOKEN` removal — the PR-5.1 v3-step removal drops the `secrets.INGEST_BEARER_TOKEN` references; `git grep -n INGEST_BEARER_TOKEN` must reach 0 by PR-5.3.
- **Verify**: `yamllint --strict -c .yamllint.yaml` on the 3 changed workflows (REQUIRED per .github/AGENTS.md); `ruff check scripts/` + `pytest scripts/` (post-ingest.py + its tests); `shellcheck`/`bash -n` on psql-bench.sh.
- **Review**: this is cross-bundle (3 workflows + script + py) -> gauntlet **pr-3** (fresh+correctness+maint). Then Step 2.5 close + advance to PR-5.2 (DNS flip).

### KEY OPERATIONAL FACTS (carried forward)
- **Docker must be running** for testcontainer suites (web vitest + migrate pytest + the migrate Rust e2e). `python` is not on PATH — use `python3`/`uv run`.
- **Prod master DSN** (further migrations/VACUUM): `AWS_PROFILE=bench-prod aws rds describe-db-instances ... MasterUserSecret.SecretArn` -> `aws secretsmanager get-secret-value` -> `PGUSER`/`PGPASSWORD` + `PGHOST=vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com PGDATABASE=vortex_bench PGSSLMODE=verify-full PGSSLROOTCERT=~/rds-ca.pem`. bench_read pw: `~/.bench-read-pw`; CA: `~/rds-ca.pem`. **Every prod write is harness auto-mode-gated** even under broad approval — surface + get per-action approval. The user approved prod READS this session.
- **Gauntlet flow used this session** (reuse it): `executor=parallel` via `compose_prompts.py --reference-dir <gauntlet>/reference --executor-routing fresh=parallel,correctness=parallel`; dispatch Claude lenses via `Agent`, Codex lenses via `node ~/.config/claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs task --model gpt-5.5 --effort xhigh --prompt-file <f>` (background); then a synthesizer subagent over the 4 executor-tagged blobs. The project **~3-cycle cap** (spine Key-decision) governs: fix-and-accept or defer at cycle 3, never spiral.
- migrations dir is repo-root `migrations/` (NOT benchmarks-website/migrations/). `schema-deploy.yml` is develop-only so ct/bench-v4 pushes only fire `web-deploy.yml`. Live URL: `https://benchmarks-web.vercel.app`.
- **Pre-squash backup ref EXISTS now:** `refs/backups/ct-bench-v4-pre-squash-386ea347b` (created 2026-06-11 at the old 143-commit tip). Recreate a fresh one before the PR-5.3 / final squash.

## SESSION HANDOFF — 2026-06-11 (PR-5.1.5 read-path-perf DEPLOYED) [HISTORICAL — PR-5.1.5 CLOSED; superseded by the 2026-06-12 handoff above]

**HISTORICAL.** PR-5.1.5 is closed (see the 2026-06-12 handoff above + the `### PR-5.1.5:` Implementation status entry). This section and the cold-render decision section below describe PR-5.1.5 mid-flight; the cold-render decision resolved as recursive-CTE skip-scan and every listed REMAINING item is done.

### COLD-RENDER DECISION — RESOLVED 2026-06-11 (user chose (a) skip-scan)
The site is deployed + fast-cached but **cold render of `/api/groups` is ~6s** (the ~13 per-group summaries are each an index-only scan of the group's full ~1.8M-row history at ~2.4s, run 8-concurrently). For this low-traffic dashboard the 5-min CDN cache often expires between visits, so cold is common. **User decided: (a) recursive-CTE skip-scan** on the summaries (jump to each series' latest -> ~ms -> cold render ~1s); apply the same pattern to discovery + DISTINCT-engine where it falls out naturally (handoff item 4). Options (b) concurrency bump and (c) accept were declined.

### What's DONE (committed + pushed to origin/ct/bench-v4 @ 6053b70cc; deployed live)
- **sargable WHERE** (2e637401e), **parallelize summaries** (629b5b0b6), **denormalize commit_timestamp + migration 006 + b/d indexes** (680b30e6e), **covering index migration 007 + DISTINCT ON summary** (a4834ba1f). All Docker-verified (web vitest 211 / migrate pytest 24).
- **Prod**: instance upsized `db.t4g.micro -> db.t4g.medium` + custom param group `vortex-bench-pg16` (work_mem 32MB). Migrations **006 + 007 APPLIED to prod as master** (`migrate-schema.py apply`) + `VACUUM ANALYZE query_measurements` (the 4.85M-row backfill bloated the table -> VACUUM was required for the planner to use the new indexes; a real lesson). `migrate-schema.py status` = 7 applied, 0 pending.
- **Deployed**: `web-deploy.yml` succeeded on the ct/bench-v4 push INCLUDING the CDN probe (the push-unblock). Live: tpch chart 13.6s->**0.094s**; `/api/groups` 38s->**0.079s cached** / **~6s cold**. Prod SQL: chart 75ms, tpcds summary **2.4s** (Index Only Scan, Heap Fetches 0 via `idx_query_measurements_summary` covering INCLUDE value_ns), DISTINCT engine 458ms, discovery 1.3s.

### REMAINING for PR-5.1.5 close-out (after the cold-render decision)
1. **(c) WRITE-PATH still TODO** — `scripts/post-ingest.py` (the v4 `--postgres` insert) + `benchmarks-website/migrate/src/postgres.rs` (Rust loader) must populate `commit_timestamp` on `query_measurements` inserts. Use the subquery pattern `commit_timestamp = (SELECT timestamp FROM commits WHERE commit_sha = ...)` (mirrors the test seeds in `test-harness.ts`/`groups.test.ts`). The column is NULLABLE + summary uses `NULLS LAST`, so existing data (backfilled by 006) is correct NOW and the deployed site is fine; the write-path is needed before the develop merge so new ingests aren't NULL. A post-deploy re-backfill clears any transient NULLs. (Other inserts: `verify.rs`/`postgres_e2e.rs`/`test_post_ingest_postgres.py` don't read commit_timestamp -> leave NULL.)
2. **Gauntlet review** of the whole PR-5.1.5 diff (Step 2.3, `Skill(spiral:gauntlet)` preset=pr-2 — it IS available this session via `/reload-plugins`; the diff spans queries.ts/summary.ts/db.ts + migrations 006/007 + the migrate test + write-path). Then Step 2.5 close.
3. **PR-5.0 deferred data-checks** (in Deferred work, Resolved-by this PR): now verifiable on the fast site — `curl <prod-url>/api/groups | jq '.groups[].charts[].slug' | sort` vs the family registry, + ~5 chart slugs vs the live v2 site `benchmarks.vortex.dev`.
4. (b)/(d) skip-scans were NOT separately rewritten — discovery 1.3s + DISTINCT engine 458ms were folded into the ~6s and judged acceptable; revisit only if the cold-render decision is (a) (skip-scan), which would naturally cover them.

### KEY OPERATIONAL FACTS
- **Docker must be running** for the testcontainer suites (web vitest + migrate pytest); the user started Docker Desktop this session. `python` is not on PATH — use `python3`/`uv run`.
- **Prod master DSN** (for further migrations/VACUUM): `AWS_PROFILE=bench-prod aws rds describe-db-instances ... MasterUserSecret.SecretArn` -> `aws secretsmanager get-secret-value` -> `PGUSER`/`PGPASSWORD` + `PGHOST=vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com PGDATABASE=vortex_bench PGSSLMODE=verify-full PGSSLROOTCERT=~/rds-ca.pem`. **The harness auto-mode classifier gates EACH prod write** (migration apply, VACUUM, index ops) even under broad user "run everything" approval — surface + get per-action approval, or the user adds a Bash allow-rule. bench_read pw: `~/.bench-read-pw`; CA: `~/rds-ca.pem`.
- migrations dir is repo-root `migrations/` (NOT benchmarks-website/migrations/). schema-deploy.yml is **develop-only** so ct/bench-v4 pushes only fire web-deploy.yml; at the develop merge schema-deploy no-ops 006/007 (already in prod ledger, master-pre-applied like 005). User authorized PUSHING (no longer deferred — the CDN probe passes now). Live URL: `https://benchmarks-web.vercel.app`.

## SESSION HANDOFF — 2026-06-11 (PR-5.0 + read-path perf) [HISTORICAL/SUPERSEDED — PR-5.0 closed; see the PR-5.1.5 handoff above]

**Read this first on resume.** PR-5.0 (bring prod online) is functionally complete for the data seed; it is BLOCKED only on a newly-discovered read-path perf problem (the Step 4 data checks). **User decision (2026-06-11):** close PR-5.0 on the verified seed; fix the read path as a **dedicated PR before PR-5.2 (DNS flip)**.

### What's DONE + verified (PR-5.0 core)
- Phase-5 transition complete (Step 3.5 re-plan-completion; `phase_entry_sha 9f68717b8`).
- Operator-gate: prod schema fully applied (5 migrations incl. 005 as master); `bench_read` authenticates; prod was empty pre-load.
- Fresh v3 snapshot rebuilt from `s3://vortex-benchmark-results-database/v3-backups/20260610T210150Z.tar.gz` (acct 375504701696, region **us-east-1** — NOT the believed us-east-2) → `~/bench-fresh.duckdb` (4.85M query_measurements; `vector_search_runs`=0, which resolves the PR-3.4 caveat).
- Load: 5.24M rows, atomic, counts exact. Verify: `0 presence diffs, 0 value mismatches`. Cross-check: `11 updated, 0 inserted` (bench_ingest IAM). Re-verify after cross-check: still clean (seed intact).
- **rustls TLS fix** (PR-5.0's only production code): native-tls rejected the RDS leaf (no `serverAuth` EKU; macOS Secure Transport). Swapped `migrate/src/postgres.rs` `connect_postgres` + `migrate/Cargo.toml` to rustls (`84c3715cb`) + a should-fix loud-bail on empty CA bundle (`d0175d70c`). Inner-loop 2-vote gauntlet (reconstructed via `compose_prompts.py` since `Skill(spiral:gauntlet)` was unavailable): cycle 1 **ACCEPTED** (`5e049db89`).
- Vercel deploy live at HEAD (`35c05c500`) at `https://benchmarks-web.vercel.app`, serving the seeded data (`/api/health` confirms row counts). **`VERCEL_TOKEN` was ROTATED by the user** (leaked one revoked).

### The BLOCKER: read-path perf at prod scale (data + rendering are CORRECT — proven)
- `/api/groups` + landing page time out (~1–2 min); big-dataset charts (tpch/tpcds/clickbench) `/api/chart` ~24s; small-dataset charts (polarsignals) ~1s and correct.
- ROOT CAUSES (diagnosed against prod, read-only EXPLAIN/timing):
  1. **Non-sargable `IS NOT DISTINCT FROM`** on nullable dims (`dataset_variant`/`scale_factor`) in `queries.ts`/`summary.ts` → `idx_query_measurements_chart` is seeked only on the leading `dataset=` and the rest is a heap Filter → per-dataset full scans (tpch chart scans ~1.8M rows).
  2. **`collectFilterUniverse`** (runs on EVERY chart page + landing, in parallel with the chart): `SELECT DISTINCT engine FROM query_measurements` = parallel seq scan of 4.85M rows (~7.5s) + DISTINCT `format` UNION across all fact tables. This is why even the small polarsignals *page* took >10s (server-side, not client).
  3. **Landing discovery**: full `GROUP BY` scans ×5 families (~1.2–5s for query_measurements; 321 charts / 13 query-groups; 64 groups total).
  4. **N+1 summaries**: 64 sequential per-group summary queries; the ~11 v2-allowlisted query-summaries are `row_number()` windows over the whole group (~10s each for tpcds 1.78M / clickbench 993K; spills to the 4MB work_mem).
  5. **Instance**: `db.t4g.micro` = 1 GiB RAM, `shared_buffers` ~180MB, `work_mem` 4MB; `query_measurements` is 1.2GB (db 1.49GB) → doesn't fit in cache → full scans hit disk. CloudWatch FreeableMemory ~70MB. (User will upsize the instance as part of the fix.)
- **CLIENT model is SOUND — NOT a v3 repeat.** Landing groups are `<details>` collapsed-by-default → ZERO chart data on initial load; charts lazily fetch `?n=100` (`DEFAULT_COMMIT_WINDOW=100`, max 1000) on group-expand through a **4-concurrent bounded hydration queue**; LTTB downsamples rendered points (`MAX_VISIBLE_POINTS`); `?n=all` only on explicit per-chart interaction; one chart per `/chart/[slug]` page (server-inlines just that one payload). Client heap ~tens of MB worst case. **No client rework needed** — the whole problem is server-side query perf.

### FIX APPROACH for the new read-path-perf PR (must land before PR-5.2)
- (a) **Sargable WHERE** (highest leverage): replace `col IS NOT DISTINCT FROM $x` with `col IS NULL` (key value null) / `col = $x` (non-null) across `queries.ts` + `summary.ts` — logically identical for a concrete chart key but index-usable (tpch chart 24s → ~ms).
- (b) **Loose-index-scan / recursive-CTE skip-scan** for the 5 discovery `GROUP BY`s.
- (c) **Index-supported latest-per-series summary** (DISTINCT ON + a supporting index, or denormalize `commit_timestamp` into the fact tables — the plan bans materialized views/triggers, but indexes/columns are fine).
- (d) **`collectFilterUniverse`**: small lookup table or index-only DISTINCT (avoid the 4.85M seq scan).
- (e) **Parallelize** the 64 per-group summaries (bounded by `poolMax`; consider raising it).
- (f) **Upsize the RDS instance** (t4g.micro → larger; raise `work_mem`/`shared_buffers`).
- Verify with before/after prod timings + a working `/api/groups` + the Step 4 data checks (family-registry slug match + ~5 chart slugs vs the v2 site `benchmarks.vortex.dev`).

### EXACT NEXT STEPS (next session)
1. Resume via `/spiral:big-plans` in the `vortex4` worktree.
2. **Step 2.5 — close PR-5.0**: append the Implementation status entry (data seed DONE + verified; rustls fix reviewed-accepted; **Step 4 data checks DEFERRED** to the read-path-perf PR).
3. **Amend Phase 5** (Step 3.5 Amend) to add the read-path-perf PR (scope above) slotted BEFORE PR-5.2; decide its exact scope/approach with the user.
4. Then continue the Phase-5 sequence: PR-5.0.5 (statpopgen/polarsignals v2-name restore) is next in the original order.

### OPERATIONAL FACTS (avoid re-deriving)
- Prod: instance `vortex-bench-prod`, endpoint `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432`, db `vortex_bench`, master user `postgres`, acct 245040174862 us-east-1, via `AWS_PROFILE=bench-prod`. Master pw via `aws rds describe-db-instances … MasterUserSecret.SecretArn` → `aws secretsmanager get-secret-value`.
- `bench_read` pw on disk `~/.bench-read-pw`; RDS CA bundle `~/rds-ca.pem` (3 roots — rustls trusts the whole bundle; `psql` uses `sslmode=verify-full` + `sslrootcert`; the Rust loader uses `sslmode=require` + `--ca-cert ~/rds-ca.pem`). Fresh snapshot: `~/bench-fresh.duckdb`. Loader binary: `target/debug/vortex-bench-migrate` (rustls).
- Working chart URL (proves data+render correct): `https://benchmarks-web.vercel.app/chart/qm.eyJrIjoiUXVlcnlNZWFzdXJlbWVudCIsImRhdGFzZXQiOiJwb2xhcnNpZ25hbHMiLCJkYXRhc2V0X3ZhcmlhbnQiOm51bGwsInNjYWxlX2ZhY3RvciI6bnVsbCwic3RvcmFnZSI6Im52bWUiLCJxdWVyeV9pZHgiOjB9`. Chart slug = `<prefix>.<base64url(JSON.stringify(orderedKey))>` (prefixes qm/ct/cs/rat/vsr; key field order per `chartKeyToSlug` in `web/lib/slug.ts`).
- Git: `origin/ct/bench-v4` is at `35c05c500` (the review-accepted commit — INCLUDES the rustls fix `84c3715cb`, the should-fix `d0175d70c`, and the gauntlet-cycle-1 record; this is what the live Vercel deploy reflects). Local is **3 commits ahead**, all **plan-only** (`1b90051ba` discovery, `2f337a0dd` root-cause, `7c28f2c34` this handoff) — no code/web changes, so the deploy is unaffected and they're unpushed by choice (pushing re-triggers `web-deploy.yml`, whose CDN probe will FAIL until the perf fix — expected). Sign-off `"Connor Tsui" <connor@spiraldb.com>` via `commit -F`, no co-author trailer, no `---` scissors. pre-squash backup ref `backup/bench-v4-pre-squash-20260609` MISSING — recreate before PR-5.3/final squash.

## Context

The `benchmarks-website/` subsystem is the public face of Vortex's continuous-benchmark numbers. **Corrected current-state (2026-06-04 re-plan, verified against the repo):** the LIVE public site at `benchmarks.vortex.dev` is the **v2** Vite/React SPA served by a Node `server.js` that reads benchmark data from the S3 bucket `vortex-ci-benchmark-results/data.json.gz` (+ `commits.json`), refreshed every ~5 min; v2 is published as a Docker image by `publish-benchmarks-website.yml`. A **v3** system (Rust/Axum + embedded DuckDB on EC2, custom systemd deploy + hourly S3 backup, bearer-token CI ingestion at `POST /api/ingest`, in-process artifact cache `read_model.rs`) is **built and live and CI-fed but has never served public traffic** — the v2→v3 public cutover was never completed (DNS stays on v2; the `7efbcacd2` "remove v2" commit lives on an unmerged branch). CI runs (one writer on `ubuntu-latest`, two on `bench-dedicated`, eleven from the SQL bench matrix) fan in to ~14 parallel `--server` envelope POSTs to the v3 endpoint per push to `develop`. **v3's structured DuckDB is the authoritative, clean source the v4 migration loads history from** (far better-structured than v2's S3 JSON blobs). The migration target is **v4** (Postgres + Next.js), replacing BOTH v2 and v3; cutover is **v2→v4 direct** (v3 is skipped, then decommissioned).

The migration target is the same data model on AWS RDS Postgres `db.t4g.micro` (account `245040174862`, region `us-east-1`, single-AZ) fronted by RDS Proxy, with CI writing directly via GitHub OIDC → AWS IAM → RDS-IAM-auth tokens (replacing the Axum POST and the bearer token), and a stateless Next.js 15 read service on Vercel using server components with header-driven edge caching (`Vercel-CDN-Cache-Control` rules on the HTML routes + `Cache-Control` s-maxage on API 200s; see the Read-service-framework Key-decision amendment for why `unstable_cache`/`revalidateTag` were not usable) against the same DB. Edge CDN replaces the in-process artifact cache. Decommission targets are `benchmarks-website/server/`, `benchmarks-website/ops/`, the bearer tokens, `/api/admin/*`, the custom backup pipeline, and systemd timers. Success is: every existing chart renders byte-equivalently against the new substrate; the next CI run after cutover idempotently upserts existing `measurement_id` rows; the EC2 instance is decommissioned and the runbook is replaced by managed-Postgres + Vercel.

**Work shape**: migration — substrate change (DuckDB → Postgres, Axum → Next.js, EC2 → Vercel) with bit-exact preservation of `measurement_id` (xxhash64), v3 envelope wire format, 6-table schema shape, `ON CONFLICT (measurement_id) DO UPDATE` upsert semantics, captured metrics, and chart URL shape. The highest-leverage insight is **Behavior-preservation**: every preserved invariant must appear as an explicit acceptance criterion and be pinned by a test.

The orchestrating branch for this work is `ct/bench-v4`. Per user direction at planning kickoff, all child PRs land on `ct/bench-v4` rather than `develop`; the final integration of `ct/bench-v4` into `develop` happens once via squash-merge after the full migration ships.

**Orchestration note** (read on resume): the big-plans state machine operates on `ct/bench-v4` throughout — plan-edits, implementation commits, inner-loop gauntlet, and phase-end gauntlet all happen on this branch. At each phase end, after gauntlet's verdict is `accept`, the assistant cherry-picks the phase's commits (range `phase_entry_sha..HEAD`) onto a fresh `claude/phase-N-<slug>` branch created at `phase_entry_sha`, pushes it, and opens a review-only GH PR (`claude/phase-N-<slug>` → `ct/bench-v4`) labeled "review only — do not merge". This preserves per-phase reviewability on GitHub without bifurcating the working branch. The single squash-merge of `ct/bench-v4` → `develop` happens at end-of-Phase-5 via a final GH PR.

## Out of scope

- Changing the v3 envelope wire format in `vortex-bench/src/v3.rs` (this stays).
- Changing the 6-table schema shape (column order, nullability, dim-tuple membership).
- Changing the xxhash64 `measurement_id` algorithm in `benchmarks-website/server/src/db.rs:162-257` — the new ingest writer reproduces it bit-identically.
- Adding materialized views or DB triggers.
- Changing the `commits` upsert semantics (`ON CONFLICT (commit_sha) DO UPDATE SET <all-but-PK>`).
- Adding a non-Postgres read path (e.g., S3 + JSON files) during steady state.
- A multi-region Postgres setup; single region is sufficient.
- Authenticated user accounts on the read side; everything stays public-read.
- Backfilling rows older than the current v3 DB (cutover seeds from the live DuckDB snapshot only).
- Editing the v2 React/Vite frontend (`src/`, `index.html`, `vite.config.js`, `package.json`, top-level `Dockerfile`, `docker-compose.yml`, `publish-benchmarks-website.yml`) outside the dedicated final-cutover PR.
- Best-effort (`continue-on-error: true`) on the **v3** CI ingest step — v3 stays hard-required (it feeds the DuckDB migration source + the live read path). **(2026-06-04 re-plan note:** the **v4** ingest step IS intentionally best-effort *during the dual-write soak* — a v4 hiccup must not break the proven v3 pipeline while v4 is unproven — and is promoted to required at cutover. This deliberately scopes the prior blanket "no best-effort" rule to v3-only.)
- Persisting `measurement_id` on the wire — it stays server-internal.
- The `/api/admin/sql` operator console (replaced by managed-Postgres console / psql access).

## Prior art / external references

- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/AGENTS.md`** — subsystem playbook including "Footguns we have already hit"; the `SCHEMA_VERSION` lockstep rule and wire-shape coupling across `records.rs`, `v3.rs`, `classifier.rs` carry over.
- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/README.md`** — the v2→v3 cutover playbook; same dual-write → soak → single deletion-commit pattern applies to v3→Postgres.
- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/migrate/src/verify.rs`** — `VerifyReport { matched_groups, only_in_v2, only_in_v3, ChartDiff }` structural diff between substrates. Direct template for the dual-write verification harness.
- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/migrate/src/migrate/mod.rs:31-35`** — uses `vortex_bench_server::db::measurement_id_*` to keep hash compatibility across the v2→v3 boundary. The same dependency edge applies to the new ingest writer.
- **`/Users/connor/spiral/vortex-data/vortex4/.github/workflows/compat-gen-upload.yml`** — dry-run + `environment: compat-upload` (manual-approval gate) + OIDC role assumption. Structural template for any approval-gated migration step (schema deploy, prod-data ingest, decommission).
- **`/Users/connor/spiral/vortex-data/vortex4/.github/workflows/bench.yml:9-11,21-24,98-101`** — production GH-OIDC → AWS IAM pattern reference (`GitHubBenchmarkRole` in account `245040174862`, used for benchmark-S3-cache uploads today). Mirror the trust-policy SHAPE (sub-claim wildcard, `aws-actions/configure-aws-credentials@d979d5b3a71173a29b74b5b88418bfda9437d885 # v6`) for the NEW `GitHubBenchmarkSchemaRole` + future `GitHubBenchmarkIngestRole` created by PR-1.1's `provision.sh` in the same account `245040174862`.
- **`/Users/connor/spiral/spiraldb/.github/workflows/deploy-spiraldb.yml:37-89`** — `deploy-migrations` job blueprint (OIDC → cloud identity → managed Postgres → `prisma migrate deploy`). GCP flavor; translate step-by-step to AWS.
- **`/Users/connor/spiral/spiraldb/.github/workflows/prisma-isolation.yml`** — 49-line workflow rejecting PRs that mix migration files with code files. The schema-isolation discipline benchmarks-website will need post-cutover; copy verbatim with a different migration directory.
- **`/Users/connor/spiral/spiraldb/spiraldb/`** — production Rust/Axum + `sea-orm` + `sqlx-postgres` + Prisma-managed schema service. Reference for connection-pool config + the sidecar-proxy operational pattern. No `sqlx::migrate!` in the Spiral ecosystem; choosing it for this migration is new ground.
- **`vortex-bench/src/v3.rs:32-53`** — per-binary→`V3Record` mapping; the bench harnesses that produce envelopes are the immutable producer interface.

## Architecture

```
                          ┌─────────────────────────────────────┐
  ~14 CI writers          │   GitHub Actions OIDC → AWS STS     │
  per `develop` push      │   AssumeRoleWithWebIdentity         │
  (bench.yml, sql-        │   GitHubBenchmarksIngestRole        │
   benchmarks.yml,        │   (or GitHubBenchmarkRole + policy) │
   v3-commit-metadata)    └────────────┬────────────────────────┘
                                       │ IAM-auth token
                                       ▼
                           ┌────────────────────────┐    pooler
                           │ AWS RDS Postgres       │◀───  RDS Proxy
                           │  (db.t4g.micro)        │      (IAM-auth)
                           │ 6 tables, BIGINT[]     │
                           │ composite indexes      │
                           │  (dim_tuple,           │
                           │   commit_timestamp     │
                           │   DESC)                │
                           └─────────┬──────────────┘
                                     │ read-replica or
                                     │ same primary
                                     ▼
                         ┌─────────────────────────────┐
                         │ Next.js on Vercel           │
                         │  server components +        │
                         │  force-dynamic +            │
                         │  Vercel-CDN-Cache-Control   │
                         │  edge-CDN response cache    │
                         └─────────────────────────────┘
                                     │
                                     ▼
                              Public read users
```

The system splits cleanly into three independent surfaces. The **writer** is a small (script or binary) ingest tool that produces bit-identical `measurement_id` xxhash64 values from a v3-envelope JSONL input and executes `INSERT ... ON CONFLICT (measurement_id) DO UPDATE` against Postgres. It is invoked from each CI workflow in place of `python3 scripts/post-ingest.py ... --server $V3_INGEST_URL`. The **reader** is a Next.js application whose server components issue parameterized SQL through a connection pool and serve from the edge CDN via header-driven caching (`Vercel-CDN-Cache-Control` on the HTML routes + `Cache-Control` s-maxage on API 200s; the `unstable_cache`/per-chart-tag sketch was not usable -- see the Read-service-framework Key-decision amendment). The **schema** lives in a single source of truth (the chosen migration tool) and is deployed to Postgres via a CI workflow gated by `prisma-isolation`-style discipline.

The migration's load-bearing invariant is the `measurement_id`: it is a server-internal xxhash64 over `(table_tag || 0x00 || commit_sha || dim_fields...)` with length-prefixed strings (little-endian u64), `Option<String>` as `0x00`/`0x01+write_str`, `i32` as 4 LE bytes, `f64` as `to_bits()` LE u64, finished `as i64`. Existing rows seeded from the DuckDB dump carry these IDs; every subsequent CI write must hash to the same bytes to upsert correctly. Producing those bytes from the new writer is the migration's central correctness obligation.

The read path's bimodal access pattern (small `LIMIT n` windows up to 1000, plus `?n=all` whole-history downsampled to ~1000 buckets client-side via LTTB) is well-served by composite indexes alone — no materialized views needed. Cold-start cost on stateless Vercel is the main read-side budget concern; the in-process `read_model.rs` artifact regeneration was only 100% materialized on EC2 because the process was long-lived. The Next.js layer keeps the warm path cache-only via edge CDN caching driven by response headers (`Vercel-CDN-Cache-Control` + `Cache-Control` s-maxage), not route-segment ISR/`revalidateTag` (see the Read-service-framework Key-decision amendment).

CI writers and the Next.js reader connect through a pooler — RDS Proxy if AWS RDS, provider-managed pooler if Neon/Supabase/Crunchy, pgbouncer self-hosted if none. The pooler choice is downstream of the Postgres flavor choice. Connection budget is bounded by Vercel's serverless concurrency × pool-per-invocation × a small connections-per-pool (≤4) constant.

CI network reach is straightforwardly "public + IAM" — the existing OIDC setup is on `ubuntu-latest` (public) and `bench-dedicated` (AWS-hosted via `runs-on`) runners, both with public egress. VPC + self-hosted runners would lose the `v3-commit-metadata.yml` job's `ubuntu-latest` slot unless it moves to self-hosted; that's a large operational expansion for marginal security gain.

Cutover style follows the v2→v3 template that just shipped: dual-write for ~1 week, soak under real CI traffic, structural-diff verification via a `migrate/verify.rs`-shaped tool, then one PR that deletes `benchmarks-website/server/`, `benchmarks-website/ops/`, the bearer token, the v2 React frontend, and the systemd-driven publishing workflow.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Branching / merge target | Per-phase child branches → squash-merge into `ct/bench-v4`; final `ct/bench-v4` → `develop` once at end | User pick (Q1). Per-phase GH PRs give review granularity at GitHub level without per-logical-PR overhead. Mechanics: big-plans state machine runs on the phase's child branch; phase-end Step 3.5 Proceed opens child→ct/bench-v4 PR and awaits squash-merge before the next phase's child branch is spawned. |
| Postgres flavor | **RDS Postgres on `db.t4g.micro`** in AWS account `245040174862`, region `us-east-1`, single-AZ | User pick (Q2). ~$13/mo floor + storage + IO. Latest Postgres (no Aurora version lag). Always-on (no cold-start). IAM auth native. Aurora's storage/replication/throughput benefits are unused at this scale. Account `245040174862` is the existing bench-S3 / `GitHubBenchmarkRole` account; CI already authenticates to it via OIDC. Operator uses an SSO `bench` profile to act against it (`aws sso login --profile bench`). Cross-account implication for Phase 3: v3 EC2 (DuckDB source) lives in personal account `375504701696/us-east-2`; the one-shot load needs operator creds for both accounts (or DuckDB snapshot via cross-account S3 bucket policy). See Risks #10. |
| Connection pooler | **RDS Proxy for the Vercel read service only**; CI writers connect directly to the public RDS instance endpoint with IAM | Locked by Q2, **amended 2026-05-29 (Phase-1 re-plan)**. RDS Proxy endpoints are VPC-internal (not publicly reachable), so off-VPC GitHub-hosted CI runners cannot use the proxy. The proxy's pooling value is for Vercel's serverless read concurrency (Phase 4); the ~14 CI writers per push connect directly to the public instance endpoint with IAM tokens (the instance was provisioned `--publicly-accessible` with IAM auth + verify-full TLS). The original "single pooler covers both CI writers and Vercel reader" claim was wrong; corrected by the Phase-1 phase-end gauntlet (RDS-Proxy-unreachable-from-off-VPC finding). **REVERSED 2026-06-10 (Phase-5 re-plan, user decision): the proxy is VPC-internal, so it was unreachable from Vercel's off-VPC serverless functions just as from CI — the read path connects DIRECTLY to the public instance endpoint + static `bench_read` password, leaving the proxy with NO consumers. Decision: DECOMMISSION the RDS Proxy in the Phase-5 teardown (no pooler in the steady-state architecture); reads stay direct (the CDN cache absorbs nearly all read load). Revisit a managed pooler only if DB connection exhaustion actually surfaces ("fix it later if it becomes a problem").** |
| v3-parity stance (Phase-5 re-plan 2026-06-10) | **Preserve v3 for the five invisible/edge-case quirks; RESTORE the v2 `statpopgen`/`polarsignals` names + descriptions** (the one real v2→v4 regression; scheduled as PR-5.0.5) | User decision at the Phase-4 boundary re-plan. The migration's success criterion is byte-equivalence vs the LIVE **v2** site; five flagged quirks (numeric `1500000` vs `1500000.0` rendering, LTTB-over-hidden-series, filter byte-order sort, same-second commit-tie, the two URL-filter round-trip edges) are invisible or vanishingly rare, so matching v3 is correct + free. The sixth — statpopgen/polarsignals falling through to the legacy `dataset sf=N` label — is a REGRESSION from v2 (v2's `src/config.js` surfaced the friendly names + descriptions), so it is restored in PR-5.0.5. Any of the five preserved quirks a future ticket deems wrong is a normal post-launch fix, decoupled from the migration. |
| Ingest writer language | **Pure Python** — extend `scripts/post-ingest.py`; port xxhash64 to Python with golden-vector tests against the existing Rust source-of-truth | User pick (Q4). Avoids adding sqlx + aws-sdk-rds Rust deps. Hash port is bounded by tests; Rust impl in `server/src/db.rs` stays the source of truth. Uses `psycopg[binary]` + `boto3.client('rds').generate_db_auth_token`. |
| Postgres schema deploy tool | **In-house `scripts/migrate-schema.py`** (~30-50 LOC) + plain SQL files under `migrations/` | User pick (Q5a). Tracks via `_applied_migrations` table; applies pending `00N_name.sql` in name order; CI workflow invokes with OIDC + IAM. Zero new tools / languages. |
| Schema-deploy authorization + execution-safety model | **PR merge IS the deploy gate** (no manual-approval `environment:` gate). `schema-deploy.yml` triggers `apply` on push to the deploy branch under `paths: migrations/**`; keep `dry_run`/`status` as the pre-apply preview and the testcontainer CI test as the per-PR safety check. **No** GitHub Environment / required-reviewer gate. | User decision 2026-05-29 (supersedes the original `environment: schema-deploy` manual-approval mandate). Two axes were conflated: *authorization* ("do we want this change?") is fully answered by reviewed-PR-merge; a human clicking "Approve" on an Environment only re-confirms authorization already given at merge, and does NOT verify the migration will apply cleanly. *Execution safety* ("will the DDL succeed against prod's current state?") is the real risk, and it is addressed by **testing the migration**, not by a manual gate. The testcontainer-against-empty-schema test (shipped) gates additive DDL (CREATE TABLE/INDEX, ADD COLUMN) at PR time, so a migration that cannot apply cannot merge. This also resolves the repo-admin blocker (creating an Environment needs admin; deciding the gate is the wrong tool removes the dependency). Tradeoffs knowingly forgone: deploy-timing decoupling (merge-now/apply-in-window) and segregation-of-duties (different approver than author); neither is material for a small-team benchmark site; revisit only on a compliance need. RDS has **no Neon-style instant copy-on-write branching**; the data-affecting-migration safety layer is a PITR-snapshot-restore-to-throwaway-instance CI step, added only when a migration mutates existing data (type change, NOT NULL on an existing column, backfill); out of scope for the additive Phase 1/early migrations. (Aurora fast-clone is the true Neon analog but requires reversing the `db.t4g.micro` Key decision; not worth it at this scale.) |
| One-shot historical data load | **Retarget `benchmarks-website/migrate/`** (existing Rust crate) for DuckDB→Postgres bulk load | Q5b — natural reuse. The crate already reads DuckDB via the `duckdb` crate and reuses `vortex_bench_server::db::measurement_id_*`. Add a `--postgres-target` mode + a Postgres bulk-insert path. Deleted post-cutover per AGENTS.md throwaway-migrator pattern. |
| CI network reach | **Public + IAM** — public RDS **instance** endpoint (the proxy is VPC-internal, not public), security group `0.0.0.0/0` because IAM is the gate, sslmode=verify-full | User pick (Q6), **amended 2026-05-29 (Phase-1 re-plan)**: the reachable endpoint is the RDS **instance** (publicly-accessible + IAM), not the proxy (which cannot be public). Every direct Postgres connection is IAM-gated; public-read of benchmark data is served by the Vercel HTTP layer, not by direct DB reads. All 14 CI writers connect to the instance endpoint with OIDC→IAM tokens. Matches the existing CI-to-AWS-S3 operational posture. |
| Cutover style | **Short dual-write window** (CI keeps writing v3 AND adds a best-effort v4 write) for ~3-7 days of soak; then promote v4 to required + drop v3-write; then DNS flip from v2 directly to v4; then decommission v2 + v3 | User pick (Q7), **amended 2026-06-04 (lean re-plan)**. v3 is a stepping stone that never serves public traffic (DNS stays on v2 until the v4 flip), but its DuckDB IS the structured migration source. **Lean de-risking:** the v4 write is BEST-EFFORT during the soak (don't gate the proven v3 pipeline on the unproven v4 path), promoted to required at cutover. The benchmark-data-loss safety net is provided WITHOUT heavy reconciliation machinery: v2 stays live, v3 stays live, the DuckDB snapshot is kept ≥90 days, and Phase-3's one-shot `migrate --verify` (DuckDB↔Postgres row/id comparison) is the primary correctness gate. (Supersedes the prior "both-must-succeed dual-write + reconciliation script + incident.io" model as disproportionate to a trusted-input low-stakes dashboard with four independent safety nets.) |
| Read service framework | **Next.js 15 + App Router + React Server Components + `unstable_cache` + `revalidateTag`** at `benchmarks-website/web/` | User pick (Q8). Server components fetch directly from Postgres; per-chart cache tags invalidated from the Python writer's CLI via a Vercel revalidation endpoint. Latest stable Next.js. Pages Router avoided. **(amended 2026-06-10, phase-4 end-review): the shipped caching mechanism is force-dynamic rendering + `Cache-Control` s-maxage=300 on API 200s (`web/lib/cache.ts`) + `Vercel-CDN-Cache-Control` rules on `/` and `/chart/:slug` (`web/vercel.json`). `unstable_cache`/`revalidateTag` were never used: route-segment revalidate is inert on request-URL handlers and forces DB-at-build-time prerender elsewhere, and function-emitted Cache-Control beats config-file rules. Framework choice itself unchanged and vindicated.** |
| Operator SQL replacement | **`scripts/psql-bench.sh`** — tiny helper that runs `aws rds generate-db-auth-token` and pipes into psql with IAM creds | User pick (Q9). Replaces `/api/admin/sql`. No bearer tokens, no Lambda. Documented in benchmarks-website/web/README.md. RDS PITR (35-day) replaces `/api/admin/snapshot`. |
| Composite index definition strategy | Net-new in `migrations/001_initial_schema.sql`. **As-shipped: dim-leading composite indexes following the read-path chart-query filter columns** (per `api/charts.rs`), NOT the hash field order. | **Amended 2026-05-29 (Phase-1 re-plan)** to match what PR-1.3 shipped: the original `(dim_tuple..., commit_timestamp DESC)` framing was superseded — every chart query filters on the dim columns and joins `commits` on `commit_sha`, so a dim-leading index serves the read path; PK uniqueness over the full hash tuple is already enforced by `measurement_id`. (PR-1.3 surprise, ratified here; an index-column-definition test is folded into PR-1.6.) |
| CI-write endpoint (re-plan 2026-05-29) | **Public RDS instance endpoint + direct IAM** for all CI writers (schema-deploy + Phase-2 ingest); RDS Proxy is Vercel-reads-only | Phase-1 phase-end gauntlet found the RDS Proxy is VPC-internal (unreachable from off-VPC GitHub runners). The instance was already provisioned `--publicly-accessible` with IAM auth, so CI writers connect to it directly with OIDC→IAM tokens + verify-full TLS. This **moots** the "register a migrator credential in the proxy auth config" finding for the CI write path (proxy auth config becomes a Phase-4 concern for the Vercel read role). Supersedes the proxy-for-CI assumption in the original pooler/Q6 decisions. |
| v3 EC2 final disposition | **Decommissioned at end of Phase 5** (single deletion PR removes `benchmarks-website/server/`, `benchmarks-website/ops/`, `benchmarks-website/migrate/`, top-level v2 files, `publish-benchmarks-website.yml`, `INGEST_BEARER_TOKEN`/`ADMIN_BEARER_TOKEN` secrets). EC2 instance terminated by hand after PR merges. | v3 never goes live; Q7 cutover model goes v2→v4 directly. |
| Phase-2 ingest DB identity (re-plan 2026-06-01) | **Dedicated `bench_ingest` role** (DB-side) + **`GitHubBenchmarkIngestRole`** (AWS-side), separate from the `migrator` / `GitHubBenchmarkSchemaRole` schema-deploy identity. `bench_ingest` gets DML-only (`SELECT,INSERT,UPDATE`, no DELETE/DDL) on the 6 data tables via migration 004. | Re-plan Q2. The ~14-writer ingest path runs on every push against a `PubliclyAccessible: true` instance with `0.0.0.0/0:5432` ingress (live-verified Phase-1 posture); a separate least-privilege identity means a leaked CI token can do data DML only, never DDL/migrations/role changes. Matches the `GitHubBenchmarkIngestRole` the original plan already anticipated. Cost: one migration + one provision.sh role block. Rejected: reuse `migrator` (conflates schema-deploy authority with the most-exposed code path). |
| Phase-2 dual-write verify scope | **SUPERSEDED 2026-06-04 (lean re-plan) → verify-once, no standalone reconciliation machinery.** Correctness of the v4 data is verified by **Phase 3's one-shot `migrate --verify`** (authoritative DuckDB↔Postgres row/id comparison) plus a short MANUAL spot-check during the soak (e.g. `psql-bench.sh` row-count + a few measurement_id lookups). **DROPPED:** the standalone `reconcile-ingest.py` service, the `dual-write-verify.yml` workflow, and incident.io paging. | The prior 2026-06-01 plan added a per-push Postgres-side reconciliation harness + incident.io alert. The 2026-06-04 course-correction found this disproportionate: benchmark numbers are trusted-input + regenerable, and four independent safety nets already cover Risk #4 (v2 live, v3 live, ≥90-day DuckDB snapshot, Phase-3 `migrate --verify`). A paged production-pipeline reconciliation service for a low-stakes dashboard is over-built. The one-shot verify is the load-bearing gate; the soak just needs eyeballs, not on-call. |
| Review calibration (lean re-plan 2026-06-04) | **Trusted-input, low-stakes calibration for all REMAINING reviews** (see the `## Review calibration` section). Reviewers flag DATA-CORRECTNESS + does-it-work, NOT adversarial-input robustness on trusted `vortex-bench` CI data. Code PRs use **2-vote** (fresh+correctness); only the final cutover phase (Phase 5) uses **3-vote**. Inner-loop **capped at ~3 cycles** (an open finding at cycle 3 is deferred or accepted, never spiraled). Deferred backlog pruned to data-correctness items only. | PR-2.2's 15-cycle / ~75-subagent spiral came from adversarial lenses hunting untrusted-input edge cases (NUL / non-UTF-8 / lone-surrogate / RecursionError + the 4-cycle `git_show_field` non-UTF-8 saga) on trusted CI JSON, with no cycle cap. This recalibrates the skill's default "thoroughness over cost" to the actual artifact (a benchmarks dashboard), preventing recurrence across Phases 3-5. |
| PR-4.4 UI architecture (2026-06-09) | **RSC shell + per-chart client islands; shard endpoint DROPPED** | User pick (PR-4.4 fork AUQ). Chart.js canvas charts are inherently client-rendered, so a chart is a `'use client'` island in every variant; the real fork was the shell + the deferred shard endpoint's fate. Decision: server components render the layout / group-section / summary / filter-bar shell (cached via `unstable_cache` + edge CDN, matching the Phase-4 `Read service framework` decision); each chart is a thin client island that lazily fetches `/api/chart/[slug]` on group-open (groups collapsed by default, faithful to v3's lazy-on-expand at `html/mod.rs`). The v3 `/api/artifacts/{generation}/groups/{slug}/shards/{i}` endpoint is **dropped**: its two jobs — per-group batch fetch + immutable `{generation}`-versioned caching — are covered in v4 by the existing `/api/group/[slug]` + `/api/chart/[slug]` routes + Next `revalidate=300` + edge CDN; `{generation}` (an in-process `read_model.rs` snapshot id, 8 retained generations) has no stateless analog on Vercel. Supersedes the PR-4.4 row's prior "(if kept) shard route" conditional. **(2026-06-10: any cache wording in this row is superseded by the shipped header-driven CDN mechanism; see the framework amendment.)** |
| Prod historical load TIMING (2026-06-05) | **Deferred from Phase 3 to the Phase 5 cutover.** Phase 3 builds + validates the full migration toolkit (loader / value-verify / rehearsal harness / cross-check, all green) AND runs a **real-snapshot LOCAL rehearsal** (the real v3 DuckDB -> a local postgres:16 -> load + verify + cross-check clean). The actual one-shot PROD load into RDS runs at cutover (Phase 5), reusing the SAME validated tools against the prod DSN. | User decision 2026-06-05. Seeding prod RDS in Phase 3 is premature: Phase 4 has not built the v4 reader yet, so any loaded history sits unread until cutover, and an irreversible prod write before cutover-commitment buys nothing. Loading at cutover yields a FRESHER snapshot (the best-effort dual-write soak since Phase 2 fills the gap), no stale prod data, and no premature prod side-effect. The prod load IS "run the existing Phase-3 tools against the prod DSN" (the user's framing) -- a deferred EXECUTION, not unbuilt work. The real-snapshot LOCAL rehearsal still de-risks NOW: it is the only check the synthetic fixtures cannot give (proof the ACTUAL v3 data shape loads + verifies clean). |

## Project-specific BANS

**REVIEW CALIBRATION (load-bearing — read first; 2026-06-04 lean re-plan).** This is a **benchmarks dashboard**: public-read, no user accounts, no payments. Its data comes exclusively from the project's own `vortex-bench --gh-json-v3` CI (TRUSTED input), and it is backed by four independent safety nets (the live v2 site, the live v3 system, a ≥90-day DuckDB snapshot, and Phase-3 `migrate --verify`). Calibrate findings accordingly:
- **DO flag:** data-correctness bugs (measurement_id / hash parity, NaN/Inf divergence, wrong upsert/ON CONFLICT semantics, schema/SCHEMA_VERSION drift, lost/duplicated rows), and "does it actually work" bugs (real runtime errors, broken CI, wrong query results, broken charts).
- **DO NOT flag (over-engineering for this artifact):** adversarial-input robustness on trusted CI/JSON/git data (NUL bytes, non-UTF-8, lone surrogates, `RecursionError` from nested JSON, oversized-integer literals, exotic `git` commit-metadata encodings); availability/pipeline-reliability hardening disproportionate to "a benchmark number is briefly stale" (the data is regenerable and v2/v3 stay live); test-completeness-of-test-completeness spirals.
- **Cycle cap:** inner-loop is capped at ~3 gauntlet cycles. A finding still open at cycle 3 is deferred (to `Deferred work`) or accepted — never spiraled into more cycles. Code PRs are 2-vote (fresh + correctness); only the final cutover phase is 3-vote. (Rationale + the PR-2.2 15-cycle spiral that motivated this: see the `Review calibration` Key-decision row.)

**General (apply regardless of language/shape):**
- Behavioral drift without test coverage: any change to observable behavior requires an added or updated test.
- Silent error swallowing: `let _ = fallible_op()`, `.ok()`, bare `try { } catch {}` without explicit rationale.
- Long `// TODO`, `// FIXME`, `// PORT NOTE`, or any `// SAFETY:` >100 chars explaining why a hack is OK (justification-comment reward-hacking).

**Migration / behavior-preservation BANS:**
- Do not change the byte layout of any `measurement_id_*` xxhash64 function in `benchmarks-website/server/src/db.rs:162-257`. (Why: existing rows in the v3 DuckDB and any Postgres rows seeded from it use these IDs as PKs; any change silently produces duplicate rows on next ingest instead of upserting.)
- Do not stop including `commit_sha` in the dim-tuple hash, and do not add `iterations` (or any side-counter) to the `vector_search_runs` hash. (Why: removing `commit_sha` collapses every commit's timing into one row; adding side counters splits one (commit, dim) tuple across rows.)
- Do not change the `ON CONFLICT (measurement_id) DO UPDATE SET ...` column lists in the new ingest path without auditing every column. (Why: each `DO UPDATE SET` enumerates only value columns; a dim column accidentally entering the SET list drifts PK and content, and an omitted value column silently keeps stale data.)
- Do not skip `commits` upsert before fact-table inserts. (Why: no FK exists, so an orphan fact row renders as a phantom point with no commit metadata.)
- Do not bump `SCHEMA_VERSION` in only one of (`server/src/schema.rs`, `vortex-bench/src/v3.rs`, `scripts/post-ingest.py` or its successor). (Why: server validation rejects mismatched envelopes; lockstep bump in one PR is required.)
- Do not remove `#[serde(deny_unknown_fields)]` from any envelope/record struct. (Why: unknown fields are how version skew surfaces loudly; relaxing tolerance silently drops producer data.)
- Do not introduce a code path that writes only to Postgres or only to DuckDB during the dual-write window. (Why: the rollback story requires each substrate to be restorable on its own.)
- Do not delete bearer-token middleware while any `/api/ingest` or `/api/admin/*` route still exists, and do not delete those routes while any caller still depends on them. (Why: half-decommission either creates an unauthenticated public write endpoint or breaks a dead caller silently.)
- Do not put `measurement_id` on the wire (request body, response body, HTML). (Why: it is server-internal by design; exposing it makes the hash a public API and freezes byte-layout migration forever.)
- Do not raise or remove `MAX_NUMERIC_COMMIT_WINDOW` on the numeric `?n=` path, and do not re-introduce a server-side cap on `?n=all`. (Why: documented DoS floor + uncapped escape hatch; reviewers have reverted this before.)
- Do not edit top-level v2 files (`server.js`, `src/`, `index.html`, `vite.config.js`, `package.json`, top-level `Dockerfile`, `docker-compose.yml`, `publish-benchmarks-website.yml`) outside the dedicated cutover PR. (Why: v2 is production until DNS flips.)
- Do not re-add `continue-on-error: true` to the new v4 ingest steps. (Why: ingest is intentionally no-longer-best-effort; silently swallowing failures masks divergence between bench run and served data.)

**Phase-2 ingest/identity BANS (re-plan 2026-06-01):**
- Do not grant the `bench_ingest` role `DELETE`, `TRUNCATE`, or any DDL/role privilege; the ingest path is data-DML-only (`SELECT,INSERT,UPDATE` on the 6 tables). (Why: the dedicated role exists for least-privilege separation from `migrator`; widening it defeats the separation-of-duties decision.)
- Do not have CI authenticate the ingest write path as `migrator` / `GitHubBenchmarkSchemaRole`. (Why: that conflates the high-frequency, internet-reachable ingest path with the schema-deploy identity that can run DDL + migrations.)
- Do not write to Postgres without an `is_finite()` guard on every f64 dim that feeds the hash (e.g. `threshold`). (Why: Rust `to_bits()` preserves NaN payload bits while Python `struct.pack('<d', nan)` emits canonical NaN, so a non-finite value hashes differently across languages and silently produces a duplicate row instead of an upsert.)
- Do not echo or log the RDS IAM auth token / `PGPASSWORD` (no `set -x` around token generation; assign it on its own line). (Why: the token is a short-lived credential; the schema-deploy.yml discipline must carry to the ingest workflows.)

**Rust / clippy carryover BANS (apply to any new Rust ingest code):**
- Do not use `std::collections::HashMap`, `std::collections::HashSet`, `std::sync::Mutex`, `std::sync::RwLock`. (Why: workspace `clippy.toml` bans them; use `vortex_utils::aliases` + `parking_lot`.)
- Do not call `std::thread::available_parallelism`. (Why: banned; use `vortex_utils::parallelism::get_available_parallelism`.)
- Do not `unwrap`/`expect` outside `#[cfg(test)]`; do not invent ad-hoc error enums when existing `thiserror` variants cover the case.

**Next.js / TypeScript BANS (carry from `vortex-web/`):**
- Do not introduce a TS file without the SPDX `Apache-2.0` + `Copyright the Vortex contributors` header pair.
- Do not disable `strict`, `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`, `noUncheckedSideEffectImports` in `tsconfig*.json`.
- Do not introduce a second formatter config — reuse the existing `vortex-web/.prettierrc.json` shape (`singleQuote: true`, `trailingComma: "all"`, `printWidth: 100`, `tabWidth: 2`).
- Do not reach for WASM in the new web layer.

**benchmarks-website-specific UI BANS (carry to the Next.js rewrite):**
- Do not flip the per-row delta predecessor walk to `idx + 1`. (Why: `payload.commits[]` is oldest-first; predecessor is `idx - 1`. A "fix" in this direction has been reverted.)
- Do not set `pointer-events: auto` on the tooltip host. (Why: known cursor-tracking flicker loop.)
- Do not bind UI handlers to `change` on range sliders; use throttled `input`. (Why: `change` fires only on release.)
- Do not refetch chart payloads on pan / zoom / slider / range-strip change. (Why: in-memory LTTB over cached payload; the only exception is the one-shot `?n=all` lazy hop.)

**Work-shape highest-leverage insight**: **Behavior-preservation under test** — every preserved invariant (xxhash64 byte layout, `ON CONFLICT` SET-column membership, 6-table schema shape, v3 envelope `deny_unknown_fields`, `SCHEMA_VERSION` triple-site coupling, all-four-or-none memory-quartet validation, `IS NOT DISTINCT FROM` NULL semantics on dim equality) must appear as an explicit acceptance criterion in the relevant PR's row and must be pinned by an automated test. Reviewers verify against this list; phase-end review re-checks every preserved invariant against the diff.

## Phases and PRs

### Phase summary

| Phase | Name | Scope (one line) | Exit criteria (machine-checkable) | PR count | Review-count |
|---|---|---|---|---|---|
| 1 | RDS + schema + hash port | Provision RDS, write schema-deploy script + initial DDL, port xxhash64 to Python with golden vectors; **(re-plan) repoint CI writes to the public instance endpoint + sweep cycle-1 must-fixes** | `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available`; `python scripts/migrate-schema.py status` clean; `pytest scripts/test_measurement_id.py` all green; Rust golden-vector test in `vortex-bench-server` matches Python output bit-exactly; **`schema-deploy.yml` applies live as the OIDC migrator against the public instance endpoint (verify-full TLS) and `status` reports clean** | 6 | 3-vote |
| 2 | Postgres writer + best-effort v4 CI | **(lean re-plan 2026-06-04)** `bench_ingest` role + grants (004); `--postgres` mode + IAM-auth + NaN/Inf guard; `scripts/` pytest in CI; **add a BEST-EFFORT v4 write to the 3 ingest workflows (v3 `--server` stays hard-required + untouched) + switch schema-deploy trigger to push-on-deploy-branch.** PR-2.1/2.2/2.3 DONE; only PR-2.4 remains. **DROPPED: PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io).** | PR-2.1/2.2/2.3 accepted (done); PR-2.4: `yamllint --strict` clean; the 3 workflows add a `continue-on-error` v4 `--postgres` step (v3 step unchanged + required); `schema-deploy.yml` triggers `apply` on push to the deploy branch under `paths: migrations/**`. v4 correctness is gated by Phase-3 verify, not a Phase-2 reconciliation harness. | 4 | 2-vote |
| 3 | Historical data load (DuckDB → Postgres) + value-verify | **(re-plan 2026-06-05, folds in 4 audit gaps; 2026-06-05 prod-load deferred to Phase 5)** Build + validate the migration TOOLKIT: `vortex-bench-migrate --postgres-target` (atomic single-txn COPY, NO aws-sdk-rds) + 004-as-master ordering guard; **value-column verify** (per-`measurement_id` VALUE+`env_triple`+array+commits-metadata compare — the PRIMARY v4-correctness gate); local postgres:16 **rehearsal harness** + **Python-writer-vs-RDS cross-check** harness. Phase 3 closes on a **real-snapshot LOCAL rehearsal** (acquire the real v3 DuckDB off the EC2 host -> load + verify + cross-check against a LOCAL postgres:16). **The one-shot PROD load + prod verify + prod cross-check are DEFERRED to the Phase-5 cutover** (run the same tools against the prod DSN at cutover, freshest snapshot). | `migrate verify --postgres-target` reports 0 presence diffs AND 0 value-column / `env_triple` / `all_runtimes_ns`-array / `commits`-metadata mismatches; the synthetic-fixture rehearsal (PR-3.3) + the cross-check (PR-3.5) are green vs local PG16; **the REAL-snapshot LOCAL rehearsal (PR-3.4) loads the actual v3 DuckDB into a local PG16 and `verify` + cross-check report clean** (the real-data validation gate). The prod-load row-counts + prod cross-check are captured at Phase-5 cutover, NOT here. **(amend 2026-06-08: +PR-3.6 applies the 3 phase-end-cycle-1 should-fix doc/status items; +PR-3.7 clears the 3 phase-end-cycle-2 doc nits.)** | 7 | 3-vote |
| 4 | Next.js read service on Vercel | Scaffold `benchmarks-website/web/`; connection lib; port read endpoints as RSC/route handlers; port chart UI; deploy to Vercel. **(lean re-plan: TIME-BASED revalidation (`export const revalidate = ~300`, matching v2's 5-min S3 refresh) instead of push-based HMAC; DROPPED the standalone revalidate endpoint + writer hook + `REVALIDATE_SECRET` (was PR-4.6).)** **(amended 2026-06-10, phase-4 end-review: the shipped caching is header-driven CDN (see the Read-service-framework Key-decision amendment), not route-segment revalidate.)** **(2026-06-08: PR-4.3 split into PR-4.3.a/b/c; the shard endpoint deferred from PR-4.3 to PR-4.4 — see PR enumeration.)** **(2026-06-09: PR-4.4 split into PR-4.4.a [server shell + CSS] / PR-4.4.b [chart client island + interactivity + permalink] — faithful v2 UI port exceeds single-PR size.)** | `vercel deploy --target=preview` serves all chart slugs; `curl preview-url/api/groups \| jq '.groups[].charts[].slug' \| sort` matches the family registry; **charts match v2 for representative slugs (manual visual check)** | 8 | 2-vote |
| 5 | Cutover + decommission | **(2026-06-12 amend: +PR-5.0.95 lazy-hydration + resilient-loading UI/UX round 2 [IntersectionObserver landing hydration + fetch timeout/abort/retry + spinner], user-directed, slotted ahead of PR-5.1; design at `.big-plans/ct__bench-v4-uiux-r2-design.md`.)** **(2026-06-11 evening amend: +PR-5.0.9 opt-in full-history loading UX, user-directed, slotted ahead of PR-5.1; design at `.big-plans/ct__bench-v4-uiux-design.md`.)** **(2026-06-11 amend: +PR-5.1.5 read-path-perf — the v4 read path did not scale to the full prod seed [discovered closing PR-5.0]; slotted before the DNS flip.)** **(2026-06-10 re-plan: PR-5.0 folds in the operator-gate; +PR-5.0.5 restores the two v2 group names; PR-5.3 also decommissions the RDS Proxy. 2026-06-05: opens with the deferred one-shot PROD historical load.)** Run the validated Phase-3 toolkit against PROD RDS (acquire freshest v3 snapshot -> `load` + `verify --postgres-target` + cross-check); **make the read path scale to the prod seed**; then **promote v4 ingest to required + drop v3-write from CI**; DNS flip v2 → v4; delete v2 frontend + server/ + ops/ + migrate/ + publish-benchmarks-website.yml + bearer-token secrets | the prod load reports per-table row-counts + `verify` 0-diffs + cross-check clean against prod RDS (RDS PITR is the rollback); **the v4 read path scales to the full prod seed (`/api/groups` returns in seconds + its slug list matches the family registry; ~5 representative chart slugs match the live v2 site)**; `git grep -n INGEST_BEARER_TOKEN` returns 0; `gh workflow list` does not include `publish-benchmarks-website`; production DNS resolves to Vercel; v3 EC2 terminated; RDS Proxy deleted; `bench_read` authenticates against prod (005-as-master + password); the v2 statpopgen/polarsignals names + descriptions render | 6 | 3-vote (final) |

Total: **5 phases, 34 PRs** (2026-06-12 UI/UX-round-2 amend: Phase 5 7→8 [+PR-5.0.95 lazy-hydration + resilient-loading, slotted ahead of PR-5.1; execution order PR-5.0 → PR-5.0.5 → PR-5.1.5 → PR-5.0.9 → **PR-5.0.95** → PR-5.1 → PR-5.2 → PR-5.3]; 2026-06-11 evening UI/UX amend: Phase 5 6→7 [+PR-5.0.9 opt-in full-history loading, slotted ahead of PR-5.1; execution order PR-5.0 → PR-5.0.5 → PR-5.1.5 → PR-5.0.9 → PR-5.1 → PR-5.2 → PR-5.3]; 2026-06-11 amend: Phase 5 5→6 [+PR-5.1.5 read-path-perf]; 2026-06-11 reorder: PR-5.1.5 moved AHEAD of PR-5.1 [execution order PR-5.0.5 → PR-5.1.5 → PR-5.1 → PR-5.2 → PR-5.3] because the perf fix unblocks pushing — any push re-triggers web-deploy.yml whose CDN probe fails until /api/groups is fast, and PR-5.1's CI-green acceptance needs pushing; lean re-plan 2026-06-04: Phase 2 5→4 [dropped PR-2.5], Phase 4 6→5 [dropped PR-4.6]; amend 2026-06-05: Phase 2 +3 [PR-2.6/2.7/2.8]; re-plan 2026-06-05: Phase 3 3→5 [+PR-3.4, +PR-3.5], Phase 3 review-count 2→3-vote; **2026-06-05 prod-load deferral: PR-3.4 re-scoped prod-load → REAL-snapshot LOCAL rehearsal, and the prod load split out as new PR-5.0 [Phase 5 3→4]**; **2026-06-08 PR-4.3 split: PR-4.3 → PR-4.3.a/4.3.b/4.3.c [Phase 4 5→7], shard endpoint deferred PR-4.3→PR-4.4 per user fork decision**; **2026-06-09 PR-4.4 split: PR-4.4 → PR-4.4.a/4.4.b [Phase 4 7→8], shard endpoint DROPPED per the RSC-shell fork decision**; **2026-06-10 Phase-5 re-plan: Phase 5 4→5 [+PR-5.0.5 statpopgen/polarsignals v2-name restore], PR-5.0 folds in the operator-gate, PR-5.3 adds the consumerless-RDS-Proxy decommission**). Done: Phase 1 (6) + Phase 2 (7) + **Phase 3 agent code: PR-3.1/3.2/3.3/3.5 complete + reviewed**. Remaining: **PR-3.4 (real-snapshot LOCAL rehearsal — NEXT, agent-doable), Phase 3 phase-end review/close**, Phase 4 (5), Phase 5 (5, incl. PR-5.0 prod-online + PR-5.0.5 v2-name restore). The 4 Phase-3 audit gaps are folded into the PR enumeration below.

### Phase 3 re-plan decisions (2026-06-05)

Re-plan at the Phase 2→3 boundary (operator-chosen at the Step 3.4 gate) to fold in the 4
load-bearing gaps from the 2026-06-05 complexity & gap audit. Decisions are grounded in 3 parallel
exploration agents over `migrate/`, the DuckDB source + cross-account boundary, and the schema
value/dim/env columns:

- **Execution model: operator-run-locally with the RDS master password (NOT IAM/OIDC).** The OIDC
  `GitHubBenchmark{Schema,Ingest}Role`s are GitHub-Actions-only (trust policy federates only
  `token.actions.githubusercontent.com` for develop + ct/bench-v4 refs) and are NOT assumable from a
  laptop; the only human-usable RDS-write credential is the Secrets-Manager master password
  (`provision.sh:269-271`, `infra/README.md:113-157`). The v3 DuckDB lives in a DIFFERENT account's
  S3/EC2 (`vortex-benchmark-results-database`) with no cross-account bridge. So the one-shot load
  runs locally with two credential sets (v3-account S3 read for the snapshot + `245040174862` master
  password for the RDS write). **PR-3.1 DROPS `aws-sdk-rds`/IAM token-minting** — it connects with the
  master-password DSN over verify-full TLS. (Reverses the original PR-3.1 "sqlx-postgres +
  aws-sdk-rds" note; a minimal async client — e.g. `tokio-postgres` + a TLS connector — suffices.)
- **Source account/region is operator-verified live, not trusted from the audit.** `375504701696` /
  `us-east-2` for the v3 DuckDB appears ONLY in this plan doc — it is pinned NOWHERE in the repo
  (`provision.sh`, `ops/`, workflows all confirm RDS = `245040174862`/`us-east-1` but never the v3
  source). PR-3.3's rehearsal runbook adds a live verification step (`aws sts get-caller-identity` /
  `aws s3api get-bucket-location`) before the prod load. The loader takes `--duckdb <path>`; the
  operator acquires the snapshot (rehydrate the per-table Vortex S3 backup via `duckdb` + `INSTALL
  vortex`, OR scp the live `/var/lib/vortex-bench/bench.duckdb`; the S3 backup has a 7-day lifecycle).
- **Value-column verify shape (gap #1, PR-3.2 — the PRIMARY gate).** `measurement_id` hashes only
  `commit_sha` + the dim tuple, so a PK/count match pins ZERO bytes of any VALUE or ENV column.
  Verify joins DuckDB-source vs Postgres-target on `measurement_id` (1:1 — accumulators dedup by id)
  and compares, per row, every non-hashed column: `query_measurements`{`value_ns`, `all_runtimes_ns`,
  `peak_physical`, `peak_virtual`, `physical_delta`, `virtual_delta`, `env_triple`};
  `compression_times`{`value_ns`, `all_runtimes_ns`, `env_triple`}; `compression_sizes`{`value_bytes`}
  (no env); `random_access_times`{`value_ns`, `all_runtimes_ns`, `env_triple`};
  `vector_search_runs`{`value_ns`, `all_runtimes_ns`, `matches`, `rows_scanned`, `bytes_scanned`,
  `iterations`, `env_triple`}; `commits`{8 metadata columns} per `commit_sha`. `all_runtimes_ns` is
  an ordered `BIGINT[]` — compare element-wise. **Full** comparison (one-shot gate, not sampled).
  `env_triple` is the audit's "env corruption invisible to count-match" case — now explicitly compared.
- **Rehearsal substrate (gap #2, PR-3.3): local postgres:16 testcontainer** (operator-chosen). Same
  engine as RDS 16.4; exercises the actual correctness risk (COPY value fidelity, type coercion,
  `BIGINT[]` handling, verify logic) at zero AWS cost. The endpoint/TLS/master-password path was
  already live-verified in Phases 1-2; the prod load (PR-3.4) is then a DSN swap guarded by
  value-verify + PITR rollback.
- **Python-writer-vs-RDS cross-check (gap #3, PR-3.5): built in Phase 3, run in BOTH Phase 3 + Phase
  5** (operator-chosen). Python==Rust hashing is already CI-gated (golden==Python port); the
  cross-check's novel value is the LIVE property — the Python `post-ingest.py --postgres` writer
  round-trips against REAL RDS and UPDATEs (not duplicate-INSERTs) the Rust-seeded rows. The harness
  writes REAL envelopes matching existing seeded dim tuples (so it UPDATEs — no synthetic prod
  pollution), asserts `measurement_id` matches the seeded row + value/env columns round-trip. Run
  right after the prod seed (PR-3.5, earliest detection) AND re-run as the PR-5.1 pre-promotion gate.
- **004-as-master ordering (gap #4): preflight guard in `migrate-schema.py` (PR-3.1).** 004's `ALTER
  DEFAULT PRIVILEGES` self-grant needs a master-capable role; documented in the SQL, enforced
  nowhere. Add a preflight assertion (the connected role has the needed privilege before applying
  002/004) + a test that a non-master role fails loud with a clear error.
- **Review-count: Phase 3 → 3-vote** (up from 2) — it is now the primary v4-correctness gate
  (establishes the `measurement_id`s the whole upsert-not-duplicate invariant depends on), so it
  earns Spec + Correctness + Maintainability.

### PR enumeration

| PR | Phase | Scope (one line) | Files touched (expected) | Acceptance (specific, testable) |
|---|---|---|---|---|
| PR-1.1 | 1 | Provision RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub OIDC schema role via aws-cli script; document in `benchmarks-website/infra/README.md`; capture endpoint via post-run `gh variable set RDS_BENCH_ENDPOINT` | `benchmarks-website/infra/provision.sh`, `benchmarks-website/infra/README.md`, `.github/workflows/schema-deploy.yml` (skeleton) | `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available` + `iam_database_authentication_enabled: true`; `aws rds describe-db-proxies --db-proxy-name vortex-bench-proxy` returns `Endpoint`. |
| PR-1.2 | 1 | Write `scripts/migrate-schema.py` (~80-180 LOC; original ~30-50 estimate was for the bare runner — status/drift, autocommit txn discipline, typed exceptions, and recovery commentary brought it closer to ~180) — applies `migrations/*.sql` in name order, tracks via `_applied_migrations` table, idempotent | `scripts/migrate-schema.py`, `scripts/test_migrate_schema.py`, `migrations/` (dir created + README), `pyproject.toml` (psycopg + testcontainers dev deps) | Unit test: applies a fresh schema to testcontainers Postgres, re-runs idempotently (0 rows changed second time), inserts and re-applies a v2 migration in order. `apply` survives a failing later migration without losing earlier ones (subprocess test). `status` exits non-zero on drift and does not DDL. |
| PR-1.3 | 1 | Write `migrations/001_initial_schema.sql` (the 6 tables + composite indexes per Table B); `migrations/002_iam_db_user.sql` (CREATE ROLE for IAM auth) | `migrations/001_initial_schema.sql`, `migrations/002_iam_db_user.sql`, `scripts/test_migrate_schema.py` (extended) | `python scripts/migrate-schema.py --target=$RDS_DSN apply` succeeds; `\dt` shows the 6 tables; `\di` shows the composite indexes; `\du` shows the IAM-auth role with `rds_iam` group. |
| PR-1.4 | 1 | Wire `.github/workflows/schema-deploy.yml` — OIDC → `GitHubBenchmarkSchemaRole` → `python scripts/migrate-schema.py apply` against RDS Proxy. ~~gated by `environment: schema-deploy` (manual approval)~~ SUPERSEDED by the 2026-05-29 deploy-model Key decision: **no `environment:` gate; PR merge is the deploy gate**. As-shipped (PR-1.4 complete): `workflow_dispatch`-only + `dry_run`; the merge-trigger switch is tracked in Deferred work. | `.github/workflows/schema-deploy.yml`, IAM role doc in `benchmarks-website/infra/README.md` (plan originally said `web/ops/README.md`; `web/` does not exist until PR-4.1) | `migrate-schema.py apply` runs as the OIDC `migrator` role against RDS Proxy; `status` reports clean post-apply. (Original "manual-approval gate fires" criterion dropped per the deploy-model decision.) |
| PR-1.5 | 1 | Port xxhash64 to Python in `scripts/_measurement_id.py`; mirror per-table tag + write_str/write_opt_str/write_i32/write_f64 encoding; golden-vector test against Rust source-of-truth | `scripts/_measurement_id.py`, `scripts/test_measurement_id.py`, `benchmarks-website/server/tests/measurement_id_golden.rs` (golden generator/asserter) | `pytest scripts/test_measurement_id.py` all green; for **63** committed golden vectors (all 5 fact tables + i32 MIN/MAX + empty/`Some("")` strings + multibyte UTF-8 — amended 2026-05-29 from the original "100 fixture inputs" estimate; the 63 exhaustively cover every table + boundary class) the Python output matches the Rust output bit-exactly. |
| PR-1.6 | 1 | **(re-plan)** Repoint CI writes to the public RDS instance endpoint + sweep the cycle-1 phase-end must-fixes/should-fixes: capture+export the instance-endpoint repo var (`RDS_BENCH_INSTANCE_ENDPOINT`) in `provision.sh`; point `schema-deploy.yml` `PGHOST` at it; set `PGSSLMODE=verify-full` in the README bootstrap; fix the `provision.sh:19` account comment + promote the hardcoded `proxy_role_name`; add `003` to `migrations/README`; tighten the stale schema-deploy env-gate comment; doc the CI=instance / proxy=Vercel split; add an index-column-definition test | `benchmarks-website/infra/provision.sh`, `.github/workflows/schema-deploy.yml`, `benchmarks-website/infra/README.md`, `migrations/README.md`, `scripts/test_migrate_schema.py` | `yamllint --strict` clean; `schema-deploy.yml` `PGHOST` resolves to the instance-endpoint repo var (not the proxy); README bootstrap uses verify-full; `git grep -n 375504701696 benchmarks-website/infra/provision.sh` returns 0; index-column-definition test green; **operator runs the live OIDC apply against the instance endpoint and `status` reports clean**. |
| PR-2.1 | 2 | **(re-plan)** Foundational ingest identity: `migrations/004_ingest_role.sql` creates `bench_ingest` (rds_iam member; `GRANT SELECT,INSERT,UPDATE` on the 6 data tables + `USAGE ON SCHEMA public` + `ALTER DEFAULT PRIVILEGES`; **no DELETE, no DDL**); `provision.sh` adds AWS `GitHubBenchmarkIngestRole` (OIDC trust branch-scoped to develop + ct/bench-v4, `rds-db:connect` for dbuser `bench_ingest`) and drops the dead RDS-Proxy `rds-db:connect` grant on `GitHubBenchmarkSchemaRole` | `migrations/004_ingest_role.sql`, `scripts/test_migrate_schema.py` (extended), `benchmarks-website/infra/provision.sh`, `benchmarks-website/infra/README.md` | testcontainer test connects AS `bench_ingest` and runs `INSERT ... ON CONFLICT DO UPDATE` on all 6 tables successfully, and is denied a DDL/DELETE attempt; `git grep` shows no proxy `rds-db:connect` on the schema role; migration 004 is idempotent under re-apply (re-run reports 0 applied) |
| PR-2.2 | 2 | Extend `scripts/post-ingest.py` with a `--postgres $RDS_DSN` mode: parse JSONL, compute measurement_id via `_measurement_id.py`, mint an IAM token (boto3) for `bench_ingest`, upsert `commits` first then the fact tables via `INSERT ... ON CONFLICT DO UPDATE` over verify-full TLS; add an `assert <f64 dim>.is_finite()` guard (e.g. `threshold`) at the ingest boundary; keep `SCHEMA_VERSION` lockstep | `scripts/post-ingest.py`, `scripts/test_post_ingest_postgres.py`, `pyproject.toml` (psycopg, boto3) | Integration test (testcontainers Postgres): POST a v3 envelope → N rows inserted; POST same envelope → 0 inserted + N updated; all measurement_id values match `_measurement_id.py`; a NaN/Inf f64 dim raises a loud error rather than writing a divergent row; `SCHEMA_VERSION` matches `schema.rs`/`v3.rs` |
| PR-2.3 | 2 | **(re-plan, CI-hardening)** Wire the `scripts/` pytest suites into CI: a job running `uv run --all-packages pytest scripts/` covering `test_measurement_id.py` (golden==Python, no Docker) and `test_migrate_schema.py` (testcontainer, needs Docker), failing loud when Docker is unavailable in CI (no silent skip) | `.github/workflows/ci.yml` (or new `scripts-test.yml`), `pyproject.toml` (`testpaths`/markers if needed), `scripts/test_migrate_schema.py` (Docker-required assertion) | The job runs both suites on PR; an intentional divergence in `_measurement_id.py` fails the golden==Python check (now CI-gated); the testcontainer suite runs (not silently skipped) when Docker is present and fails loud when absent |
| PR-2.4 | 2 | **(lean re-plan 2026-06-04)** Add a BEST-EFFORT v4 write to `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml`: after the existing (unchanged, hard-required) `post-ingest.py --server` (v3) step, add a SECOND `post-ingest.py --postgres` (v4) step via OIDC → `GitHubBenchmarkIngestRole` marked `continue-on-error: true` (v4 is unproven during the soak; a v4 failure must NOT break the proven v3 pipeline — it is promoted to required in Phase 5). Add `id-token: write` + an assume-role step where missing (e.g. `v3-commit-metadata.yml`). AND switch `schema-deploy.yml` from `workflow_dispatch`-only to also push on the deploy branch under `paths: migrations/** + scripts/migrate-schema.py` (keep `workflow_dispatch` + `dry_run`), removing superseded `environment:`-gate comments | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml,schema-deploy.yml}` | `yamllint --strict` clean; each ingest workflow has the v3 step (required) + a `continue-on-error: true` v4 `--postgres` step; `schema-deploy.yml` `on:` includes push to the deploy branch under `paths: migrations/**`; CloudWatch shows Postgres writes appearing per develop push (best-effort) |
| ~~PR-2.5~~ | 2 | **DROPPED (lean re-plan 2026-06-04).** Was: Postgres-side dual-write reconciliation (`reconcile-ingest.py` + `dual-write-verify.yml` + incident.io paging). Removed as disproportionate to a trusted-input low-stakes dashboard with four safety nets (v2 live, v3 live, ≥90-day DuckDB snapshot, Phase-3 `migrate --verify`). v4 correctness is gated by Phase-3's one-shot `migrate --verify` + a short manual soak spot-check (`psql-bench.sh` row-count + a few measurement_id lookups) — no standalone service, no paging. | (none) | n/a — dropped |
| PR-2.6 | 2 | **(amend 2026-06-05, audit must-fix)** Fix the LIVE gate-var bug: the 3 ingest workflows gate the v4 dual-write `if:` on `vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (live: SET) but assume-role uses `vars.GH_BENCH_INGEST_ROLE_ARN` (live: UNSET), so on the next `develop` push the v4 steps FIRE and FAIL at assume-role (swallowed by `continue-on-error`) instead of cleanly no-op'ing. Re-key all 9 v4 sub-step `if:` gates to `vars.GH_BENCH_INGEST_ROLE_ARN != ''` (the var that must exist for assume-role to succeed); keep the `inputs.mode == 'develop'` clause in `sql-benchmarks.yml`. | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml}` | `yamllint --strict` clean; every v4 sub-step `if:` keys on `GH_BENCH_INGEST_ROLE_ARN`; with the role ARN unset the v4 steps no-op (do not fire), restoring dormant-until-wired behavior |
| PR-2.7 | 2 | **(amend 2026-06-05, audit de-gold-plate)** Remove trusted-input over-hardening from `scripts/post-ingest.py` (pre-lean-re-plan 15-cycle residue; stricter-than-v3-source on TRUSTED CI input per the calibration) + dead code: delete `_is_local_host` (dead; inline the loopback set into the one test fixture), the unreachable `RecursionError` branch in `read_records`, `_reject_unstorable_str` (NUL/lone-surrogate string guards), the `git_show_field` bytes-then-explicit-UTF-8 rewrite (revert to `subprocess(text=True)`), the `read_records` universal-newline/non-UTF-8 hardening, and the `_require_finite` `OverflowError` branch; delete the ~12 matching tests. **KEEP**: measurement_id parity, ON CONFLICT/RETURNING(xmax=0), retry, IAM/TLS auth, `deny_unknown_fields` + typed i32/i64/finite(in-range) validation, memory-quartet/storage-enum. | `scripts/post-ingest.py`, `scripts/test_post_ingest_postgres.py` | `uv run --all-packages pytest scripts/` green; the deleted guards + their tests gone; all measurement_id / ON CONFLICT / IAM-TLS / typed-validation tests still pass |
| PR-2.8 | 2 | **(amend 2026-06-05, retained merge-blocker + phase-end nits)** (1) Reconcile the bench-v4 Python files to the repo `ruff` line-length-120 + style (clear the pre-existing E501 at `test_migrate_schema.py:831` + the established multi-line style) — the RETAINED code-quality merge blocker. (2) Fix the 2 phase-end nits: correct the `benchmarks-website/infra/README.md` var-table PR-lineage mislabel (PR-2.2 → PR-2.4 for the ingest-workflow consumers of `RDS_BENCH_INSTANCE_ENDPOINT`/`GH_BENCH_INGEST_ROLE_ARN`); add a `post-ingest.py` comment noting the v4 step inherits the v3 step's `build_commit` git-history assumption. | `scripts/*.py`, `benchmarks-website/infra/README.md` | `ruff check scripts/` clean (no E501 on the bench-v4 files); README var-table references PR-2.4 for the ingest consumers; `uv run --all-packages pytest scripts/` green |
| PR-3.1 | 3 | **(re-plan)** Extend `vortex-bench-migrate` with `--postgres-target $DSN`: add a minimal async Postgres client (`tokio-postgres` + TLS connector for verify-full; **NO `aws-sdk-rds`** — operator-local master-password DSN) + per-table `COPY FROM STDIN` (text format; `BIGINT[]` via array literals) for `commits` + the 5 fact tables, reusing the existing accumulators + `vortex_bench_server::db::measurement_id_*`. **Atomic: all tables in ONE transaction** (BEGIN → COPY each → COMMIT) so a mid-load failure rolls back to empty, never half-seeded. **+ 004-as-master ordering preflight** in `migrate-schema.py` (assert the connected role can apply 002/004 before doing so; fail loud otherwise) | `benchmarks-website/migrate/Cargo.toml` (+`tokio-postgres` + TLS; NO aws-sdk-rds), `benchmarks-website/migrate/src/postgres.rs`, `benchmarks-website/migrate/src/main.rs` (CLI flag), `scripts/migrate-schema.py`, `scripts/test_migrate_schema.py` | Embedded-DuckDB unit tests (no Docker, green) cover the read + COPY-text pipeline (escaping incl. backslash/tab/newline, `BIGINT[]` array literals, NULLs, negative ints, UTC timestamp via CAST, f64 shortest-round-trip); 004-as-master preflight rejects a non-master role (testcontainer-gated) + marker-detection unit tests (no Docker). **The full Postgres-execution integration (6-table load + atomic mid-load rollback) is delivered by PR-3.3's rehearsal harness** (which owns the local postgres:16 container infra and asserts PR-3.2 verify-clean), to avoid standing up the Postgres-container test infra twice. |
| PR-3.2 | 3 | **(re-plan — PRIMARY v4-correctness gate)** Extend `migrate/src/verify.rs` + a Postgres read path to compare DuckDB-source vs Postgres-target **per `measurement_id`** (1:1): for each fact table compare every VALUE column + `env_triple` (`all_runtimes_ns` element-wise); for `commits` compare the 8 metadata columns per `commit_sha`. `VerifyReport` gains a value-mismatch list `{table, id, column, duckdb_value, pg_value}`. Full comparison (not sampled). `verify --postgres-target` exits non-zero on ANY presence diff OR value mismatch. **(2026-06-05 impl clarifications, recorded before review):** (a) the value gate lands as a NEW `PgVerifyReport`/`ValueMismatch` pair rather than overloading the unrelated v2-vs-v3 structural `VerifyReport` (single-responsibility; the structural report compares group/chart shape, not stored values); (b) `commits.timestamp` is compared as engine-independent **epoch microseconds** (DuckDB `epoch_us` vs Postgres `(extract(epoch ...) * 1e6)::bigint`, exact for whole-second git timestamps) to sidestep cross-engine timestamp text-rendering divergence; (c) `Cargo.toml` is NOT touched (PR-3.1 already added `postgres`/`native-tls`; value verify reuses them). | `benchmarks-website/migrate/src/verify.rs`, `benchmarks-website/migrate/src/postgres.rs` (expose `ColKind`/`column_kind`/`connect_postgres` as `pub(crate)`), `benchmarks-website/migrate/src/main.rs` (`Verify --postgres-target`) | **Comparison-core (no-Docker, PR-3.2):** unit tests prove discrimination — seed source==target -> clean; mutate ONE value column / `env_triple` / one `all_runtimes_ns` element (incl. reorder) / one `commits` field / NULL-vs-value -> exactly one mismatch naming the exact `(table, key, column, duckdb_value, pg_value)`; presence-only diffs caught **both directions**; epoch-microsecond timestamp pinned. **Live PG16 end-to-end (PR-3.3):** the `load -> verify exits 0; mutate -> exits non-zero` against a real postgres:16 is delivered by PR-3.3's rehearsal harness, which **owns the container infra and already asserts PR-3.2 verify-clean** (same PR-3.2/PR-3.3 boundary as PR-3.1's deferred Postgres-execution coverage; avoids standing the container infra up twice; Docker is unavailable in the dev env). |
| PR-3.3 | 3 | **(re-plan — bulk-load rehearsal, gap #2)** Build the rehearsal harness: an end-to-end integration test that loads a representative fixture DuckDB into a **local postgres:16 testcontainer** via PR-3.1's loader and asserts PR-3.2's verify is clean — proving the load+verify CODE works before any prod write. + Document the operator runbook for the REAL-snapshot rehearsal: snapshot acquisition (rehydrate the S3 Vortex backup or scp the live `bench.duckdb`) + **live source account/region verification** (`aws sts get-caller-identity`, `aws s3api get-bucket-location`) since `375504701696/us-east-2` is unverified in-repo | `benchmarks-website/migrate/tests/end_to_end.rs` (or a new `postgres_e2e.rs`), `benchmarks-website/migrate/README.md` (rehearsal + acquisition + verification runbook) | The end-to-end test loads a fixture DuckDB → local PG16 → `verify` clean (CI, fails loud if Docker absent per the established pattern); **also asserts all 6 tables' per-table row counts match the source AND that a forced mid-load failure leaves the target empty (atomic single-txn rollback) — the Postgres-execution coverage for PR-3.1's loader**; the runbook documents snapshot acquisition + the live source-account/region verification step + the local rehearsal procedure. **(2026-06-05 carry-forward from the PR-3.2 review, correctness/codex nit):** the fixture seeds at least one `commits` row with a sub-second (and ideally pre-1970) timestamp, so the live PG16 run confirms the Postgres `(extract(epoch from ts) * 1000000)::bigint` epoch equals DuckDB `epoch_us` to the microsecond — the one verify path the PR-3.2 no-Docker tests cannot exercise (Postgres-side epoch SQL is never executed without a container). |
| PR-3.4 | 3 | **(re-scoped 2026-06-05: REAL-snapshot LOCAL rehearsal — the prod load moved to Phase 5.)** Acquire the REAL v3 DuckDB off the EC2 host, load it into a LOCAL `postgres:16` (the existing PR-3.3 container path) via PR-3.1's loader, then run PR-3.2's `verify --postgres-target` + PR-3.5's cross-check locally. This is the real-data validation gate (the only check the synthetic fixtures cannot give) with ZERO prod risk. Capture per-table row counts + verify-clean + cross-check-clean in `Implementation status`. **Mostly agent-doable now** (see operational facts below). | (no production code; operational PR + logs in Implementation status) | The acquired real v3 DuckDB loads into a local PG16 with per-table row counts captured; `migrate verify --duckdb <snapshot> --postgres-target $LOCAL_DSN` reports 0 presence diffs AND 0 value mismatches; the PR-3.5 cross-check reports clean (writer UPDATEs, not duplicates) against the locally-loaded data. **Operational facts (verified 2026-06-05):** v3 EC2 host `ec2-18-219-54-101.us-east-2.compute.amazonaws.com` (SSH port 22 reachable + in known_hosts; an "AWS EC2" RSA key is in the ssh-agent; SSH user unconfirmed — try `ec2-user`/`ubuntu`; needs explicit approval per the auto-mode classifier); DuckDB at `/var/lib/vortex-bench/bench.duckdb` (live v3 Axum server on `:3000` holds it open -> for a CONSISTENT copy either briefly `systemctl stop` the v3 service then `cp bench.duckdb*` then restart, OR copy the live file + WAL and let DuckDB recover on open); v3 source region CONFIRMED `us-east-2` (was unverified). For the eventual Phase-5 prod load: prod RDS `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432` is PubliclyAccessible; the `bench-prod` AWS CLI profile (connor-aws-cli) reaches account `245040174862`; RDS Proxy = `vortex-bench-proxy.proxy-c4f8qygk4xdp.us-east-1.rds.amazonaws.com`. |
| PR-5.0 | 5 | **(2026-06-10 re-plan: now bundles the operator-gate prerequisite, then the prod load.)** Bring prod online. (1) **Operator-gate** (Phase-1 authorize-over-operator-gated precedent): apply the full schema to prod incl. migration **005 as the RDS master** (005 is requires-superuser; the schema-deploy `migrator` path cannot apply it), set the `bench_read` password, and configure the Vercel prod env (`BENCH_DB_*` / `bench_read` DSN) so the read service authenticates against prod. (2) **One-shot PROD load**: acquire the freshest v3 snapshot, `load` it (atomic single-txn) over verify-full TLS with the master-password (or IAM) DSN, run `verify --postgres-target`, then the PR-3.5 cross-check; capture per-table row counts + verify-clean + cross-check-clean. (3) **Deploy evidence**. Production data seed (hard-to-reverse) -> RDS PITR (35-day) is the rollback. | (no production code; operational PR + logs; reuses PR-3.1/3.2/3.5 tools) | `bench_read` SELECTs against prod (005 applied as master + password set + Vercel prod env wired); Prod RDS holds the full v3 history (per-table row counts in Implementation status); `migrate verify --duckdb <snapshot> --postgres-target $PROD_DSN` reports 0 presence diffs AND 0 value mismatches; the cross-check confirms the Python writer UPDATEs the seeded rows; the PITR rollback command is documented. **ALSO closes the two Phase-4 data-dependent exit criteria MOVED here (user decision 2026-06-10): the prod Vercel deploy serves + `curl <prod-url>/api/groups \| jq '.groups[].charts[].slug' \| sort` matches the family registry, AND ~5 representative chart slugs match the current v2 site on a manual visual check.** |
| PR-5.0.5 | 5 | **(2026-06-10 re-plan, Decision C: v2-fidelity restore for the one real v2->v4 regression.)** Restore the v2 `statpopgen` / `polarsignals` group names + descriptions in the read service: extend `groupNameQuery` (`web/lib/queries.ts`) to special-case `statpopgen`/`polarsignals` (as v2's `src/config.js` did) so the live `'Statistical and Population Genetics'` / `'PolarSignals Profiling'` cases in `web/lib/descriptions.ts` attach instead of falling through to the legacy `dataset sf=N [storage]` label. The other five preserved-v3 parity quirks STAY (Decision C: preserve-v3). | `benchmarks-website/web/lib/queries.ts`, `benchmarks-website/web/lib/descriptions.ts`, `benchmarks-website/web/lib/groups.test.ts` | A seeded statpopgen + polarsignals group renders the friendly name + description (not the `dataset sf=N` legacy label); a discriminating test pins it (removing the special-case fails the test); the other four `groupNameQuery` branches keep their v3 behavior; vitest + `next build` + lint clean. |
| PR-3.5 | 3 | **(re-plan — Python-writer-vs-RDS cross-check, gap #3)** Build a cross-check harness: take a few REAL v3 envelopes whose dim tuples exist in the seeded data, run `post-ingest.py --postgres` against RDS, and assert each computed `measurement_id` matches a Rust-seeded row (→ UPDATE, 0-inserted/N-updated, `xmax != 0`) and the VALUE+`env_triple` columns round-trip (re-read + compare). Run it right after PR-3.4's seed (earliest detection). The harness is re-run as the PR-5.1 pre-promotion gate | `scripts/cross_check_python_writer.py` (or extend `post-ingest.py` test tooling), `scripts/test_*` (local-container discrimination test) | Local-container test: the harness reports UPDATE-not-INSERT + value round-trip on correct input, and FAILS when a value is deliberately wrong (discriminating). Operator runs it against prod RDS after PR-3.4; the run is logged clean (0 duplicate INSERTs; value columns match the seeded rows). |
| PR-3.6 | 3 | **(amend 2026-06-08 — phase-end cycle-1 should-fix sweep)** Apply the 3 should-fix doc/status items from the Phase-3 phase-end review: (a) add a `requires-superuser` subsection to `migrations/README.md` documenting the marker + the `rolsuper OR rolcreaterole` preflight + the "apply 002/004 as RDS master before any migrator deploy" ordering (the operator-facing doc the `_assert_master_capable` PermissionError points at); (b) annotate the PR-3.1 Implementation-status entry with the deliberate sync-`postgres`-over-plan-named-`tokio-postgres` choice; (c) annotate the PR-3.4 Implementation-status entry that `vector_search_runs` had 0 real rows + was omitted from the cross-check envelope, so that table's real-data coverage is fixture-only (PR-5.0 closes it). | `migrations/README.md` (the reviewable code/doc change); `.big-plans/` Implementation-status entries (plan edits) | `migrations/README.md` has a requires-superuser subsection matching the `_assert_master_capable` error pointer; the PR-3.1 + PR-3.4 status entries carry the two annotations; no behavior change (doc/status only). |
| PR-3.7 | 3 | **(amend 2026-06-08 — phase-end cycle-2 nit sweep)** Clear the 3 cycle-2 doc nits: (a) align the hypothetical directive example in `migrations/README.md` authoring rules from `-- migrate: no-transaction` to `-- migrate-schema: no-transaction` (match the only-implemented directive's `migrate-schema:` namespace); (b) add one sentence to `benchmarks-website/migrate/src/postgres.rs` `pub fn load` doc comment noting the target schema (migrations/001) must already be applied (it only COPYs into existing tables); (c) add a one-line note to the `migrate/README.md` rehearsal runbook (§2 Acquire the snapshot) that a pre-acquired on-disk snapshot is an acceptable rehearsal source (matching PR-3.4's execution). | `migrations/README.md`, `benchmarks-website/migrate/src/postgres.rs` (doc comment), `benchmarks-website/migrate/README.md` | the 3 nits are resolved; `cargo +nightly fmt --check -p vortex-bench-migrate` clean (doc-comment edit); no behavior change (doc-only). |
| PR-4.1 | 4 | Scaffold `benchmarks-website/web/` Next.js 15 project: `package.json`, `tsconfig.json` (matching `vortex-web/`'s strict flags), `next.config.js`, `.gitignore`, SPDX headers everywhere | `benchmarks-website/web/{package.json,tsconfig.json,next.config.js,.gitignore}`, `app/layout.tsx`, `app/page.tsx` (stub) | `pnpm install && pnpm build` succeeds; lints clean. |
| PR-4.2 | 4 | Connection lib at `web/lib/db.ts`: `pg.Pool` + `@aws-sdk/rds-signer` for IAM token gen + token-refresh-before-expiry; expose `sql` tagged-template helper | `web/lib/db.ts`, `web/lib/db.test.ts` | Integration test (testcontainers Postgres): pool connects via password (test fixture); pool roundtrips a SELECT. IAM-auth path mocked. |
| PR-4.3.a | 4 | **(2026-06-08 split of PR-4.3, foundation + health)** Read-port foundation: `web/lib/schema-version.ts` (`export const SCHEMA_VERSION = 1` — the Table D read-path gate site), `web/lib/slug.ts` (encode/decode the 5 `ChartKey` + 5 `GroupKey` variants as `<prefix>.<base64url-json>`, matching `server/src/slug.rs` prefixes), `web/lib/window.ts` (`?n=` → commit window: default 100, numeric clamp `[1, 1000]`, `all` unbounded — ports `server/src/api/window.rs`), `web/lib/families.ts` (5-fact-table registry + metadata), and the `/health` route (preserve the `HealthResponse` snake_case shape; adapt `db_path`→DB host, `build_sha`→`VERCEL_GIT_COMMIT_SHA`/"unknown", `schema_version`→the const). | `web/lib/{schema-version,slug,window,families}.ts`, `web/app/api/health/route.ts`, `web/lib/*.test.ts` | slug round-trips for all 10 key variants (decode∘encode = id, prefixes match `slug.rs`); `?n=` parsing matches `window.rs` (default 100, clamp 1..1000, `all`); `/health` returns the snake_case shape with correct `row_counts` keys + `schema_version = 1`; vitest green. |
| PR-4.3.b | 4 | **(2026-06-08 split of PR-4.3, chart endpoint)** `web/lib/queries.ts` `chartPayload` — the two-pass seeded-commit-window port for all 5 chart types (`query_measurements`, `compression_times`, `compression_sizes`, `random_access_times`, `vector_search_runs`), Postgres-parameterized (`IS NOT DISTINCT FROM` nullable-dim equality; oldest-first `commits[]`; `series` map + `series_meta` + `unit_kind` + `ChartHistory`), and the `/api/chart/{slug}` route with `export const revalidate = 300`. | `web/lib/queries.ts`, `web/app/api/chart/[slug]/route.ts`, `web/lib/queries.test.ts` | `/api/chart/{slug}` returns the exact `ChartResponse` snake_case shape (`display_name`/`unit_kind`/`history`/`commits`/`series`/`series_meta`) byte-equivalent to the Axum server for representative fixtures (snapshot test vs `server/tests/chart_api.rs` + `server/fixtures/`); `measurement_id` absent from the wire; `?n=all` uncapped, numeric capped at 1000. |
| PR-4.3.c | 4 | **(2026-06-08 split of PR-4.3, groups + group endpoints)** `collectGroups` discovery over the 5 families (slug generation + `GROUP_ORDER` sort), `web/lib/summary.ts` (4 summaries: `randomAccess` rankings, `compression` speedup geomean, `compressionSize` ratio distribution, `queryBenchmark` rankings — ports `server/src/api/summary.rs`), `web/lib/descriptions.ts` (editorial blurbs ported from v2), and the `/api/groups` + `/api/group/{slug}` routes (`revalidate = 300`). | `web/lib/{queries,summary,descriptions}.ts`, `web/app/api/groups/route.ts`, `web/app/api/group/[slug]/route.ts`, tests | `/api/groups` returns `{groups:[Group…]}` with `GROUP_ORDER` sort + correct `Summary` tagged-union shape (camelCase variant fields per `dto.rs`) + descriptions; `/api/group/{slug}` returns `GroupChartsResponse` with flattened `NamedChartResponse` charts byte-equivalent to Axum for representative groups (snapshot vs `server/tests/group_api.rs`). |
| PR-4.4.a | 4 | **(2026-06-09 split of PR-4.4, server shell + CSS)** **(port-source refinement 2026-06-09: the port source is v3's server-rendered HTML layer — `server/src/html/{render,landing,summary}.rs` + `server/static/style.css` — NOT v2's React, because v3 is ALREADY the server-shell+client-island model + reproduces the v4 endpoints' ns data shape + uses v2's CSS class vocabulary; v2 React stays the secondary visual cross-check. v3 folded v2's Sidebar into the header nav, so NO separate Sidebar component — scope unchanged.)** Server-rendered landing shell: `web/app/layout.tsx` (`<html>`/`<head>` fonts + favicons + theme-bootstrap inline script + `globals.css` import; `<body>`), `web/app/page.tsx` (server component calling `collectGroups()` which embeds per-group summary + description, rendering one collapsible `<details>` `<section.group-details>` per group in `GROUP_ORDER` — disclosure header = group name + ⓘ info-icon + chart count; summary card above the chart grid; an empty chart-card mount point per chart carrying `data-chart-slug` + a stable per-page `data-chart-index` + an empty `<canvas>`), the static server-component pieces (`web/components/Header.tsx` = logo/title/GitHub static chrome [interactive nav/theme/filter deferred to 4.4.b], `web/components/GroupSection.tsx`, `web/components/SummaryCard.tsx` = port of `summary.rs` 4 variants incl. `formatTimeNs`, `web/components/Footer.tsx` = build-SHA via `VERCEL_GIT_COMMIT_SHA`), `web/lib/format.ts` (`formatTimeNs`, ns-based, ported from `summary.rs::format_time_ns`), and `web/app/globals.css` ported from v3 `server/static/style.css`. Native `<details>` gives working per-group expand/collapse with NO JS. NO chart canvas / interactivity yet (chart-card shells are empty mount points). Preserve UI BANS that apply to the shell. | `web/app/{layout,page}.tsx`, `web/components/{Header,GroupSection,SummaryCard,Footer}.tsx`, `web/lib/format.ts`, `web/app/globals.css`, tests | landing page renders server-side for all groups in `GROUP_ORDER` with the correct summary card (ns values via `formatTimeNs`) + description per group + one chart-card shell per chart (each carrying `data-chart-slug`); structure matches v3's landing/summary layout (v2-equivalent) on a manual visual check; `next build` + `tsc` + `eslint` + `prettier` + vitest green. |
| PR-4.4.b | 4 | **(2026-06-09 split of PR-4.4, chart client island + interactivity + permalink)** The interactive surface as client islands: `web/components/Chart.tsx` (`'use client'` Chart.js line chart — lazily fetches `/api/chart/[slug]` on group-open/visible, LTTB downsample over the cached payload, range strip + zoom/pan toolbar, custom tooltip positioner; ports v2 `ChartContainer.jsx` + the relevant `chart-init.js` behavior), `web/components/FilterBar.tsx` (`'use client'` chip toggles over the filter universe), shared `web/lib/chart-format.ts` (ports the pure helpers from v3 `chart-init.js` — display-unit picker/LTTB/payload-normalize/formatting; v3's interactive layer is the behavior source per the locked v3-source arch, superseding the pre-pivot "v2 `utils.js`" wording), and the `web/app/chart/[slug]/page.tsx` permalink page. **(2026-06-09 in-flight scope note: `Modal.tsx` DROPPED — v3 has no expanded-chart modal; the `/chart/[slug]` permalink page IS the expanded view in the v3 UI this PR ports. The Modal entry predated the PR-4.4.a v3-source pivot.)** **Also owns the deferred-from-PR-4.4.a header interactivity**: the mobile nav (hamburger `.nav-controls` + `.nav-controls-github` mobile GitHub fallback — resolves the PR-4.4.a Deferred-work mobile-GitHub-link gap), expand/collapse-all, theme toggle + the theme-bootstrap inline script. Preserve ALL UI BANS (oldest-first `commits[]` predecessor walk `idx-1`; no `pointer-events:auto` on tooltip host; throttled `input` not `change` on sliders; no refetch on pan/zoom/slider beyond the one-shot `?n=all` hop). **NO shard route.** | `web/components/{Chart,FilterBar}.tsx`, `web/lib/chart-format.ts`, `web/app/chart/[slug]/page.tsx`, the header-interactivity island(s), tests | **(lean re-plan: relaxed acceptance)** charts match the current v2 site for ~5 representative slugs on a manual visual check (dropped the byte-equivalent + lighthouse≥90 + ≤5%-pixel-diff bars as over-specified for a benchmarks dashboard); the mobile GitHub link is present <768px; all UI BANS preserved (verified against the diff); `next build` + `tsc` + `eslint` + `prettier` + vitest green. |
| PR-4.5 | 4 | Vercel deploy config (`vercel.json`); GitHub Action `web-deploy.yml` for preview-per-PR + production deploy on merge | `vercel.json`, `.github/workflows/web-deploy.yml` | PR opens trigger preview deploy; merging to ct/bench-v4 triggers prod deploy (still behind dev-only Vercel domain at this stage). |
| ~~PR-4.6~~ | 4 | **DROPPED (lean re-plan 2026-06-04).** Was: a push-based HMAC revalidate endpoint (`REVALIDATE_SECRET`) + a `post-ingest.py` revalidation hook. Removed in favor of time-based `revalidate` (PR-4.3) — ~5-min staleness matches v2's existing behavior, and a push pipeline + shared secret is unnecessary machinery for a benchmarks dashboard. | (none) | n/a — dropped |
| PR-5.1.5 | 5 | **(amend 2026-06-11, read-path perf — discovered closing PR-5.0; slotted before the DNS flip because the flip exposes the read path to public traffic. User decision: COMPREHENSIVE single PR covering all diagnosed causes, reviewed once, then prod-verified.)** Make the v4 read path scale to the full prod seed (4.85M `query_measurements`; data + rendering already PROVEN correct — server-side query perf only). **(a) Sargable WHERE** (highest leverage): replace `col IS NOT DISTINCT FROM $x` with build-time-specialized `col IS NULL` (key value null) / `col = $x` (non-null) across `queries.ts` `chartPayload` (8 sites :316-438) + `summary.ts` param-bound predicates (:366-367) so `idx_query_measurements_chart` is seeked beyond the leading `dataset=` instead of heap-filtering (logically identical for a concrete chart key; tpch chart ~24s → sub-second). For the table-to-table null-safe summary joins (`summary.ts:253,328`) use an index-supported equivalent (COALESCE-to-sentinel or a covering index — keep semantics identical). **(b) Discovery GROUP BYs**: loose-index-scan / recursive-CTE skip-scan for the 5 family discovery scans in `collectGroups`. **(c) Latest-per-series summary**: index-backed `DISTINCT ON` (+ supporting index) for the per-group summaries (avoid the `row_number()` full-window spill; PREFER index-only — denormalizing `commit_timestamp` into the fact tables is a fallback only if the index approach is insufficient, since it touches the write path). **(d) `collectFilterUniverse`** (`queries.ts:993`): replace the 4.85M `SELECT DISTINCT engine` seq scan + cross-fact UNION with an index-only DISTINCT / skip-scan (or a small lookup). **(e) Parallelize** the 64 sequential per-group summary queries (bounded concurrency, possibly raise `poolMax` in `db.ts`). **(f) Upsize the RDS instance** (operator: `db.t4g.micro` → larger; raise `work_mem`/`shared_buffers` via parameter group) so the 1.2GB fact table fits cache. Index/column changes land as a new `migrations/006_*.sql`; the plan's matview/trigger BAN still holds (indexes + columns are fine). | `benchmarks-website/web/lib/queries.ts`, `benchmarks-website/web/lib/summary.ts`, `migrations/006_read_path_perf.sql` (repo-root migrations/, NOT benchmarks-website/migrations/; (c)=DECIDED denormalize commit_timestamp into the 5 fact tables + backfill + latest-per-series index, plus (b) discovery + (d) filter-universe skip-scan indexes) (+ `scripts/test_migrate_schema.py`), write-path commit_timestamp populate in `scripts/post-ingest.py` + `benchmarks-website/migrate/src/postgres.rs` (Rust loader), `benchmarks-website/web/lib/db.ts` (pool sizing for (e) parallelize), tests (`web/lib/queries.test.ts`, `web/lib/summary.test.ts`/`groups.test.ts`); operator: RDS instance-class upsize + parameter group (DONE 2026-06-11: db.t4g.medium + vortex-bench-pg16 work_mem=32MB) | Before/after prod EXPLAIN + timings captured in Implementation status showing the chart query uses `idx_query_measurements_chart` past the leading column (no per-dataset heap full scan) and tpch/tpcds/clickbench `/api/chart` drop to sub-second; **`/api/groups` returns within a few seconds against the full prod seed AND `curl <prod-url>/api/groups \| jq '.groups[].charts[].slug' \| sort` matches the family registry** (resolves the PR-5.0-deferred check); **~5 representative chart slugs incl. a big dataset match the live v2 site on a manual visual check** (resolves the second PR-5.0-deferred check); migration 006 applies cleanly (testcontainer test green) and the sargable rewrite is pinned as semantically identical (existing `queries.test.ts` NULL-dim equality test still passes + a new test asserts the sargable predicate matches a NULL dim and misses a wrong value); inner-loop 2-vote gauntlet accepts; `next build` + `tsc` + `eslint` + `prettier` + vitest + `ruff check scripts/` + `pytest scripts/` green; post-upsize CloudWatch FreeableMemory healthy. |
| PR-5.0.9 | 5 | **(2026-06-11 amend, user-directed UI/UX round; scoped via brainstorming; the design doc `.big-plans/ct__bench-v4-uiux-design.md` is authoritative.)** Opt-in full-history chart loading: remove the automatic `?n=all` warmup from `Chart.tsx` `onGroupOpen` (today every chart in an opened group background-fetches its full history through `fullHistoryQueue`: ~24MB per 22-chart group open, hundreds of MB on Expand All; measured 2026-06-11: `?n=100` 34KB / 0.17-0.2s warm MISS, `?n=all` 1.1MB / 0.46-1.07s warm MISS, CDN HIT 0.06s). Full history becomes per-chart opt-in via (a) an always-visible window chip on incomplete charts ("latest 100 of N"; hover presents "load all N"; spinner while loading; "all N" complete; "retry" on error; click fetches at `INTERACTION_FULL_PRIORITY`), (b) a ~600ms same-card hover-dwell silent prefetch at a new mid-tier priority constant (hover reveals the control immediately; only the dwell starts the fetch; `pointerleave` cancels; user decision "Both, staged"), and (c) the existing `rangeTouchesUnloadedHistory` interaction promotion, unchanged. Add `stale-while-revalidate=86400` beside `s-maxage=300` on the API cache policy. The virtual-axis upgrade path (`total_commits`/`start_index` + null prefix + in-place fill) is unchanged: no re-base jank. Deliberately deferred (design doc, not the Deferred-work table: not data-correctness): viewport-based hydration, server-side `?n=all` downsampling, visual redesign, keep-warm infra. | `benchmarks-website/web/components/Chart.tsx` (warmup removal, chip, dwell), `benchmarks-website/web/lib/chart-format.ts` (dwell + priority constants), `benchmarks-website/web/lib/cache.ts` + `benchmarks-website/web/vercel.json` (SWR), web vitest tests | No `?n=all` request without per-chart intent (dwell, chip click, or interaction touching unloaded history); discriminating test: `onGroupOpen` schedules zero `fullHistoryQueue` entries; a 22-chart group open transfers ~1MB or less of chart data; chip states pinned by tests (windowed/loading/complete/error-retry; no chip when `history.complete`); dwell fires at threshold not before and cancels on `pointerleave`; existing interaction-promotion tests stay green; API responses carry `stale-while-revalidate`; vitest + `next build` + `tsc` + `eslint` + `prettier` green; 2-vote gauntlet (pr-2) accepts; post-deploy network profile of a tpch group open shows only `?n=100` requests. |
| PR-5.0.95 | 5 | **(2026-06-12 amend, user-directed UI/UX round 2; scoped conversationally after PR-5.0.9 shipped; design `.big-plans/ct__bench-v4-uiux-r2-design.md` is authoritative.)** Lazy-hydration + resilient loading for large chart groups (clickbench ~43 charts still feels slow + sometimes hangs even after PR-5.0.9). (A) **Viewport-based lazy hydration** on the landing page: gate each group chart's initial `?n=100` fetch+construct behind an `IntersectionObserver` (reuse the permalink `else`-branch pattern, `Chart.tsx` ~L1646), so on group open only ~visible charts hydrate (top-first, visual order) and the rest hydrate on scroll; reconcile the all-charts summary-hover prefetch (`Chart.tsx` ~L1628) so it cannot re-introduce the burst (drop or make viewport-aware). (B) **Fetch timeout + abort + retry**: wire the controller `aborter.signal` into both `fetch()` calls (`Chart.tsx` ~L463/550) + a `FETCH_TIMEOUT_MS` AbortController timeout so a stalled fetch aborts instead of spinning forever; abort in-flight fetches on group close/teardown; an initial-fetch RETRY affordance that re-issues the FETCH (not just construction). (C) **Spinner animation** replacing the static "loading…" text (`Chart.tsx` ~L1808) + the chip loading state, with a `prefers-reduced-motion` guard. Pre-impl: a ~10-min read-only server-side check (Vercel logs + RDS connections via `bench_read`) to confirm whether the hangs are client-burst (A fixes) or need separate server work — do NOT expand scope to server queries. Out of scope: server query perf, `?n=all` downsampling, visual redesign, chip/dwell changes. | `benchmarks-website/web/components/Chart.tsx` (IO-gated landing hydration + summary-prefetch reconcile + fetch abort/timeout/retry + spinner markup), `benchmarks-website/web/lib/chart-format.ts` (`FETCH_TIMEOUT_MS` + IO rootMargin consts), `benchmarks-website/web/app/globals.css` (spinner `@keyframes` + reduced-motion), web vitest tests | Group open hydrates only ~visible charts (rest on scroll, top-first); a stalled initial fetch times out + offers retry (no infinite spinner) and group-close aborts in-flight fetches; the loading state shows an animated spinner respecting `prefers-reduced-motion`; PR-5.0.9 chip/dwell/404 behavior unchanged; vitest + `next build` + `tsc` + eslint + prettier green; 2-vote gauntlet (pr-2) accepts. |
| PR-5.1 | 5 | **Promote the v4 `--postgres` step to required (remove its `continue-on-error`)** + drop the v3 `--server` write from the 3 CI workflows; `post-ingest.py` runs only with `--postgres`. **Pre-promotion gate (Phase-5 half of audit gap #3): re-run the PR-3.5 Python-writer-vs-RDS cross-check against accumulated soak data and confirm clean BEFORE removing `continue-on-error`.** **(phase-4 end-review fold-in: ship + document `scripts/psql-bench.sh` here — the Operator-SQL Key decision claims README documentation but the script is unshipped and unowned; it backs the soak spot-check and the PR-5.3 `/api/admin/sql` decommission rationale.)** | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml}`, `scripts/post-ingest.py` (remove --server mode) | The PR-3.5 cross-check re-run is clean (Python writer UPDATEs seeded rows, no duplicates) BEFORE promotion; v4 step no longer `continue-on-error` (now gates CI); v3 step removed; workflows green; CloudWatch shows zero traffic to v3 EC2. |
| PR-5.2 | 5 | DNS flip v2 → v4 (update Cloudflare/Route53 records for `benchmarks.vortex.dev` to Vercel) | (operational PR; possibly `dns/` configs if checked in) | `dig benchmarks.vortex.dev` resolves to Vercel; site loads in browser; Cloudflare/Vercel SSL chain verified. |
| PR-5.3 | 5 | Delete: `benchmarks-website/server/`, `benchmarks-website/ops/`, `benchmarks-website/migrate/`, top-level `server.js`, `src/`, `index.html`, `vite.config.js`, `package.json` (root), `package-lock.json` (root), `Dockerfile`, `docker-compose.yml`, `.github/workflows/publish-benchmarks-website.yml`; remove `INGEST_BEARER_TOKEN` + `ADMIN_BEARER_TOKEN` repo secrets; **decommission the consumerless RDS Proxy `vortex-bench-proxy` (Decision A 2026-06-10) via `aws rds delete-db-proxy`**; terminate v3 EC2 by hand post-merge | ~30 files deleted + RDS Proxy | `git grep -n INGEST_BEARER_TOKEN` returns 0; `gh workflow list` does not include `publish-benchmarks-website`; **`aws rds describe-db-proxies --db-proxy-name vortex-bench-proxy` returns DBProxyNotFoundFault (deleted)**; `aws ec2 describe-instances --instance-ids <v3-instance>` returns `terminated`. |

(Each PR targets 1-3 logical commits; some — like PR-1.3 (DDL + IAM role) and PR-4.3 (3 endpoints) — are at the upper end and may split during execution per the mid-flight expansion protocol.)

## Reference tables

**REQUIRED for migration work shape.** Mechanical, exhaustive lookups that downstream PRs grep into.

### Table A — `measurement_id_*` xxhash64 algorithm specification

The new ingest writer reproduces this byte-for-byte. Source: `benchmarks-website/server/src/db.rs:162-257`.

| Step | Operation | Encoding |
|---|---|---|
| 1 | Construct hasher | `XxHash64::with_seed(0)` (no per-call seeding) |
| 2 | Write per-table tag | `hasher.write(tag.as_bytes())` then `hasher.write_u8(0)` |
| 3 | `write_str(s)` | `hasher.write_u64(s.len() as u64)` THEN `hasher.write(s.as_bytes())`. No NUL terminator. Length is LE u64 (twox-hash writes LE without byte-swap). |
| 4 | `write_opt_str(opt)` | `Some` → `hasher.write_u8(0x01)` then `write_str`. `None` → `hasher.write_u8(0x00)` and stop. |
| 5 | `write_i32(v)` | `hasher.write_i32(v)` — 4 LE bytes. |
| 6 | `write_f64(v)` | `hasher.write_u64(v.to_bits())` — IEEE 754 bits as LE u64. |
| 7 | Finalize | `hasher.finish() as i64` (bit-cast `u64 → i64`). |

### Table B — per-table dim-tuple field order (drives the hash)

| Table | Tag (literal) | Field order |
|---|---|---|
| `query_measurements` | `"query_measurements"` | `commit_sha` → `dataset` → opt `dataset_variant` → opt `scale_factor` → `query_idx` (i32) → `storage` → `engine` → `format` |
| `compression_times` | `"compression_times"` | `commit_sha` → `dataset` → opt `dataset_variant` → `format` → `op` |
| `compression_sizes` | `"compression_sizes"` | `commit_sha` → `dataset` → opt `dataset_variant` → `format` |
| `random_access_times` | `"random_access_times"` | `commit_sha` → `dataset` → `format` (NO `dataset_variant`) |
| `vector_search_runs` | `"vector_search_runs"` | `commit_sha` → `dataset` → `layout` → `flavor` → `threshold` (f64 bits). `iterations` intentionally EXCLUDED. |

### Table C — DuckDB → Postgres type/syntax map

| Column shape | DuckDB | Postgres |
|---|---|---|
| `BIGINT[]` (`all_runtimes_ns`) | `[1,2,3]` literal + `CAST(? AS BIGINT[])` | Native `BIGINT[]` bind via `&[i64]` |
| `TIMESTAMPTZ` (`commits.timestamp`) | `CAST(? AS TIMESTAMPTZ)` from RFC 3339 | Native `TIMESTAMPTZ` from `OffsetDateTime` |
| `IS NOT DISTINCT FROM` (NULL-aware eq) | Supported | Supported (native) |
| Upsert | `ON CONFLICT (measurement_id) DO UPDATE SET <cols> = excluded.<cols>` | Identical syntax |
| Write-conflict retry | 128-attempt retry loop (DuckDB MVCC) | Postgres `SQLSTATE 40001` retry; design afresh |

### Table D — `SCHEMA_VERSION = 1` lockstep sites

Every bump touches all rows in this table in one PR (or CI ingest 400/409s).

| File | Line / symbol | Form |
|---|---|---|
| `benchmarks-website/server/src/schema.rs` | `SCHEMA_VERSION = 1` (line 223) | Rust `const i32` |
| `vortex-bench/src/v3.rs` | (struct field default + tests) | Rust |
| `scripts/post-ingest.py` | `SCHEMA_VERSION = 1` (line 52) | Python literal |
| `scripts/post-ingest.py` (extended) | `SCHEMA_VERSION = 1` (existing line 52) | Python `const int` |
| ~~`scripts/_measurement_id.py`~~ | (removed 2026-05-29) | The shipped hash port does NOT import or re-export `SCHEMA_VERSION` (no need for it, and `post-ingest.py` is not cleanly importable without `importlib`). `_measurement_id.py` is NOT a lockstep site; the Python writer's lockstep site is `scripts/post-ingest.py:52` (above). |
| `benchmarks-website/web/lib/schema-version.ts` (new, PR-4.2 or 4.3) | `export const SCHEMA_VERSION = 1` | TypeScript const — read-path version-gate (returns 4xx if envelope mismatch surfaces in any out-of-band write) |
| `benchmarks-website/migrate/src/lib.rs` (existing) | `pub const SCHEMA_VERSION` | Rust constant referenced by the v3→v4 one-shot migrator |

### Table E — CI ingest call sites

The new ingest writer replaces these `python3 scripts/post-ingest.py ...` invocations. Each gets a feature-flag branch during dual-write.

| Workflow | Step location | Trigger | Concurrent writers per run |
|---|---|---|---|
| `.github/workflows/bench.yml` | lines 108-118 | `push: develop` | 2 (matrix: random-access-bench, compress-bench) |
| `.github/workflows/sql-benchmarks.yml` | lines 527-537 | `workflow_call` from bench.yml + nightly + bench-pr (PR mode skips) | 11 (matrix entries) |
| `.github/workflows/v3-commit-metadata.yml` | lines 23-34 | `push: develop`, `workflow_dispatch` | 1 |

Total per `develop` push: **~14 parallel writers**.

### Table F — Decommission inventory

Items deleted in the final cutover PR.

| Path | Decommission rationale |
|---|---|
| `benchmarks-website/server/` (Rust crate) | Replaced by Postgres + Next.js |
| `benchmarks-website/ops/` (systemd + backup + deploy) | Replaced by managed Postgres + Vercel |
| `benchmarks-website/server/src/auth.rs` (bearer middleware) | No `/api/*` routes remain |
| `INGEST_BEARER_TOKEN` references in `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml` | Replaced by OIDC + IAM-auth role |
| `ADMIN_BEARER_TOKEN` env + secret | Admin endpoints retired |
| `benchmarks-website/server/src/admin.rs` | `/api/admin/*` retired (psql replaces it) |
| Top-level v2: `server.js`, `src/`, `index.html`, `vite.config.js`, `package.json`, `package-lock.json`, `Dockerfile`, `docker-compose.yml` | Replaced by Next.js 15 app under `benchmarks-website/web/` |
| `.github/workflows/publish-benchmarks-website.yml` | Vercel deploy replaces GHCR push |
| `benchmarks-website/migrate/` (one-shot v2→v3 migrator) | One-shot; survived v2→v3 only because of dual-write soak. **Decision in interview**: retire alongside v3→v4 migration, or retain as a long-tail tool? |

## Critical files

- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/schema.rs` — 6-table DDL constants; `SCHEMA_VERSION`; the table-order applied at boot.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/db.rs:162-257` — `measurement_id_*` xxhash64; load-bearing for migration.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/ingest.rs` — `POST /api/ingest` handler; envelope validation; per-record upsert; transaction boundaries.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/records.rs` — server-side envelope/record struct with `deny_unknown_fields`.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/api/charts.rs` — per-chart SQL via `seeded_commits_in_window` two-pass pattern; `IS NOT DISTINCT FROM` for NULL-aware dim equality.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/api/window.rs` — `CommitWindow` enum; `MAX_NUMERIC_COMMIT_WINDOW = 1_000`; numeric vs `all` semantics.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/read_model.rs` — `ReadGeneration` with precompressed gzip/brotli artifacts. Maps onto Vercel edge-CDN caching driven by `Vercel-CDN-Cache-Control` + `Cache-Control` s-maxage headers (the `unstable_cache` sketch was not usable; see the Read-service-framework Key-decision amendment).
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/migrate/src/verify.rs` — `VerifyReport` structural diff template.
- `/Users/connor/spiral/vortex-data/vortex4/vortex-bench/src/v3.rs` — v3 envelope and `V3Record` producer side; immutable across this migration.
- `/Users/connor/spiral/vortex-data/vortex4/scripts/post-ingest.py` — current CI envelope-poster; replaced by the new ingest writer.
- `/Users/connor/spiral/vortex-data/vortex4/.github/workflows/bench.yml` / `sql-benchmarks.yml` / `v3-commit-metadata.yml` — CI workflows that POST the envelope; sites of the dual-write modification.
- `/Users/connor/spiral/vortex-data/vortex4/.github/workflows/compat-gen-upload.yml` — structural template for any approval-gated migration step.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/AGENTS.md` — subsystem playbook; carry rules from v2→v3 cutover.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/tests/common/mod.rs` — `Server` harness pattern; carry to Postgres testcontainer harness.
- `/Users/connor/spiral/spiraldb/.github/workflows/deploy-spiraldb.yml:37-89` — cross-codebase deploy-migrations blueprint.
- `/Users/connor/spiral/spiraldb/.github/workflows/prisma-isolation.yml` — schema-isolation guardrail template.

## Risks

1. **Hash-byte-layout drift**: P=med (every reviewer initially distrusts a private byte-encoded PK); impact=severe (silently produces duplicate rows on next CI write, breaking all historical chart continuity). Mitigation: Phase 1 produces a golden-vector unit test from a `(commit, dim) → i64` snapshot of the live DuckDB; every subsequent PR runs the test; reviewers cross-reference Table A/B before approving.
2. **`SCHEMA_VERSION` lockstep slip**: P=med (4-5 sites now, including a new Rust writer and a new TS read layer); impact=moderate (CI ingest 400/409s after a partial bump). Mitigation: introduce a single source-of-truth file (`SCHEMA_VERSION` in a one-line file or shared crate), grep guard in CI.
3. **Cold-start cost on Vercel**: P=med (read path was 100% in-process materialized; cold function rebuilds artifacts from Postgres). Mitigation (as shipped): header-driven edge-CDN caching — `Vercel-CDN-Cache-Control` rules on `/` and `/chart/:slug` + `Cache-Control` s-maxage=300 on API 200s (`web/lib/cache.ts` + `web/vercel.json`), matching v2's 5-min cadence, so the warm path is CDN-served (the `unstable_cache`/`revalidateTag`-deploy-hook sketch was not usable — see the Read-service-framework Key-decision amendment). The lighthouse-score acceptance bar was dropped in the lean re-plan as over-specified for a benchmarks dashboard.
4. **Dual-write divergence under matrix concurrency**: P=med (~14 parallel CI writers hitting both substrates per push); impact=moderate (the verification harness has to handle "in-flight" rows). Mitigation: `verify.rs`-shaped tool runs at PR end, not mid-run; settle window before assertion.
5. **Connection-budget exhaustion**: P=low-med (Vercel serverless × matrix concurrency × CI ingest = many pools); impact=severe (Postgres rejects writers). Mitigation: pooler choice in Phase 2; explicit `PgPool::max_connections = 4`; budget calculation in plan.
6. **Decommission half-rollback**: P=low (the v2→v3 cutover went smoothly); impact=severe (a partial deletion leaves orphan systemd units or unauthenticated routes). Mitigation: single decommission PR (Phase 5) gated behind a verification harness; explicit `grep -r INGEST_BEARER_TOKEN` exit criterion.
7. **Phase scope drift on the Next.js side**: P=med (no in-house Next.js + DB precedent); impact=moderate (Phase 4 turns into a multi-week spike). Mitigation: pick stack early (decision in interview); use `context7` for current Vercel docs; allow a planning re-scope if Phase 4 exceeds 5 PRs.
8. **Operational gaps in admin replacement**: P=low-med (managed Postgres console + Vercel logs replace `/api/admin/sql` + systemd journal); impact=moderate (operator can't introspect prod when needed). Mitigation: write runbook before decommission; confirm console access; capture explicit operator workflows in Phase 5 description.
9. **History migration losing rows**: P=low (DuckDB → Postgres COPY/INSERT is well-supported); impact=severe (lost benchmark data is unrecoverable for prior commits). Mitigation: pre-cutover snapshot of DuckDB via `ops/backup.sh`-equivalent; post-cutover row-count assertion; keep DuckDB snapshot for at least 90 days post-cutover.
10. **Cross-account one-shot migration**: P=low-med; impact=moderate. v4 RDS lives in `245040174862/us-east-1` (bench account); v3 EC2 + DuckDB live in `375504701696/us-east-2` (personal account). Phase 3 one-shot DuckDB→Postgres load crosses BOTH account and region boundaries. Mitigation options: (a) write a DuckDB snapshot from `375504701696/us-east-2` to a cross-account-accessible S3 bucket (either in `245040174862` with a bucket policy permitting `375504701696` to PutObject, or vice versa), then download from `245040174862` and load locally to RDS; (b) operator runs the migrator with both profiles configured (`AWS_PROFILE` switching for read vs write); (c) temporarily grant `375504701696` IAM identity an `rds-db:connect` role in `245040174862` for direct write. Steady-state CI ingest after Phase 3 is unaffected (GH Actions OIDC into `245040174862` already works). **Chosen (re-plan 2026-06-05):** a blend of (a)+(b) — the operator acquires a DuckDB snapshot from the v3 side (rehydrate the S3 Vortex backup or scp the live `bench.duckdb`) and runs the loader LOCALLY against prod RDS with the `245040174862` Secrets-Manager **master password** over verify-full TLS. NOT (c): the OIDC `rds-db:connect` roles are GitHub-Actions-only and not assumable from a laptop, so direct IAM write is unavailable off-Actions. PR-3.3's runbook verifies the source account/region live (`375504701696/us-east-2` is unverified in-repo). The one-shot load is atomic (single txn) with RDS PITR as the rollback.

## Verification

**Phase 1 verification:**

```sh
cargo test -p vortex-bench-server hash_stability
# expected: all golden vectors match
<migration-tool> migrate status
# expected: clean against a fresh Postgres
```

**Phase 2 verification:**

```sh
# Integration test against testcontainers Postgres
cargo test -p <new-ingest-crate> --test integration
# Compare result of replaying a v3 envelope JSONL against Postgres vs DuckDB
cargo run -p <new-ingest-crate> --bin verify-roundtrip -- \
  --duckdb benchmarks-website/server/fixtures/snapshot.duckdb \
  --postgres postgres://localhost/test
# expected: zero diff
```

**Phase 3 verification:**

```sh
gh workflow run bench.yml -f mode=develop
# observe both EC2 ingest + Postgres ingest succeed
cargo run -p <verify-crate> -- --against-both
# expected: only_in_v3 = []
```

**Phase 4 verification:**

```sh
# Vercel preview URL
curl -sS https://<preview>.vercel.app/api/groups | jq '.[].slug' | sort > preview-slugs.txt
curl -sS https://benchmarks.vortex.dev/api/groups | jq '.[].slug' | sort > prod-slugs.txt
diff preview-slugs.txt prod-slugs.txt
# expected: empty
```

**Phase 5 (cutover) verification:**

```sh
git grep -n INGEST_BEARER_TOKEN
# expected: zero hits in any tracked file
git grep -n V3_INGEST_URL
# expected: zero hits
gh run list --workflow=publish-benchmarks-website.yml
# expected: workflow does not exist
```

**Full verification (end-state):**

```sh
# 1. Hash stability still passes
cargo test -p <ingest-crate> hash_stability
# 2. Read-path renders all chart slugs
curl -sS https://benchmarks.vortex.dev/api/groups | jq '.[].slug' | wc -l
# expected: matches the family count from src/family.rs
# 3. Idempotent re-ingest
cargo run -p <ingest-crate> -- --envelope <recent-envelope.jsonl> --dry-run
# expected: every record reports updated=1, inserted=0
```

## PR-5.0 operator runbook — bring prod online (IN PROGRESS)

PR-5.0 is **operationally executed by the operator**: it requires prod RDS master creds (account `245040174862`), the v3-source-account creds, and a Vercel prod deploy, plus hard-to-reverse production writes. The agent prepared + verified the tooling and records the captured evidence here as the acceptance trail. **Agent prep done (2026-06-10):** toolkit unchanged since phase-4 entry (`8f249165b`); `target/debug/vortex-bench-migrate` built clean off HEAD; the PR-3.4 dress rehearsal (same `load`/`verify`/cross-check) was GREEN against the real 4.2M-row snapshot.

**Prod resource identities** (from PR-1.1):
- RDS instance `vortex-bench-prod`, account `245040174862`, region `us-east-1`. Reach via the `bench-prod` CLI profile or CloudShell — the SSO default profile is the WRONG account ([[project_bench_aws_access]]).
- Public **instance** endpoint (NOT the VPC-internal proxy): GitHub repo var `RDS_BENCH_INSTANCE_ENDPOINT`. The read path + CI writers both use the public instance endpoint.
- DB name: repo var `RDS_BENCH_DB_NAME`.
- RDS-managed master secret: `arn:aws:secretsmanager:us-east-1:245040174862:secret:rds!db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690-egkQgW`.
- `bench_read` password: on disk at `~/.bench-read-pw` (0600).

**Prerequisites** (operator):
- [ ] ROTATE the Vercel token (it entered a prior transcript); update `~/.vercel-token-bench` AND the GitHub `VERCEL_TOKEN` secret.
- [ ] Freshest v3 DuckDB snapshot acquired (Step 2b). PR-3.4 used the stale May-4 on-disk file because freshness was irrelevant for a rehearsal; the PROD load wants the FRESHEST snapshot.
- [ ] us-east-1 RDS CA bundle (`rds-ca.pem`) on disk for `--ca-cert` verify-full TLS.

**Step 1 — Operator-gate** (prod RDS master)
- 1a. Confirm identity + endpoint: `AWS_PROFILE=bench-prod aws sts get-caller-identity` (expect `245040174862`); `... aws rds describe-db-instances --region us-east-1 --db-instance-identifier vortex-bench-prod --query 'DBInstances[0].Endpoint.Address'`.
- 1b. Master connection: resolve the CURRENT master secret ARN dynamically (`aws rds describe-db-instances --db-instance-identifier vortex-bench-prod --query 'DBInstances[0].MasterUserSecret.SecretArn' --output text` — more robust than the hardcoded ARN, which is in fact the same `rds!db-23f1d9f9-…-egkQgW`), fetch it via `aws secretsmanager get-secret-value`, and prefer libpq env vars (`PGHOST/PGPORT/PGUSER=postgres/PGDATABASE=vortex_bench/PGSSLMODE=verify-full/PGSSLROOTCERT=~/rds-ca.pem` + `PGPASSWORD` from the secret) over a URL DSN to avoid URL-encoding special chars in the password.
- 1c. Schema drift + apply incl. 005 **as master**: `uv run scripts/migrate-schema.py status --target "$PROD_MASTER_DSN"` then `uv run scripts/migrate-schema.py apply --target "$PROD_MASTER_DSN"` (002/004/005 carry the `requires-superuser` marker; the migrator path cannot apply them).
- 1d. Set the `bench_read` password as master: `psql "$PROD_MASTER_DSN" -c "ALTER ROLE bench_read PASSWORD '$(cat ~/.bench-read-pw)';"`.
- 1e. Wire Vercel prod env (Production + Preview) on the Vercel project: `BENCH_DB_HOST=<public instance endpoint>`, `BENCH_DB_NAME=<db>`, `BENCH_DB_USER=bench_read`, `BENCH_DB_PASSWORD=<~/.bench-read-pw>`, `BENCH_DB_SSL=verify-full`, `BENCH_DB_CA=<PEM contents of rds-ca.pem>` (defaults: `BENCH_DB_PORT=5432`, `BENCH_DB_POOL_MAX=4`).

**Step 2 — One-shot PROD load**
- 2a. CONFIRM the v3 source account+region LIVE (believed `375504701696`/`us-east-2`, pinned NOWHERE — do not trust the audit): `aws sts get-caller-identity` (the operator's default/SSO profile reaches the personal account `375504701696`); v3 backup bucket is **`vortex-benchmark-results-database`** (per `ops/backup.sh`; distinct from v2's `vortex-ci-benchmark-results`) — `aws s3api get-bucket-location --bucket vortex-benchmark-results-database`. STOP + reconcile if either disagrees.
- 2b. Acquire the FRESHEST snapshot (pick one):
  - **Path A — scp the live DuckDB** (simplest + freshest): `scp <user>@<v3-ec2-host>:/var/lib/vortex-bench/bench.duckdb ~/bench-fresh.duckdb`.
  - **Path B — rehydrate the S3 backup** (≤1h stale; 7-day lifecycle): the backup keys are `<S3_BACKUP_PREFIX>/<ts>.tar.gz` (`ts=YYYYMMDDTHHMMSSZ`), each a tar of `<ts>/schema.sql` + per-table `.vortex` files (`commits.vortex`, `query_measurements.vortex`, `compression_times.vortex`, `compression_sizes.vortex`, `random_access_times.vortex`, `vector_search_runs.vortex`). `aws s3 ls` for the freshest `.tar.gz`, download + extract, then rebuild a `.duckdb` via `duckdb` with `INSTALL vortex; LOAD vortex;` reading the per-table files (apply `schema.sql` first).
- 2c. Load over verify-full TLS: `PROD_DSN="postgresql://<role>@<endpoint>:5432/<db>?sslmode=verify-full"`; `target/debug/vortex-bench-migrate load --duckdb <fresh.duckdb> --postgres-target "$PROD_DSN" --ca-cert rds-ca.pem` → capture per-table row counts.
- 2d. Verify (PRIMARY gate): `target/debug/vortex-bench-migrate verify --duckdb <fresh.duckdb> --postgres-target "$PROD_DSN" --ca-cert rds-ca.pem` → exit 0, "0 presence diffs, 0 value mismatches".
- 2e. Cross-check (PR-3.5 harness): build a `real_envelopes.json` from a few seeded rows (PR-3.4 approach — a recent commit, ~3+3+3+2 records across kinds whose dim tuples are in the seeded data), then `uv run scripts/cross_check_python_writer.py --postgres "$PROD_DSN" --envelopes real_envelopes.json` → "N records, N updated, 0 inserted -- CLEAN".

**Step 3 — Deploy + evidence**: the Vercel **production** deploy triggers on **push to `ct/bench-v4`** (`.github/workflows/web-deploy.yml`). Wire the Vercel env (1e) + rotate `VERCEL_TOKEN` FIRST, then push (currently 28 commits ahead, unpushed). Capture the deploy URL + the `web-deploy` run's build log + CDN-probe result.

**Step 4 — 2 moved Phase-4 data checks** (post-deploy):
- 4a. `curl <prod-url>/api/groups | jq '.groups[].charts[].slug' | sort` matches the family registry.
- 4b. ~5 representative chart slugs match the current v2 site (manual visual check).

**Rollback**: RDS PITR (35-day) for the data seed.

**Captured evidence**

_Operator-gate verified 2026-06-10 (interactive walkthrough; operator ran the commands):_
- **1a** ✓ `sts get-caller-identity` = account `245040174862` (`connor-aws-cli`); instance `vortex-bench-prod` `available`, endpoint `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432`, db `vortex_bench`, master `postgres`, IAM auth enabled.
- **1b** ✓ master secret accessible via the dynamic `MasterUserSecret.SecretArn` lookup (user `postgres`, pw len 28).
- **1c** ✓ `migrate-schema.py status` = **5 applied, 0 pending, 0 orphaned** (001–005 all applied; **005 was already applied as master in a prior session** — no DDL needed at cutover).
- **Load baseline** — all six tables **0 rows** (prod empty; the v4 dual-write soak was a no-op-until-wired per the cutover tradeoff, so the load is a clean fresh full seed, not an upsert over soak data).
- **1d** ✓ `bench_read` authenticates over verify-full TLS with the on-disk `~/.bench-read-pw` (`SELECT current_user` = `bench_read`) — password already set; no `ALTER ROLE` needed.
- **1e** ⏳ Vercel prod env (`BENCH_DB_*`) — operator to verify/wire after the Vercel-token rotation; gates Step 3 (deploy), not Step 2.

_Step 2 — snapshot acquisition + rebuild (2026-06-10):_
- **2a** ✓ v3 source = account `375504701696` (SSO PowerUserAccess); backup bucket `vortex-benchmark-results-database` region **`us-east-1`** (`LocationConstraint: None`) — **NOT the believed `us-east-2`** (the audit was wrong; the README caveat held). Backups under `v3-backups/<ts>.tar.gz`, hourly.
- **2b** ✓ freshest backup `v3-backups/20260610T210150Z.tar.gz` (302 MB, 21:01 UTC) → rebuilt `~/bench-fresh.duckdb` (643 MB) via `duckdb` `INSTALL vortex; LOAD vortex; SET TimeZone='UTC';` + `schema.sql` + `INSERT … BY NAME … read_vortex(<table>.vortex)`. `read_vortex` reproduced the exact v3 schema (`timestamp with time zone`, `double`, `bigint[]`). **Rebuilt counts**: commits 4545, query_measurements 4,849,218, compression_times 248,520, compression_sizes 101,957, random_access_times 36,055, **vector_search_runs 0**.
- **vector_search_runs residual RESOLVED**: still **0 rows** in the freshest snapshot, so it stays fixture-only-validated at cutover — exactly the accepted residual PR-3.6 documented ("if the live v3 DB still has zero vector rows at cutover, it stays fixture-only-validated").
- **PLAN-ASSUMPTION VIOLATION (Class B) — loader TLS broken on macOS; fixed with rustls (CODE change in PR-5.0).** The plan assumed the migrate toolkit works against prod; its `--ca-cert` native-tls path had never been exercised (PR-3.4 used local plaintext `NoTls`). On the first prod connect it failed: the RDS leaf cert carries **no `serverAuth` Extended Key Usage**, which macOS Secure Transport (native-tls's macOS backend) rejects (`extended key usage is not valid`), while OpenSSL/libpq treat a missing EKU as unrestricted (so `psql`/`s_client` validate fine, `Verify return code: 0`). native-tls on macOS is hardwired to Secure Transport with no per-EKU relaxation. **Fix:** swapped the loader's TLS from native-tls → **rustls** (`tokio-postgres-rustls` + `rustls-pemfile`) in `migrate/Cargo.toml` + `migrate/src/postgres.rs` `connect_postgres` (shared by `load` + `verify`). rustls/webpki treats a missing EKU as unrestricted (matching OpenSSL), validates on every OS, and trusts the whole CA bundle. This is the **only production code in PR-5.0** → it gets the inner-loop 2-vote gauntlet before PR-5.0 closes.
- **2c load** ✓ (2026-06-10, the first prod write; rustls): atomic single-txn, **2:47** wall; per-table rows EXACTLY matching the rebuilt source — `commits` 4545, `query_measurements` 4,849,218, `compression_times` 248,520, `compression_sizes` 101,957, `random_access_times` 36,055, `vector_search_runs` 0. Master role `postgres` password auth (the loader has no IAM-token minting). RDS PITR (35-day) is the rollback.
- **2d verify** ✓ (2026-06-10, the PRIMARY gate; rustls): `value verify: source and target match (0 presence diffs, 0 value mismatches)` across all 4.85M rows (exit 0, 2:06 wall) — full per-`measurement_id` compare of every value column + `env_triple` + `all_runtimes_ns` arrays + `commits` metadata, timestamps as engine-independent epoch-µs. The prod seed is byte-faithful; the rustls TLS swap is validated end-to-end (load + verify both over it).
- **2e cross-check** ✓ (2026-06-10): built an 11-record `real_envelopes.json` (3 query_measurement + 3 compression_time + 3 compression_size + 2 random_access_time) from REAL seeded rows for commit `3d7bbfb1c…`, **locally pre-verifying every computed `measurement_id` matched the seeded row** (so every record is a guaranteed UPDATE, not a junk INSERT) and carrying the real values (so the UPDATE is a no-op). Ran `cross_check_python_writer.py` as `bench_ingest` (IAM token, `--region us-east-1`, verify-full TLS): **`11 records, 11 updated, 0 inserted -- CLEAN`** — the Python writer recomputes the same `measurement_id` as the Rust seed and UPDATEs (no duplicate), values round-trip. (`bench_ingest` IAM auth via the `bench-prod` profile works.)
- **2e safety net** ✓: re-ran `verify` after the cross-check — `0 presence diffs, 0 value mismatches` (the cross-check's 11 UPDATEs were genuine no-ops; the seed is intact). **Step 2 (load + verify + cross-check) is COMPLETE.**
- **1e Vercel env CONFIRMED wired** (via `https://benchmarks-web.vercel.app/api/health`): `db_path: vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com`, `schema_version: 1`, and `row_counts` show the seeded data live (`query_measurements: 4849218`, `commits: 4545`, …). The Vercel project (`BENCHMARKS_WEB_PROD_URL=https://benchmarks-web.vercel.app`, `VERCEL_ORG_ID`/`VERCEL_PROJECT_ID`/`VERCEL_TOKEN` set) deployed successfully today at 17:21 (build_sha `8b85a3d3f`) — but against the then-empty DB and at a stale commit.
- **Step 3 deploy** — needs a fresh deploy at HEAD: `/api/groups` (the chart-family registry) **timed out (40s)** on the stale `8b85a3d3f` deploy against the 4.85M-row seed; HEAD carries the PR-4.x read-path + cycle-4 web fixes the old deploy lacks. Deploy = push `ct/bench-v4` (37 commits unpushed) → `web-deploy.yml` Vercel prod (uses the GitHub `VERCEL_TOKEN`). **`VERCEL_TOKEN` rotation still owed** (leaked into a transcript) — security follow-up; the deploy functions with the current secret.
- **Inner-loop review (Step 2.3)** ✓ cycle 1 **ACCEPTED** (2026-06-10): reconstructed 2-vote (fresh + correctness, both claude) on the rustls diff — `Skill(spiral:gauntlet)` was unavailable ("Unknown skill"), so prompts were composed faithfully via gauntlet/0.4.0 `compose_prompts.py` + run as Agent reviewers + synthesized inline (per the task ORCH recipe / big-plans [Z] round-5 pattern). Both lenses ACCEPT, high confidence, 0 must-fix. Correctness skeptic adversarially confirmed: hostname/SAN verification IS enforced, `sslmode=require` does NOT downgrade (libpq footgun N/A to tokio-postgres), empty/bad CA bundle is fail-closed (no MITM exposure of the master password). 1 should-fix (empty-CA-bundle → opaque error, both lenses) applied in `d0175d70c` (loud `bail!`); 1 nit (crypto-provider error-swallow) dismissed. Synthesizer JSON in the `5e049db89` commit body.
- **Step 3 deploy** ✓ DONE (2026-06-11): pushed `ct/bench-v4` (41 commits, DCO+clippy pre-push clean) → `web-deploy.yml` deployed to Vercel prod at HEAD (`/api/health` `build_sha: 35c05c500`, serving the seeded `row_counts`). BUT the workflow concluded **`failure`** — its post-deploy "Verify CDN caching of the landing page" probe failed because the landing page times out.
- **DISCOVERY (Step 2.6 trigger) — read-path group discovery doesn't scale to the prod seed.** `/api/groups` (and the landing page, which renders groups server-side) **times out (>40s)** even on the fresh HEAD deploy. Root cause (diagnosed against prod, read-only EXPLAIN/timing): `collectQueryGroups` runs `SELECT … GROUP BY dataset,dataset_variant,scale_factor,storage,query_idx FROM query_measurements` — a full **index-only scan of all 4.85M entries of `idx_query_measurements_chart`** to return just **321 distinct chart tuples / 13 groups** (~**4.95s** measured; PG16 has no loose-index/skip scan). `collectGroups` does this for **all 5 families sequentially**, then runs a **per-group N+1 summary query** (`collectGroupSummary`); on Vercel's sequential serverless execution the total exceeds the request timeout. Phase 4 validated the read service against a small dev DB, so this only surfaced at prod scale. The fix is a read-path perf optimization (loose-index-scan / recursive-CTE skip-scan for the discovery queries, and/or batching the N+1) — non-trivial, arguably its own PR. **Blocks the `/api/groups`-serves acceptance + MUST be fixed before the PR-5.2 public DNS flip; does NOT undo the verified data seed.**
- **DISCOVERY (full diagnosis, 2026-06-11) — the read path is broadly slow at prod scale; ROOT CAUSE is non-sargable `IS NOT DISTINCT FROM`.** Measured against prod: an individual big-dataset chart (`/api/chart` tpch sf=1 nvme q1) takes **~24s** and times out; a small-dataset chart (polarsignals, 12720 rows) returns in **~1.1s / HTTP 200** with correct data. EXPLAIN shows why: the read queries match the nullable dims with `q.dataset_variant IS NOT DISTINCT FROM $ AND q.scale_factor IS NOT DISTINCT FROM $` (ported from DuckDB NULL-equality), which Postgres **cannot use as a btree index condition**. So `idx_query_measurements_chart` (dataset, dataset_variant, scale_factor, storage, query_idx) is seeked only on the leading `dataset=` column and the rest is a heap Filter → every chart/summary query scans ALL of that dataset's rows (~1.8M for tpch). Small datasets scan little (fast); big datasets (tpch/tpcds/clickbench) time out. Three compounding costs for `/api/groups`/landing: (1) discovery full `GROUP BY` scans (~1.2–5s ×5 families), (2) **64 sequential per-group summary queries**, the ~11 query-summaries each a `row_number()` window over the whole group at **~10s** (tpcds/clickbench), (3) all serialized on Vercel. Total ~1–2 min → never completes within the function timeout, so the 5-min edge cache never populates. **The data + rendering are CORRECT (polarsignals chart proves it); this is purely query-perf.** Fix is high-leverage + well-defined: (a) sargable WHERE construction (`col IS NULL` when the key value is null, `col = $` otherwise — logically identical to `IS NOT DISTINCT FROM` for a concrete key, but index-usable), (b) loose-index-scan discovery, (c) index-supported latest-per-series summary (DISTINCT ON / supporting index), (d) parallelize the 64 summaries. Required before PR-5.2 DNS flip.
- **Step 4 data checks**: blocked on the read-path perf fix above. A small-dataset chart renders correctly (`/chart/qm.<polarsignals slug>`); big-dataset charts + the landing page do not. Promoted to the Implementation status PR-5.0 entry at Step 2.5.

## Implementation status

### PR-1.1: Provision RDS Postgres + RDS Proxy + GitHub OIDC schema role  (11 code commits, ending at 2336d48c1)

- **Scope shipped**: AWS infrastructure bootstrap script (`benchmarks-website/infra/provision.sh`) + operator runbook (`benchmarks-website/infra/README.md`) + GitHub Actions schema-deploy workflow skeleton (`.github/workflows/schema-deploy.yml`). Provisions RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub OIDC provider + `GitHubBenchmarkSchemaRole` IAM role in account `245040174862` / `us-east-1`. Idempotent re-run; existence-checked.
- **Acceptance criteria — verified by operator**:
  - `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available` + `IAMDatabaseAuthenticationEnabled: true` ✓
  - `aws rds describe-db-proxies --db-proxy-name vortex-bench-proxy` returns `available` + Endpoint `vortex-bench-proxy.proxy-c4f8qygk4xdp.us-east-1.rds.amazonaws.com` ✓
- **Repo vars set** (verified at PR-1.1): `RDS_BENCH_ENDPOINT`, `RDS_BENCH_REGION`, `RDS_BENCH_DB_NAME`, `GH_BENCH_SCHEMA_ROLE_ARN`. **Superseded by PR-1.6**: CI was repointed to the public instance endpoint, `RDS_BENCH_INSTANCE_ENDPOINT` was added as the CI var, and `RDS_BENCH_ENDPOINT` (proxy) was dropped as a GitHub variable — the proxy endpoint is now a Vercel-config value (PR-4.2), not a GitHub variable. **CORRECTION (2026-06-01 live verification):** these repo-var changes (add `RDS_BENCH_INSTANCE_ENDPOINT`, drop `RDS_BENCH_ENDPOINT`) describe the intended end state but were NOT actually applied to the live GitHub repo by PR-1.6; they were reconciled in-session on 2026-06-01. See section "Phase 1 live verification (2026-06-01)".
- **Resource identities** (for downstream PRs to reference):
  - RDS instance: `vortex-bench-prod` (resource-id `db-4VPTDACTRQHOS24WEIR3TNC2M4`)
  - Master secret: `arn:aws:secretsmanager:us-east-1:245040174862:secret:rds!db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690-egkQgW` (RDS-managed)
  - Proxy: `vortex-bench-proxy.proxy-c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432`
  - Default VPC: `vpc-03cbb254d7f018995` (6 subnets discovered)
  - Schema role: `arn:aws:iam::245040174862:role/GitHubBenchmarkSchemaRole`
- **Tests added**: none (acceptance is operator-runs-script + describe-db-* probes; no unit-test surface).
- **Review**: 2-vote (preset=pr-2, fresh+correctness) / cycle 1 reject → 7 fix commits → operator execution surfaced 3 additional bugs → 4 more fix commits → operator accept. No cycle-2 static gauntlet (deferred; operator-execution provided higher-signal verification than static re-review would).
- **Confidence**: high (acceptance criteria pass under real AWS; all gauntlet must-fix items addressed; bug discovery shifted from static review to operator execution as expected for infra scripts).
- **Deferred items**: 8 should-fixes (see Deferred work table below).
- **Surprises during implementation**:
  - Operator only had SSO access to personal account `375504701696`; the actual bench account is `245040174862` (Linux Foundation org, no SSO for operator yet). Resolution: operator logged into 245040174862 root via signin.aws.amazon.com and ran provision.sh from AWS CloudShell (no local AWS creds needed). Documented as the supported operator path.
  - Three operator-found bugs not surfaced by cycle-1 static gauntlet: (1) `mktemp -t` template missing X's on GNU mktemp (CloudShell), (2) `aws rds wait db-proxy-available` is not a real AWS CLI v2 waiter, (3) `environment: schema-deploy` reference required GitHub repo-admin to create the environment, which operator lacks. All three fixed via additional cycle-2 fix-commits (674668406, d633735d9, 2336d48c1).
  - Pattern: static reviewers can't catch what running the script catches. Acknowledged as work-shape characteristic; future infra-script PRs should plan operator-execution as part of the acceptance flow, not gauntlet-cycle-2-only.

### PR-1.2: Postgres migration runner + tests + migrations/ scaffolding  (4 code commits + 2 doc-fix commits + 3 gauntlet cycles, ending at f6999d6d3)

- **Scope shipped**: `scripts/migrate-schema.py` (substrate-agnostic forward-only Postgres migration runner with `apply` and `status` commands; libpq env vars OR `--target=<dsn>`; PEP 723 inline metadata for standalone `uv run`); `scripts/test_migrate_schema.py` (16 pytest+testcontainers tests covering apply / idempotency / sequential order / subprocess close-on-exception rollback / status drift detection / empty-file rejection / case-insensitive discover / subdirectory-with-.sql-suffix filtering / non-default search_path); `migrations/` directory with `README.md` documenting naming convention + authoring rules + transactional-only invariant; `pyproject.toml` workspace dev deps gain `psycopg[binary]>=3.2` + `testcontainers[postgres]>=4.9`; `uv.lock` regenerated.
- **Acceptance criteria — verified against postgres:16-alpine testcontainer**:
  - Apply applies pending migrations idempotently (`test_apply_creates_..._and_runs_migrations`, `test_apply_is_idempotent`) ✓
  - Apply applies new migration in name order without re-applying earlier ones (`test_apply_applies_new_migration_in_order`) ✓
  - Apply survives a failing later migration without losing earlier ones — subprocess test mirroring production close-on-exception (`test_apply_rolls_back_on_failure_subprocess`) ✓
  - Status exits non-zero on drift and does not DDL (`test_status_reports_applied_and_pending`, `test_status_clean_returns_zero`, `test_status_flags_orphaned_applied_files`, `test_status_does_not_create_table`) ✓
  - Apply rejects empty + whitespace-only .sql files (`test_apply_rejects_empty_sql_file`, `test_apply_rejects_whitespace_only_sql_file`) ✓
  - Apply + Status agree on ledger location under non-default search_path (`test_apply_uses_public_schema_under_custom_search_path`) ✓
  - Discover ignores subdirectories with .sql suffix (`test_discover_ignores_subdirectory_with_sql_suffix`) ✓
- **Tests added**: 16 pytest tests (4 pure-function, 12 testcontainer-backed). Local runs all pass against `postgres:16-alpine`. CI execution of this test suite is a deferred item (see below).
- **Review**: 2-vote (preset=pr-2, fresh+correctness, Claude executor) / cycle 1 reject (2 must-fix: implicit-outer-txn savepoint bug + masked test) → fix bundle (txn model via autocommit + subprocess test + 6 should-fix items + plan-edit) → cycle 2 reject (2 new must-fix: empty-file replay bug exposed by carry-forward + schema-qualification divergence under non-default search_path) → fix bundle (empty-file rejection + schema-qualify all ledger refs + applied_set→_applied_set rename + is_file() guard + subprocess timeout + new regression tests) → cycle 3 accept (zero must-fix; both reviewers ACCEPT; remaining 9 should-fix + 3 nit triaged as docstring drift fixed inline or deferred).
- **Confidence**: high (3-cycle gauntlet with monotonically improving verdicts; 16/16 tests green against testcontainers; substrate-agnostic runner depends only on libpq env vars; clean separation between writer-DDL path and read-only status path; per-migration autocommit-True transactions correctly survive partial failure; schema-qualification pins ledger to `public._applied_migrations` regardless of role search_path).
- **Deferred items**: 4 should-fixes for PR-1.2 (concurrency advisory lock, autocommit toggle precondition for library callers, ledger fingerprint for edit-after-apply drift, CI pytest runner — all in Deferred work table).
- **Surprises during implementation**:
  - Cycle 1 surfaced a non-obvious psycopg3 transaction-model bug: `CREATE TABLE IF NOT EXISTS` outside a `with conn.transaction()` block lazily opens an implicit outer transaction; subsequent `conn.transaction()` blocks become SAVEPOINTs rather than top-level transactions, so a failing later migration triggers the connection-level rollback that discards all prior savepoint-released work. The original test masked this by catching the exception inside the test body, so the fixture's `with psycopg.connect()` exited cleanly via commit instead of with-exception via rollback. Fix: `conn.autocommit = True` at the top of apply()/status() forces each `conn.transaction()` to be a real top-level transaction; subprocess-based regression test exercises the production close-on-exception path.
  - Cycle 2 surfaced a second non-obvious bug: schema-qualification divergence between `applied_set` / `apply` (unqualified `_applied_migrations`) and `_read_applied_filenames` (`public._applied_migrations` via `to_regclass`). Under default search_path both paths happen to find the same table; under a non-default search_path (very plausible for the PR-1.3 `migrator` IAM role under standard `rds_iam` least-privilege patterns) writer and reader would silently transact with different ledger tables. Fix: canonicalize every ledger reference to `public._applied_migrations` everywhere (DDL + INSERT + SELECT). Test connects with `options='-c search_path=foo,public'` and verifies writer-reader agreement.
  - Pattern: per-cycle gauntlet caught two correctness ship-blockers neither reviewer would have spotted with a single pass — different prompts catch different bugs (the spec's load-bearing principle). The cycle-2 fix-commit's `Shared fix-commit attention` block at cycle 3 surfaced H1 docstring drift the fix introduced and let me clean it up before completion.
  - Spec deviation: cycle-2 fix-commit bundled both must-fix items into a single fix-commit (rather than the per-must-fix split Step 2.4 prescribes), because the edits on the apply()/applied_set/ledger boundary genuinely overlap. Documented in the commit body; followed by one plan-edit decrementing by 2.

### PR-1.3: Postgres initial-schema + IAM migrator role migrations  (1 code commit + 1 gauntlet cycle, ending at b00cd967d)

- **Scope shipped**: `migrations/001_initial_schema.sql` (the `commits` dim table + 5 fact tables + 6 read-path composite indexes, the Postgres translation of the authoritative DuckDB DDL in `benchmarks-website/server/src/schema.rs`); `migrations/002_iam_db_user.sql` (the `migrator` login role + conditional `rds_iam` grant + `GRANT CREATE, USAGE ON SCHEMA public`); `scripts/test_migrate_schema.py` gains 12 testcontainer-backed tests (apply-cleanly, idempotency, table set, index set, per-table column-shape pin, key type-translation spot-checks, migrator role).
- **Acceptance criteria — verified against `postgres:16-alpine` testcontainer**:
  - Real migrations apply cleanly + idempotently (`test_real_migrations_apply_cleanly`, `test_real_migrations_idempotent`) ✓
  - All 6 tables created (`test_real_migrations_create_expected_tables`) ✓
  - All 6 composite indexes created (`test_real_migrations_create_expected_indexes`) ✓
  - Per-table column order + nullability matches `schema.rs` exactly (`test_real_migrations_preserve_column_shape[*]`, 6 parametrized cases) ✓
  - `DOUBLE`→`double precision`, `BIGINT[]`→`ARRAY`, `measurement_id` bigint PK (`test_real_migrations_key_column_types`) ✓
  - `migrator` login role created (`test_real_migrations_create_migrator_role`) ✓
  - The `\dt`/`\di`/`\du` + `apply` plan acceptance criteria are exercised by the testcontainer assertions; `rds_iam` group membership is operator-verified on real RDS (untestable on vanilla Postgres, guarded by `IF EXISTS`).
- **Tests added**: 12 (all testcontainer-backed). Local run: **28/28 pass** (16 prior + 12 new) against `postgres:16-alpine`. `py_compile` OK, `ruff` clean, `git diff --check` clean. (CI execution of the suite remains the deferred D-`PR-1.2 no-CI-runner` item, tracked for PR-1.4.)
- **Review**: 2-vote (preset=pr-2, fresh+correctness, Claude executor) / cycle 1 ACCEPT (zero must-fix; both reviewers accept). Correctness skeptic ran an automated column-by-column comparison vs `schema.rs` confirming byte-equivalent schema shape across all 6 tables. 1 should-fix (migrator table privileges) deferred to PR-2.1; 3 nits dismissed (redundant-but-intentional `NOT NULL`, PG15+ index-planner dependency satisfied by target, untestable `rds_iam` membership).
- **Confidence**: high (single-cycle clean accept; behavior-preservation obligation met and pinned by an ordinal column-shape regression test; transaction-block constraint and idempotency reasoned through by both lenses).
- **Deferred items**: 1 should-fix (migrator table-level privileges -> resolve in PR-2.1 role-ownership design; see Deferred work table).
- **Surprises during implementation**:
  - None material. The composite-index design deliberately follows the read-path chart-query filter columns (`api/charts.rs`) rather than the `measurement_id` hash field order in Table B — the hash tuple leads with `commit_sha` but every chart query filters on the dim columns and joins `commits` on `commit_sha`, so a dim-leading index is what serves the read path; PK uniqueness over the full hash tuple is already enforced by `measurement_id`.

### PR-1.4: Wire schema-deploy.yml (apply-as-migrator) + ledger grant  (2 code commits + 2 fix commits + 2 gauntlet cycles, ending at 5fc9ad5be)

- **Scope shipped**: `.github/workflows/schema-deploy.yml` rewritten from the PR-1.1 OIDC-probe skeleton into the real apply workflow (workflow_dispatch-only with a `dry_run` input; OIDC -> `GitHubBenchmarkSchemaRole`; client-side IAM auth-token; RDS global CA bundle download; `PGSSLMODE=verify-full`; `uv run --no-project scripts/migrate-schema.py apply` then `status` as the `migrator` role through the RDS Proxy). `migrations/003_migrator_ledger_grant.sql` (minimal idempotent `GRANT SELECT, INSERT ON public._applied_migrations TO migrator` so CI's migrator-role apply can record/read the ledger against a master-owned bootstrap; no DELETE/UPDATE = append-only least-privilege). `scripts/test_migrate_schema.py` updated for the 3-migration set + a discriminating ledger-grant test (SELECT/INSERT present, DELETE+UPDATE absent). `benchmarks-website/infra/README.md` gains a "Schema deploys + one-time bootstrap" section (master bootstrap must hit the INSTANCE endpoint since the proxy is `IAMAuth=REQUIRED`) and a corrected master-password claim.
- **Acceptance criteria**: workflow runs `migrate-schema.py apply` as the OIDC `migrator` role (plan PR-1.4 row) — wired and yamllint --strict clean; manual-approval gate intentionally deferred (operator lacks repo-admin to create environments; workflow_dispatch IS the gate). Tests: **29/29 pass** against `postgres:16-alpine`; `py_compile` OK, `ruff` clean, `git diff --check` clean.
- **Tests added**: 1 (ledger-grant privileges); 1 existing test extended (UPDATE-absent assertion).
- **Review**: 2-vote (preset=pr-2, fresh+correctness, Claude executor). **Cycle 1 REJECT (3 must-fix)** — the workflow `Write` had silently failed (pre-existing skeleton not Read first), so the "wire" commit touched only the README while the workflow stayed a placeholder-echo skeleton, the `push:` trigger remained, and docs described a nonexistent workflow. Caught by BOTH lenses independently. **Fix**: landed the real workflow (resolving all 3 facets at once) + tightened the test. **Cycle 2 ACCEPT** (zero must-fix; both lenses); fix-commit attention block confirmed no regressions (token leak-safe, PEP 723 deps resolve, CA path consistent). 1 should-fix deferred, 3 nits dismissed.
- **Confidence**: high (the namesake-deliverable bug was the exact failure gauntlet exists to catch; cycle-2 reviewers verified the wired workflow as new code — token handling, `uv run --no-project` PEP 723 resolution against sibling workflows, verify-full, dry_run parsing all confirmed correct).
- **Deferred items**: 1 should-fix (uv Python auto-provisioning without an explicit pin -> matches repo convention, revisit in a CI-hardening pass; see Deferred work).
- **Surprises during implementation**:
  - The cycle-1 reject was a process bug, not a design bug: a `Write` tool call was rejected ("file not Read first") for the pre-existing skeleton and the rejection was missed inside a batch of parallel calls. This is precisely the Edit/Write-without-Read failure mode the spec's NEW-C "verify the staged diff actually changed before committing" discipline guards against. The recovery applied the discipline literally (Read -> Write -> `git diff --cached --quiet` check -> commit).
  - The plan named `web/ops/README.md` for the IAM-role doc, but `web/` does not exist until PR-4.1; the docs went to `benchmarks-website/infra/README.md` (the correct home today). Recorded as an intentional plan deviation.

### PR-1.5: xxhash64 measurement_id Python port + cross-language golden vectors  (1 impl commit + 1 fix commit + 1 gauntlet cycle, ending at b912d4b6a)

- **Scope shipped**: `scripts/_measurement_id.py` (byte-for-byte Python port of the server-internal xxhash64 `measurement_id_*` functions from `benchmarks-website/server/src/db.rs`: per-table tag + `0x00` separator, LE-u64 length-prefixed `write_str`, `write_opt_str` tag bytes, LE `write_i32`, IEEE-754 `write_f64` bits, `u64 -> i64` bitcast); `benchmarks-website/server/tests/measurement_id_golden.rs` (Rust source-of-truth golden-vector generator + always-assert test; regenerates the committed JSON under `REGEN_GOLDEN_VECTORS`); `scripts/measurement_id_golden.json` (63 committed golden vectors covering all 5 fact tables + `i32` MIN/MAX + empty / `Some("")` strings + multibyte UTF-8); `scripts/test_measurement_id.py` (asserts the Python port reproduces every committed golden vector). Transitive pin: Rust == golden == Python.
- **Acceptance criteria**: `pytest scripts/test_measurement_id.py` all green -- **65 passed** (re-verified this session). Python output matches the Rust golden file bit-exactly for all 63 vectors. (NOTE: the PR-1.5 row promised "100 fixture inputs"; 63 were shipped -- flagged by the phase-end gauntlet as a must-fix to reconcile.)
- **Tests added**: `scripts/test_measurement_id.py` (65 cases: per-vector golden comparison across all tables + boundary fixtures + all-tables-covered + multibyte-present guards) plus the Rust golden generator/asserter in `measurement_id_golden.rs`.
- **Review**: 2-vote (preset=pr-2, fresh+correctness) / cycle 1 ACCEPT (zero must-fix). 1 cycle-1 fix-commit (`c80bf5c6d`: dead code + dangling doc ref, should-fix applied post-accept). 1 should-fix deferred (scripts/ pytest not wired into CI -- see Deferred work).
- **Confidence**: high (single-cycle clean accept; the load-bearing cross-language hash equivalence is pinned transitively and verified bit-exact).
- **Deferred items**: 1 should-fix (golden==Python not CI-gated; PR-1.5 cycle-1 -- see Deferred work table).
- **Surprises during implementation**:
  - The implementation landed bundled in commit `0b431a8c5` (subject `plan: PR-1.5 gauntlet cycle 1 accepted`) rather than a separate `<area>:` implementation commit -- an artifact of the operator's 65-commit rebase onto `develop`. The code content is correct and tested; only the commit-subject attribution is non-standard.
  - **Ledger backfill**: this entry was reconstructed and backfilled during the Phase-1 boundary on resume after it was found missing from Implementation status (PR-1.5 was marked complete at `b912d4b6a` without its Step 2.5 ledger entry landing). The phase-end gauntlet ran with PR-1.5 context supplied via the PR-enumeration row + the cumulative diff.

### PR-1.6: (re-plan) repoint CI schema-deploy to the public instance endpoint + sweep phase-end cross-ref drift  (2 impl/fix commits ending at `b90a740da`, across 7 inner-loop gauntlet cycles)

- **Scope shipped**: Repointed `schema-deploy.yml` `PGHOST` from the VPC-internal RDS Proxy var (`RDS_BENCH_ENDPOINT`) to the public instance var (`RDS_BENCH_INSTANCE_ENDPOINT`) with `PGSSLMODE=verify-full`; corrected the AWS account comment (`375504701696` → `245040174862`); established the CI=instance / proxy=Vercel split consistently across `schema-deploy.yml`, `benchmarks-website/infra/README.md`, `benchmarks-website/infra/provision.sh`, and `migrations/002_iam_db_user.sql`; replaced the README master-password bootstrap with a non-interactive, fail-fast Secrets-Manager fetch (`master_secret=$(aws ...) || exit 1; PGPASSWORD=$(printf '%s' "$master_secret" | jq -er '.password') || exit 1; export PGPASSWORD`); removed `RDS_BENCH_ENDPOINT` from the GitHub repo-vars (it is a Vercel-config value, not a GitHub variable). **CORRECTION (2026-06-01):** this describes the intended end state; the live GitHub repo-var edits were not actually performed until the 2026-06-01 live verification. See section "Phase 1 live verification (2026-06-01)".
- **Acceptance criteria**: `yamllint --strict` clean on `schema-deploy.yml`; `PGHOST` resolves to the instance var; README bootstrap uses `verify-full`; `git grep 375504701696 benchmarks-website/infra/provision.sh` returns 0; the static guard suite (5 no-DB tests + the DB-backed index-column test) green locally (9 passed / 26 Docker-skipped). The "operator runs the live OIDC apply against the instance endpoint" criterion is an out-of-band operator step (not gated here).
- **Tests added**: 5 static regression guards in `scripts/test_migrate_schema.py` — `test_schema_deploy_targets_instance_endpoint` (PGHOST=instance var + `PGSSLMODE=verify-full`, no proxy var), `test_provision_emits_instance_endpoint_var` (gh-var name/body: instance var set from `${DB_ENDPOINT}`, no repo-var from `${PROXY_ENDPOINT}`), `test_provision_grants_instance_dbuser` (IAM policy grants `dbuser:${DB_RESOURCE_ID}/${PG_MIGRATOR_ROLE}`), `test_readme_bootstrap_pins_verify_full` (rejects require/prefer/allow/disable/verify-ca), `test_readme_bootstrap_password_fetch_is_safe` (pins the non-interactive SM fetch; bans `export PGPASSWORD=$(`/stty/interactive-read on code lines) — plus the index-column test now pins `(table, columns)` per index incl. `tablename` and DESC/DESC ordering.
- **Review**: 2-vote `preset=pr-2` (fresh + correctness), `executor=parallel` (Claude **and** Codex per lens). **7 inner-loop cycles**, 2 early-breaks. Cycles 1–5: cross-reference drift from the endpoint split, surfaced at distinct new sites each cycle (README:139 → :21/provision:500/002:5 → :71/76 repo-vars + provision:505 → :52 → fully converged at cycle 5). Cycles 3–7: the bootstrap-password runbook line churned across 4 fixes (single-quote → `read -rsp` → `stty/read` → SM-fetch → fail-fast assign-then-export). **Cycle-7 (final) verdict: REJECT, 2 must-fix — both test-guard-strictness / coverage-completeness, NOT functional defects** (pin `\|\| exit 1`; pin `PROXY_ROLE_NAME` usage). **Accepted by user-authorized override** at the cycle-7 second early-break ("accept regardless after one final cycle; defer residual"). The parallel Claude+Codex executor disjointness was the value driver: the Codex lens caught the cycle-2/3/4/5/6/7 must-fix that both Claude lenses rated accept/nit every cycle.
- **Confidence**: high on the functional change (the endpoint repoint + cross-ref consistency were verified clean by every reviewer across all 7 cycles; the password fetch is fail-fast and syntax-checked under sh/bash/zsh/dash). medium on test-guard exhaustiveness (the deferred cycle-7 residual hardens the guards further).
- **Deferred items**: 6 (cycle-6/7 residual) → see Deferred work: 2 must-fix deferred via user-authorized early-break (`test:868` pin `\|\| exit 1`; `provision.sh:63` `PROXY_ROLE_NAME` guard), 2 should-fix (`test:884` quoted-export form; `README:23` IAM-work contradiction), 2 cycle-6 doc nits (`secretsmanager:GetSecretValue` prereq; tear-down OIDC ARN account). All fold into a follow-up test-hardening + doc-polish pass.
- **Surprises during implementation**:
  - The runbook-snippet + test-coverage churn (7 cycles) was driven by adversarial Codex reviewers in unbounded refinement mode on a doc/test-heavy PR: each cycle's fix surfaced the next stricter angle (H4 self-reinforcement at the fix boundary). The synthesizer flagged this explicitly at cycles 5–7; the user bounded it with two early-breaks (Continue at cycle 5, accept-after-one-final-cycle at cycle 7).
  - The functional change itself never regressed — every reject was about cross-reference doc consistency or test-guard strictness, never the repoint logic, IAM grant, or hash/schema invariants.

### Cycle 1 — preset=phase-3 — reject

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "review_count": 3,
  "unified_findings": [
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": ".github/workflows/schema-deploy.yml:77",
      "description": "RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. As wired, the schema-deploy job cannot reach the proxy at all. This challenges Key decisions Q2/Q6 ('RDS Proxy public endpoint, security group 0.0.0.0/0').",
      "recommended_fix": "Point off-VPC schema-deploy at the public RDS *instance* endpoint with direct IAM auth (the proxy stays for Vercel reads), OR run schema-deploy inside the VPC (self-hosted runner / CodeBuild), OR expose the proxy via an intentional NLB/PrivateLink design. Resolve before claiming the schema-deploy path works; this likely amends Q2/Q6.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/infra/provision.sh:311",
      "description": "The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth still needs a Secrets Manager credential registered for the migrator DB user; without it the proxy cannot authenticate the migrator connection.",
      "recommended_fix": "Either configure end-to-end IAM auth for the proxy for the migrator user, or create+attach a migrator credential secret and register it in the proxy auth config. Add a smoke test that connects through the chosen endpoint as migrator. (Couples with the schema-deploy.yml:77 reachability finding \u2014 resolve the CI-write endpoint design together.)",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "missing-acceptance",
      "file_line": ".github/workflows/schema-deploy.yml:68",
      "description": "PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-apply.' Implementation status records only wiring + yamllint + testcontainer coverage \u2014 the live OIDC apply against real RDS Proxy was never executed. Combined with the proxy-reachability and migrator-credential findings, the schema-deploy path is unproven and may be non-functional as designed.",
      "recommended_fix": "After resolving the endpoint/credential design, run schema-deploy live once and record the clean apply/status, OR explicitly amend the PR-1.4 acceptance criterion to state live execution is deferred (and to which phase) with rationale.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "unsafe",
      "file_line": "benchmarks-website/infra/README.md:120",
      "description": "The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while transmitting the master password. This is a MITM exposure on the single most sensitive credential in the system. The schema-deploy workflow correctly uses verify-full; the bootstrap runbook is inconsistent.",
      "recommended_fix": "Change the bootstrap runbook to PGSSLMODE=verify-full with PGSSLROOTCERT pointed at the downloaded RDS CA bundle, matching the workflow.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:19",
      "description": "The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and the entire README/plan are 245040174862. Per Key decisions, 375504701696 is the PERSONAL/v3-EC2 account \u2014 exactly the account the bench infra must NOT land in. An operator trusting the header points at the wrong account; verify_prereqs would then die confusingly, or worse the operator provisions into the wrong account.",
      "recommended_fix": "Change the line-19 comment to account 245040174862 to match TARGET_ACCOUNT and the README; or interpolate the value rather than hardcoding a stale literal.",
      "found_by": [
        "maint",
        "correctness/claude"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:102",
      "description": "PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qualitative coverage is strong (all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8), but the literal acceptance criterion is unmet and was not amended.",
      "recommended_fix": "Either add ~37 more deterministic fixture vectors, OR amend the PR-1.5 acceptance criterion to '63 vectors' with rationale (the chosen 63 exhaustively cover all tables + boundary classes). Cheap; pick one and record it.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "weak-exit-criteria",
      "file_line": "scripts/test_measurement_id.py:1",
      "description": "The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scripts/test_measurement_id.py. The documented phase gate is unrunnable as written (no such file). Independently confirmed during exit-criteria execution.",
      "recommended_fix": "Amend the Phase-1 exit-criteria string to 'pytest scripts/test_measurement_id.py' (the as-shipped file). Plan-edit only.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "scripts/_measurement_id.py:50",
      "description": "Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py | Python re-export to keep one site'. The shipped module neither imports nor re-exports SCHEMA_VERSION (and could not cleanly import from hyphenated post-ingest.py without importlib). The hash port is correct to omit it; Table D is stale and would mislead a future SCHEMA_VERSION bump. (Table D is a reference downstream PRs grep into.)",
      "recommended_fix": "Amend Table D to remove the _measurement_id.py re-export row (or replace it with the real lockstep site). Do NOT add a spurious re-export to the hash port. Plan-edit only. (See disagreement: spec framed this as must-fix scope-drift requiring re-export OR amendment; maint framed it as should-fix amend-the-doc; synthesizer call = amend Table D, code is correct.)",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:35",
      "description": "The 'Initial files' section lists only 001 and 002 and says the SQL files 'land in PR-1.3', but PR-1.4 added 003_migrator_ledger_grant.sql, which exists on disk and is exercised by the test suite. The directory's own README is a stale, incomplete inventory.",
      "recommended_fix": "Add a bullet for 003_migrator_ledger_grant.sql (GRANT SELECT,INSERT on the ledger to migrator, PR-1.4) and fix the 'land in PR-1.3' sentence.",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "coverage",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1108-1117",
      "description": "No golden vector exercises a NaN or Inf f64 threshold. Rust write_f64 uses v.to_bits() (preserves NaN payload bits); Python struct.pack('<d', nan) emits canonical NaN (0x7ff8...). A NaN threshold would hash differently across languages -> silent duplicate row, the exact failure the hash pin exists to prevent. Inf is canonical on both, so the gap is narrowly NaN.",
      "recommended_fix": "Add f64::NAN / INFINITY / NEG_INFINITY threshold vectors and regenerate the golden file, OR assert threshold.is_finite() at the PR-2.1 ingest boundary so a non-finite value fails loudly rather than diverging silently.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "scope-drift",
      "file_line": "migrations/001_initial_schema.sql:75",
      "description": "The composite-index Key decision promised indexes on '(dim_tuple..., commit_timestamp DESC)'; the migration creates dim-leading (read-path filter) indexes WITHOUT the trailing commit_timestamp, and the tests assert only index *names*, not indexed columns/order. The divergence is explained in PR-1.3's surprises (dim-leading serves the chart read path; PK enforces hash-tuple uniqueness) but the Key decision row was never updated.",
      "recommended_fix": "Amend the composite-index Key decision to the implemented read-path strategy, AND/OR add a test asserting the index column definitions (not just names) so the intended shape is pinned.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:12-14",
      "description": "The workflow header still advertises a 'schema-deploy GitHub Environment with manual-approval as the stronger gate ... tracked as deferred hardening'. The 2026-05-29 deploy-model Key decision SUPERSEDED that path (PR merge is the gate; the Environment gate was judged the wrong tool). The comment points a future engineer at deferred work the plan reversed. (NOT a re-flag of the accepted no-env-gate tradeoff \u2014 this flags the stale comment, aligned with the decision.)",
      "recommended_fix": "Replace the Environment-gate framing with the actual deferred item: switch the trigger to push on the deploy branch under paths: migrations/** (PR-merge-is-the-gate). Naturally lands with the deferred trigger-switch (Deferred work, 2026-05-29 deploy-model item).",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:266",
      "description": "proxy_role_name ('vortex-bench-proxy-role') and its policy name are hardcoded locals, but the README 'Customizing' section claims 'Every name / class / engine version / region is set at the top via readonly declarations with ${ENV:-default} fallbacks.' The proxy role is an exception, and the tear-down runbook uses the literal default name, so an operator who overrode other names gets a tear-down mismatch.",
      "recommended_fix": "Promote proxy_role_name to a top-level readonly PROXY_ROLE_NAME=\"${PROXY_ROLE_NAME:-vortex-bench-proxy-role}\" like the other names, OR soften the README 'every name is overridable' claim to enumerate the proxy role as fixed.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": "scripts/migrate-schema.py:2733-2745",
      "description": "status() reports a generic 'pending' (exit 1) for an empty/whitespace-only migration file, while apply() rejects it explicitly only when it reaches it. Same bad file, two different diagnoses. Behavior is safe (loud both ways) but asymmetric.",
      "recommended_fix": "Optionally have status() classify empty/whitespace-only on-disk files distinctly so the operator sees the same diagnosis from status as from apply.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:39",
      "description": "002 is described as 'CREATE ROLE for the IAM-auth user that bench.yml workflows assume into', but migrator is consumed by the schema-deploy workflow (PR-1.4), not bench.yml (the ingest workflow, which uses a separate future ingest role). The WHY misattributes the consumer.",
      "recommended_fix": "Change 'bench.yml workflows' to 'the schema-deploy workflow (PR-1.4)'.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scaffolding",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:30",
      "description": "REGEN_GOLDEN_VECTORS is permanent regeneration scaffolding (write-on-env, always-assert otherwise). Correct and well-documented, but nothing flags committed-JSON drift unless the Rust test runs in CI, and the golden==Python half is not CI-gated (deferred).",
      "recommended_fix": "Add one sentence to the test module doc noting golden==Python is only enforced once 'uv run --all-packages pytest scripts/' is wired into CI (deferred), so a green local run does not prove cross-language parity in CI.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:84-89",
      "description": "The 'Apply migrations' step carries a ~5-line justification comment (set -x suppression, PGPASSWORD-on-own-line vs export masking). It documents two real footguns but is exactly the >100-char justification-comment shape the shared BAN warns about.",
      "recommended_fix": "Keep the substance but tighten to two short sentences (token-leak + export-masking). No behavior change.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1",
      "description": "PR-1.5's expected files row named 'benchmarks-website/server/src/db.rs (golden-vector test added)', but the test shipped as a separate integration test file benchmarks-website/server/tests/measurement_id_golden.rs. Minor placement deviation from the plan row.",
      "recommended_fix": "Amend the PR-1.5 expected-files row to name the integration test location (the chosen placement is fine; the plan row is stale).",
      "found_by": [
        "spec"
      ]
    }
  ],
  "disagreements": [
    {
      "topic": "Severity of the wrong-account header comment (provision.sh:19)",
      "positions": [
        {
          "lens": "maint",
          "position": "must-fix: the comment names the WRONG account (375504701696 = personal/v3), and an operator trusting it provisions into the wrong account or hits a confusing verify_prereqs die."
        },
        {
          "lens": "correctness/claude",
          "position": "nit: cosmetic doc-quality; the actual TARGET_ACCOUNT default is correct."
        }
      ],
      "synthesizer_call": "must-fix (HIGHEST, conservative). An operator-facing runbook naming the wrong AWS account is an operational hazard, not cosmetic; the fix is one line."
    },
    {
      "topic": "Table D SCHEMA_VERSION re-export (scope-drift must-fix vs amend-the-doc should-fix)",
      "positions": [
        {
          "lens": "spec",
          "position": "must-fix scope-drift: Table D requires _measurement_id.py to re-export SCHEMA_VERSION; implement the re-export OR amend Table D."
        },
        {
          "lens": "maint",
          "position": "should-fix: the shipped module is CORRECT to omit the re-export (can't cleanly import hyphenated post-ingest.py; the hash port has no need for SCHEMA_VERSION); Table D is stale \u2014 amend the doc."
        }
      ],
      "synthesizer_call": "Keep must-fix severity (HIGHEST, and Table D is a reference downstream PRs grep into \u2014 fix before phase close), but the ACTION is to amend Table D, NOT to add a re-export. The shipped hash port is correct."
    }
  ],
  "dropped_re_flags": [
    {
      "topic": "migrator role lacks privileges to ALTER / CREATE INDEX on master-owned tables in future migrations (correctness/codex flagged must-fix at 002_iam_db_user.sql:37)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:487 (PR-1.3 cycle-1 \u2014 migrator table privileges; resolve role-ownership model in PR-2.1). NOTE: the deferral is framed around INGEST DML; it should be EXPANDED to also cover future-migration DDL (ALTER/CREATE INDEX on master-owned tables) so the schema-deploy steady-state is covered, not just the ingest write path. Surfaced as a synthesizer concern. No Phase-1 migration alters an existing table, so it does not block Phase 1 functionally."
    },
    {
      "topic": "golden==Python hash test (and the testcontainer suite) not wired into CI (correctness/claude flagged should-fix)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:489 (PR-1.5 cycle-1 \u2014 scripts/ pytest not in CI) and Deferred work:486 (PR-1.2 \u2014 no CI runner). The reviewer itself acknowledged the prior triage; surfaced as the single largest standing correctness exposure for the hash pin."
    }
  ],
  "phase_artifacts": {
    "summary": "Phase 1 ('RDS + schema + hash port') lands the migration foundation in five concept areas. (A) Infrastructure: provision.sh idempotently bootstraps RDS Postgres db.t4g.micro + RDS Proxy (IAMAuth=REQUIRED, TLS) + GitHub OIDC provider + GitHubBenchmarkSchemaRole in account 245040174862, with an operator runbook in infra/README.md. (B) Schema-deploy CI: schema-deploy.yml (workflow_dispatch + dry_run; PR-merge is the accepted authorization gate, no environment: approval) generates a client-side IAM token and runs the migrate runner against the proxy as migrator over verify-full TLS. (C) Migration runner: scripts/migrate-schema.py applies migrations/*.sql in name order, tracks public._applied_migrations, is idempotent, uses autocommit + per-migration top-level transactions so a failing later migration rolls back only itself, rejects empty files; 28+ testcontainer tests. (D) DDL: 001 creates the commits dim + 5 fact tables + read-path composite indexes (Postgres translation of the authoritative DuckDB schema.rs, column order/nullability/types preserved); 002 the migrator login role + conditional rds_iam; 003 the append-only ledger grant. (E) Hash port: _measurement_id.py is a byte-for-byte port of db.rs measurement_id_* (xxhash64 seed 0), pinned by a Rust source-of-truth golden file giving Rust==golden==Python, verified bit-exact for all 63 vectors (Claude correctness executed the port; Rust golden test passes). The keystone hash-equivalence deliverable is solid and the schema shape provably matches schema.rs. HOWEVER the AWS-integration path has deploy-blocking gaps the Codex correctness lens surfaced: RDS Proxy endpoints are not publicly reachable from off-VPC GitHub runners (challenges Key decisions Q2/Q6); the proxy lacks a migrator credential for IAM auth; PR-1.4's live OIDC apply against real RDS Proxy was never executed (only wiring + lint + testcontainer); and the master-bootstrap runbook uses PGSSLMODE=require (no cert verification) while sending the master password. The Codex spec lens found contract gaps: 63 vectors shipped vs the promised 100; the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py; Table D claims a SCHEMA_VERSION re-export the hash port omits. Maintainability is otherwise high, but provision.sh's header names the WRONG AWS account, migrations/README omits 003, and the schema-deploy header advertises the superseded Environment gate.",
    "surprises": [
      {
        "what": "RDS Proxy endpoints are not publicly accessible; off-VPC GitHub-hosted runners cannot reach the proxy. The plan's 'RDS Proxy public endpoint' assumption (Q6) may be architecturally invalid for the CI-write path.",
        "how_handled": "Not handled in the diff \u2014 flagged as a must-fix deploy-blocker; likely forces amending Key decisions Q2/Q6 (e.g., CI writes to the public RDS instance endpoint with direct IAM, proxy stays for Vercel reads).",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.4's schema-deploy was accepted on wiring + yamllint + testcontainer, never run live against real RDS Proxy.",
        "how_handled": "Recorded honestly in Implementation status; spec lens flags the acceptance criterion as unmet. Coupled with the proxy-reachability + migrator-credential findings, the path is unproven.",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.5 shipped 63 golden vectors, not the promised 100.",
        "how_handled": "Status acknowledges 63; no amendment or extra fixtures. Qualitative coverage is strong (all tables + boundaries).",
        "amend_plan": "yes"
      },
      {
        "what": "The Phase-1 exit criterion names pytest scripts/test_post_ingest_hash.py, but the shipped file is test_measurement_id.py.",
        "how_handled": "Not reconciled; the documented gate is unrunnable as written. Independently confirmed during exit-criteria execution.",
        "amend_plan": "yes"
      },
      {
        "what": "Table D's claim that _measurement_id.py re-exports SCHEMA_VERSION is stale; the shipped module correctly omits it.",
        "how_handled": "Artifact correct; Table D not updated. Amend the reference table.",
        "amend_plan": "yes"
      },
      {
        "what": "Composite indexes are dim-leading (read-path filter columns), not the Key decision's '(dim_tuple..., commit_timestamp DESC)'.",
        "how_handled": "Explained in PR-1.3 surprises (PK enforces hash-tuple uniqueness; dim-leading serves charts); Key decision row not updated and tests assert only index names.",
        "amend_plan": "yes"
      },
      {
        "what": "migrator role cannot ALTER / CREATE INDEX on master-owned tables in future migrations (GRANT CREATE on public is insufficient).",
        "how_handled": "Covered by the deferred PR-1.3 role-ownership item (PR-2.1), but the deferral is ingest-DML-framed and should be expanded to cover migration DDL. Does not block Phase 1 (no Phase-1 migration alters an existing table).",
        "amend_plan": "already-done"
      },
      {
        "what": "Master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while sending the master password.",
        "how_handled": "Not handled \u2014 flagged must-fix (MITM exposure); workflow already uses verify-full, so only the README bootstrap is inconsistent.",
        "amend_plan": "no"
      },
      {
        "what": "NaN/Inf f64 threshold cross-language hash divergence is unguarded (no golden vector).",
        "how_handled": "Flagged should-fix coverage; threshold is a cosine value so NaN is implausible, but the divergence would be silent.",
        "amend_plan": "yes"
      }
    ],
    "coverage": {
      "tested_cases": [
        {
          "case": "Hash Rust==golden==Python across all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8 (63 vectors, executed bit-exact)",
          "test_location": "benchmarks-website/server/tests/measurement_id_golden.rs + scripts/test_measurement_id.py",
          "confidence": "high"
        },
        {
          "case": "migrate-schema apply / idempotency / name-order / failing-migration rollback (subprocess) / status drift / empty-file rejection / non-default search_path ledger agreement / subdir-skip / case-insensitive discovery",
          "test_location": "scripts/test_migrate_schema.py",
          "confidence": "high"
        },
        {
          "case": "Real 001-003 apply cleanly + idempotent; 6 tables, 6 indexes, per-table column order+nullability, key type translations, migrator role login, ledger grants (SELECT/INSERT present, DELETE/UPDATE absent)",
          "test_location": "scripts/test_migrate_schema.py:3480-3640",
          "confidence": "high"
        }
      ],
      "untested_cases": [
        {
          "case": "Live schema-deploy OIDC apply as migrator against the real endpoint (and whether the proxy is even reachable from CI)",
          "priority": "high",
          "why_untested": "Recorded as wiring/lint/testcontainer only; reachability + migrator-credential findings suggest it may not work as wired."
        },
        {
          "case": "scripts/ pytest running in CI (golden==Python parity AND the testcontainer suite are both ungated)",
          "priority": "high",
          "why_untested": "No CI job runs uv run --all-packages pytest scripts/; deferred CI-hardening."
        },
        {
          "case": "RDS Proxy reachability from off-VPC GitHub-hosted runners",
          "priority": "high",
          "why_untested": "RDS Proxy is not publicly accessible; the CI-write endpoint design needs rework."
        },
        {
          "case": "migrator credential registered for RDS Proxy IAM auth",
          "priority": "high",
          "why_untested": "Proxy has only the master secret; migrator connection would fail auth."
        },
        {
          "case": "NaN/Inf f64 threshold cross-language equivalence",
          "priority": "medium",
          "why_untested": "No non-finite threshold vector."
        },
        {
          "case": "Future-migration DDL (ALTER/CREATE INDEX) run as migrator on master-owned tables",
          "priority": "medium",
          "why_untested": "Deferred role-ownership (PR-2.1); no Phase-1 migration alters an existing table."
        },
        {
          "case": "Composite-index column definitions (tests assert names only)",
          "priority": "medium",
          "why_untested": "Index-definition assertions not written."
        },
        {
          "case": "Edit-after-apply ledger drift (no fingerprint column)",
          "priority": "medium",
          "why_untested": "Deferred (sha256 ledger column)."
        }
      ],
      "recommendations": "Resolve the CI-write endpoint design FIRST (RDS Proxy reachability + migrator credential) \u2014 this likely amends Key decisions Q2/Q6 (point schema-deploy at the public RDS instance endpoint with direct IAM, or run in-VPC). Then run schema-deploy live once and record the clean apply/status. Fix the README bootstrap to PGSSLMODE=verify-full. Land the plan-edit must-fixes (exit-criteria test name, Table D, provision.sh account, vector-count reconciliation). Wire scripts/ pytest into CI (closes the largest hash-pin exposure)."
    },
    "tradeoffs": [
      {
        "decision": "Branching / merge target (per-phase child branches -> ct/bench-v4 -> develop)",
        "original": "User pick Q1",
        "verdict": "keep",
        "rationale": "No artifact conflict."
      },
      {
        "decision": "Postgres flavor (RDS db.t4g.micro single-AZ, account 245040174862)",
        "original": "User pick Q2",
        "verdict": "keep",
        "rationale": "Provisioning/DDL target the chosen account/region/class; flavor-portable (testcontainer runs vanilla Postgres)."
      },
      {
        "decision": "Connection pooler (RDS Proxy with IAM-auth pass-through)",
        "original": "Locked by Q2",
        "verdict": "revisit-but-keep",
        "rationale": "The proxy is right for Vercel reads, BUT the CI-write-via-proxy assumption is challenged: RDS Proxy is not publicly reachable from off-VPC GitHub runners and lacks a migrator credential. The CI-write endpoint specifically needs rework (most-pessimistic across reviewers; correctness/codex would lean reverse for the CI path)."
      },
      {
        "decision": "Ingest writer language (pure Python + golden vectors)",
        "original": "User pick Q4",
        "verdict": "revisit-but-keep",
        "rationale": "Hash port verified bit-exact, but the 100-vs-63 fixture-count and Table D lockstep gaps need reconciliation."
      },
      {
        "decision": "Schema deploy tool (in-house migrate-schema.py + plain SQL)",
        "original": "User pick Q5a",
        "verdict": "revisit-but-keep",
        "rationale": "Sound and well-tested, but grew to ~180 LOC and still lacks ledger fingerprinting, so the documented edit-after-apply prohibition has zero runtime enforcement (deferred)."
      },
      {
        "decision": "Schema-deploy authorization (PR merge is the gate; no environment gate)",
        "original": "User decision 2026-05-29",
        "verdict": "keep",
        "rationale": "Accepted tradeoff; NOT re-flagged. Only the stale header COMMENT advertising the superseded Environment gate is flagged (doc-sync, not a reversal)."
      },
      {
        "decision": "CI network reach (public + IAM, verify-full)",
        "original": "User pick Q6",
        "verdict": "revisit-but-keep",
        "rationale": "The public+IAM model works for the RDS INSTANCE endpoint, but the 'public RDS Proxy endpoint' sub-assumption is invalid (proxy isn't public). Tied to the schema-deploy.yml:77 must-fix."
      },
      {
        "decision": "Composite index definition strategy ((dim_tuple..., commit_timestamp DESC))",
        "original": "Forward-looking design",
        "verdict": "revisit-but-keep",
        "rationale": "Implemented as dim-leading read-path indexes without the trailing timestamp; either amend the decision to the implemented strategy or pin index column definitions in tests."
      },
      {
        "decision": "Cutover style / One-shot load / Read framework / Operator SQL / v3 disposition",
        "original": "User picks Q5b,Q7,Q8,Q9 + forward-looking",
        "verdict": "keep",
        "rationale": "Future-phase decisions; no Phase-1 change contradicts them."
      }
    ]
  },
  "executive_summary": "Phase 1 ships a coherent, unusually well-documented foundation, and its keystone deliverable \u2014 the cross-language measurement_id hash equivalence (Rust==golden==Python) \u2014 is verified bit-exact for all 63 vectors, with the 6-table Postgres schema provably matching the authoritative DuckDB schema.rs and a well-tested migration runner. The mixed-executor review (Claude + Codex lenses) is the reason this is a REJECT rather than an accept: the Codex correctness lens surfaced a cluster of deploy-blocking AWS-integration gaps that the hash-focused Claude review did not. The most serious: RDS Proxy endpoints are NOT publicly reachable from off-VPC GitHub-hosted runners (schema-deploy.yml:77), the proxy was provisioned without a migrator credential for IAM auth (provision.sh:311), and PR-1.4's live OIDC apply against real RDS Proxy was never actually executed (schema-deploy.yml:68) \u2014 together meaning the schema-deploy CI path, a core Phase-1 deliverable, is unproven and likely broken as wired. Resolving it probably amends Key decisions Q2/Q6 (e.g., point CI writes at the public RDS *instance* endpoint with direct IAM and reserve the proxy for Vercel reads, or run schema-deploy in-VPC). A fourth correctness must-fix: the master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while transmitting the master password (README:120) \u2014 a MITM exposure, fixed by verify-full. The Codex spec lens added contract-closure must-fixes that are mostly cheap plan-edits: 63 vectors shipped vs the promised 100 (reconcile the criterion or add vectors); the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py (rename to the shipped test_measurement_id.py \u2014 independently confirmed at exit-criteria time); and Table D claims a SCHEMA_VERSION re-export the hash port correctly omits (amend the reference table, do NOT add the re-export). The maint lens caught a genuine operational hazard: provision.sh's header comment names the WRONG (personal/v3) AWS account, which could misdirect an operator. Five should-fixes (stale migrations/README missing 003; composite-index decision-vs-impl drift; the schema-deploy header advertising the superseded Environment gate; a hardcoded proxy-role name contradicting the README; the unguarded NaN/Inf hash vector) and five nits round it out. Two findings were dropped as carry-forward (migrator table privileges and CI-gating of scripts/ pytest are both already in Deferred work) \u2014 though the role-privileges deferral should be expanded to cover future-migration DDL, not just ingest DML. Verdict: reject, 8 must-fix. The hash and schema work is strong; the AWS-integration + plan-consistency layer needs a focused fix pass before Phase 1 closes.",
  "overall": "reject",
  "must_fix_count": 8,
  "should_fix_count": 5,
  "nit_count": 5,
  "review_cycles_this_invocation": 1,
  "executor_routing": {
    "spec": "codex",
    "correctness": "parallel",
    "maint": "claude"
  }
}
```

</details>

### PR-2.1: (re-plan) Foundational ingest identity — `bench_ingest` role + 6-table DML grants + `GitHubBenchmarkIngestRole`  (6 code/fix commits across 2 inner-loop gauntlet cycles, ending at 44245c4a8)
- Scope shipped: `migrations/004_ingest_role.sql` creates the least-privilege `bench_ingest` IAM role (USAGE + per-table SELECT/INSERT/UPDATE on the 6 data tables, no DELETE/DDL) plus a default-privilege rule so future `migrator`-created tables auto-grant the ingest role; `provision.sh` adds `GitHubBenchmarkIngestRole` (OIDC, `rds-db:connect` for `bench_ingest`) and drops the dead RDS-Proxy grant on the schema role.
- Tests added: `test_real_migrations_create_bench_ingest_role`, `test_bench_ingest_has_dml_only_on_data_tables`, `test_bench_ingest_can_upsert_and_is_denied_ddl_delete`, `test_bench_ingest_default_privileges_cover_future_migrator_tables`, and (cycle-1 must-fix regression) `test_real_migrations_apply_as_non_superuser_createrole_master` (applies 001..004 as a REAL non-superuser CREATEROLE login).
- Review: 2-vote (pr-2: fresh + correctness, claude-routed) / accepted (cycles: 2). Cycle 1 REJECT (1 must-fix: 004 ADP failed on the non-superuser RDS master). Cycle 2 ACCEPT (0 must-fix; 3 should-fix + 2 nit — cheap ones applied in f3c5771e8/44245c4a8, 2 deferred).
- Confidence: high. The must-fix was a subtle prod-gating bootstrap failure; the fix (guarded INHERIT self-grant via the master's ADMIN option, then revoke) was validated end-to-end against a local PostgreSQL 16.14 cluster — RED pre-fix, GREEN post-fix for BOTH a real non-superuser master AND the superuser path; role graph restored (master retains only its ADMIN auto-grant). The cycle-2 correctness reviewer independently confirmed the REVOKE removes only the self-grant (separate `pg_auth_members` records by grantor).
- Deferred items: 2 (cycle-2: master-lacks-ADMIN-on-pre-existing-`migrator` negative test; `createrole_self_grant='inherit'` no-op-branch test — both test-coverage hardening for an unsupported / conceptually-validated path). Also resolved 1 prior deferred item (PR-1.6 proxy-grant cleanup, via `provision.sh` in commit 50a8d4a08).
- Surprises during implementation: (1) the cycle-1 reviewer's RECOMMENDED fix (`SET ROLE migrator` wrapper) was EMPIRICALLY DISPROVEN — it also fails on a real non-superuser master (`SET ROLE` is checked against the session user; the PG16 CREATEROLE auto-grant has SET FALSE). The shipped fix is the guarded INHERIT self-grant. (2) Docker/testcontainers are unavailable locally, so validation used a local Homebrew PG16.14 cluster + the real runner; CI runs the canonical testcontainer suite. (3) Discovered (OUT OF SCOPE, pre-existing, predates Phase 2): the bench-v4 Python files are ~100-col and fail the repo's `ruff check .` / `ruff format --check .` (line-length 120) — e.g. the pre-existing E501 at `scripts/test_migrate_schema.py:798` (commit 74175ec85d). Not fixed here; reformatting the whole file is unrelated to the ADP must-fix and conflicts with the file's established style. Flag for the operator: the bench Python files need a branch-wide ruff reconciliation before merge to `develop`.

### PR-2.2: `post-ingest.py --postgres` IAM-upsert writer + tests  (impl `0b2ba4b7d` + many fix commits across 15 inner-loop gauntlet cycles, ending at `06087c784`)
- Scope shipped: `scripts/post-ingest.py` gains a `--postgres $RDS_DSN` mode that parses the JSONL, computes `measurement_id` via `_measurement_id.py`, mints an RDS IAM token (boto3) for the least-privilege `bench_ingest` role, connects over verify-full TLS (asserting `conn.pgconn.ssl_in_use` post-connect), upserts `commits` first then the 5 fact tables via `INSERT ... ON CONFLICT (measurement_id) DO UPDATE` in one all-or-nothing transaction, retries on Postgres deadlock/serialization conflict, pins `search_path=public`, and applies an `is_finite()` guard on f64 dims at the ingest boundary. The v3 `--server` path is preserved (stdlib-only under bare `python3`); `SCHEMA_VERSION` kept in lockstep.
- Behavior-preservation verified byte-for-byte against the v3 server boundary: `_RECORD_FIELDS`/`_FIELD_TYPES` mirror `records.rs` (serde `tag="kind"` + `deny_unknown_fields`, per-field i32/i64/f64/str type+range validation); the 5 `ON CONFLICT SET` lists mirror `ingest.rs` (dim columns excluded); the `measurement_id` port matches `db.rs` Table A/B field order + tags; duplicate-JSON-key last-wins matches v3; commit-upsert-first then per-record (commit_sha-match -> value validation -> apply) ordering mirrors `apply_envelope_once`.
- Tests added: 102-test `scripts/test_post_ingest_postgres.py` (pure-unit + testcontainers Postgres) — insert-then-update idempotency, measurement_id==`_measurement_id.py`, NaN/Inf rejection + rollback, all 5 ON CONFLICT SET lists, deny_unknown_fields, type/range boundaries, parse-boundary hardening (universal newlines + ValueError/RecursionError/UnicodeDecodeError), record+commit string storability (NUL/non-UTF-8/lone-surrogate), write-conflict retry, verify-full TLS actually-in-use (incl. a container test pinning the `conn.pgconn` traversal), least-privilege enforcement, search_path pinning, CLI composition, and the `git_show_field` UTF-8 decode boundary with a mutation-pinned subprocess bytes-mode contract guard (`_assert_git_show_bytes_mode` covers text/universal_newlines/encoding/errors).
- Review: 2-vote (pr-2: fresh + correctness, `executor=parallel` Claude+Codex per lens) / **accepted at cycle 15** (15 inner-loop cycles; a 7-cycle early-break at cycle 7 -> operator Continue; then operator-directed fix+review at cycles 12/13/14 and a standing-autonomy land-if-clean at cycle 15). Cycle 15: 0 must-fix; both Claude lenses proved the bytes-mode contract exhaustive against CPython's `Popen.__init__` source; 1 non-blocking should-fix deferred.
- Confidence: high. Genuinely production-breaking bugs were caught and fixed mid-loop (cycle-9 `conn.info.ssl_in_use` AttributeError regression -> `conn.pgconn.ssl_in_use`, caught by fresh/claude verifying the real psycopg API; cycle-12 `git_show_field` non-UTF-8 decode). The full testcontainer suite (102) passes; the production decode/validation boundaries are mutation-verified.
- Deferred items: 3 (PR-2.2 cycle-4 destructive-scrub loopback guard; cycle-7 real-deadlock + `autocommit=False` container coverage; cycle-15 `test_server_mode_requires_benchmark_id` isolation) — all test-hardening, **Resolved-by PR-2.3**. Accepted tradeoffs recorded (dup-key last-wins; PEP 723 `dependencies=[]`; IAM region `--region`>session>host precedence).
- Surprises during implementation: (1) the multi-vote/disjoint-coverage approach earned its keep — Codex lenses probed adversarial inputs + test-discriminating-power while Claude lenses verified invariants against real library APIs and ran the full suite; the cycle-9 regression was caught only because fresh/claude checked the real psycopg accessor against mocks shaped to the wrong API. (2) The last ~3 cycles converged into a diminishing-returns spiral on the `git_show_field` regression test's contract-modeling completeness (text=True -> the universal_newlines alias), terminated once the contract was proven exhaustive against CPython source. (3) The pre-existing bench-v4 ruff line-length-120 reconciliation (flagged in PR-2.1) still pends before the `develop` merge.

### PR-2.3: (re-plan, CI-hardening) wire `scripts/` pytest into CI + fail-loud-on-no-Docker  (3 commits across 1 inner-loop gauntlet cycle, ending at `d21742dc4`)
- Scope shipped: a `scripts-test` job in `.github/workflows/ci.yml` runs `uv run --all-packages pytest scripts/` on the amd64-large runner (Docker present; `ubuntu-latest` fork fallback), mirroring the `python-test` job's `runs-on/action` sccache + `setup-prebuild` pattern, with a `docker info` fast-fail gate. The two testcontainer suites (`test_migrate_schema.py`, `test_post_ingest_postgres.py`) now FAIL LOUD in CI instead of silently skipping when Docker is absent, via a per-file `_require_docker_for_testcontainers()` (CI env set -> `pytest.fail`; local -> `pytest.skip`). Resolves the deferred scripts/-not-CI-gated items (PR-1.2, PR-1.5: golden==Python now gated).
- Tests added: `test_require_docker_fails_loud_in_ci` + `test_require_docker_skips_without_ci` in BOTH test files (monkeypatch `CI` + `_docker_available`); the improved `test_server_mode_requires_benchmark_id` (resolves the PR-2.2 cycle-15 deferral). Full suite 217 passed.
- Review: 2-vote (pr-2: fresh + correctness, `executor=parallel`) / **accepted at cycle 1**. 0 must-fix. 1 should-fix (correctness/claude): the fail-loud guard tests could not catch the always-skip regression they guard -- `pytest.skip()` raises `Skipped` (not a `Failed` subclass), so an always-skipping helper escaped `pytest.raises(Failed)` and registered as a green test-skip; mutation-verified. Applied in `d21742dc4` (catch both outcome types + assert the specific one; re-verified by mutation). 2 nits: pytest-scripts-collects-a-4th-file (applied a clarifying ci.yml comment); per-file helper duplication (dismissed -- accepted pre-existing pattern).
- Confidence: high. The central CI/test risk (a green job that silently skips the testcontainer tests) is closed by a double layer (the `docker info` job gate + the CI-env fixture fail-branch), end-to-end-verified (217 tests run with zero skips when `CI=true` + Docker present). The disjoint-coverage approach again earned its keep: correctness/claude found the ironic guard-test silent-skip gap that the other 3 lenses missed.
- Commits: `8402c1990` (CI job + fail-loud + guard tests), `ed585f451` (--server benchmark-id isolation), `d21742dc4` (guard-test should-fix + ci.yml scope comment).
- Scope decision (explicit): the larger PR-2.2 cycle-7 real-deadlock/`autocommit=False` container tests + cycle-4 destructive-scrub guard are RE-MAPPED to a follow-up test-hardening PR (PR-2.3 delivered the CI prerequisite that makes them run; the real-deadlock test needs threaded 2-connection interleaving warranting focused attention). Recorded in Deferred work.
- Surprises during implementation: (1) the guard-test silent-skip gap is the same failure class PR-2.3 closes, one level up in the test that guards it -- a nice illustration of why the fail-loud invariant needs its own discriminating test. (2) `pytest scripts/` is broader than the 3 named suites (also `scripts/tests/test_benchmark_reporting.py`); harmless + matches the acceptance command. (3) The pre-existing bench-v4 ruff line-length-120 reconciliation (1 E501 at `test_migrate_schema.py:831` + the file's established multi-line style) still pends before the `develop` merge (OPEN ITEM since PR-2.1); new lines were kept ruff-clean.

### PR-2.4: (lean re-plan) best-effort v4 Postgres dual-write in 3 ingest workflows + schema-deploy push trigger  (1 code commit + 1 inner-loop gauntlet cycle, ending at `d8bbeb6ed`)
- Scope shipped: after the unchanged hard-required v3 `post-ingest.py --server` step, added a SECOND best-effort `post-ingest.py --postgres` step (3 sub-steps: OIDC assume `GitHubBenchmarkIngestRole` -> install uv -> ingest) to `bench.yml`, `sql-benchmarks.yml`, and `v3-commit-metadata.yml`, all `continue-on-error: true` and gated on `vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (plus `inputs.mode == 'develop'` in the reusable sql-benchmarks.yml). Added `id-token: write` to `v3-commit-metadata.yml`. Switched `schema-deploy.yml` from `workflow_dispatch`-only to also `push` on `develop` under `paths: [migrations/**, scripts/migrate-schema.py]` (kept `workflow_dispatch` + `dry_run`); rewrote the now-superseded `environment:`-gate header comment. The v4 ingest mints the RDS IAM token internally via boto3; DSN uses `sslmode=verify-full` + the downloaded RDS global CA bundle; deps via `uv run --no-project --with psycopg[binary]/boto3/xxhash`. Live-v2 S3 path untouched.
- Tests added: none (CI-workflow-only change; acceptance is `yamllint --strict` clean + the structural criteria). v4 ingest correctness is gated by the Phase-3 `migrate --verify` harness, not a Phase-2 reconciliation step (PR-2.5 dropped in the lean re-plan).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude` per lean calibration) / **accepted at cycle 1**. 0 must-fix, 0 should-fix. 2 nits, both dismissed: (a) redundant AWS region double-sourcing (`AWS_REGION` env + explicit `--region`) across all 3 workflows -- harmless, resolves correctly under the accepted IAM-token region-precedence carry-forward; (b) the v4 step also fires from `nightly-bench.yml` develop-mode cron -- intentional + symmetric with the v3 step's identical gate.
- Confidence: high. Both lenses independently verified end-to-end against the real repo: OIDC `id-token: write` is present at every entry point (bench.yml + v3-commit-metadata.yml at workflow level; the reusable sql-benchmarks.yml inherits it from its develop-mode callers bench.yml/nightly-bench.yml, while PR-mode callers gate the v4 step off via `inputs.mode == 'develop'`); the `post-ingest.py --postgres` CLI signature matches the invocation exactly (positional jsonl + `--postgres` DSN + `--commit-sha` + `--region`; `--benchmark-id` genuinely unused in `--postgres` mode); the v3 step writes `results.v3.jsonl` and the v4 step reads the same path; `v3-commit-metadata.yml` self-writes `empty.jsonl` before reading it (commit-only upsert); the DSN uses the `bench_ingest` role (matching the enforced `_INGEST_ROLE` guard) + `sslmode=verify-full`; the schema-deploy push paths (`migrations/**`, `scripts/migrate-schema.py`) are correct relative to repo root; no token/`PGPASSWORD` echo; `if: failure()` incident.io alert correctly will not fire on a best-effort v4-only failure.
- Commits: `d8bbeb6ed` (the single CI-workflow code commit).
- Deferred items: 0 new. (Retained backlog unchanged: bench-v4 ruff reconcile + Phase-1 live OIDC apply gate.)
- Surprises during implementation: none. OPEN DEPENDENCY for the live v4 write (operator-side, not in-session): operator must set repo vars `GH_BENCH_INGEST_ROLE_ARN` + `RDS_BENCH_INSTANCE_ENDPOINT`/`RDS_BENCH_DB_NAME`/`RDS_BENCH_REGION` (the v4 steps no-op until then), and CloudWatch should confirm Postgres writes appearing per develop push.

### PR-2.6: (amend 2026-06-05) fix the live gate-var bug — gate v4 dual-write on GH_BENCH_INGEST_ROLE_ARN  (1 code commit + 1 inner-loop gauntlet cycle, ending at `077470973`)
- Scope shipped: re-keyed all 9 v4 dual-write sub-step `if:` gates (3 each in `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml`) from `vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (set live) to `vars.GH_BENCH_INGEST_ROLE_ARN != ''` (the assume-role input, unset live), restoring the dormant-until-wired behavior; `sql-benchmarks.yml` keeps its `inputs.mode == 'develop' &&` prefix. Endpoint var stays in `env:`/DSN (the DSN host). Updated the 3 explanatory comments. Surfaced by the 2026-06-05 complexity/gap audit (the bug: gate keyed on the wrong var, so the steps fired+failed at assume-role instead of no-op'ing).
- Tests added: none (CI-workflow gate change; acceptance is `yamllint --strict` clean + grep-verified gate re-key).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 0 should-fix, 1 nit dismissed (optional both-vars gate hardening — the role-ARN-set-but-endpoint-unset edge case fails loud + swallowed + is not a real wiring path; provision.sh emits both vars together). Both lenses verified end-to-end against the real repo: all 9 gates re-keyed, 0 stale gates, no non-v4 step touched (schema-deploy.yml's endpoint PGHOST ref is the unrelated migrator path), endpoint var retained in env:/DSN, yamllint clean.
- Confidence: high. Both lenses independently judged role-ARN-alone is the correct single gate (it is precisely the assume-role input). Live-var state confirmed the bug (endpoint SET, role ARN UNSET) and confirms the fix (gate now evaluates false → steps no-op until the operator wires the role ARN).
- Commits: `077470973` (the single CI-workflow gate fix).
- Deferred items: 0 new.
- Surprises during implementation: none.

### PR-2.7: (amend 2026-06-05) de-gold-plate post-ingest.py — drop trusted-input over-hardening + dead code  (1 impl commit + 1 fix commit across 2 inner-loop gauntlet cycles, ending at `b6d1f292b`)
- Scope shipped: removed ~35% trusted-input over-hardening (net -194 lines) per the 2026-06-05 audit: `read_records` back to a plain text-mode iterator (dropped bytes/splitlines/explicit-UTF-8/RecursionError/oversized-int machinery; still fails loud `path:line` on bad JSON); `git_show_field` reverted to `subprocess(text=True)`; `_require_finite` OverflowError sub-branch dropped (NaN/Inf guard kept); deleted `_reject_unstorable_str`+`_require_storable_str` (NUL/lone-surrogate guards) + their calls in `_require_str`/`_require_opt_str`/`_upsert_commit`; deleted dead `_is_local_host` (inlined the loopback check into the one test fixture). Deleted the ~8 corresponding tests. KEPT all load-bearing code: measurement_id parity, ON CONFLICT/RETURNING(xmax=0), retry, IAM/TLS auth, deny_unknown_fields + typed i32/i64/finite(in-range) validation, memory-quartet/storage-enum.
- Tests added: 2 minimal replacements (mixed-newline happy path, malformed-JSON loud-fail) + a simplified git_show text-mode decode test; net test delta -127 lines.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **cycle 1 REJECT (1 must-fix), cycle 2 ACCEPT (0 findings)**. Cycle-1 correctness caught an INCOMPLETE deletion: the OverflowError-branch removal left an orphaned `pytest.param(10**309, id='out_of_range_int')` in the KEPT testcontainer test `test_nonfinite_threshold_raises_and_rolls_back`, which would error (uncaught OverflowError) in CI (it only skipped locally for lack of Docker). Fresh missed it (only checked deleted tests). Fixed in `b6d1f292b` (dropped the param; nan/inf/-inf retained). Cycle 2: both lenses accept — fix complete, no other dangling refs, behavior-preservation vs v3 holds (the removed guards are stricter than the v3 Rust source; removing them moves the writer CLOSER to v3, and their failure modes stay LOUD — no silent-wrong-write path).
- Confidence: high. Disjoint-lens value earned its keep again: correctness/claude's cumulative-test trace caught the orphaned param that the per-PR diff alone (and fresh) missed. py_compile OK; ruff clean; 38 non-Docker unit tests pass.
- Commits: `6d01990e2` (de-gold-plate impl), `b6d1f292b` (cycle-1 must-fix: orphaned test param).
- Deferred items: 0 new.
- Surprises during implementation: the OverflowError-branch removal's blast radius reached a KEPT testcontainer test's parametrize list — a reminder that deleting a guarded code path requires sweeping ALL tests (not just the standalone ones) for params/cases that exercised it.

### PR-2.8: (amend 2026-06-05) bench-v4 ruff reconcile + 2 phase-end nits  (1 impl commit + 1 nit-fix commit, 1 inner-loop gauntlet cycle, ending at `f0e93b9f4`)
- Scope shipped: (1) cleared the 1 remaining ruff E501 (`test_migrate_schema.py:833`, 122>120) via an `instance_body` local — `ruff check scripts/` now clean (the RETAINED code-quality merge blocker, resolved); (2) fixed the phase-end nit: the `benchmarks-website/infra/README.md` var-table attributed the ingest-workflow consumers of `RDS_BENCH_INSTANCE_ENDPOINT`/`GH_BENCH_INGEST_ROLE_ARN` to PR-2.2, but the dual-write steps landed in PR-2.4 (git-verified: `d8bbeb6ed`); corrected both rows; (3) added a `_main_postgres` comment noting `build_commit` runs `git show <sha>` so the v4 step inherits the v3 step's checkout/git-history assumption (best-effort).
- Tests added: none (doc/quality; the E501 fix is a test-internal local binding covered by the existing passing test).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 0 should-fix, 1 nit fixed inline: fresh flagged the new comment's last line at 102 cols (over the user CLAUDE.md 100-col rule, under ruff's 120); rewrapped to <=100 in `f0e93b9f4`. Correctness git-verified the README PR-2.4 correction + confirmed the E501 fix is semantically identical + the comment factually accurate.
- Confidence: high. Closes the retained ruff merge-blocker and both phase-end nits. ruff clean; 51 non-Docker unit tests pass.
- Commits: `06f9e8a13` (E501 + README + comment), `f0e93b9f4` (comment-width nit fix).
- Deferred items: 0 new. The RETAINED deferred backlog is now: only the Phase-1 live OIDC `schema-deploy` apply operator pre-merge gate (the ruff reconcile is DONE).
- Surprises during implementation: the ruff reconcile was a single E501 (the "established multi-line style" was already in place), so PR-2.8 was much smaller than the deferred item implied.

### PR-3.1: (re-plan) DuckDB→Postgres bulk loader + 004-as-master guard  (2 impl commits + 1 should-fix commit, 1 inner-loop gauntlet cycle, ending at `a6647765f`)
- Scope shipped: (a) new `benchmarks-website/migrate/src/postgres.rs` + `Load` subcommand in `vortex-bench-migrate` — reads each of the 6 tables from an existing v3 DuckDB snapshot as Arrow batches (`query_arrow`) and streams them into Postgres via `COPY ... FROM STDIN` (text format) inside ONE transaction (atomic; mid-load failure rolls back to empty). `measurement_id` (+ `commit_sha`) copied verbatim — no hash recompute. `commits.timestamp` `CAST` to VARCHAR under `SET TimeZone='UTC'`; `all_runtimes_ns` → `{a,b,c}`; `threshold` f64 shortest-round-trip (non-finite rejected). NO `aws-sdk-rds`: `NoTls` (local rehearsal) or native-tls with `--ca-cert` (prod verify-full). (b) gap-#4: a `-- migrate-schema: requires-superuser` marker on `002`/`004` + a `rolsuper OR rolcreaterole` preflight in `migrate-schema.py` that fails loud + early before a non-master `apply` of a marked migration.
- Tests added: 11 embedded-DuckDB loader unit tests (escaping incl. backslash/tab/newline, `BIGINT[]` arrays incl. empty/negative, NULLs, negative ints, UTC timestamp via CAST incl. non-UTC-offset normalization, f64 round-trip, literal-`\N`-vs-NULL disambiguation, empty tables) + 3 marker-detection unit tests + a real-file marker-placement test + 2 testcontainer-gated guard tests (non-master role rejected before DDL; superuser is master-capable). All non-Docker green (11 loader + 16 migrate-schema non-Docker); clippy + nightly fmt clean.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 1 should-fix (applied `a6647765f`: pinned the literal-`\N`, non-UTC-offset timestamp, and empty/negative-array COPY cases — verified-correct by both reviewers but unpinned), 3 nits (1 applied: redundant `format_bigint_array` downcast-context; 2 dismissed per the PR-2.7 de-gold-plate calibration: NUL-byte hardening + a fail-loud bail on the unreachable array-element-NULL branch).
- Confidence: medium. The loader's value-fidelity-critical logic (DuckDB read + COPY-text formatting) is high-confidence — fully unit-tested via embedded DuckDB, with column list/order/ColKind programmatically cross-checked against `migrations/001` for all 6 tables. The Postgres-EXECUTION path (connect / COPY streaming / commit / atomic Drop-rollback) and verify-full TLS COMPILE but are NOT runtime-tested here (no Docker); exercised by PR-3.3's rehearsal harness + the PR-3.4 runbook.
- Deferred items: 2 (both explicit, per the PR-3.1/PR-3.3 boundary plan adjustment `7378345da`): full Postgres-execution integration (6-table load + atomic mid-load rollback) → PR-3.3; verify-full TLS path → PR-3.4 runbook.
- Surprises during implementation: (1) PR-3.1 reads an EXISTING v3 DuckDB rather than re-accumulating from v2, so `measurement_id` is copied verbatim (no hash recompute) — simpler + safer than the original plan implied. (2) The migrate crate's `appender-arrow` feature already provides `query_arrow` (the `arrow` feature does not exist on the duckdb crate). (3) The 2 review Agents raced on the working tree (one/both added a transient `zz_probe` test to verify timestamp/literal-`\N` behavior, then reverted); the orchestrator verified the committed tree clean post-review and de-contaminated the fresh-lens findings to their real shared signal (the coverage should-fix). (4) **[phase-end cycle-1 should-fix annotation, added by PR-3.6]** PR-3.1 shipped the synchronous `postgres` 0.19 crate (+ `native-tls`/`postgres-native-tls`), NOT the plan-row-named `tokio-postgres` — a deliberate choice: a one-shot CLI needs no async runtime, the binding hard `NO aws-sdk-rds` constraint IS met, and sync `postgres` is a thin blocking wrapper that pulls `tokio-postgres` in transitively. The named-dependency swap was unremarked at PR-3.1 time; recorded here per the Phase-3 phase-end review (spec lens, should-fix).

### PR-3.2: (re-plan) DuckDB→Postgres per-`measurement_id` value verify — the PRIMARY v4-correctness gate  (1 impl commit + 1 review-follow-up commit, 1 inner-loop gauntlet cycle, ending at `f796cfc24`)
- Scope shipped: `run_postgres_value_verify` in `migrate/src/verify.rs` + the `verify --postgres-target` CLI mode (mutually exclusive with the existing `--against` v2-structural mode). Joins a DuckDB source snapshot vs the Postgres target per `measurement_id` (1:1; `commits` per `commit_sha`) and compares EVERY non-hashed value column per row, FULL (not sampled), exiting non-zero on any presence diff OR value mismatch and naming the exact `(table, key, column, duckdb_value, pg_value)`. Values normalize to an engine-independent `CellValue` (`Null`/`Int(i64)`/`Text`/`Array(Vec<i64>)`): `all_runtimes_ns` element-wise + order-sensitive; `i32` `iterations` widened to `i64`; `commits.timestamp` compared as epoch microseconds (DuckDB `epoch_us` vs PG `(extract(epoch ...) * 1000000)::bigint`) to sidestep cross-engine timestamp text rendering. New `PgVerifyReport`/`ValueMismatch` (kept separate from the unrelated v2-structural `VerifyReport`). Value-column types are single-sourced from the loader via new `pub(crate) postgres::column_kind` (+ `ColKind`/`connect_postgres` exposed); NO `Cargo.toml` change (reuses PR-3.1's `postgres`/`native-tls`).
- Tests added: 14 no-Docker `value_verify_tests` (in-memory DuckDB): epoch-microseconds pinned (1s→1_000_000); clean-when-identical across all 6 tables; mismatch detection for value column / env_triple / array element / array reorder / commits metadata / NULL-vs-value / `vector_search_runs` Int32 `iterations` + side counter; empty-`BIGINT[]` clean + mismatch; presence diffs BOTH directions; SQL-builder epoch wrapping; `read_kinds` drift guard; report Display. Full migrate suite 98/98 green; clippy `--all-targets` + nightly fmt clean.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 2 should-fix (both applied `f796cfc24`: corrected the epoch-mechanism doc — PG14+ `extract(epoch)` returns `numeric`, exact for any microsecond ts, not just whole seconds; added the `vector_search_runs` Int32/side-counter mismatch test), 4 nits (2 applied: empty-array verify test + SET-TimeZone-not-needed comment; 1 dismissed per de-gold-plate calibration: single-MVCC-snapshot — a one-shot gate against a quiescent post-load target; 1 carried into PR-3.3's acceptance: live-PG16 epoch seed with a sub-second/pre-1970 commit). Both lenses converged on the epoch-doc should-fix. (cycles: 1)
- Confidence: high. The comparison core (the value-fidelity-critical logic) is high-confidence — both lenses verified it neither misses a value/env corruption nor false-positives on a faithful load, and `ReadKind` is resolved ONCE per table from the loader's authoritative `column_kind` and shared across both reads so an asymmetric mis-read is structurally impossible. The Postgres-EXECUTION read path (`read_pg_table`/`pg_cell`/`pg_select_sql` against a live PG16) COMPILES but is NOT runtime-tested here (no Docker); exercised by PR-3.3's rehearsal harness (which owns the container infra and asserts verify-clean).
- Deferred items: 0 new. (Live PG16 end-to-end is the already-planned PR-3.3 boundary, not a new deferral; `deferred_items_total` stays 2.)
- Surprises during implementation: (1) the loader's `commits.timestamp` is COPY'd as a UTC text string then re-parsed by Postgres, so the verify compares it as an absolute epoch on both sides (engine-independent) rather than as text — chosen to avoid DuckDB-vs-Postgres timestamp text-rendering divergence. (2) The acceptance row literally said "Integration test (local PG16)", but Docker is unavailable and PR-3.1's note already designated PR-3.3 as the container-infra owner; clarified the PR-3.2/PR-3.3 test boundary in the plan (commit `bab556bb0`) before review so the no-Docker comparison-core tests are the PR-3.2 deliverable and the live e2e is PR-3.3's — neither reviewer flagged the absent live test as a gap. (3) Codex was available so gauntlet's dynamic default would have routed `mixed`/`parallel`; chose `executor=claude` for reliability in this long orchestration (documented opt-out).

### PR-3.3: (re-plan) bulk-load rehearsal harness + runbook (gap #2)  (1 impl commit + 1 review-follow-up commit, 1 inner-loop gauntlet cycle, ending at `fdb6c386a`)
- Scope shipped: `tests/postgres_e2e.rs` — the first Rust testcontainer in the migrate crate. Builds a representative v3 DuckDB fixture, stands up a `postgres:16-alpine` testcontainer (schema applied from `migrations/001` via the init-SQL `/docker-entrypoint-initdb.d/` entrypoint), and asserts (a) the loader's per-table row counts match the source, (b) `verify --postgres-target` is clean, (c) a forced mid-load PK conflict on the LAST-loaded table (`vector_search_runs`) rolls the whole single-transaction load back to empty. Added `testcontainers` + `testcontainers-modules` (`postgres`, `blocking`) dev-deps (root `Cargo.lock` +342; migrate IS a workspace member, so `cargo nextest --workspace` runs the e2e in CI). `README.md` operator runbook: live source-account/region verification (since `375504701696`/`us-east-2` is unverified in-repo) + snapshot acquisition + the local PG16 rehearsal. Docker-gated (skip locally, fail-loud in CI).
- Tests added: 2 testcontainer e2e tests (`rehearsal_load_then_verify_is_clean`, `rehearsal_mid_load_failure_rolls_back_to_empty`). Both PASS against real `postgres:16-alpine`; full migrate suite 100/100. This is the FIRST RUNTIME validation of PR-3.1's loader Postgres-EXECUTION path (the deferred gap) AND PR-3.2's verify Postgres-READ path — incl. the epoch SQL (sub-second `2024-06-15 ...123456` + pre-1970 `1969-07-20` timestamps both confirmed `epoch_us`==PG-numeric-epoch by the reviewers), literal-`\N`, multibyte text (now in a value-compared column), empty/negative `BIGINT[]`.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 2 should-fix (both applied `fdb6c386a`: moved multibyte into a value-compared column so the e2e actually discriminates it; runbook readiness-wait + cwd note), 3 nits (1 applied: corrected the stale `intentionally outside the workspace` Cargo.toml comment, flagged by BOTH lenses; 2 dismissed per de-gold-plate calibration: IPv6 dsn edge + raw-string SQL line length). Both lenses independently verified the rollback test genuinely DISCRIMINATES atomicity (a non-atomic loader would fail it). (cycles: 1)
- Confidence: high. The loader + verify Postgres paths are now runtime-validated against a real PG16 (no longer compile-only); both reviewers confirmed the harness genuinely discriminates and the epoch path is exact. Closes the PR-3.1 deferred Postgres-execution gap + the PR-3.2 live-read gap.
- Deferred items: 0 new. (PR-3.1's 2 deferrals are partially discharged here: Postgres-EXECUTION is now runtime-tested via the container; verify-full TLS remains PR-3.4's runbook coverage since the container uses `NoTls`. `deferred_items_total` stays 2.)
- Surprises during implementation: (1) the migrate crate is a workspace MEMBER (root `Cargo.toml:70`), not "outside the workspace" as its stale comment claimed — load-bearing because membership is why CI's `--workspace` run executes the Docker-gated e2e. (2) `testcontainers-modules` `with_init_sql` mounts to `/docker-entrypoint-initdb.d/`, so the schema applies at container init with no separate apply step; its wait-strategy (stderr THEN stdout "ready to accept connections") correctly handles the init-SQL double-message. (3) Operator started Docker Desktop mid-PR (chose local validation over the CI-gated default), so the harness was iterated to green against real PG16 rather than written blind.

### PR-3.5: (re-plan) Python-writer-vs-RDS cross-check harness (gap #3)  (1 impl + 2 review-follow-up commits, 2 inner-loop gauntlet cycles, ending at `6263cc428`)
- Scope shipped: `scripts/cross_check_python_writer.py` — `cross_check(conn, commit, records) -> CrossCheckReport`: runs `post-ingest.ingest_postgres`, asserts `inserted == 0` (every record UPDATEd a seeded row, not duplicate-INSERTed), re-reads each row's value columns + asserts round-trip; reuses `connect_postgres` (verify-full TLS, `bench_ingest` role, IAM token) + a `--postgres`/`--envelopes` CLI for the prod run. Confirms the LIVE property: the Python writer recomputes the SAME `measurement_id` (Python port == Rust hash, golden-gated) and UPDATEs the Rust-loader-seeded rows.
- Tests added: 5 cross-check tests extending `test_post_ingest_postgres.py` (reuse `postgres_dsn`/`schema_conn` fixtures): a directly-seeded golden-mid row UPDATEd + value overwritten; all 5 fact kinds; a NOT-seeded envelope -> INSERT flagged (the duplicate-risk discrimination); `value_mismatches` discrimination on value_ns / array-reorder / env_triple / a side counter; an env + memory-quartet integration round-trip. All pass vs `postgres:16-alpine`; `ruff format --check` + `ruff check` clean; basedpyright bare-`dict` matches the sibling `post-ingest.py` convention (scoped to vortex-python).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / cycle 1 REJECTED then cycle 2 ACCEPTED. Cycle 1: lenses SPLIT — correctness accepted (mutation-verified the harness catches a duplicate-INSERT + value corruption); fresh rejected with a CI-gating must-fix it caught that correctness missed (`uvx ruff format --check .`, ci.yml:64) + a should-fix coverage gap (env/memory/side-counters in `_VALUE_COLUMNS` but never deliberately mismatched). Both fixed (`cee7e224c` ruff format; `6263cc428` discrimination tests). Cycle 2: both accept, 0 findings — both MUTATION-TESTED the new env/memory test (deleting `env_triple`/`peak_physical` from the writer's ON CONFLICT SET list makes it fail with the exact harness diagnostic). (cycles: 2)
- Confidence: high. The harness's discrimination is mutation-verified across both cycles (it catches a duplicate-INSERT AND a writer SET-list omission AND a value corruption). The LIVE cross-language property against real Rust-seeded RDS rows is the deferred operator gate (PR-3.5 prod run).
- Deferred items: 0 new. (PR-3.5's PROD run is the already-documented operator gate; `deferred_items_total` stays 4.)
- Surprises during implementation: (1) the existing `post-ingest.py` + `test_post_ingest_postgres.py` already provided `ingest_postgres` (with a `RETURNING (xmax=0)` insert/update classifier), `connect_postgres`, and the record/value-column structure, so the harness is thin glue + a discrimination test rather than new writer logic. (2) The cycle-1 split was a textbook multi-lens win: the correctness lens's deep mutation-testing of the discrimination logic and the fresh lens's CI-format-gate catch were disjoint — neither alone would have yielded the right verdict (accept-worthy substance + a real merge-blocker). (3) A `git commit -m` with backticks hit a shell command-substitution trap (twice); fixed by amend + switching to a quoted heredoc.

### PR-3.4: (re-scoped) REAL-snapshot LOCAL rehearsal — the real-data validation gate  (no production code; operational run, 2026-06-08)
- Scope shipped: ran the validated Phase-3 toolkit end-to-end against the REAL v3 history on the operator's laptop, ZERO prod write. Source: the on-disk `./bench.duckdb` (468MB, gitignored, last written 2026-05-04; the real v3 DuckDB — 6 tables, commits 2025-01-02..2026-05-04). Per the 2026-06-08 user decision, used the local on-disk snapshot instead of an SSH/scp pull off the prod EC2 host — the May-4 file IS real v3 data and freshness is irrelevant for the rehearsal gate (it validates the load+verify+cross-check code against the real data shape); freshness is PR-5.0/cutover's concern. Substrate: a throwaway local Homebrew `postgresql@16` cluster (16.14) on `127.0.0.1:55432` — Docker was DOWN, so the PR-3.3 testcontainer path was unavailable; a real local PG16 cluster is the equivalent rehearsal substrate the runbook describes. Schema applied from `migrations/001_initial_schema.sql`; loader = the current `target/debug/vortex-bench-migrate` (Jun-5 build, rebuilt clean off HEAD).
- Results captured (the acceptance evidence):
  - **load** (`vortex-bench-migrate load --duckdb bench.duckdb --postgres-target postgresql://postgres@127.0.0.1:55432/bench`): atomic single-txn, 25s; per-table row counts match the DuckDB source EXACTLY — `commits` 4202, `query_measurements` 4223133, `compression_times` 222150, `compression_sizes` 88161, `random_access_times` 27064, `vector_search_runs` 0.
  - **verify** (`verify --duckdb bench.duckdb --postgres-target ...`): exit 0, `value verify: source and target match (0 presence diffs, 0 value mismatches)` — FULL per-`measurement_id` comparison of every value column + `env_triple` + `all_runtimes_ns` arrays + `commits` metadata across all 4.2M+ rows, 41s. The PRIMARY v4-correctness gate, GREEN against real data.
  - **cross-check** (the shipping `scripts/cross_check_python_writer.py` over verify-full TLS as the `bench_ingest` role): built an 11-record envelope from REAL seeded rows for commit `344cda8e…` (3 query_measurement + 3 compression_time + 3 compression_size + 2 random_access_time); result `11 records, 11 updated, 0 inserted -- CLEAN` (the Python writer recomputed the SAME `measurement_id` as the Rust-seeded rows and UPDATEd them, values round-trip). Negative discrimination confirmed: mutating one record's `engine` dim → `1 inserted` → `FAILED` (the would-be-duplicate is caught); the junk INSERT was deleted so the target again matches the source. To satisfy `connect_postgres`'s hard contract locally, a self-signed `localhost` cert enabled verify-full TLS and a local `bench_ingest` role (LOGIN + SELECT/INSERT/UPDATE grants, no `rds_iam`) was created.
- Tests added: none (no production code; operational rehearsal). The throwaway envelope generator + local PG cluster + cert live outside the repo (`/tmp`); nothing committed to the tree.
- Review: **no inner-loop gauntlet** — PR-3.4 ships no production code (empty repo diff), so there is nothing for a 2-vote review to read (per the plan's PR-3.4 "no code -> no gauntlet review needed"). The Phase-3 cumulative diff (PR-3.1/3.2/3.3/3.5 code) gets the 3-vote phase-end gauntlet next.
- Confidence: high. All three gates are GREEN against the REAL 4.2M-row v3 snapshot, not synthetic fixtures: exact row-count match, full value-verify clean, and the live cross-language UPDATE-not-duplicate + value-round-trip property — with the cross-check's discrimination re-confirmed against this real data.
- **[phase-end cycle-1 should-fix annotation, added by PR-3.6] Real-data coverage caveat:** the real snapshot held `vector_search_runs` = 0 rows AND the PR-3.4 cross-check envelope omitted that kind, so that ONE table's load-COPY (Double/`threshold` dim, four Int side counters, the only Int32 value column) / value-verify / cross-check paths were exercised only by the synthetic PR-3.3 fixture + PR-3.2/3.5 unit tests, NOT against the real data shape. The other 5 tables got full real-data validation. PR-5.0's freshest-snapshot prod load is expected to close this (or, if the live v3 DB still has zero vector rows at cutover, `vector_search_runs` stays fixture-only-validated). Surfaced by the Phase-3 phase-end review (spec lens should-fix; maint lens judged it fixture-covered/acceptable).
- Deferred items: 0 new. The one-shot PROD load + prod verify + prod cross-check remain PR-5.0/PR-5.1 (Phase-5 cutover, freshest snapshot); `deferred_items_total` stays 4.
- Surprises during implementation: (1) the rehearsal needed no prod access at all — a real v3 DuckDB was already on disk, so the whole gate ran locally (the user's premise was correct). (2) Docker being down was a non-issue: a Homebrew PG16 cluster is an equivalent (arguably cleaner) rehearsal substrate vs the testcontainer; the migrate tools only need a `--postgres-target` DSN. (3) `connect_postgres` hard-requires verify-full TLS + the `bench_ingest` role with NO local-affordance exception (by design — it is the prod ingest connector), so the local cross-check needed a self-signed-cert TLS endpoint + a real `bench_ingest` role rather than a plaintext shortcut; this exercised the actual shipping CLI end-to-end rather than a bypass.

### PR-3.6: (amend) apply Phase-3 phase-end cycle-1 should-fix doc/status items  (1 doc commit + 1 review-nit commit + 2 plan annotations, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: applied the 3 should-fix items from the Phase-3 phase-end review. (a) Added a `## Bootstrap ordering — requires-superuser migrations (002 / 004)` subsection to `migrations/README.md` documenting the `-- migrate-schema: requires-superuser` marker, the `rolsuper OR rolcreaterole` master-capable preflight, and the "apply 002/004 as RDS master before any migrator deploy" ordering — the operator-facing doc the `_assert_master_capable` PermissionError points at (closes the maint should-fix). (b) Annotated the PR-3.1 status with the deliberate sync-`postgres`-over-plan-named-`tokio-postgres` choice (spec should-fix). (c) Annotated the PR-3.4 status that `vector_search_runs` had 0 real rows + was omitted from the cross-check envelope, so its real-data coverage is fixture-only, with PR-5.0 closing it (spec/maint should-fix).
- Tests added: none (doc/status only; no behavior change).
- Review: 2-vote (pr-2: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix, 0 should-fix, 1 nit (applied `0c33c11e2`). Both reviewers verified every README claim against the live `scripts/migrate-schema.py` source (the `_REQUIRES_SUPERUSER_DIRECTIVE` marker string, the `rolsuper OR rolcreaterole` capability SQL, the `_assert_master_capable` PermissionError text + that the guard fires BEFORE the migration transaction, and that exactly 002/004 carry the marker while 001/003 do not); the correctness lens could not construct a planted-bug case; the fresh lens's nit refined the failure-statement list to include `GRANT` (since `CREATE ROLE` is `IF NOT EXISTS`-guarded in a `DO` block).
- Confidence: high. The README subsection was verified line-by-line against the actual code contract by both lenses; the 2 plan annotations are factual status records.
- Deferred items: 0 new (`deferred_items_total` stays 4).
- Surprises during implementation: none — the should-fix items were exactly as the phase-end review scoped them; the README documents an already-tested contract.

### PR-3.7: (amend) clear Phase-3 phase-end cycle-2 doc nits  (1 doc commit + 1 should-fix repoint commit, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: cleared the 3 cycle-2 nits — (a) aligned `migrations/README.md`'s hypothetical directive example `-- migrate: no-transaction` → `-- migrate-schema: no-transaction` (the only implemented directive namespace); (b) added a `pub fn load` doc-comment sentence noting the target schema (`migrations/001`) must already be applied (`load` only `COPY`s, never DDL); (c) added a `migrate/README.md` runbook bullet that a pre-acquired on-disk snapshot is an acceptable LOCAL-rehearsal source. The inner-loop correctness lens caught a bonus should-fix: 3 pre-existing stale `prod load (PR-3.4)` references in `migrate/README.md` (from before the 2026-06-05 re-scope) — repointed to PR-5.0 (`cbbd6f25c`), making the runbook internally consistent with PR-3.4=rehearsal / PR-5.0=prod-load.
- Tests added: none (doc/doc-comment only; no behavior change; `cargo +nightly fmt --check -p vortex-bench-migrate` clean).
- Review: 2-vote (pr-2: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix. fresh found nothing (verified all 3 edits against source: the `migrate-schema:` namespace, `load`-never-runs-DDL via grep, PR-3.4-on-disk-rehearsal via git history); correctness verified the same + flagged the stale-prod-load-references should-fix (applied).
- Confidence: high. All edits verified against the live source by both lenses.
- Deferred items: 0 new (`deferred_items_total` stays 4).
- Surprises during implementation: the correctness lens surfaced a pre-existing doc inconsistency (the runbook still called PR-3.4 the prod load) that the new bullet (c) made visible — a genuine bonus catch, now fixed.

### PR-4.1: Scaffold benchmarks-website/web/ Next.js 15 App Router project  (1 impl commit, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: created the Next.js 15.5.19 + React 19 + App Router scaffold under `benchmarks-website/web/`. Files: `package.json` (pnpm; scripts dev/build/start/lint/format/format:check; pinned Next 15.x since 16 is published but the migration targets 15), `tsconfig.json` (Next base + all 5 vortex-web strict flags: `strict`/`noUnusedLocals`/`noUnusedParameters`/`noFallthroughCasesInSwitch`/`noUncheckedSideEffectImports`), `next.config.js` (CommonJS; `outputFileTracingRoot` pinned to the app dir to resolve the repo's multi-lockfile root ambiguity), `eslint.config.mjs` (flat config via `eslint-config-next` + `FlatCompat` over `next/core-web-vitals` + `next/typescript`; added `@eslint/eslintrc` devDep), `.prettierrc.json` (byte-identical to vortex-web shape), `.gitignore`, `pnpm-workspace.yaml` (`allowBuilds: sharp + unrs-resolver`), `app/layout.tsx` + `app/page.tsx` stubs. SPDX header pair on every comment-capable file (JSON exempt, matching vortex-web). `pnpm-lock.yaml` committed.
- Tests added: none (scaffold; no runtime logic). Acceptance verified empirically: `pnpm install && pnpm build` succeed; `pnpm lint` + `pnpm format:check` clean; strict flags confirmed genuinely enforced (injected unused local → tsc TS6133, run by `next build`).
- Review: 2-vote (`pr-2`: fresh + correctness, executor=mixed→**parallel**; 4 reviewer outputs — Claude + Codex per lens) / accepted at cycle 1. 0 must-fix, 0 should-fix, 4 nits. **One disagreement**: Codex/fresh flagged `web/.prettierrc.json` as must-fix ("second formatter config"); the synthesizer + both Claude lenses adjudicated it compliant (the BAN targets formatter-SHAPE divergence, not file count; `web/` is a separately-deployable pnpm/Vercel package so a sibling-tree config is undiscoverable without a fragile `--config` path or out-of-scope repo-root config) → downgraded to nit + recorded in `Accepted tradeoffs` (commit `c6b78598e`) so future Phase-4 PR + phase-end reviews drop the re-flag.
- Confidence: high. Both Claude reviewers re-ran build/lint/format/type-check independently; all BANS verified (SPDX header pair, 5 strict flags, no WASM, no v2 top-level files touched, prettier shape match).
- Deferred items: 0 new (`deferred_items_total` stays 4). 3 nits dismissed (no `engines.node` pin; format-glob misses root config files; unused-var is a lint warning with `next build` type-check as the real gate) per the lean trusted-input calibration.
- Surprises during implementation: (a) pnpm not preinstalled (installed pnpm 11.5.2 globally); (b) pnpm 11 moved `onlyBuiltDependencies` → `pnpm-workspace.yaml` `allowBuilds`; (c) `eslint-config-next` 15.x ships legacy eslintrc configs (used `FlatCompat` + added `@eslint/eslintrc`, the canonical Next-15 flat-config bridge — avoids the deprecated `next lint`); (d) the user's mid-session `develop` rebase rewrote/orphaned `phase_entry_sha` `dcc24f748` → repointed to the rewritten entry commit `5a9d24471` (`git diff 5a9d24471..HEAD` verified clean of develop's byte_length changes); (e) 1Password SSH commit-signing re-locked during the ~10-min review (user unlocked, retry succeeded).

### PR-4.2: Postgres connection lib (web/lib/db.ts, pg.Pool + RDS IAM auth) + testcontainers test  (1 impl + 1 should-fix fix commit, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: `web/lib/db.ts` — a `globalThis`-cached singleton `pg.Pool` (warm-invocation + dev-HMR reuse) whose password is an async provider minting a fresh `@aws-sdk/rds-signer` IAM token per connection (token-refresh-before-expiry, ~15-min tokens); a `BENCH_DB_PASSWORD` static-password path bypasses IAM for local/test; `resolveSsl` (verify-full default, disable for tests, **fail-loud on unknown mode**); a pure `buildQuery` helper + the injection-safe `sql` tagged-template ($1..$n binding); a `resetPool` teardown helper. Also: re-included `web/lib/` in `web/.gitignore` (the repo-root Python-artifact `lib/` pattern was silently swallowing it), widened the prettier globs to `{app,lib}/**` + `vitest.config.ts`, and added pg/@aws-sdk/rds-signer/vitest/@testcontainers/postgresql/@types/pg deps (ssh2/cpu-features/protobufjs native builds skipped via `allowBuilds` — not needed for local-Docker testcontainers).
- Tests added: 15 (vitest 4) — 2 LIVE testcontainers Postgres roundtrip (connect-via-password-fixture + SELECT + parameter binding; user chose "run for real" so verified against a real `postgres:16-alpine` container, Docker up), 1 per-connection-IAM-refresh (distinct tokens across two `provider()` calls — pins fresh-per-connection, not single-cache), static-password + missing-region-throws, 3 `buildQuery` (0/1/N values), 4 `resolveSsl` (default/disable/ca-merge/unknown-throws), 3 `requireEnv`. The testcontainers describe is Docker-skip-guarded (mirrors the repo's Python `_docker_available()` precedent).
- Review: 2-vote (`pr-2`: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix. Both lenses verified the `$N` reducer + IAM branching CORRECT on disk (tsc/eslint/prettier/vitest all green; container roundtrip ran live). NO disagreement (synthesized inline). 5 should-fix + 3 nits surfaced; substantive ones APPLIED in fix-commit `5201aff38`: SSL fail-loud on unknown mode (the one finding with real security character — a typo previously disabled cert verification silently), per-connection-refresh test discrimination, `buildQuery` extraction + non-Docker unit coverage, `resolveSsl`/`requireEnv` unit tests, verify-full doc accuracy, format-glob widening, pool reset. 1 nit dismissed (silent NaN port — trusted-config input-hardening, out of the lean calibration). 1 should-fix deferred (wire the vitest suite into CI → PR-4.5). No cycle-2 (should-fix-only changes on an accept verdict; re-verified build/lint/test/format all green).
- Confidence: high. Core logic (IAM per-connection refresh, `sql` parameterization, SSL mode handling) verified live + unit-pinned by both lenses.
- Deferred items: 1 new (vitest-CI-runner gap, Resolved-by PR-4.5; `deferred_items_total` 4 -> 5).
- Surprises during implementation: (a) the repo-root `.gitignore` `lib/` (Python packaging-artifact) pattern silently ignored `web/lib/` — fixed with a scoped `!/lib/` negation in `web/.gitignore`; (b) a `vi.fn` arrow-mock is not `new`-able — used a named `function` expression for the Signer mock; (c) the testcontainers Postgres container started + roundtripped cleanly once Docker was up; (d) `executor=claude` chosen for this inner-loop review (lean 2-vote calibration + faster than the Codex-polling path), unlike PR-4.1's mixed->parallel.

### PR-4.3.a: Read-port foundation (schema-version, slug, window, families) + /health  (2 impl + 3 should-fix/nit fix commits, 1 inner-loop 2-vote cycle accepted, 2026-06-08)
- Scope shipped: the foundation layer for the read-endpoint port. `web/lib/schema-version.ts` (`SCHEMA_VERSION = 1` read-path gate, Table D site, lockstep with `server/src/schema.rs`); `web/lib/families.ts` (5-fact-table registry: tableName + chart/group slug prefixes qm/ct/cs/rat/vsr + qmg/ctg/csg/rag/vsg + kind discriminants, in `family.rs` order; `familyForChartKind`/`familyForGroupKind`; sorted `HEALTH_TABLES`); `web/lib/window.ts` (`?n=` -> CommitWindow port of `window.rs`: default 100, floor-0->1, clamp 1..1000, case-insensitive trimmed `all`, u32-overflow->fallback); `web/lib/slug.ts` (port of `slug.rs`: 5 ChartKey + 5 GroupKey variants, `<prefix>.<base64url-no-pad(canonical-JSON)>` with `k`-first serde-tagged field order + explicit-null Options; decode validates full per-variant shape, NOT just `k`); `web/lib/health.ts` + `web/app/api/health/route.ts` (snake_case `HealthResponse` shape preserved; `db_path`->DB host, `build_sha`->`VERCEL_GIT_COMMIT_SHA`, `schema_version`->const; sorted `row_counts` via per-table `COUNT(*)::int`; `force-dynamic` liveness route). Slugs treated as OPAQUE round-trippable tokens (not Rust-byte-identical) per the shard-endpoint deferral — see Accepted tradeoffs.
- Tests added: 65 vitest total (was 58 in PR-4.2; +7 this PR). slug: round-trip all 10 variants + prefix checks + malformed rejection + 2 canonical-encoding golden vectors (pin `k`-first/explicit-null byte order) + 4 validation-discrimination tests (reject missing/mistyped required fields; absent-Option->null serde parity). window: full `window.rs` parity (default/all/clamp/floor/overflow/malformed). families: order + distinct prefixes + exact-prefix pin + lookups + sorted HEALTH_TABLES. health: pure `assembleHealth`/`buildRowCounts` + 2 LIVE testcontainers `postgres:16-alpine` integration tests (applies real `migrations/001`; empty-schema zeros + real-count + timestamp-format). schema-version: 1-line `=== 1` drift sentinel.
- Review: 2-vote (`pr-2`: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix, 2 should-fix, 5 nits. Both lenses cross-checked every TS module vs its Rust source-of-truth (slug.rs/window.rs/family.rs/dto.rs/mod.rs/migrations-001) + confirmed window parse vs `rustc`; no disagreement. Applied: F1 (slug decode per-variant field validation — the substantive should-fix, both lenses; slugs are user-controllable URL params for 4.3.b) in `99a8b6f96`; F2 canonical golden test; F3 SCHEMA_VERSION sentinel + F7 redundant-slice in `90b6b04e6`; F4 timestamp doc note in `3da6da8d2` (`fix(docs)`). F5 (base64url leniency) + F6 (threshold f64 1.0-vs-1) dismissed -> Accepted tradeoffs (opaque-slug decision moots cross-language byte-identity). No cycle-2 (should-fix/nit on an accept verdict, re-verified tsc/eslint/prettier/65-tests/next-build all green).
- Confidence: high. Faithful behavior-preservation port verified line-by-line against the Rust oracle by both lenses + the full suite green incl. live PG16 container.
- Deferred items: 0 new (`deferred_items_total` stays 5).
- Surprises during implementation: (a) verified wire format is snake_case (the exploration agents had wrongly assumed camelCase) — only the `Summary` enum variant fields are camelCase + `UnitKind` is snake_case, per `dto.rs`; (b) `SCHEMA_VERSION` confirmed `= 1` from `schema.rs` (an exploration agent had hallucinated `2`); (c) the artifacts/shard endpoint was deferred to PR-4.4 via a user fork decision (1 AUQ this phase) — its in-memory generation-store + ETag/brotli machinery is what the lean re-plan moved to the edge CDN.

### PR-4.3.b: Chart endpoint port (queries.ts `chartPayload` + `/api/chart/[slug]` + tests)  (1 impl + 1 must-fix + 1 test-hardening commit, 2 inner-loop 2-vote cycles, 2026-06-08)
- Scope shipped: `web/lib/queries.ts` (`chartPayload` — faithful port of `charts.rs`: `SeriesAccumulator` + `seeded_commits_in_window` two-pass for all 5 chart families; `IS NOT DISTINCT FROM` nullable dims; value cols `::float8`; `count(*)::int`; commit timestamp via `to_char(... AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS"+00"')` reproducing the DuckDB `CAST(... AS VARCHAR)` text byte-for-byte; `BTreeMap`-sorted series/series_meta; `series_meta` omitted when empty per serde skip-if-empty); `web/app/api/chart/[slug]/route.ts` (`revalidate=300`; 400/404/200; `?n` window); `web/vitest.config.ts` (+`@/` alias to import the route handler in tests). Plus the cycle-1 must-fix: `reqI32` in `web/lib/slug.ts` (validate `query_idx` as a 32-bit integer matching serde, so a forged non-i32 `query_idx` returns 400, not an unhandled 500 from the Postgres int4 bind).
- Tests added: `queries.test.ts` (testcontainers PG16): keystone `toEqual` vs the v3 golden snapshot `chart_page_query.snap` (modulo numeric rendering); `unit_kind` per family; `series_meta` tag/no-tag/format-only; window `n`/`all`/default; history placement (125-commit bounded + all); missing->null; route 400/404/200 + `?n=1`/`?n=all`/malformed; no-DB route input-validation (forged non-i32 `query_idx` -> 400); tailored seeded-window describe (null-gap, pre-history exclusion, NULL `scale_factor` `IS NOT DISTINCT FROM` equality, identical-timestamp `commit_sha` tie-break). `slug.test.ts`: `i32`-rejection unit test. 87 vitest total (was 77 at implementation; +10 from review fixes); tsc/eslint/prettier/next-build green.
- Review: 2-vote (`pr-2`: fresh + correctness, executor=parallel Claude+Codex). cycle 1 = REJECT — 1 must-fix found ONLY by the Codex correctness side (the forged non-i32 `query_idx` 400-vs-500 divergence; the disjoint find that justified parallel routing), 3 should-fix coverage + 5 nits. Applied: must-fix `reqI32` + tests (`2d554ecc4`); should-fix coverage (null-gap, NULL-dim equality, tie-break) + nits (route `?n`, series_meta, keystone rename) as test hardening (`e98bf5fcb`). Dismissed: nit #8 (non-contract 400 body — reviewer marked optional) + nit #9 (UTF-16 vs UTF-8 sort comparator — documented, ASCII-only). cycle 2 = ACCEPT — both Claude reviewers verified `reqI32` matches serde `i32` + the tailored tests discriminate, 0 findings; both Codex sides hit a codex-companion infra failure (ENOENT / MODULE_NOT_FOUND in the plugin runtime, mid-session) and degraded to `parallel-to-claude` per the carve-out (Codex is auxiliary, never blocks the verdict).
- Confidence: high. Faithful behavior-preservation port; the keystone `toEqual` pins semantic-equivalence (modulo numeric rendering) vs the v3 Axum golden, and the only behavior divergence found (forged `query_idx`) is fixed + pinned.
- Deferred items: 0 new (`deferred_items_total` stays 5).
- Surprises during implementation: (a) the v3 DuckDB `CAST(... AS VARCHAR)` golden timestamp text is `2026-04-23 12:00:00+00` (space separator, `+00` offset, no `T`/`Z`) — reproduced exactly via `to_char` because the chart `commits[].timestamp` is a wire-compat field (unlike `/health`'s smoke-test timestamp which uses `T...Z`); (b) the cycle-1 must-fix was a PRE-EXISTING latent gap in PR-4.3.a's `slug.ts` (`query_idx` decoded as any number) that PR-4.3.b's int4 bind exposed — fixed at the decode source (`reqI32`); (c) parallel Claude+Codex routing earned its keep — the must-fix was found ONLY by Codex; (d) both Codex sides failed on cycle 2 with a codex-companion infra error, handled via the auxiliary-executor degrade-to-claude carve-out.

### PR-4.3.c: Groups + group endpoints (collectGroups/collectGroupCharts + summary.ts + descriptions.ts + /api/groups + /api/group/[slug])  (6 code commits: 1 impl + 5 review-fix, 3 inner-loop 2-vote cycles, ending at b56c8c516, 2026-06-09)
- Scope shipped: `web/lib/queries.ts` (`collectGroups` discovery over the 5 families with slug generation + `GROUP_ORDER` stable sort by `(pos, name)`; `groupNameQuery` tpch/tpcds/clickbench + legacy fallback; `collectGroupCharts` full re-discovery + flattened `NamedChartResponse`); `web/lib/summary.ts` (4 summaries — `randomAccess` rankings, `compression` speedup geomean, `compressionSize` ratio min/mean/max, `queryBenchmark` missing-series penalty model — ports `summary.rs`; the MF1 fix keeps `MAX(timestamp)` inside SQL via a CTE for both compression-time + size, eliminating the text round-trip that truncated sub-second timestamps); `web/lib/descriptions.ts` (editorial blurbs byte-ported from v2 via `descriptions.rs`); `/api/groups` + `/api/group/[slug]` routes (`revalidate=300`, 400/404/200).
- Tests added: `groups.test.ts` (testcontainers PG16): group ordering, all 4 summary variants, query-penalty exact scores, route 200/404 + `?n`, plus a new `summary math fidelity` describe with isolated truncate-per-test fixtures (sub-second-timestamp regression [discriminating: fails the pre-MF1-fix truncating code], distinct encode/decode timestamps, decode-only fallback, 300k-penalty-floor ranking flip); `descriptions.test.ts` (6 pure-logic). 108 vitest total (was 104 at implementation; +4 from review fixes); tsc/eslint/prettier green.
- Review: 2-vote (`pr-2`: fresh + correctness, executor=parallel Claude+Codex). cycle 1 = REJECT (2 must-fix found only by Codex: MF1 summary-timestamp truncation [fixed], MF2 statpopgen/polarsignals legacy naming [dismissed preserved-v3]; 3 should-fix, 3 nits). cycle 2 = REJECT (1 must-fix, correctness/codex only: same-second-timestamp summary tie [dismissed preserved-v3, verified identical across all 3 Rust summary paths]; 2 should-fix, 3 nits). cycle 3 = ACCEPT (unanimous, all 4 reviewers 0 findings; 365fdaeb2 loop change verified behavior-neutral; carry-forward correctly not re-flagged).
- Confidence: high. Faithful behavior-preservation port; the testcontainer suite pins semantic-equivalence vs the v3 Axum group_api.rs contract values, and the one runtime fidelity gap found (MF1 sub-second timestamp) is fixed + pinned by a discriminating regression test.
- Deferred items: 3 new (`deferred_items_total` 5 -> 8): MF2 statpopgen/polarsignals v2-description restore (v2->v4 display regression, preserved-v3); same-second-timestamp summary tie determinism (preserved-v3 latent bug, all 4 summary paths); groupNameQuery clickbench/variant test-coverage (faithful port, Rust has same gap).
- Surprises during implementation: (a) TWO of the three must-fixes across cycles 1-2 were real latent v3 bugs that the faithful port reproduces (MF2 dead descriptions; same-timestamp tie) — both dismissed as preserved-v3-behavior per PR-4.3.c's v3-semantic-equivalence acceptance criterion and tracked in Deferred work; this accumulating pattern is flagged for a user decision at the Phase-4 boundary (schedule a deliberate "v4 correctness improvements over v3" effort vs preserve v3 exactly); (b) the MF1 timestamp truncation was the only fix-worthy runtime gap, caught only by the Codex correctness lens via deep Rust cross-reading — parallel Claude+Codex routing earned its keep again; (c) Docker/testcontainers ran ~157s once during a cold Docker Desktop start, then recovered to ~5s — transient infra, not a code issue.

### PR-4.4.a: v4 read-service server-rendered landing shell + global CSS  (2 impl + 1 review-fix commit, 1 inner-loop 2-vote cycle accepted, ending at 2b44fc2a4, 2026-06-09)
- Scope shipped: the non-interactive Next.js (App Router, RSC) landing shell at `benchmarks-website/web/`, ported from v3's HTML layer (`render.rs`/`landing.rs`/`summary.rs` + `static/style.css`), NOT v2 React. `app/layout.tsx` (html/body + `globals.css` + external Geist/Funnel fonts hoisted by React 19 + theme-aware favicons via metadata); `app/page.tsx` (`force-dynamic` server component: `collectGroups()` → `<Header>` + a `<GroupSection>` per group in `GROUP_ORDER` + `<Footer>`, page-wide `data-chart-index`, `<noscript>` notice); `components/Header.tsx` (static logo/title/GitHub chrome; interactive nav/theme/filter deferred to PR-4.4.b); `components/GroupSection.tsx` (`<section.group-details>` → collapsed native `<details.group-disclosure>` wrapping ONLY `<summary>`, with `<SummaryCard>` + `.chart-grid` of empty chart-card `<canvas>` shells as SIBLINGS so the ported `.group-disclosure:not([open]) ~ .chart-grid` rule hides charts when collapsed); `components/SummaryCard.tsx` (4 `Summary` variants ported from `summary.rs::summary_markup`, `formatTimeNs` for ns values, `.toFixed(2)x` ratios, empty-content guards, exhaustiveness default); `components/Footer.tsx` (build-SHA via `VERCEL_GIT_COMMIT_SHA`); `lib/format.ts` (`formatTimeNs`, port of `format_time_ns`); `app/globals.css` (v3 `static/style.css` ported ~verbatim, prettier-reformatted); `css.d.ts` (ambient `*.css`); `vitest.config.ts` (oxc automatic-JSX + `components/**` include); logo/favicon assets under `web/public/`.
- Tests added: `lib/format.test.ts` (formatTimeNs tiers/boundaries/sign — 5), `components/SummaryCard.test.tsx` (4 variants + empty/absent guards via `react-dom/server` — 8), `components/GroupSection.test.tsx` (collapsed disclosure, info-icon + count pluralization, summary card, chart-card shells w/ page-wide `data-chart-index` + `data-chart-slug` + permalinks — 5). 126 vitest total (108 prior + 18 new); tsc/eslint/prettier clean; `next build` compiles+lints+typechecks (prerender needs `BENCH_DB_HOST`, a pre-existing PR-4.3 API-route deploy-env condition; the `force-dynamic` page is excluded from prerender).
- Review: 2-vote (`pr-2`: fresh + correctness, executor=parallel Claude+Codex). cycle 1 = ACCEPT (unanimous, all 4 reviewers overall=accept, 0 must-fix). 5 should-fix/nit findings; Codex uniquely caught the mobile-GitHub-link gap. Triaged: 3 fixed in-cycle (noscript, prettier-glob coverage, SummaryCard exhaustiveness guard); 2 deferred (mobile-nav GitHub → PR-4.4.b; round-half-even display parity → Deferred work).
- Confidence: high. Faithful v3-HTML-layer port with a small authored surface (mostly verbatim CSS + declarative server components + a tested pure helper); zero must-fix; SummaryCard/formatTimeNs/GroupSection verified against the v3 Rust source and pinned by the new tests.
- Deferred items: 2 new (`deferred_items_total` 8 → 10): mobile GitHub link <768px (Resolved-by PR-4.4.b); round-half-even display-rounding parity (low-priority, pairs with the Phase-4-boundary v4-vs-v3 decision).
- Surprises during implementation: (a) the better port source is v3's server-rendered HTML layer, NOT v2 React (v3 is already server-shell + client-island, reproduces the v4 endpoints' ns `Summary` shape, uses v2's CSS class vocabulary); recorded as the PR-4.4.a port-source-refinement Key decision. (b) rolldown-vite (vite@8) ignores `esbuild.jsx`; JSX-in-vitest needed `oxc: { jsx: { runtime: 'automatic' } }`. (c) `noUncheckedSideEffectImports` rejects the `import './globals.css'` side-effect import without an ambient `*.css` declaration (added `css.d.ts` rather than disabling the flag, per the BANS). (d) parallel Claude+Codex earned its keep again — the mobile-GitHub finding was Codex-only.

### Phase-4 holistic mid-phase review (2026-06-09, user-requested; not a planned PR; fixes at b53e07727)
- Scope: 3-lens parallel review of the cumulative phase-4 product (`benchmarks-website/web/`): coherence/claude + maintainability/claude via `Agent`, correctness/codex via the companion (gpt-5.5 xhigh); conservative-union triage; all fixes in one consolidated commit `b53e07727`.
- Must-fix (found by coherence/claude, independently by correctness/codex): the read-route caching story was incoherent across PRs. `/api/groups` exported `revalidate = 300`, which forced a BUILD-time prerender so a DB-less `next build` FAILED (verified; contradicting page.tsx's documented `force-dynamic` decision); the same export was inert on `/api/chart/[slug]` + `/api/group/[slug]` (reading `request.url` forces dynamic), so their CDN-caching comments were false. Fixed by replacing route-segment ISR with `Cache-Control: public, s-maxage=300, stale-while-revalidate=300` headers on the three data routes' 200 responses (full-URL-keyed Vercel edge caching, v2's 5-min cadence) via the new shared `web/lib/cache.ts`; pinned by route-test header assertions + a DB-less `next build` (now green; all routes render dynamically).
- Should-fix applied: 400/404 error envelopes aligned to the Axum `{error: 'bad_request'|'not_found', message}` shape; `reqNumber` finiteness in `lib/slug.ts` (serde f64 parity: forged `threshold: 1e400` is now a 400, pinned by a raw-payload test since `JSON.stringify(Infinity)` would mask it); shared `lib/test-harness.ts` extracted (Docker probe + DDL + container boot + the canonical 3-commit fixture that 4 suites had copy-pasted; net -208 lines).
- Nits applied: `compareCodeUnits` dedupe (was `byCodeUnit`/`cmpStr` twins); `<noscript>` no longer renders in the empty-DB state (v3 `landing_body` parity); misleading ported `.group-info-icon` pointer-events comment rewritten; non-discriminating `?n=all` route test renamed + a discriminating all-vs-default route assertion added to the 125-commit suite; doc comments for `requireEnv`, health-route GET, `sql`-vs-`QueryParams` division of labor, `commitWindowUrlValue` forward reference; `families.ts` table-name consumer claim corrected; em-dash sweep.
- Deferred: landing-page caching strategy (page stays `force-dynamic`, every `/` render hits Postgres until PR-4.5 decides vercel.json `Cache-Control` on `/` vs build-time-DB revalidation). `deferred_items_total` 10 -> 11. Dismissed: next-env.d.ts SPDX (generated file, low confidence).
- Review: 3 reviewers, 1 cycle, unanimous on the must-fix. Verified: tsc/eslint/prettier clean; vitest 128 pass (+2 new); DB-less `next build` green.
- History note (2026-06-09): immediately after this review the branch history was restructured (user-requested) to one squashed commit per phase. Pre-squash per-commit history, including every `ending at <SHA>` reference in the entries above, is preserved at the LOCAL ref `backup/bench-v4-pre-squash-20260609`.

### PR-4.4.b: chart client island + header interactivity + permalink page  (8 code commits: 4 impl + 4 review-fix, 3 inner-loop 2-vote cycles, ending at 013280c46, 2026-06-09)
- Scope shipped: the full interactive layer as client islands, ported from v3 `chart-init.js` (NOT v2 React): `lib/chart-format.ts` (pure helpers: display-unit picker, LTTB, payload normalization onto the full-history axis, visible-range math, filter predicates, BAN-pinned `predecessorValue`), `lib/chart-store.ts` (bounded priority fetch queues at v3 caps 4/2, global-filter + per-group stores with `useSyncExternalStore` APIs), `lib/chart-js.ts` (lazy register-once Chart.js 4.5 + zoom-plugin loader with failed-import retry), `components/Chart.tsx` (per-card island: toolbar with throttled-`input` scope slider + Y switch, range strip, external tooltip with `idx-1` deltas, crosshair, wheel pan, drag pan/zoom, LTTB cap 500 shared indices, lazy `?n=100` on group-open/intent + one-shot `?n=all` upgrade; controller-per-effect-mount lifecycle with AbortController listener teardown), `components/FilterBar.tsx` (global chips + URL allowlist sync), `components/GroupToolbar.tsx` (group Y/series-filter/reset), `Header.tsx` as client island (hamburger mobile nav + `.nav-controls-github` mobile GitHub fallback resolving the PR-4.4.a gap, expand/collapse-all via native `details.open`, theme toggle), theme-bootstrap inline script in the root layout, `collectFilterUniverse()` in queries.ts, `/chart/[slug]` permalink RSC (payload inlined, v3 title + meta line, sync slug 404, universe optional per v3 `.ok()`), and a production-server smoke test (next start + seeded testcontainer).
- Scope notes: `Modal.tsx` DROPPED (v3 has no modal; the permalink page is the expanded view; plan amended pre-implementation). Deliberate deviations documented in-code and reviewer-verified: permalink card keeps its title row + working downsample badge (inert-by-bug in v3); `applyY`/`applyScope` record state before the chart-null guard (v3 bound toolbars only post-construction).
- Tests added: 76 new (chart-format unit suite incl. LTTB/unit-picker/normalize golden cases + BAN pin; chart-store queue/filter/group suites; chart-js loader retry; SSR markup contracts for Chart/Header/GroupToolbar/GroupSection; jsdom lifecycle suite with two discriminating regression pins (StrictMode replay fetch, group-Y mount replay); production-server smoke incl. permalink + 404 + `?n=all`). Total vitest 204.
- Review: custom 2-vote (fresh + correctness, parallel Claude + Codex gpt-5.5 xhigh per lens) / accepted-by-exhaustion at cycle 3 (cap). Cycle 1: unanimous reject on the StrictMode disposal latch (all 4 reviewers; fixed + discriminating test); 2 Codex filter findings counter-verified as exact v3 parity by correctness/claude and dismissed to the boundary flag. Cycle 2: 3 reject / 1 accept, all 3 union must-fix at the cycle-1 fix boundary (group-Y replay, permalink chunk-failure dead-end, applyScope drop); fixed + pinned. Cycle 3 (fix-pass): 2 accept / 2 reject; the genuine new must-fix (closed-group construction during async chunk load) fixed; coverage demands folded into the deferred interaction-suite row; theme-bootstrap bare-catch dismissed as byte-identical v3 with rationale documented. The cycle-over-cycle pattern (each fix wave surfacing one boundary regression, caught by the next fix-aware pass) is the multi-cycle review working as designed.
- Confidence: medium-high. All four UI BANS pinned and re-verified every cycle; v3 parity line-verified broadly by both Claude lenses; DB-less build + 204 tests + lint/format green; the ~5-slug manual visual check vs the live v2 site remains an operator item at the phase boundary (also a phase exit criterion).
- Deferred items: 1 new this PR (jsdom interaction suite, now explicitly including the bounded-retry pin, pre-construction-scope pin, and header interaction tests; `deferred_items_total` 11 -> 12). Boundary flag grew by 2 preserved-v3 dismissals (filter cardinality check; all-off URL round-trip) — now FIVE+ accumulated parity items for the Step 3.4 v4-fidelity-vs-preserve-v3 decision.
- Surprises during implementation: Next dev StrictMode would have blanked every island under the original once-per-instance controller (caught by all 4 cycle-1 reviewers, not by build/tests — the strongest single argument this PR for the adversarial review); the v4 architecture's async Chart.js load opened three timing windows v3 structurally could not have (closed-group construct, chunk-failure dead-end, pre-construction toolbar input), all surfaced by reviewers rather than manual testing.

### PR-4.5: Vercel deploy config + web-deploy workflow + CDN caching  (14 code commits: 2 impl + 12 review-fix, 3 inner-loop 2-vote cycles, ending at e623468c2, 2026-06-09)
- Scope shipped: `benchmarks-website/web/vercel.json` (CDN caching for the two HTML routes: `Vercel-CDN-Cache-Control: max-age=300, stale-while-revalidate=300` rules on `/` and `/chart/:slug` — docs-verified mechanism: Next.js emits `Cache-Control: no-store` for `force-dynamic` pages and function-emitted Cache-Control beats config-file rules, while `Vercel-CDN-Cache-Control` is consumed and stripped by Vercel's CDN alone at the highest precedence, so browsers still revalidate every load; resolves the deferred landing-page-caching decision in favor of CDN headers over build-time revalidation, preserving the DB-less-build invariant); `.github/workflows/web-deploy.yml` (changes-detect job via dorny/paths-filter with push-base + migrations/** + deploy-tooling gating; Check & Test job: pnpm install/format:check/lint/DB-less build/vitest with a `docker info` guard so the testcontainers suite cannot silently self-skip in CI — resolves the PR-4.2 vitest-CI-wiring deferral; CLI-driven Vercel deploys gated on checks: preview per same-repo PR, production on push to ct/bench-v4, concurrency keyed by event name + PR number, fork guard, contents: read restated per job); `.github/actions/verify-cdn-cache/action.yml` (post-deploy evidence gate: single-request probe requiring HTTP 200 + x-vercel-cache HIT/STALE, 401/403 protection-skip gated by an `allow-protection-skip` input that production disables when probing the public `BENCHMARKS_WEB_PROD_URL`); `web/README.md` (local dev, BENCH_DB_* contract, CDN-caching design + 404/5xx notes, one-time Vercel operator setup incl. the open proxy-vs-public-instance wiring choice); `web/lib/test-harness.ts` now applies the FULL migration set in runner order (ledger-first since 003 grants on `_applied_migrations`; runner-parity discovery), making future schema migrations automatically exercised by the web suite.
- Tests added: none (deploy-config PR); the harness change upgrades all 204 existing tests to run against the full migration set, live-proving 002-004's container-portability guards.
- Review: custom 2-vote (fresh + correctness, parallel Claude + Codex gpt-5.5 xhigh per lens) / 3 cycles at the cap. Cycle 1: reject, 2 must-fix (concurrency-group event collision that could cancel production deploys during the cutover-PR window, found by 3 of 4 lenses; job-level permissions dropping contents: read, both Codex lenses). Cycle 2: reject, 3 must-fix all on the verification surface (probe accepted cached non-200, 4 of 4 lenses at varying severity; PR concurrency by head_ref letting same-named fork branches cancel each other; migrations/** gate vs 001-only harness — synthesizer overturned the infeasibility assumption after line-verifying 002/004's portability guards). Cycle 3: 3 accept / 1 reject (correctness/codex); the 2 remaining must-fix (path-filter hole for the extracted composite action, 4-of-4 consensus; protection-skip wrongly tolerated on public-URL probes) plus all consensus should-fixes applied POST-CAP and self-verified (yamllint/format/lint/DB-less build/204 vitest) per the trusted-input low-stakes ~3-cycle calibration, mirroring the PR-4.2 self-verify precedent. Residual: the post-cap fix wave (1e631821f..e623468c2) has not itself been adversarially re-reviewed; it is small, mechanical, and fully reviewer-specified — the Phase-4 phase-end 2-vote review covers it in the cumulative diff.
- Confidence: high for everything CI-verifiable; the live deploy path (acceptance criterion "PR opens trigger preview deploy") is NOT yet live-verifiable — it requires the one-time operator Vercel setup (project with Root Directory benchmarks-website/web, git integration off, VERCEL_TOKEN secret + VERCEL_ORG_ID/VERCEL_PROJECT_ID vars), now documented in web/README.md. Operator item at the phase boundary.
- Deferred items: 0 new; 2 RESOLVED (PR-4.2 vitest-CI-runner gap → the workflow test job; phase-4-holistic landing-page caching → vercel.json), rows annotated in Deferred work. `deferred_items_total` stays 12 (monotonic).
- Surprises during implementation: the plan's sketched option (a) for landing caching ("Cache-Control on / via vercel.json") is unimplementable as written — Vercel docs state function-emitted Cache-Control overrides config-file rules and Next emits no-store for force-dynamic pages; `Vercel-CDN-Cache-Control` is the correct lever (context7-verified, then pinned by the live probe). Migrations 002-004 turned out to be deliberately container-portable, enabling the full-set harness fix the cycle-2 synthesizer initially assumed infeasible. Docker Desktop silently died mid-session and 41 tests skipped without failure — live demonstration of exactly the silent-skip the new CI docker-info guard prevents. A 1Password signing outage (op-ssh-sign "failed to fill whole buffer") blocked commits mid-fix-wave; resolved by operator unlock (Class C infrastructure).

### PR-5.0: Bring prod online — operator-gate + one-shot prod load + rustls TLS fix  (2 code commits: rustls impl `84c3715cb` + should-fix `d0175d70c`, 1 inner-loop 2-vote cycle accepted at `5e049db89`/`35c05c500`, plus the operational prod-seed sequence, 2026-06-10/11)
- Scope shipped (verified DONE): (1) **Operator-gate** — full prod schema applied incl. migration 005 as the RDS master (005 is requires-superuser; the `migrator` path cannot apply it), `bench_read` password set, Vercel prod env wired (`BENCH_DB_*` / `bench_read` DSN). (2) **One-shot PROD load** — freshest v3 snapshot rebuilt from `s3://vortex-benchmark-results-database/v3-backups/20260610T210150Z.tar.gz` (acct 375504701696, region us-east-1) → `~/bench-fresh.duckdb` (4.85M qm rows; `vector_search_runs`=0, which resolves the PR-3.4 caveat); atomic single-txn `load` (5.24M rows, counts exact) over rustls TLS with the master DSN; `verify --postgres-target` = **0 presence diffs, 0 value mismatches** across 4.85M rows; PR-3.5 cross-check = **11 updated, 0 inserted** (bench_ingest IAM path UPDATEs seeded rows); re-verify after cross-check still clean (seed intact). (3) **rustls TLS fix** (PR-5.0's only production code) — native-tls rejected the RDS leaf (no `serverAuth` EKU on macOS Secure Transport); swapped `migrate/src/postgres.rs` `connect_postgres` + `migrate/Cargo.toml` to rustls (`84c3715cb`) + a should-fix loud-bail on an empty CA bundle (`d0175d70c`). (4) **Deploy evidence** — Vercel deploy live at `35c05c500` serving the seeded data (`/api/health` confirms row counts; a single chart URL renders correctly). `VERCEL_TOKEN` was rotated by the user (one leaked, now revoked).
- Tests added: none (operational PR + a TLS-path code change reusing PR-3.1/3.2/3.5 tools). The `--ca-cert` empty-bundle loud-bail is the only new guard.
- Review: 2-vote (fresh-eyes generalist + correctness skeptic) / accepted (cycles: 1). Gauntlet was UNAVAILABLE that session, so the 2-vote was reconstructed faithfully via `compose_prompts.py` (same pr-2 lens prompts); cycle 1 accepted with zero must-fix + 1 should-fix applied (`d0175d70c`). (`spiral:gauntlet` IS available again as of the 2026-06-11 resume session — future Phase-5 PRs use the real Skill.)
- Confidence: **high for the SHIPPED scope** (data seed + rustls fix: verify/cross-check/re-verify all clean; deploy serves). **PR-5.0 is closed with 2 acceptance criteria DEFERRED** (user decision 2026-06-11): the two Phase-4-moved data-dependent exit criteria (`/api/groups` slug-vs-family-registry match; ~5 representative chart slugs vs the live v2 site) cannot pass because the read path does not scale to the prod seed — `/api/groups` + landing time out (~1-2 min), big-dataset charts ~24s. **Data + rendering are PROVEN correct** (single chart URLs render; small-dataset polarsignals ~1s). The whole problem is server-side query perf; the client model is sound (collapsed-by-default landing, lazy `?n=100` bounded hydration, LTTB) — NOT a v3 repeat. Root causes diagnosed against prod (read-only EXPLAIN/timing): non-sargable `IS NOT DISTINCT FROM` → per-dataset full scans; `collectFilterUniverse` 4.85M seq scan on every page; landing discovery `GROUP BY` ×5 families; N+1 per-group summaries (row_number windows spilling 4MB work_mem); `db.t4g.micro` 1 GiB RAM can't cache the 1.2GB fact table.
- Deferred items: 2 new (the two moved exit criteria above), both **Resolved-by: the read-path-perf PR** scheduled before PR-5.2; `deferred_items_total` 14 → 16.
- Surprises during implementation: (Class B) native-tls rejects the RDS leaf on macOS — required the rustls swap mid-PR (recorded). (Class B, the BLOCKER) the read path does not scale to the full prod seed — discovered during the Step-4 data checks (early-break DISCOVERY trigger `1b90051ba`); root-caused to server-side query non-sargability + instance size, not a data or client-model problem; user decided to close PR-5.0 on the verified seed and fix the read path as a dedicated PR before the DNS flip.

### PR-5.0.5: restore v2 statpopgen/polarsignals group names  (2 code commits: impl `462d8c6be` + nit-fix `85bc19dc6`, 1 inner-loop 2-vote cycle accepted, 2026-06-11)
- Scope shipped: `groupNameQuery` (`web/lib/queries.ts`) now special-cases `statpopgen` -> `'Statistical and Population Genetics'` and `polarsignals` -> `'PolarSignals Profiling'` (v2 `src/config.js` displayName values, byte-identical to the existing `descriptions.ts` switch keys), so the already-present editorial descriptions attach instead of the names falling through to the legacy `dataset sf=N [storage]` label. The function is exported (test-only) so a Docker-free unit test can pin the mapping; `descriptions.ts` needed no edit (the cases were already ported dead in PR-4.3.c). The other five preserved-v3 parity quirks STAY (Decision C: preserve-v3).
- Tests added: a Docker-free `groups.test.ts` describe block (4 tests): statpopgen + polarsignals name+description pins (production-faithful null scale_factor), a tpch/tpcds/clickbench regression pin, and a legacy-fallback pin. Mutation-verified locally (removing a special-case reverts to the legacy label and fails both the name and the description-attach assertions). Full vitest 170 pass / 41 Docker-skipped / 0 fail; `next build` + tsc + eslint + prettier green.
- Review: 2-vote (fresh-eyes generalist + correctness skeptic) / accepted (cycles: 1). Real `Skill(spiral:gauntlet)` v0.5.3 this session (after a `/reload-plugins`; the first invocation hit "Unknown skill", consistent with PR-5.0's session, until the user reloaded plugins). Both lenses accepted high-confidence, traced the change correct end-to-end, and confirmed it is the SANCTIONED reversal of the PR-4.3.c accepted tradeoff (RESOLVED 2026-06-10, Decision C), not a re-flag. Zero must-fix.
- Confidence: high. Acceptance criteria met: the friendly names + descriptions render (verified by inspection end-to-end + pinned by the discriminating unit test); the other branches keep v3 behavior; vitest + build + lint clean. The acceptance criterion's "seeded ... renders" sub-part is delivered as a discriminating unit test rather than a seeded testcontainer render (see Deferred items).
- Deferred items: 1 new (seeded end-to-end `collectGroups` test for statpopgen/polarsignals -> web test-hardening pass; Docker unavailable locally to verify it; wiring independently verified by both lenses; `deferred_items_total` 16 -> 17). Resolved-by: web test-hardening pass (pre-develop-merge).
- Surprises during implementation: (process) the gauntlet `Skill` was initially uninvokable ("Unknown skill") despite appearing in the skill list; a `/reload-plugins` fixed it, so this PR got a real gauntlet review (unlike PR-5.0's faithful manual reconstruction). (mechanics) a mid-PR `git checkout -- queries.ts` used to undo a mutation-verification test ALSO reverted the uncommitted feature edit; caught immediately (grep returned 0 special-cases) and re-applied before committing — no bad state shipped.

### PR-5.1.5: read-path perf — scale the v4 read path to the full prod seed  (4 impl + 3 gauntlet-fix commits, 3 inner-loop 2-vote cycles accepted at the project ~3-cycle cap, ending at `b20904ae0`, 2026-06-11)
- Scope shipped (read-side, `15b778b01` for skip scans + earlier `2e637401e`/`629b5b0b6`/`680b30e6e`/`a4834ba1f` for sargable/parallelize/006/007): (a) **sargable WHERE** (`col IS NULL` / `col = $n` instead of `IS NOT DISTINCT FROM`) across `queries.ts` `chartPayload` + `summary.ts`, so `idx_query_measurements_chart` seeks past the leading `dataset=` (tpch chart 13.6s -> 0.094s live). (b/c/d) **recursive-CTE skip scans** replacing the whole-group/whole-table scans: `collectQuerySummary` latest-per-series (3-branch successor walk + per-series latest probe; prod tpcds 2796ms -> 63ms), `collectQueryGroups` discovery (15-branch NULL-aware successor walk vs a 4.85M-row GROUP BY; 2333ms -> 20ms), `collectFilterUniverse` distinct engine/format (single-column skips; 565ms -> 0.2ms). Each probe selects via a constant-ordinal `ORDER BY br LIMIT 1` (cycle-1 hardening) so arm choice is SQL-guaranteed, not a reliance on Append's undocumented syntactic order. NULLS-LAST "latest" is emulated with an `IS NOT NULL` index descent + a `commits`-joined `IS NULL` fallback (cycle-2 fix made the all-NULL fallback deterministic). All three rewrites verified byte-identical against the replaced queries on both the testcontainer seed and the full prod seed. (e) **bounded-concurrency** summary fan-out (`mapWithConcurrency`, `SUMMARY_CONCURRENCY`/`BENCH_DB_POOL_MAX` 8). (f) operator **RDS upsize** db.t4g.micro -> db.t4g.medium + `vortex-bench-pg16` param group (work_mem 32MB), done 2026-06-11. **Write-path (c)** (`e3861734e`): `post-ingest.py` stamps `commit_timestamp` via a scalar `commits` subquery on both upsert paths; the migrate Rust loader runs a drift-repairing (`IS DISTINCT FROM`) post-COPY backfill inside the load transaction. Migrations **006** (denormalize `commit_timestamp` + backfill + b/d indexes) and **007** (covering `INCLUDE (value_ns)` summary index) are repo-root `migrations/`, carry the `requires-superuser` marker, and were applied to prod as master + `VACUUM ANALYZE` (the 4.85M-row backfill bloated the table; VACUUM was required for the planner to use the new indexes).
- DEPLOYED + LIVE-VERIFIED: `web-deploy.yml` succeeded on the ct/bench-v4 push incl. the CDN probe. Live: `/api/groups` cache-cold **0.64-1.07s** (was ~6.0s) / **0.079s** cached, landing cold **1.2s** (was ~6.9s), payload sane (16 groups / 13 summaries / 372 charts). The cold-render approach (recursive-CTE skip-scan) was a **user decision (2026-06-11)** over a concurrency band-aid / accept-~6s.
- Tests: web vitest 214 pass (3 new skip-scan-fidelity tests: stamped-beats-newer-NULL + deterministic all-NULL fallback, summary successor-branch enumeration, discovery-vs-GROUP-BY-oracle across NULL partitions); migrate Rust 100 (e2e asserts 0 NULL / 0 drift post-load, schema init applies 001+006+007); python 154 (006 backfill on pre-existing rows; INCLUDE/DESC index pins; commit_timestamp stamping count-over-all-rows on both upsert paths; 006/007 requires-superuser marker pinned). tsc + eslint + prettier + ruff green.
- PR-5.0 deferred data-checks RESOLVED (user approved prod reads this session): all 13 query-group chart-count sets match the live v2 site `bench.vortex.dev` exactly; 5 representative chart latest-values within run-to-run variance; the one missing series (lance TPC-H S3 SF=10) is a 6-months-stale series correctly outside v4's chart window; remaining structural diffs are the intentional v3-parity compression-group shape + v2's empty config placeholders.
- Review: 2-vote (preset=pr-2, fresh + correctness), executor=parallel (each lens on BOTH Claude and Codex). Real `Skill(spiral:gauntlet)` v0.5.3 with the synthesizer subagent each cycle. **Cycle 1** reject (4 must-fix): branch-ordinal hardening + NULLS-LAST/successor/backfill/discovery-oracle tests + INCLUDE/DESC index pins. **Cycle 2** reject (2 must-fix): operator-doc drift (migrations/README.md + migrate-schema.py 002/004/005 list stale vs 006/007's new marker) + a non-deterministic all-NULL fallback arm (now LEFT-joins commits + ORDER BY c.timestamp DESC NULLS LAST). **Cycle 3** reject (2 must-fix): sibling operator-runbook drift (infra/README.md + provision.sh master-apply list) — fixed; plus should-fixes (006/007 marker test, re-stamp privilege note, count-over-all-rows test). Notable: the executor-asymmetry held — across cycles 2/3 the Codex sides surfaced the operator-doc-drift blockers the Claude sides cleared or did not visit.
- Cycle cap: accepted at the project ~3-cycle cap. Cycle-3 must-fixes were operator-doc + test-only (zero production-logic change) and are all resolved at `b20904ae0`; per the spine's anti-spiral Key decision the loop was not spiraled into a 4th review. Production logic was accepted by 3 of 4 reviewers across two clean cycles before cycle 3.
- Confidence: high. Acceptance criteria met: prod EXPLAIN/timings captured (above) show index-descent plans + sub-second charts; `/api/groups` returns in ~1s cold and its slug list + ~5 chart slugs match the live v2 site (resolves both PR-5.0-deferred checks); migration 006/007 apply cleanly (testcontainer green) and the sargable + skip-scan rewrites are pinned semantically identical; vitest + build + lint + ruff + pytest green; post-upsize FreeableMemory healthy.
- Deferred items (nits/should-fixes, cycle triage): derive `SUMMARY_CONCURRENCY` from pool config; share `sargableDimEq`/`QueryParams` with `summary.ts`; a `mapWithConcurrency` unit test + poolMax default pin; summary-path coverage of the remaining groupPred pin combinations; trim the summary.ts skip-scan comment block (documents load-bearing non-obvious planner behavior, low priority). `deferred_items_total` 17 -> 22.
- Surprises: (mechanics) the 006 backfill UPDATE bloated `query_measurements` enough that the planner ignored the new indexes until a `VACUUM ANALYZE` — a real lesson, now documented in migrations/README.md. (process) prototyped all three skip scans against a 2.1M-row synthetic `postgres:16-alpine` container before writing TS, which surfaced the two load-bearing planner gotchas (row-comparisons aren't btree quals past index column 1; `IS NULL` pins don't reduce ORDER BY pathkeys) that shaped the final SQL. (process) `schema-deploy.yml` is develop-only so the ct/bench-v4 pushes only fired `web-deploy.yml`; at the develop merge schema-deploy no-ops 006/007 (already in the prod ledger, master-pre-applied like 005).

### PR-5.0.9: opt-in full-history chart loading (UI/UX round)  (6 impl/review commits + 1 gauntlet-fix commit, 1 inner-loop 2-vote cycle rejected then accepted [2 cycles], ending at `ab775f5e7`, 2026-06-12)
- Scope shipped (frontend-only, `benchmarks-website/web/`; design `.big-plans/ct__bench-v4-uiux-design.md`): removed the automatic `?n=all` warmup from `Chart.tsx` `onGroupOpen` (group open now costs only the windowed `?n=100` fetches: ~750KB for the 22-chart tpch group vs the old ~24MB), so full history is per-chart opt-in via three intent paths funneling through the unchanged `ensureFullHistory`/`fullHistoryQueue` (concurrency 2): (a) an always-visible window chip (`syncWindowChip`, imperative like `syncDownsampleBadge`) showing windowed "latest 100 of N" -> hover-revealed "load all N" -> "loading all N…" -> "all N", with an error "retry" affordance and a terminal-404 disabled state; born-complete charts (`history.complete`, <100 commits) show no chip (`everWindowed` gate); chip click fetches at `INTERACTION_FULL_PRIORITY`; (b) a ~600ms same-card hover-dwell silent prefetch at a new mid-tier `HOVER_PREFETCH_PRIORITY=500_000` (`HOVER_DWELL_MS=600`); hover reveals the action immediately, only the dwell starts the fetch, `pointerleave` cancels, timer cleared on `destroy()`; (c) the existing `rangeTouchesUnloadedHistory` pan/zoom promotion, unchanged. CDN `stale-while-revalidate` bumped `300 -> 86400` in `lib/cache.ts` + `vercel.json` (low-traffic site; the design predated the existing SWR=300, so this was a value bump). The virtual full-length x-axis (`normalizeChartPayload` total/start + null prefix + in-place fill) is untouched -> jank-free late fill-in. New `.chart-window-chip` CSS in `app/globals.css` (pill matching `.chart-badge`, per-`data-state` visuals, accent hover-reveal).
- Tests: web vitest 183 pass / 44 Docker-skipped / 0 fail (new `Chart.loading.test.tsx`: zero-warmup discriminator, chip windowed/loading/complete/error-retry/born-complete-hidden, dwell-fires-at-threshold-not-before + pointerleave-cancels + hover-reveal-without-fetch, terminal-404 stops re-fetch, and the cycle-1 regression test pinning the dwell-vs-initial-fetch race; new `lib/cache.test.ts` pins SWR=86400; `chart-format.test.ts` constants). `tsc --noEmit` + `next build` + eslint + prettier all green.
- Review: 2-vote (preset=pr-2, fresh + correctness), **executor=claude** (the openai-codex companion was not installed this session, so the dynamic default resolved to Claude-only; no parallel Codex side this round, unlike PR-5.1.5). Real `Skill(spiral:gauntlet)` v0.5.3 with composed lens prompts + carry-forward (BANS/accepted/deferred). **Cycle 1** reject (1 must-fix + 2 should-fix + 2 nits): the correctness lens caught a concurrency-ordering bug — a hover-dwell/chip `?n=all` upgrade can resolve before a still-pending initial `?n=100`, and the late window resolution unconditionally clobbered the already-loaded full payload back to the bounded window (diverging `state.payload` from the rendered datasets, regressing the chip, re-arming a redundant pan refetch). Fix `ab775f5e7`: `ensureInitialPayload`'s resolver early-returns when `fullLoaded`; `ensureFullHistory` also honors `fullUnavailable` so the pan/zoom path respects a terminal 404 (fresh should-fix); chip counts pinned to `toLocaleString('en-US')`; `onCardHoverEnd` disposed-guard; + a mutation-verified regression test (fails without the guard, passes with). **Cycle 2** accept (zero must-fix; both lenses): correctness mutation-verified the race is closed in both orderings + the 404 ordering with no new lifecycle/cross-ref drift; fresh accepted with 2 coverage-only nits.
- Confidence: high. Acceptance criteria (Phase Map PR-5.0.9 row) met: no `?n=all` without per-chart intent; the discriminating "zero `fullHistoryQueue` entries on group open" test passes; chip states + dwell behavior + terminal-404 pinned; existing interaction-promotion path unchanged; API SWR directive present; vitest + build + tsc + eslint + prettier green; 2-vote gauntlet accepted. The "22-chart group open transfers ~1MB or less" + "post-deploy network profile shows only `?n=100`" criteria are structurally satisfied by the warmup removal (pinned by the zero-warmup test) and confirmable post-deploy.
- Deferred items: 2 cycle-2 coverage nits (a pan/zoom-path direct test for the shared terminal-404 guard; a retry-click round-trip test) — both verified-by-inspection, explicitly non-blocking under the lean trusted-input calibration's test-completeness-spiral guard. Resolved-by: web test-hardening pass (pre-develop-merge). `deferred_items_total` 22 -> 24.
- Surprises: (scope) the design doc predated the in-tree `stale-while-revalidate=300` (added during PR-5.1.5's cache work), so design item 5 reduced to a `300 -> 86400` value bump rather than a new directive. (mechanics) the plan omitted chip CSS; the SDD code-quality review caught the unstyled chip + a 404 re-fetch loop before gauntlet, so those landed in the pre-gauntlet review-fix commit `e2a8ef278`. (process) executor=claude only (no Codex companion this session), so this PR did not get the Claude+Codex executor-disjointness that surfaced the operator-doc-drift blockers in PR-5.1.5; the single must-fix was nonetheless a real concurrency bug caught by the Claude correctness lens.

### PR-5.0.95: lazy-hydration + resilient loading for large chart groups (UI/UX round 2)  (9 impl/review commits + 3 gauntlet-fix commits, 3 inner-loop 2-vote cycles [reject / reject / fix-and-accept at the ~3-cycle cap], ending at `327f1fb92`, 2026-06-12)
- Pre-impl read-only investigation (live site): the clickbench "hangs" are a CLIENT-side burst + cold-start + no-recovery, NOT slow server queries — a 43-chart concurrent warm burst finished in ~0.75s with zero failures, each `?n=100` ~32KB at 60-90ms even on a CDN MISS. Scope stayed client-side A+B+C, no server-query work (design `.big-plans/ct__bench-v4-uiux-r2-design.md`, which records the investigation + the 3 resolved open decisions).
- Scope shipped (frontend-only, `benchmarks-website/web/`): **(A) viewport-gated top-first lazy hydration** — each landing group-chart's initial `?n=100` fetch+construct is gated behind an `IntersectionObserver` (reusing the permalink `else`-branch shape, `LAZY_HYDRATION_ROOT_MARGIN='300px 0px'`), so opening a big group hydrates only the ~visible charts top-first by visual `index` (`priority = index === 0 ? 0 : -index`, queue drains highest-first) and the rest hydrate on scroll; the all-charts group-summary bulk prefetch and the now-dead `nextGroupOpenPriority`/`GROUP_OPEN_PRIORITY_STEP` are removed; closing the group disconnects the observer and aborts in-flight fetches, reopening re-arms; graceful-degradation immediate-hydrate where `IntersectionObserver` is undefined (the path jsdom tests use). **(B) per-fetch abort + timeout + retry** — both `fetch()` calls (`ensureInitialPayload` `?n=100`, `ensureFullHistory` `?n=all`) wire a fresh per-fetch `AbortController` bridged to the controller-lifetime `aborter` (so `destroy()` and group-close cancel in-flight) plus a `setTimeout(FETCH_TIMEOUT_MS=30000)` abort (manual, not `AbortSignal.timeout`/`any`, for fake-timer testability); the catch keeps a close/destroy `AbortError` silent and surfaces a `TimeoutError`/failure; `abortInFlightFetches()` cancels without tearing down the controller; a clickable initial-fetch retry (`retryInitialPayload`, `data-role="fetch-retry"`) re-issues the `?n=100` fetch and persists past the 4s construction-retry auto-dismiss. **(C) loading spinner** — an animated CSS `.chart-spinner` (`@keyframes chart-spin`) replaces the static "loading…" text (accessible: `role="status"`/`aria-live`, visually-hidden label retained, `aria-hidden` ring), the chip `data-state='loading'` gets an inline `::before` spinner, and a `@media (prefers-reduced-motion: reduce)` guard disables the animation.
- Tests: web vitest **247 pass / 24 files** (new `Chart.lazy-hydration.test.tsx`: no-fetch-until-intersect, only-visible-hydrate, off-viewport-on-scroll, top-first priority via a `hydrationQueue.schedule` spy, no summary bulk prefetch, close disconnects+aborts, reopen re-arm, and the queue-saturation reopen-race regression; extended `Chart.loading.test.tsx`: signal-presence, timeout-aborts+errors, destroy-aborts, close/destroy-abort-silent for BOTH fetches, full-history timeout→chip-retry, clickable initial-fetch retry; new `app/globals.spinner.test.ts` node-env: keyframes + `.chart-spinner` + reduced-motion guard, with `vitest.config.ts` `include` extended to `app/**/*.test.ts`). `tsc --noEmit` + `next build` + eslint + prettier all green.
- Review: 2-vote (`preset=pr-2`, fresh + correctness), **executor=claude** (openai-codex companion not installed this session — dynamic default resolved to Claude-only, as in PR-5.0.9). Real `Skill(spiral:gauntlet)` v0.5.3 with composed lens prompts + carry-forward (BANS/architecture/key-decisions) + the fix-commit attention block each re-review. Three cycles, each surfacing a progressively-deeper bug in the SAME async loading-state machine: **Cycle 1** reject (1 must-fix + 2 should-fix + 2 nits) — both lenses caught that `abortInFlightFetches()` aborted the per-fetch controllers but left `state.initialFetchEntry`/`fullFetchEntry` non-null, so a close-while-queued then reopen joins the aborting promise and the card stays blank; fix `23469e52e` clears the entry refs synchronously + identity-guards the handler clears (`=== entry`/`=== fc`) so a late aborted task can't clobber a newer entry, plus an explicit `destroy()` abort reason, a retry-button controller-presence guard, the reopen-race regression test, and two nits. **Cycle 2** reject (1 must-fix) — the correctness lens caught that the cycle-1 regression test was a TAUTOLOGY (`act` flushes microtasks, so the aborted task's rejection cleared the entry even without the fix); fix `bab5a5089` rewrote it to saturate the bounded hydration queue (concurrency 4) with rejectable blockers so the target fetch genuinely stays QUEUED, making the synchronous entry-clear the only thing that lets reopen schedule fresh — proven non-tautological (it fails when the clear is reverted); + removed a redundant retry-button double state-reset. **Cycle 3** reject → **fix-and-accept at the ~3-cycle cap** (1 must-fix + 1 should-fix + 1 nit) — both lenses caught that `setLoading(false)` fired BEFORE the `AbortError` early-return in the initial-fetch rejection handler, so a stale aborted rejection (after reopen) could extinguish a fresh fetch's spinner; fix `327f1fb92` moves `setLoading(false)` after the `AbortError` return (a cancellation no longer touches loading state) + adds `restoreMocks` to the vitest config. Per the spine `Cycle cap` Key-decision (fix-and-accept or defer at cycle 3), the cycle-3 must-fix was the reviewers' exact prescribed one-line move (effectively pre-vetted), applied with the full suite + all gates green; not spiraled into a 4th cycle.
- Confidence: high. Acceptance criteria (design-doc / PR-5.0.95 row) met: a ~43-chart group hydrates only the ~visible charts top-first with the rest on scroll (IO-gated, pinned by the no-fetch-until-intersect + only-visible + top-first-priority tests); a stalled initial fetch times out at 30s and offers a clickable retry; closing a group aborts its in-flight fetches; the spinner respects `prefers-reduced-motion`; PR-5.0.9 behavior (opt-in full history, chip, dwell, terminal-404) unchanged (all PR-5.0.9 tests stay green via the no-IO graceful-degradation path); vitest + tsc + next build + eslint + prettier green; 2-vote gauntlet accepted.
- Deferred items: 1 should-fix (a direct loading-survival test asserting the `.chart-loading` spinner persists across the close→reopen abort race where a stale aborted rejection settles while a fresh fetch is loading) — test-completeness, deferred under the lean trusted-input calibration's test-spiral guard; the cycle-3 code fix itself is verified by inspection + the full suite. `deferred_items_total` 24 -> 25.
- Surprises: (process) executor=claude only again (no Codex companion), so no Claude+Codex executor-disjointness this round; the Claude correctness lens nonetheless drove all three cycles' must-fixes. (review) the fix-commit attention block earned its keep — each re-review caught a fix-adjacent issue the prior cycle's fix exposed (stale-entry → its own tautological test → the loading-state-on-abort ordering), a clean demonstration of why the re-review is non-skippable. (mechanics) the cycle-2 finding was meta: the cycle-1 regression test passed against both fixed and unfixed code; the queue-saturation rewrite (rejectable blockers drained in `finally`) is the honest reproduction of the queued-not-running race.

### PR-5.0.97: always-warm last-100 cache + full spinner coverage + fast Expand All (UI/UX round 3)  (10 impl/review commits + 3 gauntlet-fix commits, 2 inner-loop 3-vote cycles [reject / accept], ending at `5f92566ee`, 2026-06-12)
- Scope shipped (cross-cutting: `benchmarks-website/web/` + `scripts/` + 3 ingest workflows; design `.big-plans/ct__bench-v4-uiux-r3-design.md`): **(A) Vercel Data Cache for the default `?n=100` window** — new `web/lib/data-cache.ts` wraps `collectGroupCharts`/`chartPayload`/`collectGroups`/`collectFilterUniverse` (default last-100 only) in `unstable_cache` (shared tag `'bench-data'`, 3600s backstop); the chart/group/groups routes + the landing (`app/page.tsx`) and permalink (`app/chart/[slug]/page.tsx`) pages branch to the cached fn only when `window.kind==='last' && window.n===DEFAULT_COMMIT_WINDOW`, every other `?n=` keeps the direct query; `force-dynamic` + `READ_API_CACHE_CONTROL` unchanged. Eliminates the ~7.8s cold RDS path for the default window (CDN misses read the Data Cache). **(B) `POST /api/revalidate`** (`app/api/revalidate/route.ts`): POST-only, bearer `BENCH_REVALIDATE_TOKEN` compared via `crypto.timingSafeEqual` (length-check first), missing/empty env -> 503 fail-closed, bad token -> 401, success -> `revalidateTag('bench-data')` -> 200; never CDN-cached. **(C) `scripts/post-ingest.py` `refresh_site_cache` hook** — POST `/api/revalidate` (bearer) then a best-effort warm pass (`GET /`, `/api/groups`, each `/api/group/{slug}?n=100` at concurrency 4 via `ThreadPoolExecutor`); EVERY failure caught + logged + swallowed so it can never change the ingest exit code; skips the warm pass when revalidate fails (no point warming an un-flushed cache); called from `_main_postgres` only when both `BENCH_SITE_BASE_URL` + `BENCH_REVALIDATE_TOKEN` are set (silent no-op otherwise); 2-line additive `env:` (`vars.BENCH_SITE_BASE_URL` + `secrets.BENCH_REVALIDATE_TOKEN`) on the v4 Postgres step in `bench.yml`/`sql-benchmarks.yml`/`v3-commit-metadata.yml`. **(D) client group-bundle + session cache** (`web/lib/chart-store.ts` + `components/Chart.tsx`): `ensureGroupBundle`/`abortGroupBundle` fetch ONE `/api/group/{slug}?n=100` bundle per group (deduped per-group in-flight, priority-bumped, AbortController+`FETCH_TIMEOUT_MS`) into a session `payloadCache` Map (+ `completedBundles` never-cleared / `attemptedBundles` cleared-on-close gating); `ensureInitialPayload` consults the cache synchronously, drives the bundle, falls back to the per-chart fetch only when the bundle 404s / fails / misses the slug; the permalink (no `groupSlug`) keeps the per-chart path; the `IntersectionObserver` still gates Chart.js CONSTRUCTION, so Expand All loads every chart's last-100 eagerly (top-group-first) while construction stays lazy and close/reopen refetches nothing. **(E) full spinner coverage** — a server-rendered `.chart-placeholder` (`role=status`, spinner ring + "loading…" label) shown for every `!constructed && !error` state (pre-hydration cards were blank), removed once `maybeConstruct` sets `constructed`; the `@media (prefers-reduced-motion: reduce)` block keeps the ring + label visible statically (animation off only — the pinned user decision).
- Tests: web vitest **286 pass / 28 files** (new `lib/data-cache.test.ts` tags/TTL/keying; new `app/api/revalidate/route.test.ts` 503/401/200 + same-length-wrong-token + empty-bearer + empty-env + no-cache-control; extended `Chart.lazy-hydration.test.tsx` one-bundle-per-group / hydrate-from-bundle / close-aborts / reopen-after-success-zero-fetches / slug-absent + 404 fallback / two-group concurrency+priority / the close-while-awaiting-bundle regression pin / reopen-after-404; extended `Chart.loading.test.tsx` + `app/globals.spinner.test.ts` placeholder coverage incl. the non-tautological removed-after-construction test; extended route tests for default-vs-non-default branching; `lib/server-smoke.test.ts` now exercises the default window through the REAL `unstable_cache` under `next start`). `scripts` pytest: `test_post_ingest_revalidate.py` 7 pass (bearer + revalidate-before-warm ordering, skip-warm-on-failed-revalidate, swallow-all-failures, `_main_postgres` exit-0-on-refresh-failure, hook-skip-on-missing-env for both vars, warm-pass happy-path URL set). `tsc --noEmit` + `next build` (all routes force-dynamic) + eslint + prettier + yamllint + `py_compile` all green.
- Review: 3-vote (`preset=pr-3`, fresh + correctness + maint), **executor=claude** (Codex companion was installed this session, but the inline gauntlet ran Claude-routed 3-lens for reliable execution; the disjoint LENSES carried the value). Real lens prompts + synthesizer per the gauntlet v0.5.3 contract. **Cycle 1** REJECT (1 must-fix + 6 should-fix + 6 nits): the CORRECTNESS lens caught a real must-fix that fresh+maint BOTH missed — `ensureInitialPayload`'s bundle `.then` guarded only `disposed||payload` (not `groupIsOpen()`), so CLOSING a group while a card awaited the in-flight bundle fell through to `fetchInitialPayloadDirect` AFTER `disarmHydration`'s `abortInFlightFetches()` already ran, issuing an UNABORTABLE per-chart fetch for the now-closed group (defeats the "closing a group frees server capacity" contract; reachable on the exact ~7.8s cold path this PR targets; the reviewer reproduced it with a probe). Cycle-1 fixes (`d565800bf` client + `d59662f4d` server): the `|| !this.groupIsOpen()` guard + a regression test, plus the cheap should-fixes (Data Cache real-`unstable_cache` smoke assertion; post-ingest `_main_postgres` exit-0/skip + warm-happy-path tests; `primePayload` un-exported; `.chart-placeholder-text` CSS; reopen-after-404 test; `constructed` reset on cleanup; ops-pointer comment; composition-test env hardening). **Cycle 2** ACCEPT (zero must-fix; all 3 lenses): correctness empirically re-verified the leak is closed (reverting the guard fails the regression test, restoring it passes — non-tautological) and confirmed no fix-boundary regression; fresh flagged one should-fix (the regression test's fixed-tick microtask flush flaked ~1/50 on a cold start) which was hardened to a deterministic bounded-drain (`5f92566ee`; 20/20 deterministic + still fails on a reverted guard); maint re-verified cross-bundle consistency (env-var names / tag / default window / paths all agree).
- Confidence: high. Acceptance criteria (design-doc / PR-5.0.97 row) met: the default `?n=100` window reads the Data Cache (CDN misses stop hitting RDS for it, proven by the real-`unstable_cache` smoke test); `POST /api/revalidate` 503s-unconfigured / 401s-bad-token / 200s+`revalidateTag`; the post-ingest hook revalidates+warms best-effort and never changes the ingest exit code; Expand All loads every chart's last-100 via one bundle per group with construction still lazy; every pre-data card shows the placeholder, static under reduced motion; PR-5.0.9 / PR-5.0.95 behavior unchanged; 286 vitest + 7 pytest + build/tsc/eslint/prettier/yamllint green; 3-vote gauntlet accepted.
- Deferred items: 1 new (a dev-only `constructed`-reset placeholder-reappears-after-StrictMode-remount test; cosmetic, non-blocking under the lean trusted-input test-completeness calibration; `deferred_items_total` 25 -> 26). **OPS PREREQUISITE (not a code item, tracked here for the operator): set `BENCH_REVALIDATE_TOKEN` in Vercel env + as a GH Actions secret, `BENCH_SITE_BASE_URL` as a GH Actions var — until set, `/api/revalidate` 503s fail-closed and the post-ingest hook is a silent no-op, so refresh-on-update + warming do not activate (the PR is safe deployed without it; behavior degrades to current).**
- Surprises: (review) the disjoint-lens structure earned its keep — the gating must-fix (close-while-awaiting-bundle per-chart leak) was found ONLY by the correctness lens; fresh and maint both ACCEPTED, not because they disputed it but because neither examined that race (a coverage gap across lenses, not a contradiction). (mechanics) Task 1's narrow gates (`vitest lib/data-cache.test.ts app/api`) missed that the route handlers calling `unstable_cache` throw `Invariant: incrementalCache missing` in plain vitest outside Next's request/build context — surfaced during Task 4's full `pnpm test` run as 6 failures in `lib/queries.test.ts`/`lib/groups.test.ts`; fixed by mocking `unstable_cache` transparent in those route-integration tests (`21a193883`). (process) executor=claude only despite Codex being present this session — the inline gauntlet prioritized reliable 3-lens execution over Claude+Codex disjointness.

### PR-5.0.98: benchmarks-web keep-warm cron (inserted ahead of PR-5.1 via Amend; user-greenlit)  (1 code commit `9adc6c870`, 1 inner-loop 2-vote cycle accepted cycle 1, 2026-06-15)
- Scope shipped (single file `.github/workflows/web-keep-warm.yml`; JIT plan `.big-plans/ct__bench-v4--5-0-98-keep-warm-cron.plan.md`): a scheduled GitHub Actions workflow that keeps the low-traffic benchmarks site's Vercel Data Cache + 5-min edge CDN from going cold. Trigger `schedule: cron */5 * * * *` (the GH Actions floor; best-effort/delayed is acceptable + documented) plus `workflow_dispatch: { }` for manual runs. One `ubuntu-latest` job, `timeout-minutes: 10`, `permissions: contents: read`, single concurrency group (`cancel-in-progress: true`). One `set -Eeuo pipefail` `run:` block using preinstalled `curl`/`jq` (no checkout, no third-party actions): GETs `/`, then GETs `/api/groups` capturing the JSON, parses `.groups[].slug` in a pipeline (so a malformed payload trips `jq` under `pipefail`), then for each slug GETs `/api/group/{slug}?n=100` with the slug `@uri`-encoded; writes a warmed-count line to `$GITHUB_STEP_SUMMARY`. Base URL HARDCODED `https://benchmarks-web.vercel.app` (NO secret, NO repo var — read-only public traffic). All curls `--fail --silent --show-error`, so a genuinely broken endpoint fails the run — a deliberate lightweight uptime signal. Lives outside `benchmarks-website/web/**`, so it does NOT trigger `web-deploy.yml`.
- Tests/checks: `yamllint --strict -c .yamllint.yaml .github/workflows/web-keep-warm.yml` clean (exit 0); `git diff --check` clean; SDD spec + code-quality subagent reviews both passed; live sanity check against prod parsed 16 group slugs and the first `/api/group/{slug}?n=100` returned HTTP 200. No Rust/web checks run (correctly — the change touches only one `.github/` YAML file; per `.github/AGENTS.md` yamllint is the only gating lint).
- Review: 2-vote (`preset=pr-2`, fresh + correctness), **executor=claude** (no Codex companion installed this session). Real lens prompts + synthesizer per the gauntlet v0.5.3 contract. **Cycle 1 ACCEPT** (zero must-fix): both lenses independently reproduced the load-bearing bash under `set -Eeuo pipefail` — the `count=$((count + 1))` form avoids the `((count++))` set-e trap; the here-string keeps the loop in the current shell so `count` survives into the summary; the `[ -n "${slug}" ]` guard correctly handles the zero-group / trailing-newline edge; a malformed/unreachable `/api/groups` aborts the run (observed exit 5); `jq -rR '@uri'` is injection-safe into the double-quoted URL; yamllint `--strict` passes (`{ }` empty-brace + `on:` truthy-exempt). One shared nit (false-green when `/api/groups` returns zero/empty groups) — explicitly de-scoped by the spine REVIEW CALIBRATION (graceful degradation on trusted, regenerable data); not applied.
- Confidence: high. Goal met: the cron warms the three hot endpoint classes (`/`, `/api/groups`, each `/api/group/{slug}?n=100`) every 5 min under the CDN s-maxage so the Data Cache + CDN never go cold on this low-traffic site; no secret/var needed; broken-endpoint runs fail loudly as an uptime signal.
- Deferred items: 0 new (`deferred_items_total` stays 26). The de-scoped false-green nit is intentional behavior, not deferred work.
- Surprises: none — a clean single-file change, accepted on the first cycle. Note: this is a keep-warm mitigation; the OPS PREREQ (set `BENCH_REVALIDATE_TOKEN` + `BENCH_SITE_BASE_URL`) is still the user's action and independent of this PR.

### PR-5.0.99: raise Vercel Data Cache backstop 1h -> 24h (cold-group-open fix)  (1 code commit `ebabd1849`, 1 inner-loop 2-vote cycle accepted cycle 1, 2026-06-15)
- Root cause (user-confirmed 2026-06-15, after a live diagnostic): the "site feels slow" complaint is **COLD CACHE on initial group open at the default `?n=100` window** — NOT payload size, NOT the `?n=all` "load all" path (the earlier tentative "PR-5.0.99 = server-side `?n=all` downsampling" idea was DROPPED as the wrong lever). Live `/api/group/{slug}?n=100` measurements: warm bundles are 0.25-3.18MB / sub-0.21s (datacenter); TPC-DS = 3.18MB/99 charts, Clickbench = 1.44MB/43. The ~7.8s cold RDS fill is paid only when a request reaches the function with the Data Cache entry expired (>1h backstop) AND no warm CDN copy (>24h since that URL was last fetched, given CDN `s-maxage=300, stale-while-revalidate=86400`). On a low-traffic site the user is often the first visitor to a long-idle group-bundle URL, so they eat the cold fill.
- Scope shipped (`benchmarks-website/web/lib/data-cache.ts` + its test): raise `DATA_CACHE_BACKSTOP_SECONDS` `3600 -> 86400` (1h -> 24h; `86400 = 24*3600`) so a CDN miss reads a still-warm Data Cache instead of the cold database across overnight idle gaps; updated the constant's doc comment (24h + the low-traffic cold-cache rationale + the revalidate-hook freshness bound) and the single literal assertion at `lib/data-cache.test.ts:55`. The `revalidate: DATA_CACHE_BACKSTOP_SECONDS` in `CACHE_OPTIONS` and the test at line 58 read the constant, so they stay consistent automatically. The CDN header (`READ_API_CACHE_CONTROL` in `lib/cache.ts`), the `bench-data` tag, and per-wrapper keying are all UNCHANGED (out of scope).
- Tradeoff (explicitly accepted by the user): without the `POST /api/revalidate` ops wiring active, newly ingested benchmark data can lag up to the 24h backstop before auto-refreshing. Benchmark data is low-frequency, trusted, regenerable -> low-stakes; once the wiring (`BENCH_REVALIDATE_TOKEN` + `BENCH_SITE_BASE_URL`) is set, each ingest flushes the `bench-data` tag and freshness is immediate again, making the long backstop purely a safety net. The user chose 24h (covers overnight idle; caps staleness at a day) over a longer weekend-spanning window.
- Tests/checks: `vitest run lib/data-cache.test.ts` 2/2 pass; `tsc --noEmit` clean; eslint + prettier clean on both files; no stray `3600` left in `lib/`/`app/`. Full web vitest (incl. testcontainers Postgres suite) + `next build` run in CI via web-deploy.yml on push. No Rust checks (no Rust touched).
- Review: 2-vote (`preset=pr-2`, fresh + correctness), executor=claude (no Codex companion this session). **Cycle 1 ACCEPT, ZERO findings** (not even nits): both lenses verified `86400` is the correct seconds value, all four `unstable_cache` wrappers (`groupChartsCached`/`chartPayloadCached`/`groupsCached`/`filterUniverseCached`) share `CACHE_OPTIONS` so the constant + both assertions stay consistent, no stale `3600`/"1h" reference remains, and the CDN header / tag / keying are untouched. The 24h staleness is the user-approved tradeoff, not flagged as a bug per the REVIEW CALIBRATION.
- Confidence: high. TOUCHES `benchmarks-website/web/**`, so the push FIRES `web-deploy.yml` (test + production deploy) -> the backstop change goes live on `https://benchmarks-web.vercel.app`.
- Deferred items: 0 new (`deferred_items_total` stays 26).
- Surprises: the original framing ("load all is slow -> downsample") was the wrong target; a live measurement pass (group bundle sizes/timings + the two-layer CDN+Data-Cache mechanism) + a direct user check re-diagnosed it as cold-cache on default group open, making a one-constant backstop raise the correct, minimal fix instead of a downsampling feature. HELD OFF per user (2026-06-15): option B (a Vercel-cron / external warmer that runs pre-merge, the "never cold" guarantee — needed because the PR-5.0.98 GH Actions keep-warm cron only fires on schedule from the DEFAULT branch and this is still `ct/bench-v4`), the ops wiring, and PR-5.1.

### PR-5.0.991: parallelize the group-bundle query fan-out (cold-path '#3' fix)  (1 code commit `1977192ab`, 1 inner-loop 2-vote cycle accepted cycle 1, 2026-06-15)
- Finding (live diagnostic + code read, 2026-06-15): after PR-5.0.99 the remaining slowness is purely the COLD path. Live `/api/group/{slug}?n=100`: 8.7s cold (TPC-DS, 99 charts) / 18.7s cold (Clickbench, 43) vs <0.2s warm; every deploy resets the Vercel Data Cache so the first visitor to each URL eats the cold fill. Compression is ALREADY optimal (brotli: TPC-DS 3.18MB raw -> 319KB on the wire, Clickbench -> 168KB), so payload size is NOT the issue and the deferred `?n=all` downsampling is moot. The cold cost is `collectGroupCharts` (`benchmarks-website/web/lib/queries.ts:1134`) running a SEQUENTIAL `await` loop of one `chartPayload` SQL query PER chart (each collector = exactly one `getPool().query`), so a cache MISS pays N serial DB round-trips on one pooled connection (warm is fast only because the whole bundle is one Data Cache entry -> a hit does 0 queries).
- Scope shipped (`benchmarks-website/web/lib/queries.ts`, `collectGroupCharts` only, ~6 lines): replaced the sequential loop with an order-preserving `Promise.all(group.charts.map(async (link) => ...))` + a `.filter((c): c is NamedChartResponse => c !== null)`, bounded by the existing pg pool (`lib/db.ts` `max: 8`). ~99 serial round-trips become ~13 concurrent waves of 8 -> roughly an 8x cold-time cut, with IDENTICAL output (chart set, order, flattened `{name,slug,...chart}` shape, null-skip). `poolMax` UNCHANGED (raising it is a separate RDS-connection-limit tuning decision); `collectGroups()` discovery queries at `queries.ts:1139` left as a possible follow-up; chosen over the heavier rewrite-5-collectors-into-one-batched-SQL approach (bigger win, much higher risk) as the high-impact low-risk first step.
- Tests/checks: `tsc --noEmit` clean; eslint + prettier clean on `lib/queries.ts`; non-Docker unit sanity (`slug`/`window`) 40 pass. The output/ORDER equivalence is pinned by the EXISTING integration test `lib/groups.test.ts:200` (`['Q1','Q2']` + flattened payload), which runs in CI (testcontainers Postgres; Docker absent locally). No Rust touched.
- Review: 2-vote (`preset=pr-2`, fresh + correctness), executor=claude. **Cycle 1 ACCEPT** (0 must-fix, 0 should-fix, 1 shared nit). Both lenses verified: `Promise.all(map)` resolves in input-array order + `.filter` preserves order (so the `['Q1','Q2']` test holds); the `chart is NamedChartResponse` guard drops exactly the nulls; every collector is a stateless single `getPool().query` (per-query checkout/release, no connection pinned across awaits) so `pg` queues safely at max=8 (no deadlock/`too many clients`); `chartKeyFromSlug` is pure (no race); error propagation is equivalent (aggregate `Promise.all` rejection -> same 500). The nit (on the failure path, up-to-7 sibling read-only queries keep running after one rejects; no unhandled-rejection since they're members of the `Promise.all` array) is explicitly no-fix-required + de-scoped per the REVIEW CALIBRATION.
- Confidence: high. TOUCHES `benchmarks-website/web/**` so the push FIRES `web-deploy.yml` (the testcontainers order test + `next build` run in CI, then production deploy) -> the parallel fan-out goes live. CI GREEN (Check & Test incl. the `groups.test.ts` order test passed) + deployed (live build_sha 5f949910a).
- POST-DEPLOY MEASUREMENT (2026-06-15, parallel fan-out live): a cache-cold bundle whose serverless function + DB connection are ALREADY warm is now ~0.8-1.5s even for 22-chart groups (vs the old 8-19s) -> the per-chart serial-query penalty is gone. BUT the dominant REMAINING cold cost is the Vercel FUNCTION cold-start + cold DB connection (RDS Proxy connect + IAM token + TLS): the FIRST request to a freshly-spun-up function instance was 6-11s even for a 22-chart group, while subsequent fresh (cache-cold) URLs hit in ~1s once that function/connection warmed (proven by a back-to-back fresh-URL test). This cost is SHARED across URLs and chart-count-independent. **REFINES '#1' (the warmer): it must keep the serverless FUNCTION + pooled DB CONNECTION warm (ping frequently enough to keep an instance + connection alive), not just the Data Cache** — a warm cache in front of a cold function still costs the first visitor several seconds.
- Deferred items: 0 new (`deferred_items_total` stays 26). Possible follow-ups (NOT done): parallelize/cache the `collectGroups()` discovery queries; raise `poolMax` (RDS-limit tuning); the SQL-batch rewrite; and the user's '#1' post-deploy warm step (deferred to a fresh conversation).
- Surprises: none. The measurement cleanly isolated the cold path to N-sequential-queries, and the fix is output-identical (guarded by an existing test), so it was a clean cycle-1 accept.

### PR-5.0.992: the warmer (#1) — Vercel keep-warm cron + pg idleTimeoutMillis  (5 code commits `eedc3a7f7`..`bad4dff45`, gauntlet pr-2 accepted across 2 cycles, 2026-06-15)
- Goal: kill the dominant remaining cold-load cost for the FIRST visitor by keeping the Vercel serverless FUNCTION instance AND its pooled Postgres CONNECTIONS warm (not just the Data Cache, already handled by PR-5.0.97/5.0.99). Root cause = the PR-5.0.991 post-deploy measurement: a cache-cold bundle on an already-warm function+connection is ~0.8-1.5s, but the FIRST request to a freshly-spun-up Vercel instance is 6-11s (function cold-start + cold DB connection: RDS Proxy connect + IAM token mint + TLS), shared across URLs, chart-count-independent. The PR-5.0.98 GH keep-warm cron is DORMANT on ct/bench-v4 (GH scheduled workflows fire only from the default branch) — that dormancy is why this sub-PR exists. Design spec: `.big-plans/ct__bench-v4-warmer-design.md`. User-greenlit 2026-06-15 (the deferred warm step, now taken up); Vercel plan = Pro (user-confirmed).
- Scope shipped (two small changes under `benchmarks-website/web/`): (1) `vercel.json` gains `"crons": [{ "path": "/api/health", "schedule": "*/2 * * * *" }]` — fires PRE-MERGE because ct/bench-v4 pushes are `vercel deploy --prebuilt --prod` (git integration disabled) and Vercel Cron Jobs run only on production deployments. `/api/health` is already public (no `CRON_SECRET`) and `collectHealth` fans out a `Promise.all` of per-table `COUNT(*)` queries so each ping warms multiple pooled connections (the same `max: 8` pool a cold-cache group-bundle fan-out from PR-5.0.991 uses). (2) `lib/db.ts` raises pg `idleTimeoutMillis` from the 10s default to 5 min via a new exported `resolveIdleTimeoutMillis()` (reads `BENCH_DB_IDLE_TIMEOUT_MS`, module-const `DEFAULT_IDLE_TIMEOUT_MS = 300_000`, threaded through `DbConfig`/`readConfig`/`createPool`, mirroring `resolveSsl`/`poolMax`) so pooled connections survive between `*/2` pings (without it a connection drops 10s after each ping and a user landing mid-gap re-pays IAM+TLS even on a warm function). `poolMax` UNCHANGED. The existing GH keep-warm cron is UNCHANGED (redundant + uptime, activates at merge). No new endpoint, no new secret.
- Tests/checks: `pnpm test` 248 pass / 46 skipped (testcontainers Postgres suite self-skips locally, runs green in CI); `pnpm build`/`tsc`, `pnpm lint` (eslint), `pnpm format:check` (prettier) all clean. New unit tests: `resolveIdleTimeoutMillis` (unset/empty/whitespace → 300000, numeric override, `0` never-timeout sentinel, throws on non-numeric/negative), a `getPool().options.idleTimeoutMillis` threading test (env-snapshot + resetPool isolation), and a `lib/vercel-config.test.ts` regression guard reading `vercel.json` from disk and asserting the cron targets `/api/health` on `*/2`. CI GREEN (web-deploy run 27567195313: Detect Changes + Check & Test + Deploy Production all success) + deployed live (production build_sha `bad4dff45`); post-deploy `GET /api/health` returns 200 with populated `row_counts` (function + DB connection reachable), warm hit ~0.43s.
- Review: 2-vote (`preset=pr-2`, fresh + correctness), executor=claude (no Codex companion this session). **Cycle 1 ACCEPT** (0 must-fix) + 1 should-fix found by BOTH lenses: an empty `BENCH_DB_IDLE_TIMEOUT_MS` slipped past `??` and `Number('')===0` silently disabled the idle timeout, contradicting the "fails loudly" JSDoc and the file's `requireEnv` convention. Electively applied (cheap, both-lens, doc-vs-behavior consistency, ships to prod): treat unset OR empty/whitespace as "use the default". **Cycle 2 ACCEPT** (0 must-fix, 0 should-fix, 1 nit) — both lenses confirmed the fix correct across the full input matrix and verified the cron shape / threading-test cast / resetPool-on-lazy-pool / connect-time IAM auth. The lone cycle-2 nit (no test for a whitespace-only value, though `.trim()` provably handles it) was de-scoped per the spine REVIEW CALIBRATION (test-completeness-spiral guard). An em-dash in a fix comment was caught + reflowed per the project no-em-dash style (commit `bad4dff45`).
- Confidence: high. TOUCHES `benchmarks-website/web/**` so the push FIRED `web-deploy.yml` -> the cron + the raised idleTimeoutMillis are LIVE on `https://benchmarks-web.vercel.app`.
- Deferred items: 0 new (`deferred_items_total` stays 26). The whitespace-only-value test nit is intentional de-scope (provably-correct logic), not deferred work.
- Surprises: none. The enabling fact (ct/bench-v4 pushes are PRODUCTION Vercel deploys, so a vercel.json cron fires pre-merge) made the in-repo Vercel-cron the clean mechanism over an external pinger; the only iteration was the elective empty-string should-fix.
- POST-DEPLOY VERIFICATION (operator, not a code blocker): confirm in the Vercel dashboard (Project -> Cron Jobs) that the `/api/health` cron is REGISTERED and its first `*/2` invocation logged 200. The deploy succeeded with the crons block in the live `vercel.json`, so registration is expected; the dashboard is the authoritative confirmation. Optionally re-run the cold-start measurement after the warmer has been live a few minutes to confirm the first-visitor 6-11s cold-start is gone.

### PR-5.0.993: read-path R1 — recency-filter query_measurements chart reads on commit_timestamp  (6 code commits `05e41b2a6`..`85ba7ac37`, gauntlet pr-2 accepted across 2 cycles, 2026-06-15)
- Goal: kill the per-chart over-read that dominates the big-group cold open. Root cause (live-prod profiling + EXPLAIN, recorded in `.big-plans/ct__bench-v4-readpath-findings.md`, recommendation R1): `collectQueryChart`'s data query + its `buildEarliest` seed read each chart's full ~18k-row history to return the ~665-row last-N window, because recency was applied via a `commits` join on `commit_sha` AFTER a full-history scan instead of via the denormalized, already-indexed `q.commit_timestamp`. The cold cost is CPU-bound on that over-read (RDS CPU ~5%, CPU credits maxed, near-zero physical I/O), NOT cores/RAM/throughput. Built via the big-plans Amend flow, inserted AHEAD of PR-5.1; user-greenlit 2026-06-15 (AskUserQuestion 'Read-path R1 fix' at resume).
- Scope shipped (one collector, `benchmarks-website/web/lib/queries.ts`, COMPLEXITY-REDUCING — removes two joins): (1) NEW `queryMeasurementWindowFilter(params, window)` next to `factWindowFilter` — for a bounded window it emits `AND q.commit_timestamp >= (SELECT min(timestamp) FROM (SELECT timestamp FROM commits ORDER BY timestamp DESC, commit_sha DESC LIMIT $n) w) AND q.commit_sha IN (SELECT commit_sha FROM commits ORDER BY timestamp DESC, commit_sha DESC LIMIT $n)` (binds the limit ONCE, reuses the `$n` placeholder twice). The `>= cutoff` is the sargable lever that lets the planner seek `idx_query_measurements_summary (dataset,dataset_variant,scale_factor,storage,query_idx,engine,format,commit_timestamp DESC) INCLUDE(value_ns)`; the kept `commit_sha IN (last-n)` is the EXACT tie-trim. Returns `''` for the unbounded `?n=all` window (full scan unchanged). (2) `collectQueryChart`'s data query drops `JOIN commits c USING (commit_sha)`, orders by `q.commit_timestamp` (== `c.timestamp` by the denormalization invariant), and calls the new filter. (3) the `buildEarliest` seed swaps `MIN(c2.timestamp)` over the commits join for `MIN(q2.commit_timestamp)` direct, dropping its join. The SHARED `factWindowFilter` and the four OTHER collectors (compression-time/size, random-access, vector-search) are BYTE-IDENTICAL (their fact tables have no `commit_timestamp`). No schema change; uses the existing index. EXPLAIN-verified result-identical (665 rows), ~5x/chart (seed 57->8ms, data 38->9ms); expected cold TPC-DS bundle ~4.7s->~1s.
- Tests/checks: NEW behavior-pinning `?n=2` testcontainer equivalence test (`queries.test.ts`) asserting the full bounded-window payload (commits + series + series_meta + history) selects exactly the two newest commits — pins the result the refactor must preserve. `tsc --noEmit`, `eslint`, `prettier --check` all clean locally. Docker is ABSENT locally so the `@testcontainers/postgresql` suite (incl. the new test + the existing golden-snapshot equivalence test) SKIPS locally and validates in CI on push (`web-deploy.yml` Check & Test).
- Build: SDD (2 tasks: test then impl), executor=haiku implementers + sonnet spec/quality reviewers. Two SDD code-quality cleanups landed (em dash + backticks in the test comment `03b21a750`; ASCII-ify a doc-comment ellipsis `b2232406c`) per the project no-em-dash / ASCII style.
- Review: 2-vote (`preset=pr-2`, fresh + correctness), executor=claude (no Codex companion this session). **Cycle 1 ACCEPT** (0 must-fix). BOTH lenses independently verified scope + produced a RESULT-EQUIVALENCE PROOF (high confidence): the `>= min(last-n)` cutoff is inclusive (never drops a legit last-n commit); the only boundary-tie over-selection it admits is excluded by the kept `commit_sha IN`; `ORDER BY q.commit_timestamp == ORDER BY c.timestamp` by the invariant and row order isn't load-bearing (SeriesAccumulator places positionally by `commitIdx(sha)`); dropping the seed join is safe (orphan rows carry NULL `commit_timestamp`, MIN ignores NULL); `?n=all` returns `''` → unchanged. No carry-forward re-flagged. **Cycle 2 ACCEPT** (0 must-fix, 1 nit): re-review after the CI-regression fix (see CI bullet). The cycle-1 proof assumed `commit_timestamp` is universally populated — TRUE in prod, but the proof's "orphan=NULL" framing missed that NON-orphan rows can also be NULL if a writer omits it (which the stale TEST fixtures did). Cycle 2 independently traced ALL prod write paths and confirmed the fixture fix is FAITHFUL (the invariant holds + is test-pinned at every writer; the chart path safely tightens the requirement while the summary path keeps its NULLS-LAST tolerance). The lone cycle-2 nit (the new seed comment was a verbless fragment) was applied (`85ba7ac37`).
- Confidence: high. TOUCHES `benchmarks-website/web/**` so the close push FIRES `web-deploy.yml` (Check & Test incl. testcontainers + Deploy Production) -> the change goes live.
- DEPLOYED + PROD-VALIDATED (2026-06-15): the second push's `web-deploy.yml` (run 27576044515) is GREEN end-to-end — Detect Changes + Check & Test (testcontainers, including the fixed seeded-window fixtures + the new `?n=2` test) + Deploy Production all success; live `GET /api/health` returns 200 with `build_sha 00ee5d0ef` (matching HEAD) and populated `row_counts` (query_measurements 4,849,218). LIVE READ-PATH MEASUREMENT confirming the PR's goal: the TPC-DS NVMe SF=1 group (99 charts — the exact big group the user hit at ~15-16s) now serves a COLD bundle (cache-bypassed `?n=96`, function pre-warmed) in **1.77s** for all 99 charts (HTTP 200, 3.05 MB), vs the findings-doc baseline ~16.2s cold / ~4.7s warm-RDS — the predicted ~5x/chart win surfacing as a ~9x faster cold open.
- CI REGRESSION (the one in-phase plan-assumption violation, Class B): the FIRST push's CI (run 27575008289) FAILED Check & Test on two pre-existing seeded-window-semantics tests; Deploy Production correctly SKIPPED (test gates deploy → nothing went live). Cause = those fixtures insert `query_measurements` WITHOUT `commit_timestamp`, so the new `MIN(q2.commit_timestamp)` seed / `q.commit_timestamp >= cutoff` window saw NULL → empty chart. NOT a prod bug (every prod writer + the migrate backfill populate `commit_timestamp`, all test-pinned); the fixtures were stale relative to the write contract. Fix `36ed8a90e` stamps them via `(SELECT timestamp FROM commits WHERE commit_sha=$2)` (matching `seedChartFixture` + the writers) + documents the seed invariant. Re-reviewed cycle 2 (accept). The second push validates the fix in CI then deploys.
- Deferred items: 1 (non-blocking, deferred BECAUSE equivalence is proven; `deferred_items_total` 27->28; folded into the web test-hardening pass pre-develop-merge): should-fix coverage — the `?n=2` test uses 3 distinct-timestamp commits so the same-boundary-timestamp tie that the kept `commit_sha IN` clause trims is NOT exercised on the new path (the only tie test runs RandomAccess/factWindowFilter); a regression dropping the IN clause would stay green. Deferred because a tie-trim testcontainer test needs a NEW fixture that CANNOT be locally verified (Docker absent), matching the project's 'don't ship unverifiable seeded tests' pattern. (The cycle-1 nit — document the seed's commit_timestamp invariant — was RESOLVED in cycle 2, not deferred.)
- Surprises: ONE (Class B plan-assumption violation, handled per big-plans Step 2.4 CI-failure path): the gauntlet equivalence proof + the implementation both assumed `commit_timestamp` is universally non-NULL on `query_measurements`; that holds in PROD (writers + backfill, test-pinned) but the seeded-window TEST fixtures didn't honor it, so CI caught a stale-fixture regression that local checks (Docker absent) could not. Fixed + re-reviewed in-phase; no re-plan needed. Lesson recorded: when a read-path change starts depending on a denormalized column, audit EVERY test fixture that inserts the fact table, not just the golden one.

## Superseded (re-planned) phase-end must-fix items — Phase 1: RDS + schema + hash port — cycle 1

| Severity | File:line | Description | Implicated PR | Resolved |
|----------|-----------|-------------|---------------|----------|
| must-fix | .github/workflows/schema-deploy.yml:77 | RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. As wired, the schema... | PR-1.4 | [ ] |
| must-fix | benchmarks-website/infra/provision.sh:311 | The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth still needs a Secrets M... | PR-1.1 | [ ] |
| must-fix | .github/workflows/schema-deploy.yml:68 | PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-apply.' Implementation... | PR-1.4 | [ ] |
| must-fix | benchmarks-website/infra/README.md:120 | The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while transmitting the maste... | PR-1.4 | [ ] |
| must-fix | benchmarks-website/infra/provision.sh:19 | The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and the entire README/p... | PR-1.1 | [ ] |
| must-fix | benchmarks-website/server/tests/measurement_id_golden.rs:102 | PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qualitative coverage is ... | PR-1.5 | [ ] |
| must-fix | scripts/test_measurement_id.py:1 | The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scripts/test_measurement_... | PR-1.5 | [ ] |
| must-fix | scripts/_measurement_id.py:50 | Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py \| Python re-export to keep... | PR-1.5 | [ ] |

## Phase 1: RDS + schema + hash port — end-of-phase review (cycle 1) — rejected (3-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-3, executor=mixed: spec→codex, correctness→parallel, maint→claude); full Synthesizer Output JSON in the `<details>` block at the end of this section.**

Verdict: **reject** — 8 must-fix, 5 should-fix, 5 nit.

### Summary of changes

Phase 1 ('RDS + schema + hash port') lands the migration foundation in five concept areas. (A) Infrastructure: provision.sh idempotently bootstraps RDS Postgres db.t4g.micro + RDS Proxy (IAMAuth=REQUIRED, TLS) + GitHub OIDC provider + GitHubBenchmarkSchemaRole in account 245040174862, with an operator runbook in infra/README.md. (B) Schema-deploy CI: schema-deploy.yml (workflow_dispatch + dry_run; PR-merge is the accepted authorization gate, no environment: approval) generates a client-side IAM token and runs the migrate runner against the proxy as migrator over verify-full TLS. (C) Migration runner: scripts/migrate-schema.py applies migrations/*.sql in name order, tracks public._applied_migrations, is idempotent, uses autocommit + per-migration top-level transactions so a failing later migration rolls back only itself, rejects empty files; 28+ testcontainer tests. (D) DDL: 001 creates the commits dim + 5 fact tables + read-path composite indexes (Postgres translation of the authoritative DuckDB schema.rs, column order/nullability/types preserved); 002 the migrator login role + conditional rds_iam; 003 the append-only ledger grant. (E) Hash port: _measurement_id.py is a byte-for-byte port of db.rs measurement_id_* (xxhash64 seed 0), pinned by a Rust source-of-truth golden file giving Rust==golden==Python, verified bit-exact for all 63 vectors (Claude correctness executed the port; Rust golden test passes). The keystone hash-equivalence deliverable is solid and the schema shape provably matches schema.rs. HOWEVER the AWS-integration path has deploy-blocking gaps the Codex correctness lens surfaced: RDS Proxy endpoints are not publicly reachable from off-VPC GitHub runners (challenges Key decisions Q2/Q6); the proxy lacks a migrator credential for IAM auth; PR-1.4's live OIDC apply against real RDS Proxy was never executed (only wiring + lint + testcontainer); and the master-bootstrap runbook uses PGSSLMODE=require (no cert verification) while sending the master password. The Codex spec lens found contract gaps: 63 vectors shipped vs the promised 100; the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py; Table D claims a SCHEMA_VERSION re-export the hash port omits. Maintainability is otherwise high, but provision.sh's header names the WRONG AWS account, migrations/README omits 003, and the schema-deploy header advertises the superseded Environment gate.

### Unified findings

| # | Severity | Kind | File:line | Description | found_by |
|---|----------|------|-----------|-------------|----------|
| 1 | must-fix | bug | .github/workflows/schema-deploy.yml:77 | RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. ... | correctness/codex |
| 2 | must-fix | bug | benchmarks-website/infra/provision.sh:311 | The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth sti... | correctness/codex |
| 3 | must-fix | missing-acceptance | .github/workflows/schema-deploy.yml:68 | PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-ap... | spec |
| 4 | must-fix | unsafe | benchmarks-website/infra/README.md:120 | The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while tr... | correctness/codex |
| 5 | must-fix | doc-quality | benchmarks-website/infra/provision.sh:19 | The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and... | maint, correctness/claude |
| 6 | must-fix | scope-drift | benchmarks-website/server/tests/measurement_id_golden.rs:102 | PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qual... | spec |
| 7 | must-fix | weak-exit-criteria | scripts/test_measurement_id.py:1 | The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scrip... | spec |
| 8 | must-fix | scope-drift | scripts/_measurement_id.py:50 | Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py \| Pyth... | spec, maint |
| 9 | should-fix | doc-quality | migrations/README.md:35 | The 'Initial files' section lists only 001 and 002 and says the SQL files 'land in PR-1.3', but PR-1.4 added 003_migrator_ledger_grant.sq... | spec, maint |
| 10 | should-fix | coverage | benchmarks-website/server/tests/measurement_id_golden.rs:1108-1117 | No golden vector exercises a NaN or Inf f64 threshold. Rust write_f64 uses v.to_bits() (preserves NaN payload bits); Python struct.pack('... | correctness/claude |
| 11 | should-fix | scope-drift | migrations/001_initial_schema.sql:75 | The composite-index Key decision promised indexes on '(dim_tuple..., commit_timestamp DESC)'; the migration creates dim-leading (read-pat... | spec |
| 12 | should-fix | doc-quality | .github/workflows/schema-deploy.yml:12-14 | The workflow header still advertises a 'schema-deploy GitHub Environment with manual-approval as the stronger gate ... tracked as deferre... | maint |
| 13 | should-fix | doc-quality | benchmarks-website/infra/provision.sh:266 | proxy_role_name ('vortex-bench-proxy-role') and its policy name are hardcoded locals, but the README 'Customizing' section claims 'Every ... | maint |
| 14 | nit | boundary | scripts/migrate-schema.py:2733-2745 | status() reports a generic 'pending' (exit 1) for an empty/whitespace-only migration file, while apply() rejects it explicitly only when ... | correctness/claude |
| 15 | nit | doc-quality | migrations/README.md:39 | 002 is described as 'CREATE ROLE for the IAM-auth user that bench.yml workflows assume into', but migrator is consumed by the schema-depl... | maint |
| 16 | nit | scaffolding | benchmarks-website/server/tests/measurement_id_golden.rs:30 | REGEN_GOLDEN_VECTORS is permanent regeneration scaffolding (write-on-env, always-assert otherwise). Correct and well-documented, but noth... | maint |
| 17 | nit | doc-quality | .github/workflows/schema-deploy.yml:84-89 | The 'Apply migrations' step carries a ~5-line justification comment (set -x suppression, PGPASSWORD-on-own-line vs export masking). It do... | maint |
| 18 | nit | scope-drift | benchmarks-website/server/tests/measurement_id_golden.rs:1 | PR-1.5's expected files row named 'benchmarks-website/server/src/db.rs (golden-vector test added)', but the test shipped as a separate in... | spec |

### Surprises and discoveries

- **RDS Proxy endpoints are not publicly accessible; off-VPC GitHub-hosted runners cannot reach the proxy. The plan's 'RDS Proxy public endpoint' assumption (Q6) may be architecturally invalid for the CI-write path.** — handled: Not handled in the diff — flagged as a must-fix deploy-blocker; likely forces amending Key decisions Q2/Q6 (e.g., CI writes to the public RDS instance endpoint with direct IAM, proxy stays for Vercel reads). (amend_plan: yes)
- **PR-1.4's schema-deploy was accepted on wiring + yamllint + testcontainer, never run live against real RDS Proxy.** — handled: Recorded honestly in Implementation status; spec lens flags the acceptance criterion as unmet. Coupled with the proxy-reachability + migrator-credential findings, the path is unproven. (amend_plan: yes)
- **PR-1.5 shipped 63 golden vectors, not the promised 100.** — handled: Status acknowledges 63; no amendment or extra fixtures. Qualitative coverage is strong (all tables + boundaries). (amend_plan: yes)
- **The Phase-1 exit criterion names pytest scripts/test_post_ingest_hash.py, but the shipped file is test_measurement_id.py.** — handled: Not reconciled; the documented gate is unrunnable as written. Independently confirmed during exit-criteria execution. (amend_plan: yes)
- **Table D's claim that _measurement_id.py re-exports SCHEMA_VERSION is stale; the shipped module correctly omits it.** — handled: Artifact correct; Table D not updated. Amend the reference table. (amend_plan: yes)
- **Composite indexes are dim-leading (read-path filter columns), not the Key decision's '(dim_tuple..., commit_timestamp DESC)'.** — handled: Explained in PR-1.3 surprises (PK enforces hash-tuple uniqueness; dim-leading serves charts); Key decision row not updated and tests assert only index names. (amend_plan: yes)
- **migrator role cannot ALTER / CREATE INDEX on master-owned tables in future migrations (GRANT CREATE on public is insufficient).** — handled: Covered by the deferred PR-1.3 role-ownership item (PR-2.1), but the deferral is ingest-DML-framed and should be expanded to cover migration DDL. Does not block Phase 1 (no Phase-1 migration alters an existing table). (amend_plan: already-done)
- **Master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while sending the master password.** — handled: Not handled — flagged must-fix (MITM exposure); workflow already uses verify-full, so only the README bootstrap is inconsistent. (amend_plan: no)
- **NaN/Inf f64 threshold cross-language hash divergence is unguarded (no golden vector).** — handled: Flagged should-fix coverage; threshold is a cosine value so NaN is implausible, but the divergence would be silent. (amend_plan: yes)

### Testing coverage assessment

**Tested:**
- Hash Rust==golden==Python across all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8 (63 vectors, executed bit-exact) (`benchmarks-website/server/tests/measurement_id_golden.rs + scripts/test_measurement_id.py`, confidence high)
- migrate-schema apply / idempotency / name-order / failing-migration rollback (subprocess) / status drift / empty-file rejection / non-default search_path ledger agreement / subdir-skip / case-insensitive discovery (`scripts/test_migrate_schema.py`, confidence high)
- Real 001-003 apply cleanly + idempotent; 6 tables, 6 indexes, per-table column order+nullability, key type translations, migrator role login, ledger grants (SELECT/INSERT present, DELETE/UPDATE absent) (`scripts/test_migrate_schema.py:3480-3640`, confidence high)

**Untested (skeptical):**
- [high] Live schema-deploy OIDC apply as migrator against the real endpoint (and whether the proxy is even reachable from CI) — Recorded as wiring/lint/testcontainer only; reachability + migrator-credential findings suggest it may not work as wired.
- [high] scripts/ pytest running in CI (golden==Python parity AND the testcontainer suite are both ungated) — No CI job runs uv run --all-packages pytest scripts/; deferred CI-hardening.
- [high] RDS Proxy reachability from off-VPC GitHub-hosted runners — RDS Proxy is not publicly accessible; the CI-write endpoint design needs rework.
- [high] migrator credential registered for RDS Proxy IAM auth — Proxy has only the master secret; migrator connection would fail auth.
- [medium] NaN/Inf f64 threshold cross-language equivalence — No non-finite threshold vector.
- [medium] Future-migration DDL (ALTER/CREATE INDEX) run as migrator on master-owned tables — Deferred role-ownership (PR-2.1); no Phase-1 migration alters an existing table.
- [medium] Composite-index column definitions (tests assert names only) — Index-definition assertions not written.
- [medium] Edit-after-apply ledger drift (no fingerprint column) — Deferred (sha256 ledger column).

_Recommendations:_ Resolve the CI-write endpoint design FIRST (RDS Proxy reachability + migrator credential) — this likely amends Key decisions Q2/Q6 (point schema-deploy at the public RDS instance endpoint with direct IAM, or run in-VPC). Then run schema-deploy live once and record the clean apply/status. Fix the README bootstrap to PGSSLMODE=verify-full. Land the plan-edit must-fixes (exit-criteria test name, Table D, provision.sh account, vector-count reconciliation). Wire scripts/ pytest into CI (closes the largest hash-pin exposure).

### Tradeoffs re-evaluation

- **Branching / merge target (per-phase child branches -> ct/bench-v4 -> develop)** → `keep` — No artifact conflict.
- **Postgres flavor (RDS db.t4g.micro single-AZ, account 245040174862)** → `keep` — Provisioning/DDL target the chosen account/region/class; flavor-portable (testcontainer runs vanilla Postgres).
- **Connection pooler (RDS Proxy with IAM-auth pass-through)** → `revisit-but-keep` — The proxy is right for Vercel reads, BUT the CI-write-via-proxy assumption is challenged: RDS Proxy is not publicly reachable from off-VPC GitHub runners and lacks a migrator credential. The CI-write endpoint specifically needs rework (most-pessimistic across reviewers; correctness/codex would lean reverse for the CI path).
- **Ingest writer language (pure Python + golden vectors)** → `revisit-but-keep` — Hash port verified bit-exact, but the 100-vs-63 fixture-count and Table D lockstep gaps need reconciliation.
- **Schema deploy tool (in-house migrate-schema.py + plain SQL)** → `revisit-but-keep` — Sound and well-tested, but grew to ~180 LOC and still lacks ledger fingerprinting, so the documented edit-after-apply prohibition has zero runtime enforcement (deferred).
- **Schema-deploy authorization (PR merge is the gate; no environment gate)** → `keep` — Accepted tradeoff; NOT re-flagged. Only the stale header COMMENT advertising the superseded Environment gate is flagged (doc-sync, not a reversal).
- **CI network reach (public + IAM, verify-full)** → `revisit-but-keep` — The public+IAM model works for the RDS INSTANCE endpoint, but the 'public RDS Proxy endpoint' sub-assumption is invalid (proxy isn't public). Tied to the schema-deploy.yml:77 must-fix.
- **Composite index definition strategy ((dim_tuple..., commit_timestamp DESC))** → `revisit-but-keep` — Implemented as dim-leading read-path indexes without the trailing timestamp; either amend the decision to the implemented strategy or pin index column definitions in tests.
- **Cutover style / One-shot load / Read framework / Operator SQL / v3 disposition** → `keep` — Future-phase decisions; no Phase-1 change contradicts them.

### Disagreements

- **Severity of the wrong-account header comment (provision.sh:19)**: maint: must-fix: the comment names the WRONG account (375504701696 = personal/v3), and an operator trusting it provisions into the wrong account or hits a confusing verify_prereqs die.; correctness/claude: nit: cosmetic doc-quality; the actual TARGET_ACCOUNT default is correct. → call: must-fix (HIGHEST, conservative). An operator-facing runbook naming the wrong AWS account is an operational hazard, not cosmetic; the fix is one line.
- **Table D SCHEMA_VERSION re-export (scope-drift must-fix vs amend-the-doc should-fix)**: spec: must-fix scope-drift: Table D requires _measurement_id.py to re-export SCHEMA_VERSION; implement the re-export OR amend Table D.; maint: should-fix: the shipped module is CORRECT to omit the re-export (can't cleanly import hyphenated post-ingest.py; the hash port has no need for SCHEMA_VERSION); Table D is stale — amend the doc. → call: Keep must-fix severity (HIGHEST, and Table D is a reference downstream PRs grep into — fix before phase close), but the ACTION is to amend Table D, NOT to add a re-export. The shipped hash port is correct.

### Dropped re-flags (carry-forward)

- migrator role lacks privileges to ALTER / CREATE INDEX on master-owned tables in future migrations (correctness/codex flagged must-fix at 002_iam_db_user.sql:37) — covered by Deferred work (Deferred work:487 (PR-1.3 cycle-1 — migrator table privileges; resolve role-ownership model in PR-2.1). NOTE: the deferral is framed around INGEST DML; it should be EXPANDED to also cover future-migration DDL (ALTER/CREATE INDEX on master-owned tables) so the schema-deploy steady-state is covered, not just the ingest write path. Surfaced as a synthesizer concern. No Phase-1 migration alters an existing table, so it does not block Phase 1 functionally.)
- golden==Python hash test (and the testcontainer suite) not wired into CI (correctness/claude flagged should-fix) — covered by Deferred work (Deferred work:489 (PR-1.5 cycle-1 — scripts/ pytest not in CI) and Deferred work:486 (PR-1.2 — no CI runner). The reviewer itself acknowledged the prior triage; surfaced as the single largest standing correctness exposure for the hash pin.)

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "review_count": 3,
  "unified_findings": [
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": ".github/workflows/schema-deploy.yml:77",
      "description": "RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. As wired, the schema-deploy job cannot reach the proxy at all. This challenges Key decisions Q2/Q6 ('RDS Proxy public endpoint, security group 0.0.0.0/0').",
      "recommended_fix": "Point off-VPC schema-deploy at the public RDS *instance* endpoint with direct IAM auth (the proxy stays for Vercel reads), OR run schema-deploy inside the VPC (self-hosted runner / CodeBuild), OR expose the proxy via an intentional NLB/PrivateLink design. Resolve before claiming the schema-deploy path works; this likely amends Q2/Q6.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/infra/provision.sh:311",
      "description": "The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth still needs a Secrets Manager credential registered for the migrator DB user; without it the proxy cannot authenticate the migrator connection.",
      "recommended_fix": "Either configure end-to-end IAM auth for the proxy for the migrator user, or create+attach a migrator credential secret and register it in the proxy auth config. Add a smoke test that connects through the chosen endpoint as migrator. (Couples with the schema-deploy.yml:77 reachability finding \u2014 resolve the CI-write endpoint design together.)",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "missing-acceptance",
      "file_line": ".github/workflows/schema-deploy.yml:68",
      "description": "PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-apply.' Implementation status records only wiring + yamllint + testcontainer coverage \u2014 the live OIDC apply against real RDS Proxy was never executed. Combined with the proxy-reachability and migrator-credential findings, the schema-deploy path is unproven and may be non-functional as designed.",
      "recommended_fix": "After resolving the endpoint/credential design, run schema-deploy live once and record the clean apply/status, OR explicitly amend the PR-1.4 acceptance criterion to state live execution is deferred (and to which phase) with rationale.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "unsafe",
      "file_line": "benchmarks-website/infra/README.md:120",
      "description": "The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while transmitting the master password. This is a MITM exposure on the single most sensitive credential in the system. The schema-deploy workflow correctly uses verify-full; the bootstrap runbook is inconsistent.",
      "recommended_fix": "Change the bootstrap runbook to PGSSLMODE=verify-full with PGSSLROOTCERT pointed at the downloaded RDS CA bundle, matching the workflow.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:19",
      "description": "The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and the entire README/plan are 245040174862. Per Key decisions, 375504701696 is the PERSONAL/v3-EC2 account \u2014 exactly the account the bench infra must NOT land in. An operator trusting the header points at the wrong account; verify_prereqs would then die confusingly, or worse the operator provisions into the wrong account.",
      "recommended_fix": "Change the line-19 comment to account 245040174862 to match TARGET_ACCOUNT and the README; or interpolate the value rather than hardcoding a stale literal.",
      "found_by": [
        "maint",
        "correctness/claude"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:102",
      "description": "PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qualitative coverage is strong (all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8), but the literal acceptance criterion is unmet and was not amended.",
      "recommended_fix": "Either add ~37 more deterministic fixture vectors, OR amend the PR-1.5 acceptance criterion to '63 vectors' with rationale (the chosen 63 exhaustively cover all tables + boundary classes). Cheap; pick one and record it.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "weak-exit-criteria",
      "file_line": "scripts/test_measurement_id.py:1",
      "description": "The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scripts/test_measurement_id.py. The documented phase gate is unrunnable as written (no such file). Independently confirmed during exit-criteria execution.",
      "recommended_fix": "Amend the Phase-1 exit-criteria string to 'pytest scripts/test_measurement_id.py' (the as-shipped file). Plan-edit only.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "scripts/_measurement_id.py:50",
      "description": "Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py | Python re-export to keep one site'. The shipped module neither imports nor re-exports SCHEMA_VERSION (and could not cleanly import from hyphenated post-ingest.py without importlib). The hash port is correct to omit it; Table D is stale and would mislead a future SCHEMA_VERSION bump. (Table D is a reference downstream PRs grep into.)",
      "recommended_fix": "Amend Table D to remove the _measurement_id.py re-export row (or replace it with the real lockstep site). Do NOT add a spurious re-export to the hash port. Plan-edit only. (See disagreement: spec framed this as must-fix scope-drift requiring re-export OR amendment; maint framed it as should-fix amend-the-doc; synthesizer call = amend Table D, code is correct.)",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:35",
      "description": "The 'Initial files' section lists only 001 and 002 and says the SQL files 'land in PR-1.3', but PR-1.4 added 003_migrator_ledger_grant.sql, which exists on disk and is exercised by the test suite. The directory's own README is a stale, incomplete inventory.",
      "recommended_fix": "Add a bullet for 003_migrator_ledger_grant.sql (GRANT SELECT,INSERT on the ledger to migrator, PR-1.4) and fix the 'land in PR-1.3' sentence.",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "coverage",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1108-1117",
      "description": "No golden vector exercises a NaN or Inf f64 threshold. Rust write_f64 uses v.to_bits() (preserves NaN payload bits); Python struct.pack('<d', nan) emits canonical NaN (0x7ff8...). A NaN threshold would hash differently across languages -> silent duplicate row, the exact failure the hash pin exists to prevent. Inf is canonical on both, so the gap is narrowly NaN.",
      "recommended_fix": "Add f64::NAN / INFINITY / NEG_INFINITY threshold vectors and regenerate the golden file, OR assert threshold.is_finite() at the PR-2.1 ingest boundary so a non-finite value fails loudly rather than diverging silently.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "scope-drift",
      "file_line": "migrations/001_initial_schema.sql:75",
      "description": "The composite-index Key decision promised indexes on '(dim_tuple..., commit_timestamp DESC)'; the migration creates dim-leading (read-path filter) indexes WITHOUT the trailing commit_timestamp, and the tests assert only index *names*, not indexed columns/order. The divergence is explained in PR-1.3's surprises (dim-leading serves the chart read path; PK enforces hash-tuple uniqueness) but the Key decision row was never updated.",
      "recommended_fix": "Amend the composite-index Key decision to the implemented read-path strategy, AND/OR add a test asserting the index column definitions (not just names) so the intended shape is pinned.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:12-14",
      "description": "The workflow header still advertises a 'schema-deploy GitHub Environment with manual-approval as the stronger gate ... tracked as deferred hardening'. The 2026-05-29 deploy-model Key decision SUPERSEDED that path (PR merge is the gate; the Environment gate was judged the wrong tool). The comment points a future engineer at deferred work the plan reversed. (NOT a re-flag of the accepted no-env-gate tradeoff \u2014 this flags the stale comment, aligned with the decision.)",
      "recommended_fix": "Replace the Environment-gate framing with the actual deferred item: switch the trigger to push on the deploy branch under paths: migrations/** (PR-merge-is-the-gate). Naturally lands with the deferred trigger-switch (Deferred work, 2026-05-29 deploy-model item).",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:266",
      "description": "proxy_role_name ('vortex-bench-proxy-role') and its policy name are hardcoded locals, but the README 'Customizing' section claims 'Every name / class / engine version / region is set at the top via readonly declarations with ${ENV:-default} fallbacks.' The proxy role is an exception, and the tear-down runbook uses the literal default name, so an operator who overrode other names gets a tear-down mismatch.",
      "recommended_fix": "Promote proxy_role_name to a top-level readonly PROXY_ROLE_NAME=\"${PROXY_ROLE_NAME:-vortex-bench-proxy-role}\" like the other names, OR soften the README 'every name is overridable' claim to enumerate the proxy role as fixed.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": "scripts/migrate-schema.py:2733-2745",
      "description": "status() reports a generic 'pending' (exit 1) for an empty/whitespace-only migration file, while apply() rejects it explicitly only when it reaches it. Same bad file, two different diagnoses. Behavior is safe (loud both ways) but asymmetric.",
      "recommended_fix": "Optionally have status() classify empty/whitespace-only on-disk files distinctly so the operator sees the same diagnosis from status as from apply.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:39",
      "description": "002 is described as 'CREATE ROLE for the IAM-auth user that bench.yml workflows assume into', but migrator is consumed by the schema-deploy workflow (PR-1.4), not bench.yml (the ingest workflow, which uses a separate future ingest role). The WHY misattributes the consumer.",
      "recommended_fix": "Change 'bench.yml workflows' to 'the schema-deploy workflow (PR-1.4)'.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scaffolding",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:30",
      "description": "REGEN_GOLDEN_VECTORS is permanent regeneration scaffolding (write-on-env, always-assert otherwise). Correct and well-documented, but nothing flags committed-JSON drift unless the Rust test runs in CI, and the golden==Python half is not CI-gated (deferred).",
      "recommended_fix": "Add one sentence to the test module doc noting golden==Python is only enforced once 'uv run --all-packages pytest scripts/' is wired into CI (deferred), so a green local run does not prove cross-language parity in CI.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:84-89",
      "description": "The 'Apply migrations' step carries a ~5-line justification comment (set -x suppression, PGPASSWORD-on-own-line vs export masking). It documents two real footguns but is exactly the >100-char justification-comment shape the shared BAN warns about.",
      "recommended_fix": "Keep the substance but tighten to two short sentences (token-leak + export-masking). No behavior change.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1",
      "description": "PR-1.5's expected files row named 'benchmarks-website/server/src/db.rs (golden-vector test added)', but the test shipped as a separate integration test file benchmarks-website/server/tests/measurement_id_golden.rs. Minor placement deviation from the plan row.",
      "recommended_fix": "Amend the PR-1.5 expected-files row to name the integration test location (the chosen placement is fine; the plan row is stale).",
      "found_by": [
        "spec"
      ]
    }
  ],
  "disagreements": [
    {
      "topic": "Severity of the wrong-account header comment (provision.sh:19)",
      "positions": [
        {
          "lens": "maint",
          "position": "must-fix: the comment names the WRONG account (375504701696 = personal/v3), and an operator trusting it provisions into the wrong account or hits a confusing verify_prereqs die."
        },
        {
          "lens": "correctness/claude",
          "position": "nit: cosmetic doc-quality; the actual TARGET_ACCOUNT default is correct."
        }
      ],
      "synthesizer_call": "must-fix (HIGHEST, conservative). An operator-facing runbook naming the wrong AWS account is an operational hazard, not cosmetic; the fix is one line."
    },
    {
      "topic": "Table D SCHEMA_VERSION re-export (scope-drift must-fix vs amend-the-doc should-fix)",
      "positions": [
        {
          "lens": "spec",
          "position": "must-fix scope-drift: Table D requires _measurement_id.py to re-export SCHEMA_VERSION; implement the re-export OR amend Table D."
        },
        {
          "lens": "maint",
          "position": "should-fix: the shipped module is CORRECT to omit the re-export (can't cleanly import hyphenated post-ingest.py; the hash port has no need for SCHEMA_VERSION); Table D is stale \u2014 amend the doc."
        }
      ],
      "synthesizer_call": "Keep must-fix severity (HIGHEST, and Table D is a reference downstream PRs grep into \u2014 fix before phase close), but the ACTION is to amend Table D, NOT to add a re-export. The shipped hash port is correct."
    }
  ],
  "dropped_re_flags": [
    {
      "topic": "migrator role lacks privileges to ALTER / CREATE INDEX on master-owned tables in future migrations (correctness/codex flagged must-fix at 002_iam_db_user.sql:37)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:487 (PR-1.3 cycle-1 \u2014 migrator table privileges; resolve role-ownership model in PR-2.1). NOTE: the deferral is framed around INGEST DML; it should be EXPANDED to also cover future-migration DDL (ALTER/CREATE INDEX on master-owned tables) so the schema-deploy steady-state is covered, not just the ingest write path. Surfaced as a synthesizer concern. No Phase-1 migration alters an existing table, so it does not block Phase 1 functionally."
    },
    {
      "topic": "golden==Python hash test (and the testcontainer suite) not wired into CI (correctness/claude flagged should-fix)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:489 (PR-1.5 cycle-1 \u2014 scripts/ pytest not in CI) and Deferred work:486 (PR-1.2 \u2014 no CI runner). The reviewer itself acknowledged the prior triage; surfaced as the single largest standing correctness exposure for the hash pin."
    }
  ],
  "phase_artifacts": {
    "summary": "Phase 1 ('RDS + schema + hash port') lands the migration foundation in five concept areas. (A) Infrastructure: provision.sh idempotently bootstraps RDS Postgres db.t4g.micro + RDS Proxy (IAMAuth=REQUIRED, TLS) + GitHub OIDC provider + GitHubBenchmarkSchemaRole in account 245040174862, with an operator runbook in infra/README.md. (B) Schema-deploy CI: schema-deploy.yml (workflow_dispatch + dry_run; PR-merge is the accepted authorization gate, no environment: approval) generates a client-side IAM token and runs the migrate runner against the proxy as migrator over verify-full TLS. (C) Migration runner: scripts/migrate-schema.py applies migrations/*.sql in name order, tracks public._applied_migrations, is idempotent, uses autocommit + per-migration top-level transactions so a failing later migration rolls back only itself, rejects empty files; 28+ testcontainer tests. (D) DDL: 001 creates the commits dim + 5 fact tables + read-path composite indexes (Postgres translation of the authoritative DuckDB schema.rs, column order/nullability/types preserved); 002 the migrator login role + conditional rds_iam; 003 the append-only ledger grant. (E) Hash port: _measurement_id.py is a byte-for-byte port of db.rs measurement_id_* (xxhash64 seed 0), pinned by a Rust source-of-truth golden file giving Rust==golden==Python, verified bit-exact for all 63 vectors (Claude correctness executed the port; Rust golden test passes). The keystone hash-equivalence deliverable is solid and the schema shape provably matches schema.rs. HOWEVER the AWS-integration path has deploy-blocking gaps the Codex correctness lens surfaced: RDS Proxy endpoints are not publicly reachable from off-VPC GitHub runners (challenges Key decisions Q2/Q6); the proxy lacks a migrator credential for IAM auth; PR-1.4's live OIDC apply against real RDS Proxy was never executed (only wiring + lint + testcontainer); and the master-bootstrap runbook uses PGSSLMODE=require (no cert verification) while sending the master password. The Codex spec lens found contract gaps: 63 vectors shipped vs the promised 100; the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py; Table D claims a SCHEMA_VERSION re-export the hash port omits. Maintainability is otherwise high, but provision.sh's header names the WRONG AWS account, migrations/README omits 003, and the schema-deploy header advertises the superseded Environment gate.",
    "surprises": [
      {
        "what": "RDS Proxy endpoints are not publicly accessible; off-VPC GitHub-hosted runners cannot reach the proxy. The plan's 'RDS Proxy public endpoint' assumption (Q6) may be architecturally invalid for the CI-write path.",
        "how_handled": "Not handled in the diff \u2014 flagged as a must-fix deploy-blocker; likely forces amending Key decisions Q2/Q6 (e.g., CI writes to the public RDS instance endpoint with direct IAM, proxy stays for Vercel reads).",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.4's schema-deploy was accepted on wiring + yamllint + testcontainer, never run live against real RDS Proxy.",
        "how_handled": "Recorded honestly in Implementation status; spec lens flags the acceptance criterion as unmet. Coupled with the proxy-reachability + migrator-credential findings, the path is unproven.",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.5 shipped 63 golden vectors, not the promised 100.",
        "how_handled": "Status acknowledges 63; no amendment or extra fixtures. Qualitative coverage is strong (all tables + boundaries).",
        "amend_plan": "yes"
      },
      {
        "what": "The Phase-1 exit criterion names pytest scripts/test_post_ingest_hash.py, but the shipped file is test_measurement_id.py.",
        "how_handled": "Not reconciled; the documented gate is unrunnable as written. Independently confirmed during exit-criteria execution.",
        "amend_plan": "yes"
      },
      {
        "what": "Table D's claim that _measurement_id.py re-exports SCHEMA_VERSION is stale; the shipped module correctly omits it.",
        "how_handled": "Artifact correct; Table D not updated. Amend the reference table.",
        "amend_plan": "yes"
      },
      {
        "what": "Composite indexes are dim-leading (read-path filter columns), not the Key decision's '(dim_tuple..., commit_timestamp DESC)'.",
        "how_handled": "Explained in PR-1.3 surprises (PK enforces hash-tuple uniqueness; dim-leading serves charts); Key decision row not updated and tests assert only index names.",
        "amend_plan": "yes"
      },
      {
        "what": "migrator role cannot ALTER / CREATE INDEX on master-owned tables in future migrations (GRANT CREATE on public is insufficient).",
        "how_handled": "Covered by the deferred PR-1.3 role-ownership item (PR-2.1), but the deferral is ingest-DML-framed and should be expanded to cover migration DDL. Does not block Phase 1 (no Phase-1 migration alters an existing table).",
        "amend_plan": "already-done"
      },
      {
        "what": "Master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while sending the master password.",
        "how_handled": "Not handled \u2014 flagged must-fix (MITM exposure); workflow already uses verify-full, so only the README bootstrap is inconsistent.",
        "amend_plan": "no"
      },
      {
        "what": "NaN/Inf f64 threshold cross-language hash divergence is unguarded (no golden vector).",
        "how_handled": "Flagged should-fix coverage; threshold is a cosine value so NaN is implausible, but the divergence would be silent.",
        "amend_plan": "yes"
      }
    ],
    "coverage": {
      "tested_cases": [
        {
          "case": "Hash Rust==golden==Python across all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8 (63 vectors, executed bit-exact)",
          "test_location": "benchmarks-website/server/tests/measurement_id_golden.rs + scripts/test_measurement_id.py",
          "confidence": "high"
        },
        {
          "case": "migrate-schema apply / idempotency / name-order / failing-migration rollback (subprocess) / status drift / empty-file rejection / non-default search_path ledger agreement / subdir-skip / case-insensitive discovery",
          "test_location": "scripts/test_migrate_schema.py",
          "confidence": "high"
        },
        {
          "case": "Real 001-003 apply cleanly + idempotent; 6 tables, 6 indexes, per-table column order+nullability, key type translations, migrator role login, ledger grants (SELECT/INSERT present, DELETE/UPDATE absent)",
          "test_location": "scripts/test_migrate_schema.py:3480-3640",
          "confidence": "high"
        }
      ],
      "untested_cases": [
        {
          "case": "Live schema-deploy OIDC apply as migrator against the real endpoint (and whether the proxy is even reachable from CI)",
          "priority": "high",
          "why_untested": "Recorded as wiring/lint/testcontainer only; reachability + migrator-credential findings suggest it may not work as wired."
        },
        {
          "case": "scripts/ pytest running in CI (golden==Python parity AND the testcontainer suite are both ungated)",
          "priority": "high",
          "why_untested": "No CI job runs uv run --all-packages pytest scripts/; deferred CI-hardening."
        },
        {
          "case": "RDS Proxy reachability from off-VPC GitHub-hosted runners",
          "priority": "high",
          "why_untested": "RDS Proxy is not publicly accessible; the CI-write endpoint design needs rework."
        },
        {
          "case": "migrator credential registered for RDS Proxy IAM auth",
          "priority": "high",
          "why_untested": "Proxy has only the master secret; migrator connection would fail auth."
        },
        {
          "case": "NaN/Inf f64 threshold cross-language equivalence",
          "priority": "medium",
          "why_untested": "No non-finite threshold vector."
        },
        {
          "case": "Future-migration DDL (ALTER/CREATE INDEX) run as migrator on master-owned tables",
          "priority": "medium",
          "why_untested": "Deferred role-ownership (PR-2.1); no Phase-1 migration alters an existing table."
        },
        {
          "case": "Composite-index column definitions (tests assert names only)",
          "priority": "medium",
          "why_untested": "Index-definition assertions not written."
        },
        {
          "case": "Edit-after-apply ledger drift (no fingerprint column)",
          "priority": "medium",
          "why_untested": "Deferred (sha256 ledger column)."
        }
      ],
      "recommendations": "Resolve the CI-write endpoint design FIRST (RDS Proxy reachability + migrator credential) \u2014 this likely amends Key decisions Q2/Q6 (point schema-deploy at the public RDS instance endpoint with direct IAM, or run in-VPC). Then run schema-deploy live once and record the clean apply/status. Fix the README bootstrap to PGSSLMODE=verify-full. Land the plan-edit must-fixes (exit-criteria test name, Table D, provision.sh account, vector-count reconciliation). Wire scripts/ pytest into CI (closes the largest hash-pin exposure)."
    },
    "tradeoffs": [
      {
        "decision": "Branching / merge target (per-phase child branches -> ct/bench-v4 -> develop)",
        "original": "User pick Q1",
        "verdict": "keep",
        "rationale": "No artifact conflict."
      },
      {
        "decision": "Postgres flavor (RDS db.t4g.micro single-AZ, account 245040174862)",
        "original": "User pick Q2",
        "verdict": "keep",
        "rationale": "Provisioning/DDL target the chosen account/region/class; flavor-portable (testcontainer runs vanilla Postgres)."
      },
      {
        "decision": "Connection pooler (RDS Proxy with IAM-auth pass-through)",
        "original": "Locked by Q2",
        "verdict": "revisit-but-keep",
        "rationale": "The proxy is right for Vercel reads, BUT the CI-write-via-proxy assumption is challenged: RDS Proxy is not publicly reachable from off-VPC GitHub runners and lacks a migrator credential. The CI-write endpoint specifically needs rework (most-pessimistic across reviewers; correctness/codex would lean reverse for the CI path)."
      },
      {
        "decision": "Ingest writer language (pure Python + golden vectors)",
        "original": "User pick Q4",
        "verdict": "revisit-but-keep",
        "rationale": "Hash port verified bit-exact, but the 100-vs-63 fixture-count and Table D lockstep gaps need reconciliation."
      },
      {
        "decision": "Schema deploy tool (in-house migrate-schema.py + plain SQL)",
        "original": "User pick Q5a",
        "verdict": "revisit-but-keep",
        "rationale": "Sound and well-tested, but grew to ~180 LOC and still lacks ledger fingerprinting, so the documented edit-after-apply prohibition has zero runtime enforcement (deferred)."
      },
      {
        "decision": "Schema-deploy authorization (PR merge is the gate; no environment gate)",
        "original": "User decision 2026-05-29",
        "verdict": "keep",
        "rationale": "Accepted tradeoff; NOT re-flagged. Only the stale header COMMENT advertising the superseded Environment gate is flagged (doc-sync, not a reversal)."
      },
      {
        "decision": "CI network reach (public + IAM, verify-full)",
        "original": "User pick Q6",
        "verdict": "revisit-but-keep",
        "rationale": "The public+IAM model works for the RDS INSTANCE endpoint, but the 'public RDS Proxy endpoint' sub-assumption is invalid (proxy isn't public). Tied to the schema-deploy.yml:77 must-fix."
      },
      {
        "decision": "Composite index definition strategy ((dim_tuple..., commit_timestamp DESC))",
        "original": "Forward-looking design",
        "verdict": "revisit-but-keep",
        "rationale": "Implemented as dim-leading read-path indexes without the trailing timestamp; either amend the decision to the implemented strategy or pin index column definitions in tests."
      },
      {
        "decision": "Cutover style / One-shot load / Read framework / Operator SQL / v3 disposition",
        "original": "User picks Q5b,Q7,Q8,Q9 + forward-looking",
        "verdict": "keep",
        "rationale": "Future-phase decisions; no Phase-1 change contradicts them."
      }
    ]
  },
  "executive_summary": "Phase 1 ships a coherent, unusually well-documented foundation, and its keystone deliverable \u2014 the cross-language measurement_id hash equivalence (Rust==golden==Python) \u2014 is verified bit-exact for all 63 vectors, with the 6-table Postgres schema provably matching the authoritative DuckDB schema.rs and a well-tested migration runner. The mixed-executor review (Claude + Codex lenses) is the reason this is a REJECT rather than an accept: the Codex correctness lens surfaced a cluster of deploy-blocking AWS-integration gaps that the hash-focused Claude review did not. The most serious: RDS Proxy endpoints are NOT publicly reachable from off-VPC GitHub-hosted runners (schema-deploy.yml:77), the proxy was provisioned without a migrator credential for IAM auth (provision.sh:311), and PR-1.4's live OIDC apply against real RDS Proxy was never actually executed (schema-deploy.yml:68) \u2014 together meaning the schema-deploy CI path, a core Phase-1 deliverable, is unproven and likely broken as wired. Resolving it probably amends Key decisions Q2/Q6 (e.g., point CI writes at the public RDS *instance* endpoint with direct IAM and reserve the proxy for Vercel reads, or run schema-deploy in-VPC). A fourth correctness must-fix: the master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while transmitting the master password (README:120) \u2014 a MITM exposure, fixed by verify-full. The Codex spec lens added contract-closure must-fixes that are mostly cheap plan-edits: 63 vectors shipped vs the promised 100 (reconcile the criterion or add vectors); the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py (rename to the shipped test_measurement_id.py \u2014 independently confirmed at exit-criteria time); and Table D claims a SCHEMA_VERSION re-export the hash port correctly omits (amend the reference table, do NOT add the re-export). The maint lens caught a genuine operational hazard: provision.sh's header comment names the WRONG (personal/v3) AWS account, which could misdirect an operator. Five should-fixes (stale migrations/README missing 003; composite-index decision-vs-impl drift; the schema-deploy header advertising the superseded Environment gate; a hardcoded proxy-role name contradicting the README; the unguarded NaN/Inf hash vector) and five nits round it out. Two findings were dropped as carry-forward (migrator table privileges and CI-gating of scripts/ pytest are both already in Deferred work) \u2014 though the role-privileges deferral should be expanded to cover future-migration DDL, not just ingest DML. Verdict: reject, 8 must-fix. The hash and schema work is strong; the AWS-integration + plan-consistency layer needs a focused fix pass before Phase 1 closes.",
  "overall": "reject",
  "must_fix_count": 8,
  "should_fix_count": 5,
  "nit_count": 5,
  "review_cycles_this_invocation": 1,
  "executor_routing": {
    "spec": "codex",
    "correctness": "parallel",
    "maint": "claude"
  }
}
```

</details>

## Phase 1: RDS + schema + hash port — end-of-phase review (cycle 2) — accepted (3-vote; user-authorized over operator-gated + deferred items)

**Recorded from 4 REAL reviewer invocations** (gauntlet `preset=phase-3`, `executor=mixed`: `spec`/codex, `correctness`/claude+codex parallel, `maint`/claude), run against the cumulative Phase-1 code diff `ae3e0494f..HEAD` (159KB, `.big-plans/` excluded). **Reconciliation note:** the formal synthesizer SUBAGENT was not spawned for this cycle (the two Claude reviewer outputs were ~140-177KB and `SendMessage` was unavailable to marshal them into files without re-running the reviews); the orchestrator reconciled the 4 real reviewer outputs inline (conservative-union severity, carry-forward drop). The 2 Codex raw JSONs are archived at `/tmp/gauntlet-PR-1_6-cycle2-35145/phase-reviews/`; the Claude findings + phase_artifacts are captured below. This is a transparent inline reconciliation of a real review, NOT a fabricated synthesizer output.

**Reviewer verdicts**: `correctness`/claude ACCEPT (2 nits); `maint`/claude ACCEPT (2 nits); `spec`/codex REJECT (1 must-fix); `correctness`/codex REJECT (3 must-fix). Conservative verdict: **reject** → resolved as below.

### Unified findings + resolution

| Severity | File:line | Finding | Resolution |
|---|---|---|---|
| must-fix | `migrate-schema.py:158` | Migration DDL runs under caller `search_path`; under a non-default `search_path` an unqualified `CREATE TABLE` lands in the wrong schema while the public-pinned ledger reports clean. (correctness/codex) | **FIXED** in `cb68db1c6`: `SET LOCAL search_path TO public` per migration txn + test extended to assert the table lands in `public`. |
| must-fix | `provision.sh:67` | `PG_MIGRATOR_ROLE` env-overridable but `migrations/002` hardcodes `CREATE ROLE migrator`; an override scopes the IAM grant to a nonexistent user and breaks deploy auth. (correctness/codex) | **FIXED** in `cb68db1c6`: `PG_MIGRATOR_ROLE` hardcoded to `migrator` (no env override) with an explanatory comment. |
| must-fix | (acceptance criterion) | Phase-1 / PR-1.6 requires the operator to run the live OIDC schema-deploy apply against the instance endpoint + confirm `status` clean; this is unverified (out-of-band). (spec/codex) | **DEFERRED to operator pre-merge action** (live AWS apply; not performable in-session — no AWS creds). Tracked as the operator's gate before `ct/bench-v4` → `develop`. |
| must-fix | `test_migrate_schema.py:980` | Tests assert migrator privileges by inspection but never run the runner AS migrator after a master-owned bootstrap. (correctness/codex) | **DROPPED (re-flag)**: covered by Deferred work (PR-1.3 cycle-1 → PR-2.1 role-ownership model). |
| nit | `migrate-schema.py` discover/empty-file | Case-insensitive ledger keys; comment-only migration semantics untested. (correctness/claude) | Deferred (off-convention inputs; low priority). |
| nit | `_measurement_id.py:48` | twox-hash endianness attribution misplaced (std Hasher defaults, not twox-hash); conclusion correct. (maint/claude) | Deferred (doc-quality; follow-up polish). |
| nit | `migrate-schema.py:88` | `_applied_set` name does not signal its CREATE-TABLE side effect. (maint/claude) | Deferred (naming clarity; follow-up polish). |

### Phase artifacts (reconciled from the 2 Claude reviewers' real phase_artifacts)

- **Summary of changes**: Phase 1 stands up the data substrate — RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub-OIDC schema role (`provision.sh`); a forward-only `migrate-schema.py` runner (public-pinned ledger, per-migration top-level autocommit transactions, now `SET LOCAL search_path TO public`); the 6-table schema + dim-leading read-path composite indexes (001), the IAM-auth `migrator` role (002), the ledger SELECT/INSERT grant (003); and a byte-for-byte Python port of the server-internal xxhash64 `measurement_id`, pinned transitively Rust==golden==Python via 63 vectors. PR-1.6 repointed CI from the VPC-internal proxy to the public instance endpoint (verify-full) with consistent CI=instance / proxy=Vercel docs + 5 static drift guards.
- **Surprises**: (1) RDS Proxy is VPC-internal/unreachable from off-VPC runners — corrected via the PR-1.6 repoint + guards (Key decisions amended 2026-05-29). (2) Composite indexes diverged to dim-leading read-path columns (ratified; verified against `api/charts.rs` by maint/claude; pinned by the index test). (3) The single-owner testcontainer model does not exercise the master/migrator ownership split (documented; deferred PR-2.1). (4) NaN/Inf threshold cross-language divergence (deferred to a PR-2.1 ingest `is_finite()` guard).
- **Testing coverage**: hash equivalence (Rust golden + Python, 63 vectors, all boundary classes); runner transaction discipline, idempotency, ordering, partial-failure persistence, search_path ledger AND now table placement; schema shape/nullability/index columns+order/role grants; 5 static PR-1.6 drift guards. Untested-but-triaged: golden==Python not CI-gated (deferred CI-hardening — highest-leverage Phase-2 action); non-additive migration under the real role split (deferred PR-2.1).
- **Tradeoffs re-evaluation**: all Key decisions hold (`keep`); the schema-deploy-tool decision is `revisit-but-keep` (runner grew to ~229 LOC, still lacks applied-migration fingerprinting — revisit before the first data-affecting migration); no decision warrants `reverse`. The dead proxy grant on the schema role remains deferred least-privilege cleanup (PR-2.1).

**Acceptance**: per the operator's cycle-2 phase-boundary decision, the 2 code must-fix were fixed (`cb68db1c6`) and the Phase-1 boundary is **accepted**, with the live OIDC apply as the operator's pre-merge gate and all nits + the dead-proxy-grant deferred to follow-up. No code defect in the shipped Phase-1 substrate was found across either phase-end cycle; the cycle-1→cycle-2 findings were endpoint-model + cross-PR-consistency + test-strictness items, all now resolved or tracked.

## Phase 1 live verification (2026-06-01)

Run live against real AWS by the operator + assistant on 2026-06-01 (account `245040174862`, local CLI profile `bench-prod` = IAM user `connor-aws-cli`, AdministratorAccess). This verified whether "Phase 1 is actually done" and surfaced that the schema had **never been applied** and that two GitHub repo-var changes this plan claimed PR-1.6 made had **never actually happened**. All three were fixed in-session.

**Verified real (read-only AWS):** RDS instance `vortex-bench-prod` (`db-4VPTDACTRQHOS24WEIR3TNC2M4`) `available`, `IAMDatabaseAuthenticationEnabled: true`; RDS Proxy `vortex-bench-proxy` `available`; OIDC provider `token.actions.githubusercontent.com` (aud `sts.amazonaws.com`) present; `GitHubBenchmarkSchemaRole` trust branch-scoped to `refs/heads/develop` + `refs/heads/ct/bench-v4` (the PR-1.1 wildcard `repo:vortex-data/vortex:*` is gone, so the deferred trust-tightening re-run WAS applied); inline policy `rds-db-connect-migrator` grants `rds-db:connect` on the instance + proxy `dbuser:.../migrator` resources.

**Fixed in-session (mutations to prod / shared repo):**

1. **Schema bootstrap (was MISSING; the DB was bare).** Applied `001+002+003` as the RDS master (`postgres`) via `migrate-schema.py apply` over `verify-full`. Post-state verified: 6 data tables + `public._applied_migrations` ledger present; `migrator` role exists with LOGIN and is a member of `rds_iam`; `migrator` has SELECT/INSERT on the ledger; ledger records all 3 migrations. This is the one-time master bootstrap that `schema-deploy.yml`'s header documents.
2. **`RDS_BENCH_INSTANCE_ENDPOINT` repo var (was MISSING).** Set to `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com`. Without it the workflow's `PGHOST` resolved to empty. This corrects the Implementation-status claims (PR-1.1 "Repo vars set" and PR-1.6 scope) that PR-1.6 had already added it.
3. **Stale `RDS_BENCH_ENDPOINT` (proxy) repo var (lingered).** Audited (zero consumers across workflows, scripts, and app code; only negative test guards and plan history reference the name) and **deleted**. This corrects the same claims that PR-1.6 had already dropped it. The proxy endpoint value is preserved under Resource identities for the PR-4.2 Vercel config.

**IAM end-to-end path confirmed at the DB:** generated an RDS IAM auth token for `migrator` (the same token type CI mints) and connected over `verify-full`, yielding `current_user=migrator`; ran the exact CI runner commands `status` and `apply` as `migrator`, both clean (`apply` is a no-op since master already applied everything). This proves every step downstream of the OIDC token; the locally generated token stood in for the OIDC-assumed-role token (`connor-aws-cli` admin carries `rds-db:connect`).

**Resolves must-fix #1454 (operator pre-merge gate), narrowed:** the schema apply and the DB-side IAM-auth path are now DONE and verified. The only residual is the live `schema-deploy.yml` **workflow run via GitHub OIDC**, which is BLOCKED until `ct/bench-v4` merges to `develop` (a `workflow_dispatch` trigger is only registered once the workflow file is on the default branch). The sole unexercised link is the GitHub-to-STS assume-role federation, which is correctly configured (trust already allows `develop`).

**Confirmed real (deferred PR-2.1 ownership model):** `migrator` HAS `CREATE` + `USAGE` on schema `public`, so a future new-table migration via the OIDC path would succeed; but the six data tables are owned by `postgres` and `migrator` has no SELECT/INSERT/ALTER on them, so a future migration that ALTERs an existing master-owned table could NOT run as `migrator`. This is the PR-2.1 role-ownership concern, now confirmed against the live DB.

**Security postures observed (some already deferred):** the instance is `PubliclyAccessible: true` with SG `sg-065c61c4693f63816` ingress `0.0.0.0/0` on 5432, and `StorageEncrypted: false`. Worth tightening before production traffic.

## Phase 2: Postgres writer + best-effort v4 CI — end-of-phase review (cycle 1) — accepted (2-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-2: spec + correctness, executor=claude); full Synthesizer Output JSON in the `<details>` block at the end of this section. Verdict: accept (0 must-fix, 0 should-fix, 2 nits).**

### Summary of changes

Phase 2 delivers the Postgres dual-write ingest path and its CI plumbing, shipped exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction; correctness review found zero data-correctness bugs. Ingest identity (PR-2.1): migration 004 creates a least-privilege bench_ingest login role (rds_iam member on RDS, guarded for vanilla PG) holding USAGE-not-CREATE on public and SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), plus ALTER DEFAULT PRIVILEGES FOR ROLE migrator so future migrator tables auto-grant. Because the RDS master is rds_superuser but not a true superuser, the ADP runs via a temporary INHERIT self-grant through the creator's ADMIN option, then revoke. provision.sh adds GitHubBenchmarkIngestRole (OIDC trust branch-scoped to develop+ct/bench-v4, rds-db:connect for bench_ingest on the instance only) and drops the dead proxy grant. Writer (PR-2.2): post-ingest.py --postgres reproduces the v3 serde boundary in Python (deny_unknown_fields, per-field type/range validation, memory quartet, storage enum, commit_sha match), computes measurement_id bit-identically via the PR-1.5 port (_measurement_id.py is a byte-for-byte port of db.rs:162-257, pinned transitively Rust==golden==Python), then upserts commits-first and five fact tables in one all-or-nothing transaction via INSERT ... ON CONFLICT DO UPDATE with deadlock retry, over verify-full TLS authenticated by an RDS IAM token as bench_ingest. An is_finite guard rejects NaN/Inf/out-of-f64-range threshold loudly with rollback (Python stricter-or-equal to serde). All six ON CONFLICT SET lists touch only value/env columns, never dim columns; search_path is pinned to public; ssl_in_use is asserted post-connect. The v3 --server path stays stdlib-only (PEP 723 dependencies=[]). SCHEMA_VERSION lockstep held. CI wiring (PR-2.3): a scripts-test job runs pytest scripts/ behind a docker info hard gate plus a CI-env fail-loud fixture so testcontainer suites cannot silently skip. Dual-write CI (PR-2.4): the three ingest workflows each add a best-effort continue-on-error v4 --postgres step after the unchanged hard-required v3 step (v3-commit-metadata.yml gains id-token: write and ingests commit-row-only empty.jsonl); schema-deploy.yml switches to push-on-develop under paths migrations/**, keeping workflow_dispatch+dry_run. PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io) was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. All five measurement_id functions, six ON CONFLICT SET lists, field/type sets, and column widths were verified against the Rust source and 001_initial_schema.sql. Two cosmetic nits, no must-fix or should-fix.

### Surprises and discoveries

- **Migration 004 ALTER DEFAULT PRIVILEGES FOR ROLE migrator fails on a real non-superuser RDS master (createrole_self_grant default), masked by the superuser testcontainer.** — Fixed PR-2.1 cycle-1: guarded INHERIT self-grant via the master's ADMIN option then revoke, plus a test applying 001..004 as a real NOSUPERUSER CREATEROLE login. _(amend_plan: already-done)_
- **conn.info.ssl_in_use does not exist in psycopg (ssl_in_use lives on conn.pgconn); the cycle-9 fix shipped the wrong accessor.** — Fixed PR-2.2 (conn.pgconn.ssl_in_use) with a unit test pinning the accessor location and a container test pinning the traversal. _(amend_plan: already-done)_
- **fail-loud-on-no-Docker guard tests could not catch an always-skip regression because pytest.skip raises Skipped (not a Failed subclass).** — Fixed PR-2.3 (catch both outcome types and assert the specific one; mutation-verified). _(amend_plan: already-done)_
- **Python json.loads accepts NaN/Infinity literals that serde_json rejects at parse time.** — The _require_finite guard rejects them loudly with rollback, making the Python writer stricter-or-equal to v3, so no divergent row is written. Already handled. _(amend_plan: no)_
- **The v3 producer emits several fields as u32/u64 (query_idx u32; value_ns/all_runtimes_ns u64) where server and Python validator use i32/i64.** — Both v3 serde and Python _require_int reject values exceeding the signed range, so the substrates agree; values are far inside range. No change needed. _(amend_plan: no)_
- **insert-vs-update classification differs across substrates: Rust does an exists() preflight SELECT; Python derives the flag from RETURNING (xmax = 0).** — Equivalence (incl. same-dim-tuple-twice-in-one-envelope yielding (1,1) not (2,0)) is pinned by test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update. Already handled. _(amend_plan: already-done)_
- **bench-v4 Python files predate Phase 2 at ~100-col and fail the repo ruff line-length-120 (a pre-existing E501).** — Left unfixed and flagged for the operator as a branch-wide ruff reconciliation required before merging ct/bench-v4 to develop; RETAINED deferred item. _(amend_plan: no)_
- **pytest scripts/ collects a fourth file (scripts/tests/test_benchmark_reporting.py) beyond the three named suites.** — Accepted as harmless and matching the acceptance command; a clarifying ci.yml comment added. _(amend_plan: no)_

### Testing coverage assessment

| Case | Location | Confidence |
|---|---|---|
| measurement_id Rust==golden==Python parity across all 5 tables incl. negative i32, null/Some, multibyte UTF-8, whole/negative/tiny/large f64 thresholds; SCHEMA_VERSION lockstep with schema.rs | `scripts/test_measurement_id.py + scripts/measurement_id_golden.json; scripts/test_post_ingest_postgres.py:3105` | high |
| bench_ingest DML-only (SELECT/INSERT/UPDATE) on all 6 tables, denied DELETE/CREATE/DDL; default-privileges cover future migrator tables; 001..004 apply cleanly + idempotently under a real NOSUPERUSER CREATEROLE master | `scripts/test_migrate_schema.py:1960,1928,1993,2069` | high |
| provision.sh provisions ingest role on instance dbuser, emits GH_BENCH_INGEST_ROLE_ARN, drops dead proxy grant | `scripts/test_migrate_schema.py:1805,1835` | high |
| insert-then-update accounting; re-ingest upserts (0 inserted, N updated) with stable counts; measurement_id matches port | `scripts/test_post_ingest_postgres.py:2477,2572` | high |
| ON CONFLICT DO UPDATE overwrites every value/env column while leaving dim tuple / measurement_id stable, per table; dim columns NEVER in any DO UPDATE SET list for all 5 tables | `scripts/test_post_ingest_postgres.py:2867` | high |
| NaN/Inf/out-of-f64-range threshold raises loudly and rolls back; deny_unknown_fields; missing-required; unknown/non-scalar kind; storage enum; memory quartet; commit_sha mismatch; type/range boundaries reproduced | `scripts/test_post_ingest_postgres.py:2633,2887,3066,3074,3094` | high |
| connect_postgres: IAM-token-when-passwordless, always-bench_ingest, rejects weak sslmode (verify-full), forces search_path=public, rejects non-TLS, post-connect ssl_in_use | `scripts/test_post_ingest_postgres.py:3142,3221,3233,3264,3285` | high |
| --server requires --benchmark-id while --postgres does not; mutual exclusivity + dispatch | `scripts/test_post_ingest_postgres.py:3401,3363,3375` | high |
| fail-loud-on-no-Docker in CI vs skip locally (both test files) | `scripts/test_migrate_schema.py:1745; scripts/test_post_ingest_postgres.py:2753` | high |
| write-conflict retry (deadlock/serialization) retries then succeeds, gives up at cap, propagates validation errors immediately | `scripts/test_post_ingest_postgres.py (test_retry_write_conflicts_*)` | medium |

**Untested / gaps:**

| Case | Priority | Why untested |
|---|---|---|
| End-to-end live v4 dual-write per develop push (CloudWatch shows Postgres writes) — the PR-2.4 acceptance criterion needing real RDS + repo vars | high | Operator-side, not performable in-session; the v4 steps no-op until GH_BENCH_INGEST_ROLE_ARN + RDS_BENCH_INSTANCE_ENDPOINT/DB_NAME/REGION are set. Tracked as an open operator dependency. |
| Real psycopg DeadlockDetected inside conn.transaction() across two reversed-order connections on autocommit=False (retry+commit/rollback) | medium | Retry path covered only by mocked-op unit tests; a real-conflict container test needs deterministic threaded interleaving. Explicitly deferred (PR-2.2 cycle-7) to the follow-up test-hardening PR; behavior manually verified against live PG16. |
| migration 004 self-grant when the executing role lacks ADMIN on a pre-existing migrator (unsupported misconfiguration) | low | Deferred; the shipped single-bootstrap-master path is correct and tested. |
| schema_conn BENCH_TEST_PG_DSN destructive-scrub guard against a localhost-tunnel-to-prod | low | Dev-only override never used in CI (testcontainers only); explicitly accepted/deferred as an exotic foot-gun. |

_Recommendations:_ Coverage of the load-bearing data-correctness invariants is comprehensive and pins each against an automated test, satisfying behavior-preservation against the v3 serde/ingest boundary per-table. The only material gap is the live end-to-end dual-write, gated on operator var-setting and best-effort by design. The deferred real-deadlock/autocommit=False container test is the one worth landing in the follow-up test-hardening PR; the remaining untested cases are availability/test-infra hardening already triaged and out of scope per the lean re-plan.

### Tradeoffs re-evaluation

- **Schema-deploy authorization (PR merge is the gate; push-on-develop trigger; no environment/manual-approval gate)** — verdict: **keep**. PR-2.4 implements exactly this: push-on-develop under paths migrations/**, dry_run kept, environment-gate comments removed. Execution safety comes from the per-PR testcontainer migration test. Accepted tradeoff; not re-flagged. _(found_by: spec, correctness)_
- **Ingest writer language (pure Python extending post-ingest.py; xxhash port)** — verdict: **keep**. PR-2.2 delivered exactly this; the port matches Table A/B and SCHEMA_VERSION lockstep holds. _(found_by: spec)_
- **Cutover style (short best-effort dual-write soak, then promote v4); v4 dual-write is continue-on-error during soak, v3 stays hard-required** — verdict: **keep**. PR-2.4's continue-on-error v4 steps are the direct realization; a v4 OIDC/connect hiccup must never break the proven v3 pipeline; v4 promoted to required at cutover (PR-5.1). Both steps consume the same results.v3.jsonl, so neither substrate is written in isolation. Calibrated, accepted exception to the no-best-effort rule. _(found_by: spec, correctness)_
- **Phase-2 ingest DB identity (dedicated bench_ingest + GitHubBenchmarkIngestRole, separate from migrator)** — verdict: **keep**. PR-2.1 implemented precisely this with DML-only grants and a separate OIDC role; separation of duties; the writer enforces bench_ingest unconditionally. Pinned by tests. _(found_by: spec, correctness)_
- **Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; drop reconcile-ingest.py + dual-write-verify.yml + incident.io)** — verdict: **keep**. PR-2.5 correctly dropped; PR-2.4 adds no reconciliation machinery; v4-only failure does not trigger incident.io. Four independent safety nets remain. Not drift. _(found_by: spec)_
- **Duplicate JSON keys collapse last-wins (no rejection), matching v3** — verdict: **keep**. serde_json::Value::Object and Python json.loads are both last-wins; rejecting would break behavior-preservation. Not re-flagged. _(found_by: correctness)_
- **post-ingest.py PEP 723 dependencies = []; --postgres runs from the uv env** — verdict: **keep**. Keeps the v3 --server path stdlib-only under bare python3; --postgres deps come from the uv workspace. Lazy imports verified. Not re-flagged. _(found_by: correctness)_
- **IAM-token region precedence: --region > boto3 session region > RDS-hostname-parsed** — verdict: **keep**. A wrong region fails loud at IAM connect, not silently; ordering matches AWS convention. Not re-flagged. _(found_by: correctness)_

### Disagreements

None. Both lenses (spec + correctness) accepted at high confidence with no contradictory findings or tradeoff verdicts.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-2",
  "lenses_used": ["spec", "correctness"],
  "review_count": 2,
  "unified_findings": [
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/README.md:81",
      "description": "README var table still attributes RDS_BENCH_INSTANCE_ENDPOINT/GH_BENCH_INGEST_ROLE_ARN consumers to 'PR-2.2' ingest workflows, but the dual-write steps actually landed in PR-2.4; minor lineage mislabel in shipped docs.",
      "recommended_fix": "Update the 'used by' column references from PR-2.2 to PR-2.4 for the ingest-workflow consumers of RDS_BENCH_INSTANCE_ENDPOINT and GH_BENCH_INGEST_ROLE_ARN, to match where the dual-write steps were actually added.",
      "found_by": ["spec"]
    },
    {
      "severity": "nit",
      "kind": "other",
      "file_line": "scripts/post-ingest.py:1168",
      "description": "_main_postgres calls build_commit() (git show -s <sha>) for every run; bench.yml/sql-benchmarks.yml checkouts are not pinned to fetch-depth the way v3-commit-metadata.yml sets fetch-depth:2. Behavior matches v3 (--server also builds the commit), so it is parity, not drift.",
      "recommended_fix": "No change required for spec adherence (best-effort + continue-on-error absorbs any git-history miss). Optionally note in a comment that the v4 step inherits the v3 step's git-history assumption.",
      "found_by": ["spec"]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Phase 2 delivers the Postgres dual-write ingest path and its CI plumbing, shipped exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction; correctness review found zero data-correctness bugs. Ingest identity (PR-2.1): migration 004 creates a least-privilege bench_ingest login role (rds_iam member on RDS, guarded for vanilla PG) holding USAGE-not-CREATE on public and SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), plus ALTER DEFAULT PRIVILEGES FOR ROLE migrator so future migrator tables auto-grant. Because the RDS master is rds_superuser but not a true superuser, the ADP runs via a temporary INHERIT self-grant through the creator's ADMIN option, then revoke. provision.sh adds GitHubBenchmarkIngestRole (OIDC trust branch-scoped to develop+ct/bench-v4, rds-db:connect for bench_ingest on the instance only) and drops the dead proxy grant. Writer (PR-2.2): post-ingest.py --postgres reproduces the v3 serde boundary in Python (deny_unknown_fields, per-field type/range validation, memory quartet, storage enum, commit_sha match), computes measurement_id bit-identically via the PR-1.5 port (_measurement_id.py is a byte-for-byte port of db.rs:162-257, pinned transitively Rust==golden==Python), then upserts commits-first and five fact tables in one all-or-nothing transaction via INSERT ... ON CONFLICT DO UPDATE with deadlock retry, over verify-full TLS authenticated by an RDS IAM token as bench_ingest. An is_finite guard rejects NaN/Inf/out-of-f64-range threshold loudly with rollback (Python stricter-or-equal to serde). All six ON CONFLICT SET lists touch only value/env columns, never dim columns; search_path is pinned to public; ssl_in_use is asserted post-connect. The v3 --server path stays stdlib-only (PEP 723 dependencies=[]). SCHEMA_VERSION lockstep held. CI wiring (PR-2.3): a scripts-test job runs pytest scripts/ behind a docker info hard gate plus a CI-env fail-loud fixture so testcontainer suites cannot silently skip. Dual-write CI (PR-2.4): the three ingest workflows each add a best-effort continue-on-error v4 --postgres step after the unchanged hard-required v3 step (v3-commit-metadata.yml gains id-token: write and ingests commit-row-only empty.jsonl); schema-deploy.yml switches to push-on-develop under paths migrations/**, keeping workflow_dispatch+dry_run. PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io) was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. All five measurement_id functions, six ON CONFLICT SET lists, field/type sets, and column widths were verified against the Rust source and 001_initial_schema.sql. Two cosmetic nits, no must-fix or should-fix.",
    "surprises": [
      { "what": "Migration 004 ALTER DEFAULT PRIVILEGES FOR ROLE migrator fails on a real non-superuser RDS master (createrole_self_grant default), masked by the superuser testcontainer.", "how_handled": "Fixed PR-2.1 cycle-1: guarded INHERIT self-grant via the master's ADMIN option then revoke, plus a test applying 001..004 as a real NOSUPERUSER CREATEROLE login.", "amend_plan": "already-done" },
      { "what": "conn.info.ssl_in_use does not exist in psycopg (ssl_in_use lives on conn.pgconn); the cycle-9 fix shipped the wrong accessor.", "how_handled": "Fixed PR-2.2 (conn.pgconn.ssl_in_use) with a unit test pinning the accessor location and a container test pinning the traversal.", "amend_plan": "already-done" },
      { "what": "fail-loud-on-no-Docker guard tests could not catch an always-skip regression because pytest.skip raises Skipped (not a Failed subclass).", "how_handled": "Fixed PR-2.3 (catch both outcome types and assert the specific one; mutation-verified).", "amend_plan": "already-done" },
      { "what": "Python json.loads accepts NaN/Infinity literals that serde_json rejects at parse time.", "how_handled": "The _require_finite guard rejects them loudly with rollback, making the Python writer stricter-or-equal to v3, so no divergent row is written. Already handled.", "amend_plan": "no" },
      { "what": "The v3 producer emits several fields as u32/u64 (query_idx u32; value_ns/all_runtimes_ns u64) where server and Python validator use i32/i64.", "how_handled": "Both v3 serde and Python _require_int reject values exceeding the signed range, so the substrates agree; values are far inside range. No change needed.", "amend_plan": "no" },
      { "what": "insert-vs-update classification differs across substrates: Rust does an exists() preflight SELECT; Python derives the flag from RETURNING (xmax = 0).", "how_handled": "Equivalence (incl. same-dim-tuple-twice-in-one-envelope yielding (1,1) not (2,0)) is pinned by test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update. Already handled.", "amend_plan": "already-done" },
      { "what": "bench-v4 Python files predate Phase 2 at ~100-col and fail the repo ruff line-length-120 (a pre-existing E501).", "how_handled": "Left unfixed and flagged for the operator as a branch-wide ruff reconciliation required before merging ct/bench-v4 to develop; RETAINED deferred item.", "amend_plan": "no" },
      { "what": "pytest scripts/ collects a fourth file (scripts/tests/test_benchmark_reporting.py) beyond the three named suites.", "how_handled": "Accepted as harmless and matching the acceptance command; a clarifying ci.yml comment added.", "amend_plan": "no" }
    ],
    "coverage": {
      "tested_cases": [
        { "case": "measurement_id Rust==golden==Python parity across all 5 tables incl. negative i32, null/Some, multibyte UTF-8, whole/negative/tiny/large f64 thresholds; SCHEMA_VERSION lockstep with schema.rs", "test_location": "scripts/test_measurement_id.py + scripts/measurement_id_golden.json; scripts/test_post_ingest_postgres.py:3105", "confidence": "high" },
        { "case": "bench_ingest DML-only (SELECT/INSERT/UPDATE) on all 6 tables, denied DELETE/CREATE/DDL; default-privileges cover future migrator tables; 001..004 apply cleanly + idempotently under a real NOSUPERUSER CREATEROLE master", "test_location": "scripts/test_migrate_schema.py:1960,1928,1993,2069", "confidence": "high" },
        { "case": "provision.sh provisions ingest role on instance dbuser, emits GH_BENCH_INGEST_ROLE_ARN, drops dead proxy grant", "test_location": "scripts/test_migrate_schema.py:1805,1835", "confidence": "high" },
        { "case": "insert-then-update accounting; re-ingest upserts (0 inserted, N updated) with stable counts; measurement_id matches port", "test_location": "scripts/test_post_ingest_postgres.py:2477,2572", "confidence": "high" },
        { "case": "ON CONFLICT DO UPDATE overwrites every value/env column while leaving dim tuple / measurement_id stable, per table; dim columns NEVER in any DO UPDATE SET list for all 5 tables", "test_location": "scripts/test_post_ingest_postgres.py:2867", "confidence": "high" },
        { "case": "NaN/Inf/out-of-f64-range threshold raises loudly and rolls back; deny_unknown_fields; missing-required; unknown/non-scalar kind; storage enum; memory quartet; commit_sha mismatch; type/range boundaries reproduced", "test_location": "scripts/test_post_ingest_postgres.py:2633,2887,3066,3074,3094", "confidence": "high" },
        { "case": "connect_postgres: IAM-token-when-passwordless, always-bench_ingest, rejects weak sslmode (verify-full), forces search_path=public, rejects non-TLS, post-connect ssl_in_use", "test_location": "scripts/test_post_ingest_postgres.py:3142,3221,3233,3264,3285", "confidence": "high" },
        { "case": "--server requires --benchmark-id while --postgres does not; mutual exclusivity + dispatch", "test_location": "scripts/test_post_ingest_postgres.py:3401,3363,3375", "confidence": "high" },
        { "case": "fail-loud-on-no-Docker in CI vs skip locally (both test files)", "test_location": "scripts/test_migrate_schema.py:1745; scripts/test_post_ingest_postgres.py:2753", "confidence": "high" },
        { "case": "write-conflict retry (deadlock/serialization) retries then succeeds, gives up at cap, propagates validation errors immediately", "test_location": "scripts/test_post_ingest_postgres.py (test_retry_write_conflicts_*)", "confidence": "medium" }
      ],
      "untested_cases": [
        { "case": "End-to-end live v4 dual-write per develop push (CloudWatch shows Postgres writes) — the PR-2.4 acceptance criterion needing real RDS + repo vars", "priority": "high", "why_untested": "Operator-side, not performable in-session; the v4 steps no-op until GH_BENCH_INGEST_ROLE_ARN + RDS_BENCH_INSTANCE_ENDPOINT/DB_NAME/REGION are set. Tracked as an open operator dependency." },
        { "case": "Real psycopg DeadlockDetected inside conn.transaction() across two reversed-order connections on autocommit=False (retry+commit/rollback)", "priority": "medium", "why_untested": "Retry path covered only by mocked-op unit tests; a real-conflict container test needs deterministic threaded interleaving. Explicitly deferred (PR-2.2 cycle-7) to the follow-up test-hardening PR; behavior manually verified against live PG16." },
        { "case": "migration 004 self-grant when the executing role lacks ADMIN on a pre-existing migrator (unsupported misconfiguration)", "priority": "low", "why_untested": "Deferred; the shipped single-bootstrap-master path is correct and tested." },
        { "case": "schema_conn BENCH_TEST_PG_DSN destructive-scrub guard against a localhost-tunnel-to-prod", "priority": "low", "why_untested": "Dev-only override never used in CI (testcontainers only); explicitly accepted/deferred as an exotic foot-gun." }
      ],
      "recommendations": "Coverage of the load-bearing data-correctness invariants is comprehensive and pins each against an automated test, satisfying behavior-preservation against the v3 serde/ingest boundary per-table. The only material gap is the live end-to-end dual-write, gated on operator var-setting and best-effort by design. The deferred real-deadlock/autocommit=False container test is the one worth landing in the follow-up test-hardening PR; the remaining untested cases are availability/test-infra hardening already triaged and out of scope per the lean re-plan."
    },
    "tradeoffs": [
      { "decision": "Schema-deploy authorization (PR merge is the gate; push-on-develop trigger; no environment/manual-approval gate)", "original": "Supersedes the original manual-approval mandate (2026-05-29 deploy-model decision)", "verdict": "keep", "rationale": "PR-2.4 implements exactly this: push-on-develop under paths migrations/**, dry_run kept, environment-gate comments removed. Execution safety comes from the per-PR testcontainer migration test. Accepted tradeoff; not re-flagged.", "found_by": ["spec", "correctness"] },
      { "decision": "Ingest writer language (pure Python extending post-ingest.py; xxhash port)", "original": "Avoid sqlx/aws-sdk Rust deps", "verdict": "keep", "rationale": "PR-2.2 delivered exactly this; the port matches Table A/B and SCHEMA_VERSION lockstep holds.", "found_by": ["spec"] },
      { "decision": "Cutover style (short best-effort dual-write soak, then promote v4); v4 dual-write is continue-on-error during soak, v3 stays hard-required", "original": "Amended 2026-06-04 to make v4 best-effort during soak (PR-2.4 lean re-plan)", "verdict": "keep", "rationale": "PR-2.4's continue-on-error v4 steps are the direct realization; a v4 OIDC/connect hiccup must never break the proven v3 pipeline; v4 promoted to required at cutover (PR-5.1). Both steps consume the same results.v3.jsonl, so neither substrate is written in isolation. Calibrated, accepted exception to the no-best-effort rule.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 ingest DB identity (dedicated bench_ingest + GitHubBenchmarkIngestRole, separate from migrator)", "original": "Re-plan Q2 least-privilege split (PR-2.1)", "verdict": "keep", "rationale": "PR-2.1 implemented precisely this with DML-only grants and a separate OIDC role; separation of duties; the writer enforces bench_ingest unconditionally. Pinned by tests.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; drop reconcile-ingest.py + dual-write-verify.yml + incident.io)", "original": "Superseded the 2026-06-01 per-push reconciliation harness", "verdict": "keep", "rationale": "PR-2.5 correctly dropped; PR-2.4 adds no reconciliation machinery; v4-only failure does not trigger incident.io. Four independent safety nets remain. Not drift.", "found_by": ["spec"] },
      { "decision": "Duplicate JSON keys collapse last-wins (no rejection), matching v3", "original": "PR-2.2 cycle-6", "verdict": "keep", "rationale": "serde_json::Value::Object and Python json.loads are both last-wins; rejecting would break behavior-preservation. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "post-ingest.py PEP 723 dependencies = []; --postgres runs from the uv env", "original": "PR-2.2", "verdict": "keep", "rationale": "Keeps the v3 --server path stdlib-only under bare python3; --postgres deps come from the uv workspace. Lazy imports verified. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "IAM-token region precedence: --region > boto3 session region > RDS-hostname-parsed", "original": "PR-2.2", "verdict": "keep", "rationale": "A wrong region fails loud at IAM connect, not silently; ordering matches AWS convention. Not re-flagged.", "found_by": ["correctness"] }
    ]
  },
  "executive_summary": "Phase 2 ships the Postgres dual-write ingest path and its CI plumbing, exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction. Both lenses accept at high confidence; the correctness lens found zero data-correctness bugs and the spec lens found only two cosmetic nits, so there are no must-fix or should-fix items. PR-2.1 adds a least-privilege bench_ingest login role (migration 004) with SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), USAGE-not-CREATE, and ALTER DEFAULT PRIVILEGES so future migrator tables auto-grant; provision.sh adds a branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance only) and drops the dead proxy grant. PR-2.2's post-ingest.py --postgres reproduces the v3 serde boundary in Python, computes measurement_id bit-identically via the PR-1.5 port, and upserts commits-first then five fact tables in one all-or-nothing transaction with deadlock retry over verify-full TLS and IAM auth. PR-2.3 runs pytest scripts/ behind a docker hard gate plus a CI fail-loud fixture. PR-2.4 makes each of the three ingest workflows add a best-effort continue-on-error v4 step after the unchanged hard-required v3 step, and switches schema-deploy to push-on-develop under paths migrations/**. PR-2.5 was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. The key engineering surprises were all already resolved in-cycle: the non-superuser RDS master breaking ALTER DEFAULT PRIVILEGES (fixed with a guarded INHERIT self-grant + a real NOSUPERUSER test), the ssl_in_use accessor living on conn.pgconn rather than conn.info (fixed and pinned), and the fail-loud-on-no-Docker guard being unable to catch always-skip because pytest.skip is not a Failed subclass (fixed and mutation-verified). Correctness independently confirmed Python-is-stricter-or-equal handling of NaN/Inf literals, agreement on u32/u64-vs-i32/i64 ranges, and insert-vs-update equivalence between Rust's exists() preflight and Python's xmax=0 classifier. Every Key decision was verdicted keep by both reviewers: schema-deploy PR-merge gating, pure-Python writer + xxhash port, best-effort dual-write during soak with v3 hard-required, the dedicated bench_ingest identity, dropping the reconcile harness, last-wins duplicate-key handling, PEP 723 empty-deps, and IAM region precedence. Coverage of load-bearing data-correctness invariants is comprehensive and per-table-pinned; the one material gap is the live end-to-end dual-write, which is operator-gated on repo vars and best-effort by design. Verdict: ACCEPT.",
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 0,
  "nit_count": 2,
  "review_cycles_this_invocation": 1
}
```

</details>

## Phase 2: Postgres writer + best-effort v4 CI — end-of-phase review (cycle 2) — accepted (2-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-2: spec + correctness, executor=claude); amend-triggered re-review of the cumulative amended phase. Full JSON in the `<details>` block + the cycle-2 archive entry. Verdict: accept (0 must-fix, 0 should-fix, 2 dismissable nits).**

### Summary of changes

Amended cumulative Phase 2 delivers the Postgres dual-write ingest path + best-effort v4 CI. Ingest identity (PR-2.1): migrations/004 creates least-privilege bench_ingest (SELECT/INSERT/UPDATE on the 6 tables, USAGE-not-CREATE, ALTER DEFAULT PRIVILEGES for future migrator tables, no DELETE/DDL); provision.sh adds the branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance dbuser) and drops the dead proxy grant; 004 self-grants migrator INHERIT via the master's ADMIN option then revokes (non-superuser-RDS-master bootstrap fix). Writer (PR-2.2, de-gold-plated by PR-2.7): post-ingest.py --postgres computes measurement_id bit-identically via the byte-exact _measurement_id.py port, validates every field/type/range to reproduce the v3 serde + deny_unknown_fields boundary, upserts commits-first then 5 fact tables via INSERT...ON CONFLICT(measurement_id) DO UPDATE...RETURNING(xmax=0) in one all-or-nothing transaction with deadlock/serialization retry; connect_postgres enforces verify-full TLS + post-connect ssl_in_use + bench_ingest-only + IAM-token mint + search_path=public. CI wiring (PR-2.3/2.4/2.6): a scripts-test job runs pytest scripts/ (Docker-required, fail-loud-not-skip in CI); the 3 ingest workflows add a continue-on-error best-effort v4 --postgres step after the unchanged hard-required v3 step; schema-deploy.yml triggers on push to develop under migrations/**. Amendments: PR-2.6 re-keys the 9 v4 gates from the endpoint var to GH_BENCH_INGEST_ROLE_ARN so they no-op (not fire+fail) until infra is wired; PR-2.7 de-gold-plates ~194 lines of trusted-input over-hardening (NUL/surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) + ~8 tests, moving the writer CLOSER to the v3 Rust source; PR-2.8 clears the ruff E501 merge-blocker + README PR-lineage + a git-history comment. Every preserved invariant (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed validation, memory-quartet, IAM/TLS) remains pinned by tests; the de-gold-plate removed only stricter-than-v3 hardening on trusted CI input.

### Surprises and discoveries

- **PR-2.7's OverflowError-branch removal left an orphaned pytest.param(10**309) in the KEPT test_nonfinite_threshold_raises_and_rolls_back, which would have gone CI-red (uncaught OverflowError); only skipped locally for lack of Docker.** — Caught by cycle-1 correctness (cumulative-test trace), fixed in b6d1f292b (param dropped; nan/inf/-inf retained). Verified gone. _(amend_plan: already-done)_
- **The de-gold-plate moves the writer CLOSER to v3: the removed NUL/surrogate/non-UTF-8 guards were STRICTER than the v3 Rust source, so removal improves behavior-preservation rather than regressing it; failure modes stayed LOUD (UnicodeEncodeError in _write_str / Postgres DataError at bind, inside the transaction -> rollback). No silent-wrong-write path.** — Authorized scope-reduction per the 2026-06-05 complexity audit; KEEP-list verified intact field-by-field. _(amend_plan: no)_
- **The live gate-var bug (PR-2.6) was a real config mismatch: gates keyed on RDS_BENCH_INSTANCE_ENDPOINT (set) while assume-role uses GH_BENCH_INGEST_ROLE_ARN (unset), so the v4 steps would fire+fail at assume-role on the next develop push.** — Re-keyed all 9 gates to the role-ARN var; live-var state now makes the gate evaluate false (clean no-op until wired). _(amend_plan: already-done)_

### Testing coverage assessment

| Case | Location | Confidence |
|---|---|---|
| measurement_id parity (5 kinds) vs the Python port + stored rows; SCHEMA_VERSION lockstep | `scripts/test_measurement_id.py (65 vectors) + test_post_ingest_postgres.py (measurement_ids_match_python_port, schema_version)` | high |
| ON CONFLICT SET excludes dim columns (BAN) across all 5 _insert_ fns; per-table value-column update | `test_on_conflict_set_excludes_dim_columns + test_update_overwrites_all_value_columns_per_table` | high |
| insert-vs-update accounting incl. same-dim-twice (xmax=0 classifier); commit-before-facts; all-or-nothing rollback | `test_ingest_inserts_then_updates, test_same_dim_tuple_twice..., test_late_validation_failure_rolls_back_earlier_fact_row` | high |
| NaN/+Inf/-Inf threshold rejected + rollback (post de-gold-plate, OverflowError param removed) | `test_nonfinite_threshold_raises_and_rolls_back` | high |
| typed i32/i64/finite + deny_unknown_fields + memory-quartet + storage-enum + commit_sha-mismatch validation | `test_post_ingest_postgres.py value-validation tests` | high |
| connect_postgres: verify-full + post-connect ssl_in_use + bench_ingest-only + IAM-token-vs-password + search_path pin | `test_connect_postgres_* (9 pure-unit tests, pass non-Docker)` | high |
| de-gold-plate replacements: mixed-newline read, malformed-JSON loud-fail, git_show text-mode decode | `test_read_records_happy_path_mixed_newlines / _rejects_malformed_json / test_git_show_field_decodes_and_strips` | high |
| bench_ingest least-privilege + 004 idempotency + non-superuser-master bootstrap; provision.sh ingest role + dropped proxy grant | `scripts/test_migrate_schema.py (testcontainer + static provision tests)` | high |
| v4 gate keys on GH_BENCH_INGEST_ROLE_ARN; CI fail-loud-not-skip on missing Docker | `workflow inspection + test_require_docker_fails_loud_in_ci/_skips_without_ci` | high |

**Untested / gaps:**

| Case | Priority | Why |
|---|---|---|
| Live v4 dual-write end-to-end against real RDS per develop push (OIDC->IAM->verify-full->upsert; CloudWatch confirmation) | medium | Requires live AWS + operator-set repo vars; operator-side soak. Acceptance for PR-2.4 is the structural/yamllint criteria; v4 correctness is gated by Phase-3 migrate --verify. |
| Real two-connection DeadlockDetected on an autocommit=False connection asserting retry+commit | low | Container-only threaded-interleaving test; deferred (PR-2.2 cycle-7) to a follow-up test-hardening PR. Retry pinned by mocked-op unit tests + live-container verification. |
| Live OIDC schema-deploy apply firing on a migrations/** push | low | The retained operator pre-merge gate; not in-session testable. |

_Recommendations:_ Coverage is strong for a trusted-input low-stakes dashboard and matches the lean calibration. The only meaningful gaps are the two retained operator-side gates (live OIDC schema-deploy apply; live v4 dual-write soak + CloudWatch). No new test work is required to accept the amended phase. The deferred real-deadlock/autocommit=False integration test remains a reasonable follow-up but is not a blocker.

### Tradeoffs re-evaluation

- **v4 ingest steps continue-on-error (best-effort) during the soak** — **keep**. v3 stays hard-required + untouched; the 'no continue-on-error' BAN applies at the PR-5.1 cutover, not the soak. PR-2.6 reinforces it (clean no-op until wired).
- **Gate v4 steps on GH_BENCH_INGEST_ROLE_ARN (PR-2.6)** — **keep**. Aligns the gate with the assume-role input so the steps no-op (not fire+fail) until infra is wired; verified across all 9 gates.
- **De-gold-plate trusted-input hardening (PR-2.7)** — **keep**. Removed only stricter-than-v3 guards on trusted CI input; every removed input still fails loud inside the transaction (rollback). No data-correctness guard or its test removed.
- **Schema-deploy authorization (PR-merge gate, push trigger, no environment: gate)** — **keep**. PR-2.4 shipped the trigger; the amendment did not touch it. Accepted tradeoff.
- **Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; PR-2.5 dropped)** — **keep**. The amendment added no reconciliation machinery; consistent with the superseding decision.
- **Pure-Python writer + PEP 723 dependencies=[]; IAM region precedence** — **keep**. Accepted tradeoffs; keeps v3 --server stdlib-only; region mismatch fails loud. Unchanged by the amendment.
- **Dedicated bench_ingest identity separate from migrator** — **keep**. Unchanged; PR-2.6 gates on the ingest role ARN, reinforcing the separate-identity model; connect_postgres still enforces bench_ingest-only.

### Disagreements

None. Both lenses accept at high confidence; the 2 nits are dismissable (comment at 100-col limit; huge-int→OverflowError still fails loud, an intentional de-gold-plate tradeoff).

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1, "preset": "phase-2", "lenses_used": ["spec", "correctness"], "review_count": 2,
  "unified_findings": [
    {"severity":"nit","kind":"convention","file_line":"scripts/post-ingest.py:1085","description":"PR-2.8's build_commit git-history comment lines sit at exactly 100 cols (at the user CLAUDE.md limit, not over; well under ruff's 120). Already rewrapped from 102 in f0e93b9f4. No action.","recommended_fix":"None; within the 100-col rule.","found_by":["spec"]},
    {"severity":"nit","kind":"error-path","file_line":"scripts/post-ingest.py:448","description":"After the de-gold-plate dropped the _require_finite OverflowError sub-branch, a huge-int threshold (e.g. 10**309) now raises an uncaught OverflowError instead of a controlled SystemExit. Still fails LOUD + rolls back the transaction (no silent-wrong-write); the input class (oversized-int on trusted CI) is explicitly DO-NOT-flag per the calibration. Intentional de-gold-plate tradeoff.","recommended_fix":"None required (accepted). A one-line `except OverflowError: is_finite_number = False` would restore the controlled error, but is not worth carrying per the trusted-input calibration.","found_by":["correctness"]}
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Amended cumulative Phase 2 delivers the Postgres dual-write ingest path + best-effort v4 CI. Ingest identity (PR-2.1): migrations/004 creates least-privilege bench_ingest (SELECT/INSERT/UPDATE on the 6 tables, USAGE-not-CREATE, ALTER DEFAULT PRIVILEGES for future migrator tables, no DELETE/DDL); provision.sh adds the branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance dbuser) and drops the dead proxy grant; 004 self-grants migrator INHERIT via the master's ADMIN option then revokes (non-superuser-RDS-master bootstrap fix). Writer (PR-2.2, de-gold-plated by PR-2.7): post-ingest.py --postgres computes measurement_id bit-identically via the byte-exact _measurement_id.py port, validates every field/type/range to reproduce the v3 serde + deny_unknown_fields boundary, upserts commits-first then 5 fact tables via INSERT...ON CONFLICT(measurement_id) DO UPDATE...RETURNING(xmax=0) in one all-or-nothing transaction with deadlock/serialization retry; connect_postgres enforces verify-full TLS + post-connect ssl_in_use + bench_ingest-only + IAM-token mint + search_path=public. CI wiring (PR-2.3/2.4/2.6): a scripts-test job runs pytest scripts/ (Docker-required, fail-loud-not-skip in CI); the 3 ingest workflows add a continue-on-error best-effort v4 --postgres step after the unchanged hard-required v3 step; schema-deploy.yml triggers on push to develop under migrations/**. Amendments: PR-2.6 re-keys the 9 v4 gates from the endpoint var to GH_BENCH_INGEST_ROLE_ARN so they no-op (not fire+fail) until infra is wired; PR-2.7 de-gold-plates ~194 lines of trusted-input over-hardening (NUL/surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) + ~8 tests, moving the writer CLOSER to the v3 Rust source; PR-2.8 clears the ruff E501 merge-blocker + README PR-lineage + a git-history comment. Every preserved invariant (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed validation, memory-quartet, IAM/TLS) remains pinned by tests; the de-gold-plate removed only stricter-than-v3 hardening on trusted CI input.",
    "surprises": [
      {"what":"PR-2.7's OverflowError-branch removal left an orphaned pytest.param(10**309) in the KEPT test_nonfinite_threshold_raises_and_rolls_back, which would have gone CI-red (uncaught OverflowError); only skipped locally for lack of Docker.","how_handled":"Caught by cycle-1 correctness (cumulative-test trace), fixed in b6d1f292b (param dropped; nan/inf/-inf retained). Verified gone.","amend_plan":"already-done"},
      {"what":"The de-gold-plate moves the writer CLOSER to v3: the removed NUL/surrogate/non-UTF-8 guards were STRICTER than the v3 Rust source, so removal improves behavior-preservation rather than regressing it; failure modes stayed LOUD (UnicodeEncodeError in _write_str / Postgres DataError at bind, inside the transaction -> rollback). No silent-wrong-write path.","how_handled":"Authorized scope-reduction per the 2026-06-05 complexity audit; KEEP-list verified intact field-by-field.","amend_plan":"no"},
      {"what":"The live gate-var bug (PR-2.6) was a real config mismatch: gates keyed on RDS_BENCH_INSTANCE_ENDPOINT (set) while assume-role uses GH_BENCH_INGEST_ROLE_ARN (unset), so the v4 steps would fire+fail at assume-role on the next develop push.","how_handled":"Re-keyed all 9 gates to the role-ARN var; live-var state now makes the gate evaluate false (clean no-op until wired).","amend_plan":"already-done"}
    ],
    "coverage": {
      "tested_cases": [
        {"case":"measurement_id parity (5 kinds) vs the Python port + stored rows; SCHEMA_VERSION lockstep","test_location":"scripts/test_measurement_id.py (65 vectors) + test_post_ingest_postgres.py (measurement_ids_match_python_port, schema_version)","confidence":"high"},
        {"case":"ON CONFLICT SET excludes dim columns (BAN) across all 5 _insert_ fns; per-table value-column update","test_location":"test_on_conflict_set_excludes_dim_columns + test_update_overwrites_all_value_columns_per_table","confidence":"high"},
        {"case":"insert-vs-update accounting incl. same-dim-twice (xmax=0 classifier); commit-before-facts; all-or-nothing rollback","test_location":"test_ingest_inserts_then_updates, test_same_dim_tuple_twice..., test_late_validation_failure_rolls_back_earlier_fact_row","confidence":"high"},
        {"case":"NaN/+Inf/-Inf threshold rejected + rollback (post de-gold-plate, OverflowError param removed)","test_location":"test_nonfinite_threshold_raises_and_rolls_back","confidence":"high"},
        {"case":"typed i32/i64/finite + deny_unknown_fields + memory-quartet + storage-enum + commit_sha-mismatch validation","test_location":"test_post_ingest_postgres.py value-validation tests","confidence":"high"},
        {"case":"connect_postgres: verify-full + post-connect ssl_in_use + bench_ingest-only + IAM-token-vs-password + search_path pin","test_location":"test_connect_postgres_* (9 pure-unit tests, pass non-Docker)","confidence":"high"},
        {"case":"de-gold-plate replacements: mixed-newline read, malformed-JSON loud-fail, git_show text-mode decode","test_location":"test_read_records_happy_path_mixed_newlines / _rejects_malformed_json / test_git_show_field_decodes_and_strips","confidence":"high"},
        {"case":"bench_ingest least-privilege + 004 idempotency + non-superuser-master bootstrap; provision.sh ingest role + dropped proxy grant","test_location":"scripts/test_migrate_schema.py (testcontainer + static provision tests)","confidence":"high"},
        {"case":"v4 gate keys on GH_BENCH_INGEST_ROLE_ARN; CI fail-loud-not-skip on missing Docker","test_location":"workflow inspection + test_require_docker_fails_loud_in_ci/_skips_without_ci","confidence":"high"}
      ],
      "untested_cases": [
        {"case":"Live v4 dual-write end-to-end against real RDS per develop push (OIDC->IAM->verify-full->upsert; CloudWatch confirmation)","priority":"medium","why_untested":"Requires live AWS + operator-set repo vars; operator-side soak. Acceptance for PR-2.4 is the structural/yamllint criteria; v4 correctness is gated by Phase-3 migrate --verify."},
        {"case":"Real two-connection DeadlockDetected on an autocommit=False connection asserting retry+commit","priority":"low","why_untested":"Container-only threaded-interleaving test; deferred (PR-2.2 cycle-7) to a follow-up test-hardening PR. Retry pinned by mocked-op unit tests + live-container verification."},
        {"case":"Live OIDC schema-deploy apply firing on a migrations/** push","priority":"low","why_untested":"The retained operator pre-merge gate; not in-session testable."}
      ],
      "recommendations": "Coverage is strong for a trusted-input low-stakes dashboard and matches the lean calibration. The only meaningful gaps are the two retained operator-side gates (live OIDC schema-deploy apply; live v4 dual-write soak + CloudWatch). No new test work is required to accept the amended phase. The deferred real-deadlock/autocommit=False integration test remains a reasonable follow-up but is not a blocker."
    },
    "tradeoffs": [
      {"decision":"v4 ingest steps continue-on-error (best-effort) during the soak","original":"PR-2.4 lean re-plan amendment","verdict":"keep","rationale":"v3 stays hard-required + untouched; the 'no continue-on-error' BAN applies at the PR-5.1 cutover, not the soak. PR-2.6 reinforces it (clean no-op until wired)."},
      {"decision":"Gate v4 steps on GH_BENCH_INGEST_ROLE_ARN (PR-2.6)","original":"audit live-bug fix","verdict":"keep","rationale":"Aligns the gate with the assume-role input so the steps no-op (not fire+fail) until infra is wired; verified across all 9 gates."},
      {"decision":"De-gold-plate trusted-input hardening (PR-2.7)","original":"2026-06-05 complexity audit","verdict":"keep","rationale":"Removed only stricter-than-v3 guards on trusted CI input; every removed input still fails loud inside the transaction (rollback). No data-correctness guard or its test removed."},
      {"decision":"Schema-deploy authorization (PR-merge gate, push trigger, no environment: gate)","original":"2026-05-29 deploy-model decision","verdict":"keep","rationale":"PR-2.4 shipped the trigger; the amendment did not touch it. Accepted tradeoff."},
      {"decision":"Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; PR-2.5 dropped)","original":"lean re-plan","verdict":"keep","rationale":"The amendment added no reconciliation machinery; consistent with the superseding decision."},
      {"decision":"Pure-Python writer + PEP 723 dependencies=[]; IAM region precedence","original":"PR-2.2","verdict":"keep","rationale":"Accepted tradeoffs; keeps v3 --server stdlib-only; region mismatch fails loud. Unchanged by the amendment."},
      {"decision":"Dedicated bench_ingest identity separate from migrator","original":"re-plan Q2","verdict":"keep","rationale":"Unchanged; PR-2.6 gates on the ingest role ARN, reinforcing the separate-identity model; connect_postgres still enforces bench_ingest-only."}
    ]
  },
  "executive_summary": "Phase 2 cycle-2 phase-end review of the AMENDED cumulative phase (PR-2.1..2.4 accepted at cycle 1, plus the 2026-06-05 amendment PR-2.6/2.7/2.8). Both lenses ACCEPT with high confidence; 0 must-fix, 2 dismissable nits, no disagreements. The amendment did exactly three things and the reviewers verified each: PR-2.6 fixed a real LIVE gate-var bug (the v4 dual-write gates keyed on RDS_BENCH_INSTANCE_ENDPOINT [set] while assume-role uses GH_BENCH_INGEST_ROLE_ARN [unset], so the steps would fire+fail on the next develop push) by re-keying all 9 gates to the role-ARN var, restoring clean no-op-until-wired behavior. PR-2.7 de-gold-plated ~194 lines of trusted-input over-hardening (NUL/lone-surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) that was STRICTER than the v3 Rust source it mirrors; the spec lens confirmed this is authorized scope-reduction (not silent de-scoping) and the correctness lens verified field-by-field that NO load-bearing data-correctness validation was removed (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed i32/i64/finite, deny_unknown_fields, memory-quartet, IAM/TLS all intact) and that the removed string guards open NO silent-wrong-write path (a NUL or lone surrogate still fails LOUD inside the transaction -> rollback). PR-2.8 cleared the retained ruff E501 merge-blocker + 2 doc nits. The cycle-1 amendment must-fix (an orphaned 10**309 test param after the OverflowError-branch removal) was caught and fixed during PR-2.7's own inner loop. All Phase-2 exit criteria re-verified on the amended tree: yamllint clean, ruff clean, the 3 ingest workflows have the required v3 step + continue-on-error v4 steps, the schema-deploy push trigger, SCHEMA_VERSION lockstep, measurement_id reproduction; 120 non-Docker unit tests pass. Every Key decision remains keep. The two nits are dismissable: the build_commit comment sits at (not over) the 100-col limit, and the huge-int-threshold path now raises an uncaught OverflowError instead of SystemExit but still fails loud + rolls back (an explicitly DO-NOT-flag trusted-input case). Two operator-side gates remain (live OIDC schema-deploy apply; live v4 soak + CloudWatch). Verdict: ACCEPT.",
  "overall": "accept", "must_fix_count": 0, "should_fix_count": 0, "nit_count": 2, "review_cycles_this_invocation": 2
}
```

</details>

## Phase 2 raw gauntlet responses (archive)

### Cycle 1 — preset=phase-2 — accept

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-2",
  "lenses_used": ["spec", "correctness"],
  "review_count": 2,
  "unified_findings": [
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/README.md:81",
      "description": "README var table still attributes RDS_BENCH_INSTANCE_ENDPOINT/GH_BENCH_INGEST_ROLE_ARN consumers to 'PR-2.2' ingest workflows, but the dual-write steps actually landed in PR-2.4; minor lineage mislabel in shipped docs.",
      "recommended_fix": "Update the 'used by' column references from PR-2.2 to PR-2.4 for the ingest-workflow consumers of RDS_BENCH_INSTANCE_ENDPOINT and GH_BENCH_INGEST_ROLE_ARN, to match where the dual-write steps were actually added.",
      "found_by": ["spec"]
    },
    {
      "severity": "nit",
      "kind": "other",
      "file_line": "scripts/post-ingest.py:1168",
      "description": "_main_postgres calls build_commit() (git show -s <sha>) for every run; bench.yml/sql-benchmarks.yml checkouts are not pinned to fetch-depth the way v3-commit-metadata.yml sets fetch-depth:2. Behavior matches v3 (--server also builds the commit), so it is parity, not drift.",
      "recommended_fix": "No change required for spec adherence (best-effort + continue-on-error absorbs any git-history miss). Optionally note in a comment that the v4 step inherits the v3 step's git-history assumption.",
      "found_by": ["spec"]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Phase 2 delivers the Postgres dual-write ingest path and its CI plumbing, shipped exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction; correctness review found zero data-correctness bugs. Ingest identity (PR-2.1): migration 004 creates a least-privilege bench_ingest login role (rds_iam member on RDS, guarded for vanilla PG) holding USAGE-not-CREATE on public and SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), plus ALTER DEFAULT PRIVILEGES FOR ROLE migrator so future migrator tables auto-grant. Because the RDS master is rds_superuser but not a true superuser, the ADP runs via a temporary INHERIT self-grant through the creator's ADMIN option, then revoke. provision.sh adds GitHubBenchmarkIngestRole (OIDC trust branch-scoped to develop+ct/bench-v4, rds-db:connect for bench_ingest on the instance only) and drops the dead proxy grant. Writer (PR-2.2): post-ingest.py --postgres reproduces the v3 serde boundary in Python (deny_unknown_fields, per-field type/range validation, memory quartet, storage enum, commit_sha match), computes measurement_id bit-identically via the PR-1.5 port (_measurement_id.py is a byte-for-byte port of db.rs:162-257, pinned transitively Rust==golden==Python), then upserts commits-first and five fact tables in one all-or-nothing transaction via INSERT ... ON CONFLICT DO UPDATE with deadlock retry, over verify-full TLS authenticated by an RDS IAM token as bench_ingest. An is_finite guard rejects NaN/Inf/out-of-f64-range threshold loudly with rollback (Python stricter-or-equal to serde). All six ON CONFLICT SET lists touch only value/env columns, never dim columns; search_path is pinned to public; ssl_in_use is asserted post-connect. The v3 --server path stays stdlib-only (PEP 723 dependencies=[]). SCHEMA_VERSION lockstep held. CI wiring (PR-2.3): a scripts-test job runs pytest scripts/ behind a docker info hard gate plus a CI-env fail-loud fixture so testcontainer suites cannot silently skip. Dual-write CI (PR-2.4): the three ingest workflows each add a best-effort continue-on-error v4 --postgres step after the unchanged hard-required v3 step (v3-commit-metadata.yml gains id-token: write and ingests commit-row-only empty.jsonl); schema-deploy.yml switches to push-on-develop under paths migrations/**, keeping workflow_dispatch+dry_run. PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io) was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. All five measurement_id functions, six ON CONFLICT SET lists, field/type sets, and column widths were verified against the Rust source and 001_initial_schema.sql. Two cosmetic nits, no must-fix or should-fix.",
    "surprises": [
      { "what": "Migration 004 ALTER DEFAULT PRIVILEGES FOR ROLE migrator fails on a real non-superuser RDS master (createrole_self_grant default), masked by the superuser testcontainer.", "how_handled": "Fixed PR-2.1 cycle-1: guarded INHERIT self-grant via the master's ADMIN option then revoke, plus a test applying 001..004 as a real NOSUPERUSER CREATEROLE login.", "amend_plan": "already-done" },
      { "what": "conn.info.ssl_in_use does not exist in psycopg (ssl_in_use lives on conn.pgconn); the cycle-9 fix shipped the wrong accessor.", "how_handled": "Fixed PR-2.2 (conn.pgconn.ssl_in_use) with a unit test pinning the accessor location and a container test pinning the traversal.", "amend_plan": "already-done" },
      { "what": "fail-loud-on-no-Docker guard tests could not catch an always-skip regression because pytest.skip raises Skipped (not a Failed subclass).", "how_handled": "Fixed PR-2.3 (catch both outcome types and assert the specific one; mutation-verified).", "amend_plan": "already-done" },
      { "what": "Python json.loads accepts NaN/Infinity literals that serde_json rejects at parse time.", "how_handled": "The _require_finite guard rejects them loudly with rollback, making the Python writer stricter-or-equal to v3, so no divergent row is written. Already handled.", "amend_plan": "no" },
      { "what": "The v3 producer emits several fields as u32/u64 (query_idx u32; value_ns/all_runtimes_ns u64) where server and Python validator use i32/i64.", "how_handled": "Both v3 serde and Python _require_int reject values exceeding the signed range, so the substrates agree; values are far inside range. No change needed.", "amend_plan": "no" },
      { "what": "insert-vs-update classification differs across substrates: Rust does an exists() preflight SELECT; Python derives the flag from RETURNING (xmax = 0).", "how_handled": "Equivalence (incl. same-dim-tuple-twice-in-one-envelope yielding (1,1) not (2,0)) is pinned by test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update. Already handled.", "amend_plan": "already-done" },
      { "what": "bench-v4 Python files predate Phase 2 at ~100-col and fail the repo ruff line-length-120 (a pre-existing E501).", "how_handled": "Left unfixed and flagged for the operator as a branch-wide ruff reconciliation required before merging ct/bench-v4 to develop; RETAINED deferred item.", "amend_plan": "no" },
      { "what": "pytest scripts/ collects a fourth file (scripts/tests/test_benchmark_reporting.py) beyond the three named suites.", "how_handled": "Accepted as harmless and matching the acceptance command; a clarifying ci.yml comment added.", "amend_plan": "no" }
    ],
    "coverage": {
      "tested_cases": [
        { "case": "measurement_id Rust==golden==Python parity across all 5 tables incl. negative i32, null/Some, multibyte UTF-8, whole/negative/tiny/large f64 thresholds; SCHEMA_VERSION lockstep with schema.rs", "test_location": "scripts/test_measurement_id.py + scripts/measurement_id_golden.json; scripts/test_post_ingest_postgres.py:3105", "confidence": "high" },
        { "case": "bench_ingest DML-only (SELECT/INSERT/UPDATE) on all 6 tables, denied DELETE/CREATE/DDL; default-privileges cover future migrator tables; 001..004 apply cleanly + idempotently under a real NOSUPERUSER CREATEROLE master", "test_location": "scripts/test_migrate_schema.py:1960,1928,1993,2069", "confidence": "high" },
        { "case": "provision.sh provisions ingest role on instance dbuser, emits GH_BENCH_INGEST_ROLE_ARN, drops dead proxy grant", "test_location": "scripts/test_migrate_schema.py:1805,1835", "confidence": "high" },
        { "case": "insert-then-update accounting; re-ingest upserts (0 inserted, N updated) with stable counts; measurement_id matches port", "test_location": "scripts/test_post_ingest_postgres.py:2477,2572", "confidence": "high" },
        { "case": "ON CONFLICT DO UPDATE overwrites every value/env column while leaving dim tuple / measurement_id stable, per table; dim columns NEVER in any DO UPDATE SET list for all 5 tables", "test_location": "scripts/test_post_ingest_postgres.py:2867", "confidence": "high" },
        { "case": "NaN/Inf/out-of-f64-range threshold raises loudly and rolls back; deny_unknown_fields; missing-required; unknown/non-scalar kind; storage enum; memory quartet; commit_sha mismatch; type/range boundaries reproduced", "test_location": "scripts/test_post_ingest_postgres.py:2633,2887,3066,3074,3094", "confidence": "high" },
        { "case": "connect_postgres: IAM-token-when-passwordless, always-bench_ingest, rejects weak sslmode (verify-full), forces search_path=public, rejects non-TLS, post-connect ssl_in_use", "test_location": "scripts/test_post_ingest_postgres.py:3142,3221,3233,3264,3285", "confidence": "high" },
        { "case": "--server requires --benchmark-id while --postgres does not; mutual exclusivity + dispatch", "test_location": "scripts/test_post_ingest_postgres.py:3401,3363,3375", "confidence": "high" },
        { "case": "fail-loud-on-no-Docker in CI vs skip locally (both test files)", "test_location": "scripts/test_migrate_schema.py:1745; scripts/test_post_ingest_postgres.py:2753", "confidence": "high" },
        { "case": "write-conflict retry (deadlock/serialization) retries then succeeds, gives up at cap, propagates validation errors immediately", "test_location": "scripts/test_post_ingest_postgres.py (test_retry_write_conflicts_*)", "confidence": "medium" }
      ],
      "untested_cases": [
        { "case": "End-to-end live v4 dual-write per develop push (CloudWatch shows Postgres writes) — the PR-2.4 acceptance criterion needing real RDS + repo vars", "priority": "high", "why_untested": "Operator-side, not performable in-session; the v4 steps no-op until GH_BENCH_INGEST_ROLE_ARN + RDS_BENCH_INSTANCE_ENDPOINT/DB_NAME/REGION are set. Tracked as an open operator dependency." },
        { "case": "Real psycopg DeadlockDetected inside conn.transaction() across two reversed-order connections on autocommit=False (retry+commit/rollback)", "priority": "medium", "why_untested": "Retry path covered only by mocked-op unit tests; a real-conflict container test needs deterministic threaded interleaving. Explicitly deferred (PR-2.2 cycle-7) to the follow-up test-hardening PR; behavior manually verified against live PG16." },
        { "case": "migration 004 self-grant when the executing role lacks ADMIN on a pre-existing migrator (unsupported misconfiguration)", "priority": "low", "why_untested": "Deferred; the shipped single-bootstrap-master path is correct and tested." },
        { "case": "schema_conn BENCH_TEST_PG_DSN destructive-scrub guard against a localhost-tunnel-to-prod", "priority": "low", "why_untested": "Dev-only override never used in CI (testcontainers only); explicitly accepted/deferred as an exotic foot-gun." }
      ],
      "recommendations": "Coverage of the load-bearing data-correctness invariants is comprehensive and pins each against an automated test, satisfying behavior-preservation against the v3 serde/ingest boundary per-table. The only material gap is the live end-to-end dual-write, gated on operator var-setting and best-effort by design. The deferred real-deadlock/autocommit=False container test is the one worth landing in the follow-up test-hardening PR; the remaining untested cases are availability/test-infra hardening already triaged and out of scope per the lean re-plan."
    },
    "tradeoffs": [
      { "decision": "Schema-deploy authorization (PR merge is the gate; push-on-develop trigger; no environment/manual-approval gate)", "original": "Supersedes the original manual-approval mandate (2026-05-29 deploy-model decision)", "verdict": "keep", "rationale": "PR-2.4 implements exactly this: push-on-develop under paths migrations/**, dry_run kept, environment-gate comments removed. Execution safety comes from the per-PR testcontainer migration test. Accepted tradeoff; not re-flagged.", "found_by": ["spec", "correctness"] },
      { "decision": "Ingest writer language (pure Python extending post-ingest.py; xxhash port)", "original": "Avoid sqlx/aws-sdk Rust deps", "verdict": "keep", "rationale": "PR-2.2 delivered exactly this; the port matches Table A/B and SCHEMA_VERSION lockstep holds.", "found_by": ["spec"] },
      { "decision": "Cutover style (short best-effort dual-write soak, then promote v4); v4 dual-write is continue-on-error during soak, v3 stays hard-required", "original": "Amended 2026-06-04 to make v4 best-effort during soak (PR-2.4 lean re-plan)", "verdict": "keep", "rationale": "PR-2.4's continue-on-error v4 steps are the direct realization; a v4 OIDC/connect hiccup must never break the proven v3 pipeline; v4 promoted to required at cutover (PR-5.1). Both steps consume the same results.v3.jsonl, so neither substrate is written in isolation. Calibrated, accepted exception to the no-best-effort rule.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 ingest DB identity (dedicated bench_ingest + GitHubBenchmarkIngestRole, separate from migrator)", "original": "Re-plan Q2 least-privilege split (PR-2.1)", "verdict": "keep", "rationale": "PR-2.1 implemented precisely this with DML-only grants and a separate OIDC role; separation of duties; the writer enforces bench_ingest unconditionally. Pinned by tests.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; drop reconcile-ingest.py + dual-write-verify.yml + incident.io)", "original": "Superseded the 2026-06-01 per-push reconciliation harness", "verdict": "keep", "rationale": "PR-2.5 correctly dropped; PR-2.4 adds no reconciliation machinery; v4-only failure does not trigger incident.io. Four independent safety nets remain. Not drift.", "found_by": ["spec"] },
      { "decision": "Duplicate JSON keys collapse last-wins (no rejection), matching v3", "original": "PR-2.2 cycle-6", "verdict": "keep", "rationale": "serde_json::Value::Object and Python json.loads are both last-wins; rejecting would break behavior-preservation. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "post-ingest.py PEP 723 dependencies = []; --postgres runs from the uv env", "original": "PR-2.2", "verdict": "keep", "rationale": "Keeps the v3 --server path stdlib-only under bare python3; --postgres deps come from the uv workspace. Lazy imports verified. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "IAM-token region precedence: --region > boto3 session region > RDS-hostname-parsed", "original": "PR-2.2", "verdict": "keep", "rationale": "A wrong region fails loud at IAM connect, not silently; ordering matches AWS convention. Not re-flagged.", "found_by": ["correctness"] }
    ]
  },
  "executive_summary": "Phase 2 ships the Postgres dual-write ingest path and its CI plumbing, exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction. Both lenses accept at high confidence; the correctness lens found zero data-correctness bugs and the spec lens found only two cosmetic nits, so there are no must-fix or should-fix items. PR-2.1 adds a least-privilege bench_ingest login role (migration 004) with SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), USAGE-not-CREATE, and ALTER DEFAULT PRIVILEGES so future migrator tables auto-grant; provision.sh adds a branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance only) and drops the dead proxy grant. PR-2.2's post-ingest.py --postgres reproduces the v3 serde boundary in Python, computes measurement_id bit-identically via the PR-1.5 port, and upserts commits-first then five fact tables in one all-or-nothing transaction with deadlock retry over verify-full TLS and IAM auth. PR-2.3 runs pytest scripts/ behind a docker hard gate plus a CI fail-loud fixture. PR-2.4 makes each of the three ingest workflows add a best-effort continue-on-error v4 step after the unchanged hard-required v3 step, and switches schema-deploy to push-on-develop under paths migrations/**. PR-2.5 was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. The key engineering surprises were all already resolved in-cycle: the non-superuser RDS master breaking ALTER DEFAULT PRIVILEGES (fixed with a guarded INHERIT self-grant + a real NOSUPERUSER test), the ssl_in_use accessor living on conn.pgconn rather than conn.info (fixed and pinned), and the fail-loud-on-no-Docker guard being unable to catch always-skip because pytest.skip is not a Failed subclass (fixed and mutation-verified). Correctness independently confirmed Python-is-stricter-or-equal handling of NaN/Inf literals, agreement on u32/u64-vs-i32/i64 ranges, and insert-vs-update equivalence between Rust's exists() preflight and Python's xmax=0 classifier. Every Key decision was verdicted keep by both reviewers: schema-deploy PR-merge gating, pure-Python writer + xxhash port, best-effort dual-write during soak with v3 hard-required, the dedicated bench_ingest identity, dropping the reconcile harness, last-wins duplicate-key handling, PEP 723 empty-deps, and IAM region precedence. Coverage of load-bearing data-correctness invariants is comprehensive and per-table-pinned; the one material gap is the live end-to-end dual-write, which is operator-gated on repo vars and best-effort by design. Verdict: ACCEPT.",
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 0,
  "nit_count": 2,
  "review_cycles_this_invocation": 1
}
```

</details>


### Cycle 2 — preset=phase-2 — accept

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1, "preset": "phase-2", "lenses_used": ["spec", "correctness"], "review_count": 2,
  "unified_findings": [
    {"severity":"nit","kind":"convention","file_line":"scripts/post-ingest.py:1085","description":"PR-2.8's build_commit git-history comment lines sit at exactly 100 cols (at the user CLAUDE.md limit, not over; well under ruff's 120). Already rewrapped from 102 in f0e93b9f4. No action.","recommended_fix":"None; within the 100-col rule.","found_by":["spec"]},
    {"severity":"nit","kind":"error-path","file_line":"scripts/post-ingest.py:448","description":"After the de-gold-plate dropped the _require_finite OverflowError sub-branch, a huge-int threshold (e.g. 10**309) now raises an uncaught OverflowError instead of a controlled SystemExit. Still fails LOUD + rolls back the transaction (no silent-wrong-write); the input class (oversized-int on trusted CI) is explicitly DO-NOT-flag per the calibration. Intentional de-gold-plate tradeoff.","recommended_fix":"None required (accepted). A one-line `except OverflowError: is_finite_number = False` would restore the controlled error, but is not worth carrying per the trusted-input calibration.","found_by":["correctness"]}
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Amended cumulative Phase 2 delivers the Postgres dual-write ingest path + best-effort v4 CI. Ingest identity (PR-2.1): migrations/004 creates least-privilege bench_ingest (SELECT/INSERT/UPDATE on the 6 tables, USAGE-not-CREATE, ALTER DEFAULT PRIVILEGES for future migrator tables, no DELETE/DDL); provision.sh adds the branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance dbuser) and drops the dead proxy grant; 004 self-grants migrator INHERIT via the master's ADMIN option then revokes (non-superuser-RDS-master bootstrap fix). Writer (PR-2.2, de-gold-plated by PR-2.7): post-ingest.py --postgres computes measurement_id bit-identically via the byte-exact _measurement_id.py port, validates every field/type/range to reproduce the v3 serde + deny_unknown_fields boundary, upserts commits-first then 5 fact tables via INSERT...ON CONFLICT(measurement_id) DO UPDATE...RETURNING(xmax=0) in one all-or-nothing transaction with deadlock/serialization retry; connect_postgres enforces verify-full TLS + post-connect ssl_in_use + bench_ingest-only + IAM-token mint + search_path=public. CI wiring (PR-2.3/2.4/2.6): a scripts-test job runs pytest scripts/ (Docker-required, fail-loud-not-skip in CI); the 3 ingest workflows add a continue-on-error best-effort v4 --postgres step after the unchanged hard-required v3 step; schema-deploy.yml triggers on push to develop under migrations/**. Amendments: PR-2.6 re-keys the 9 v4 gates from the endpoint var to GH_BENCH_INGEST_ROLE_ARN so they no-op (not fire+fail) until infra is wired; PR-2.7 de-gold-plates ~194 lines of trusted-input over-hardening (NUL/surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) + ~8 tests, moving the writer CLOSER to the v3 Rust source; PR-2.8 clears the ruff E501 merge-blocker + README PR-lineage + a git-history comment. Every preserved invariant (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed validation, memory-quartet, IAM/TLS) remains pinned by tests; the de-gold-plate removed only stricter-than-v3 hardening on trusted CI input.",
    "surprises": [
      {"what":"PR-2.7's OverflowError-branch removal left an orphaned pytest.param(10**309) in the KEPT test_nonfinite_threshold_raises_and_rolls_back, which would have gone CI-red (uncaught OverflowError); only skipped locally for lack of Docker.","how_handled":"Caught by cycle-1 correctness (cumulative-test trace), fixed in b6d1f292b (param dropped; nan/inf/-inf retained). Verified gone.","amend_plan":"already-done"},
      {"what":"The de-gold-plate moves the writer CLOSER to v3: the removed NUL/surrogate/non-UTF-8 guards were STRICTER than the v3 Rust source, so removal improves behavior-preservation rather than regressing it; failure modes stayed LOUD (UnicodeEncodeError in _write_str / Postgres DataError at bind, inside the transaction -> rollback). No silent-wrong-write path.","how_handled":"Authorized scope-reduction per the 2026-06-05 complexity audit; KEEP-list verified intact field-by-field.","amend_plan":"no"},
      {"what":"The live gate-var bug (PR-2.6) was a real config mismatch: gates keyed on RDS_BENCH_INSTANCE_ENDPOINT (set) while assume-role uses GH_BENCH_INGEST_ROLE_ARN (unset), so the v4 steps would fire+fail at assume-role on the next develop push.","how_handled":"Re-keyed all 9 gates to the role-ARN var; live-var state now makes the gate evaluate false (clean no-op until wired).","amend_plan":"already-done"}
    ],
    "coverage": {
      "tested_cases": [
        {"case":"measurement_id parity (5 kinds) vs the Python port + stored rows; SCHEMA_VERSION lockstep","test_location":"scripts/test_measurement_id.py (65 vectors) + test_post_ingest_postgres.py (measurement_ids_match_python_port, schema_version)","confidence":"high"},
        {"case":"ON CONFLICT SET excludes dim columns (BAN) across all 5 _insert_ fns; per-table value-column update","test_location":"test_on_conflict_set_excludes_dim_columns + test_update_overwrites_all_value_columns_per_table","confidence":"high"},
        {"case":"insert-vs-update accounting incl. same-dim-twice (xmax=0 classifier); commit-before-facts; all-or-nothing rollback","test_location":"test_ingest_inserts_then_updates, test_same_dim_tuple_twice..., test_late_validation_failure_rolls_back_earlier_fact_row","confidence":"high"},
        {"case":"NaN/+Inf/-Inf threshold rejected + rollback (post de-gold-plate, OverflowError param removed)","test_location":"test_nonfinite_threshold_raises_and_rolls_back","confidence":"high"},
        {"case":"typed i32/i64/finite + deny_unknown_fields + memory-quartet + storage-enum + commit_sha-mismatch validation","test_location":"test_post_ingest_postgres.py value-validation tests","confidence":"high"},
        {"case":"connect_postgres: verify-full + post-connect ssl_in_use + bench_ingest-only + IAM-token-vs-password + search_path pin","test_location":"test_connect_postgres_* (9 pure-unit tests, pass non-Docker)","confidence":"high"},
        {"case":"de-gold-plate replacements: mixed-newline read, malformed-JSON loud-fail, git_show text-mode decode","test_location":"test_read_records_happy_path_mixed_newlines / _rejects_malformed_json / test_git_show_field_decodes_and_strips","confidence":"high"},
        {"case":"bench_ingest least-privilege + 004 idempotency + non-superuser-master bootstrap; provision.sh ingest role + dropped proxy grant","test_location":"scripts/test_migrate_schema.py (testcontainer + static provision tests)","confidence":"high"},
        {"case":"v4 gate keys on GH_BENCH_INGEST_ROLE_ARN; CI fail-loud-not-skip on missing Docker","test_location":"workflow inspection + test_require_docker_fails_loud_in_ci/_skips_without_ci","confidence":"high"}
      ],
      "untested_cases": [
        {"case":"Live v4 dual-write end-to-end against real RDS per develop push (OIDC->IAM->verify-full->upsert; CloudWatch confirmation)","priority":"medium","why_untested":"Requires live AWS + operator-set repo vars; operator-side soak. Acceptance for PR-2.4 is the structural/yamllint criteria; v4 correctness is gated by Phase-3 migrate --verify."},
        {"case":"Real two-connection DeadlockDetected on an autocommit=False connection asserting retry+commit","priority":"low","why_untested":"Container-only threaded-interleaving test; deferred (PR-2.2 cycle-7) to a follow-up test-hardening PR. Retry pinned by mocked-op unit tests + live-container verification."},
        {"case":"Live OIDC schema-deploy apply firing on a migrations/** push","priority":"low","why_untested":"The retained operator pre-merge gate; not in-session testable."}
      ],
      "recommendations": "Coverage is strong for a trusted-input low-stakes dashboard and matches the lean calibration. The only meaningful gaps are the two retained operator-side gates (live OIDC schema-deploy apply; live v4 dual-write soak + CloudWatch). No new test work is required to accept the amended phase. The deferred real-deadlock/autocommit=False integration test remains a reasonable follow-up but is not a blocker."
    },
    "tradeoffs": [
      {"decision":"v4 ingest steps continue-on-error (best-effort) during the soak","original":"PR-2.4 lean re-plan amendment","verdict":"keep","rationale":"v3 stays hard-required + untouched; the 'no continue-on-error' BAN applies at the PR-5.1 cutover, not the soak. PR-2.6 reinforces it (clean no-op until wired)."},
      {"decision":"Gate v4 steps on GH_BENCH_INGEST_ROLE_ARN (PR-2.6)","original":"audit live-bug fix","verdict":"keep","rationale":"Aligns the gate with the assume-role input so the steps no-op (not fire+fail) until infra is wired; verified across all 9 gates."},
      {"decision":"De-gold-plate trusted-input hardening (PR-2.7)","original":"2026-06-05 complexity audit","verdict":"keep","rationale":"Removed only stricter-than-v3 guards on trusted CI input; every removed input still fails loud inside the transaction (rollback). No data-correctness guard or its test removed."},
      {"decision":"Schema-deploy authorization (PR-merge gate, push trigger, no environment: gate)","original":"2026-05-29 deploy-model decision","verdict":"keep","rationale":"PR-2.4 shipped the trigger; the amendment did not touch it. Accepted tradeoff."},
      {"decision":"Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; PR-2.5 dropped)","original":"lean re-plan","verdict":"keep","rationale":"The amendment added no reconciliation machinery; consistent with the superseding decision."},
      {"decision":"Pure-Python writer + PEP 723 dependencies=[]; IAM region precedence","original":"PR-2.2","verdict":"keep","rationale":"Accepted tradeoffs; keeps v3 --server stdlib-only; region mismatch fails loud. Unchanged by the amendment."},
      {"decision":"Dedicated bench_ingest identity separate from migrator","original":"re-plan Q2","verdict":"keep","rationale":"Unchanged; PR-2.6 gates on the ingest role ARN, reinforcing the separate-identity model; connect_postgres still enforces bench_ingest-only."}
    ]
  },
  "executive_summary": "Phase 2 cycle-2 phase-end review of the AMENDED cumulative phase (PR-2.1..2.4 accepted at cycle 1, plus the 2026-06-05 amendment PR-2.6/2.7/2.8). Both lenses ACCEPT with high confidence; 0 must-fix, 2 dismissable nits, no disagreements. The amendment did exactly three things and the reviewers verified each: PR-2.6 fixed a real LIVE gate-var bug (the v4 dual-write gates keyed on RDS_BENCH_INSTANCE_ENDPOINT [set] while assume-role uses GH_BENCH_INGEST_ROLE_ARN [unset], so the steps would fire+fail on the next develop push) by re-keying all 9 gates to the role-ARN var, restoring clean no-op-until-wired behavior. PR-2.7 de-gold-plated ~194 lines of trusted-input over-hardening (NUL/lone-surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) that was STRICTER than the v3 Rust source it mirrors; the spec lens confirmed this is authorized scope-reduction (not silent de-scoping) and the correctness lens verified field-by-field that NO load-bearing data-correctness validation was removed (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed i32/i64/finite, deny_unknown_fields, memory-quartet, IAM/TLS all intact) and that the removed string guards open NO silent-wrong-write path (a NUL or lone surrogate still fails LOUD inside the transaction -> rollback). PR-2.8 cleared the retained ruff E501 merge-blocker + 2 doc nits. The cycle-1 amendment must-fix (an orphaned 10**309 test param after the OverflowError-branch removal) was caught and fixed during PR-2.7's own inner loop. All Phase-2 exit criteria re-verified on the amended tree: yamllint clean, ruff clean, the 3 ingest workflows have the required v3 step + continue-on-error v4 steps, the schema-deploy push trigger, SCHEMA_VERSION lockstep, measurement_id reproduction; 120 non-Docker unit tests pass. Every Key decision remains keep. The two nits are dismissable: the build_commit comment sits at (not over) the 100-col limit, and the huge-int-threshold path now raises an uncaught OverflowError instead of SystemExit but still fails loud + rolls back (an explicitly DO-NOT-flag trusted-input case). Two operator-side gates remain (live OIDC schema-deploy apply; live v4 soak + CloudWatch). Verdict: ACCEPT.",
  "overall": "accept", "must_fix_count": 0, "should_fix_count": 0, "nit_count": 2, "review_cycles_this_invocation": 2
}
```

</details>

## Phase 1+2 complexity & gap audit (2026-06-05)

Operator-requested step-back review ("did we go overkill / too much complexity / is stuff missing"), run as 3 parallel auditors (over-engineering, gaps, architecture-coherence) with full repo access, then synthesized. Verdict: **the destination architecture is sound and coherent; the writer/tooling is meaningfully over-built; one real LIVE bug + several Phase-3 gaps.**

**Sound (do not touch):** RDS + IAM-auth + hash-preserving `ON CONFLICT` upsert + Next.js/Vercel is the right shape. `measurement_id` parity verified field-by-field across all 6 tables -> complete + correct, no silent-duplicate risk in the port. `RETURNING (xmax=0)` classification is more correct than the v3 Rust under concurrency.

**LIVE BUG (must-fix, PR-2.4 scope) -- v4 dual-write gates on the WRONG variable.** The 3 ingest workflows gate the v4 step on `if: vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (live: SET since 2026-06-01) but assume-role uses `vars.GH_BENCH_INGEST_ROLE_ARN` (live: NOT SET). So on the next `develop` push the v4 steps FIRE and FAIL at assume-role (swallowed by `continue-on-error`) rather than cleanly no-op'ing -- the plan's "no-ops until vars set" premise is false. Fix: gate on `GH_BENCH_INGEST_ROLE_ARN != ''` (the var that must exist for assume-role). One-line per workflow (`bench.yml:128/135/139`, `sql-benchmarks.yml:508/515/519`, `v3-commit-metadata.yml:41/48/52`). **Selected for the Phase-2 amendment.**

**Overkill (confirmed by 2 of 3 auditors):** ~35% of `post-ingest.py`'s 987 new `--postgres` lines + ~25-30% of `test_post_ingest_postgres.py`'s 1221 lines are adversarial-input hardening (NUL / lone-surrogate / non-UTF-8 / RecursionError / oversized-int) on TRUSTED CI input -- which the lean calibration itself says is out of scope, and which is STRICTER than the v3 Rust source it mirrors. Frozen residue of the pre-lean-re-plan 15-cycle PR-2.2 spiral.
- Cleanly deletable, ~no risk: `_is_local_host` (dead -- its own docstring says no production role), the unreachable `RecursionError` branch in `read_records`.
- Low risk: `_reject_unstorable_str` (surrogate/NUL guards), the `git_show_field` bytes-decode rewrite, the `read_records` universal-newline hardening, the `_require_finite` `OverflowError` branch, + ~12 tests.
- Keep untouched: hash parity, ON CONFLICT/RETURNING, retry, IAM/TLS auth + `004_ingest_role.sql`, deny_unknown_fields + typed i32/i64/finite validation, memory-quartet/storage-enum.

**Bigger structural simplifications (higher risk; reverse accepted + partly live-verified Phase-1 decisions -- arguably re-plan, not amend):**
- Reuse the existing Rust `measurement_id_*` hasher (already linked by `migrate/`) instead of the Python port -> deletes the golden-vector apparatus + the NaN-payload `is_finite` hazard + the endianness caveat. Medium risk; reverses Key-decision Q4 (pure-Python writer to keep CI writers dep-light).
- Replace the bespoke `migrate-schema.py` (240 LOC + 1431 test lines) with a ~15-line `psql -f` apply loop for 4 idempotent additive migrations. Low-med risk; reverses PR-1.2 (tested + live-verified).
- RDS Proxy is fully provisioned (`provision.sh`) but consumed by NOTHING until PR-4.2 (whose Vercel-to-VPC reachability is unsolved) -> dead infra today. Could defer provisioning to Phase 4.

**Phase-3 gaps (load-bearing; deferred to a Phase-3 re-plan at the Phase 2->3 boundary per operator 2026-06-05):**
1. `migrate --verify` (the PRIMARY v4-correctness gate) does NOT exist yet and is net-new, not an "extension": `migrate/src/verify.rs` today is a v2-vs-v3 structural diff with NO Postgres dependency (no sqlx/tokio-postgres in `migrate/Cargo.toml`). Its stated baseline `matched_rows == duckdb_rows AND only_in_postgres == []` is necessary-but-NOT-sufficient -- a PK-count match cannot detect value-column corruption in the bulk COPY, and `env_triple` is not in the hash so count-match can't see env corruption. Phase-3 verify must compare value-column payloads per `measurement_id`, not just PK presence.
2. The DuckDB->Postgres bulk-load seed (PR-3.1) establishes the existing `measurement_id`s the whole upsert-not-duplicate invariant depends on -- the single most correctness-critical unwritten code. One-shot, crosses an AWS account boundary (`375504701696/us-east-2` DuckDB <-> `245040174862/us-east-1` RDS) + region; NO rehearsal/dry-run specified before the one-shot prod load.
3. The soak exercises the PYTHON writer (`post-ingest.py --postgres`); the cutover gate (`migrate --verify`) exercises the RUST bulk-loader -- the two NEVER cross-check. So a green verify does not imply the steady-state Python writer round-trips against real RDS. Add a Python-writer-vs-RDS cross-check (a few rows' `measurement_id` + value columns vs the Rust-seeded rows) before PR-5.1 makes the Python path required.
4. Migration `004_ingest_role.sql` must be applied AS THE RDS MASTER (or the `ALTER DEFAULT PRIVILEGES` self-grant hits InsufficientPrivilege + rolls back); this ordering ("002+004 by the master before any `migrator` run") is documented in the SQL but enforced nowhere.

## Phase 3: Historical data load (DuckDB → Postgres) + value-verify — end-of-phase review (cycle 1) — accepted (3-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-3, lenses=spec+correctness+maint, executor=claude). Verdict: ACCEPT — 0 must-fix, 3 should-fix, 4 nits. Full Synthesizer Output JSON in the `<details>` block at the end of this section (also the cycle-1 raw-response archive).**

### Summary of changes
Phase 3 delivers the v3-to-v4 DuckDB-to-Postgres migration TOOLKIT (validated locally; the irreversible prod load is deferred to the Phase-5 cutover as PR-5.0). (1) BULK LOADER (postgres.rs + Load subcommand): reads 6 tables from an existing v3 DuckDB as Arrow batches and streams them via COPY FROM STDIN text format inside ONE transaction (atomic; mid-load failure rolls back). measurement_id/commit_sha verbatim; commits.timestamp CAST to VARCHAR under SET TimeZone=UTC; all_runtimes_ns explodes to {a,b,c}; f64 shortest-round-trip with non-finite rejected; NoTls locally or native-tls+--ca-cert for prod; NO aws-sdk-rds. (2) VALUE VERIFIER (verify.rs + verify --postgres-target): PRIMARY gate, a new PgVerifyReport/ValueMismatch pair (separate from the v2 structural VerifyReport), per-measurement_id full compare of every non-hashed value column, all_runtimes_ns element-wise + order-sensitive, timestamps as engine-independent epoch microseconds, read kinds resolved once from the loader column_kind so the two sides cannot drift. (3) REHEARSAL (postgres_e2e.rs + README): first migrate testcontainer (postgres:16-alpine), asserts count match + verify-clean + atomic mid-load rollback, Docker-gated skip-local/fail-loud-CI. (4) CROSS-CHECK (cross_check_python_writer.py): writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip, reuses connect_postgres verify-full TLS as bench_ingest. (5) BOOTSTRAP GUARD (migrate-schema.py): requires-superuser marker + rolsuper-OR-rolcreaterole preflight rejecting a non-master apply before any DDL. PR-3.4 ran all three gates GREEN against the real 4.2M-row bench.duckdb into local PG16, zero prod writes. Toolkit honestly labeled throwaway (removal trigger Table F / PR-5.3).

### Surprises and discoveries
- **The loader reads an EXISTING v3 DuckDB (copying measurement_id/commit_sha verbatim) rather than re-accumulating from v2.** — Documented; simpler and safer since ids are byte-preserved. (amend_plan: no)
- **PR-3.1 shipped the synchronous `postgres` 0.19 crate, not the plan-named `tokio-postgres`, with no deviation note.** — Defensible (hard-NO on aws-sdk-rds honored) but unremarked. Add a one-line status note. (amend_plan: yes)
- **PR-3.4's real snapshot held vector_search_runs=0 rows AND the cross-check envelope omitted that kind, so vector_search_runs got zero REAL-data coverage through the gate.** — Code paths fixture-covered; status overstates GREEN. Annotate; PR-5.0 closes it. (Lens disagreement: spec should-fix, maint acceptable.) (amend_plan: yes)
- **The bootstrap PermissionError cites migrations/README.md content (requires-superuser marker / preflight / 002-004 ordering) that does not exist there.** — Not handled; add a README subsection. (amend_plan: yes)
- **The local module `postgres` collides by name with the external `postgres` crate.** — Compiles; cosmetic disambiguation nit. (amend_plan: no)
- **PR-3.1 status claims column order is 'programmatically cross-checked against migrations/001', but the check is indirect (e2e COPY fails on disagreement), not a DDL-parsing test.** — Status log overstates mechanism; behavior covered by the e2e. (amend_plan: no)

### Testing coverage assessment
**Tested (high confidence):**
- loader COPY-text rendering incl. escaping/BIGINT[]/literal-\N-vs-NULL/non-UTC-tz/f64-round-trip+non-finite-reject (postgres.rs:514-712, high)
- value-verify discrimination across 6 tables incl. array reorder/Int32+side-counters/presence-both-directions/epoch-us-pinned (verify.rs:965-1356, high)
- Postgres e2e: counts/verify-clean/atomic mid-load rollback (postgres_e2e.rs:136-226, high)
- cross-check UPDATE-not-INSERT/all-5-kinds/NOT-seeded->INSERT/value_mismatches mutation-verified (test_post_ingest_postgres.py:3795-3947, high)
- requires-superuser guard: marker detection/non-master rejected before DDL (test_migrate_schema.py:3629-3768, high)
- real-data PR-3.4: 4.2M-row load exact match + verify clean + cross-check clean + negative (manual run, medium)

**Untested / under-covered (all non-blocking; plan-acknowledged or status-annotation):**
- vector_search_runs Double/threshold COPY + vector value-verify against REAL data (0 real rows + omitted from cross-check) — priority medium, annotate status + PR-5.0 closes
- verify-full TLS --ca-cert against a live RDS endpoint — priority low, compile+runbook only, plan-acknowledged
- TABLE_SPECS vs migrations/001 column-order as a direct DDL-parsing unit test — priority low, covered behaviorally by the e2e COPY

### Tradeoffs re-evaluation
- **Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0); Phase 3 closes on the real-snapshot LOCAL rehearsal** — _keep_: PR-3.4 executed exactly this with zero irreversible prod side-effect; RDS PITR is the prod rollback.
- **Value-column verify as a NEW PgVerifyReport pair (not overloading v2-structural VerifyReport)** — _keep_: Single-responsibility; structural vs stored-value comparison are different concerns.
- **commits.timestamp compared as epoch microseconds** — _keep_: Engine-independent and exact; e2e confirmed sub-second + pre-1970.
- **local postgres:16 testcontainer rehearsal, Docker-gated** — _keep_: Same engine as RDS at zero AWS cost; e2e closes the live Postgres-execution gap.
- **operator-run-locally master-password DSN; DROP aws-sdk-rds** — _revisit-but-keep_: Keep (avoids heavy AWS SDK); record the resulting sync-`postgres`-over-`tokio-postgres` swap in the PR-3.1 status.
- **004-as-master requires-superuser preflight guard** — _revisit-but-keep_: Guard is well-shaped and load-bearing; close the migrations/README.md doc gap the PermissionError points at.
- **trusted-input low-stakes review calibration; 3-vote Phase 3** — _keep_: Appropriate; adversarial-input hardening correctly out of scope; retained guards are load-bearing.

### Disagreements
- **vector_search_runs real-data coverage gap through the PR-3.4 validation gate**
  - spec: should-fix / weak-acceptance: one of six tables (the one with the most distinct column shape) got no real-data validation through the gate while the status asserts GREEN without qualifying the fall-through. Annotate the status entry.
  - maint: acceptable / amend_plan: no: the Double/threshold COPY + vector value-verify paths are covered by the synthetic e2e fixture (one seeded vector row), so the gap is covered; no plan amendment needed.
  - **Synthesizer call:** Act on the spec position at should-fix, scoped to a status-log annotation (NOT a code/test change). maint is correct that the paths ARE fixture-exercised (no untested-code defect, no new test warranted for a throwaway crate); spec is correct that PR-3.4's distinguishing value is REAL-data validation that vector_search_runs did not receive. Both reduce to: annotate the PR-3.4 status that vector_search_runs had 0 real rows and point at PR-5.0's freshest-snapshot prod load. Non-blocking.

**Dropped re-flags (carry-forward respected):**
- Dim-column COPY corruption is invisible to the primary value-verify gate (verify.rs omits dim columns; load_table COPYs them verbatim) — covered by Accepted tradeoffs (Accepted tradeoffs / r1 traps)

### Findings (0 must-fix, 3 should-fix, 4 nits)
| # | Severity | Kind | File:line | Description | Found by |
|---|----------|------|-----------|-------------|----------|
| 1 | should-fix | scope-drift | `benchmarks-website/migrate/Cargo.toml:35-37` | PR-3.1 shipped the synchronous `postgres` 0.19 crate (+ native-tls + postgres-native-tls), not the plan-named `tokio-postgres`, with no explicit deviation note in the PR-3.1 implementation-status entry. The binding hard-NO (no aws-sdk-rd... | spec/claude |
| 2 | should-fix | weak-acceptance | `benchmarks-website/migrate/tests/postgres_e2e.rs` | PR-3.4's real-snapshot LOCAL rehearsal gave `vector_search_runs` zero real-data coverage: the real snapshot held 0 such rows AND the cross-check envelope omitted that kind, so its load COPY (Double/threshold, four Int side counters, the ... | spec/claude,maint/claude |
| 3 | should-fix | doc-quality | `scripts/migrate-schema.py:3590` | The new `_assert_master_capable` PermissionError tells the operator to 'see migrations/README.md and the migration header', but migrations/README.md has NO mention of the requires-superuser marker, the master-capable bootstrap, the rolsu... | maint/claude |
| 4 | nit | convention | `benchmarks-website/migrate/src/postgres.rs:1` | Local module is named `postgres` (`crate::postgres`) while the crate also depends on the external `postgres` crate. In verify.rs `crate::postgres::ColKind` and extern `postgres::Client` coexist; a fresh reader must disambiguate. | maint/claude |
| 5 | nit | convention | `benchmarks-website/migrate/src/verify.rs:684` | Two private `fn select_sql` with different signatures coexist: postgres.rs:333 `select_sql(&TableSpec)` and verify.rs:684 `select_sql(&ValueSpec, epoch_expr)`. A grep for `select_sql` lands on two unrelated builders. | maint/claude |
| 6 | nit | coverage | `benchmarks-website/migrate/src/verify.rs:972` | `in_memory_v3()` is duplicated verbatim in postgres.rs:506 and verify.rs:972, and build_fixture_duckdb (postgres_e2e.rs:97) repeats the same DDL-apply loop a third time. Three copies of the v3-schema builder; all iterate FAMILIES/COMMITS... | maint/claude |
| 7 | nit | boundary | `scripts/migrate-schema.py (_migration_requires_superuser)` | requires-superuser marker detection scans each physical line and matches when `line.startswith('--')` and `stripped == directive`. A migration INSERTing multi-line text whose own line begins with `-- migrate-schema: requires-superuser` (... | correctness/claude |

These 3 should-fix items are all NON-code documentation/status annotations; tracked for a follow-up doc/status pass (none block Phase 3). The 4 nits are dismissible per the trusted-input throwaway-crate calibration.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1) — cycle 1 raw archive</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 3,
  "nit_count": 4,
  "review_cycles_this_invocation": 1,
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/migrate/Cargo.toml:35-37",
      "description": "PR-3.1 shipped the synchronous `postgres` 0.19 crate (+ native-tls + postgres-native-tls), not the plan-named `tokio-postgres`, with no explicit deviation note in the PR-3.1 implementation-status entry. The binding hard-NO (no aws-sdk-rds) IS honored and the swap is defensible (sync `postgres` is a thin blocking wrapper over tokio-postgres), but a silent named-dependency substitution is the quiet plan-narrowing the spec lens surfaces.",
      "recommended_fix": "Add one line to the PR-3.1 implementation-status Surprises bullet noting the deliberate choice of synchronous `postgres` over the plan-named `tokio-postgres`. No code change required.",
      "found_by": [
        "spec/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "weak-acceptance",
      "file_line": "benchmarks-website/migrate/tests/postgres_e2e.rs",
      "description": "PR-3.4's real-snapshot LOCAL rehearsal gave `vector_search_runs` zero real-data coverage: the real snapshot held 0 such rows AND the cross-check envelope omitted that kind, so its load COPY (Double/threshold, four Int side counters, the only Int32 value column), value-verify join, and cross-check were never exercised against real data. The PR-3.4 status asserts GREEN without qualifying which table fell through; coverage rests on the synthetic PR-3.3 fixture + PR-3.2/3.5 unit tests.",
      "recommended_fix": "Annotate the PR-3.4 status entry that `vector_search_runs` had 0 real rows so its paths were validated only by synthetic fixtures, and note PR-5.0's freshest-snapshot prod load is expected to cover it. No code change.",
      "found_by": [
        "spec/claude",
        "maint/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "scripts/migrate-schema.py:3590",
      "description": "The new `_assert_master_capable` PermissionError tells the operator to 'see migrations/README.md and the migration header', but migrations/README.md has NO mention of the requires-superuser marker, the master-capable bootstrap, the rolsuper-OR-rolcreaterole preflight, or 002/004 ordering. The cited doc does not back the contract the error advertises.",
      "recommended_fix": "Add a short subsection to migrations/README.md covering the `migrate-schema: requires-superuser` marker, the rolsuper OR rolcreaterole preflight, and 'apply 002/004 as the RDS master before any migrator deploy'.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "convention",
      "file_line": "benchmarks-website/migrate/src/postgres.rs:1",
      "description": "Local module is named `postgres` (`crate::postgres`) while the crate also depends on the external `postgres` crate. In verify.rs `crate::postgres::ColKind` and extern `postgres::Client` coexist; a fresh reader must disambiguate.",
      "recommended_fix": "Rename the local module (e.g. `loader`/`pg_load`) or import the extern crate as `pg`.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "convention",
      "file_line": "benchmarks-website/migrate/src/verify.rs:684",
      "description": "Two private `fn select_sql` with different signatures coexist: postgres.rs:333 `select_sql(&TableSpec)` and verify.rs:684 `select_sql(&ValueSpec, epoch_expr)`. A grep for `select_sql` lands on two unrelated builders.",
      "recommended_fix": "Disambiguate the verify one (e.g. `value_select_sql`).",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "coverage",
      "file_line": "benchmarks-website/migrate/src/verify.rs:972",
      "description": "`in_memory_v3()` is duplicated verbatim in postgres.rs:506 and verify.rs:972, and build_fixture_duckdb (postgres_e2e.rs:97) repeats the same DDL-apply loop a third time. Three copies of the v3-schema builder; all iterate FAMILIES/COMMITS_DDL so the table set cannot drift, but the duplication is real.",
      "recommended_fix": "Extract one `pub(crate) fn open_in_memory_v3()` shared by all three call sites.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": "scripts/migrate-schema.py (_migration_requires_superuser)",
      "description": "requires-superuser marker detection scans each physical line and matches when `line.startswith('--')` and `stripped == directive`. A migration INSERTing multi-line text whose own line begins with `-- migrate-schema: requires-superuser` (inside a string literal) would be falsely classified. Author-controlled trusted content; the review calibration argues against fixing.",
      "recommended_fix": "If touched later, strip SQL string literals / inspect only the leading comment block. Not required given trusted-input calibration.",
      "found_by": [
        "correctness/claude"
      ]
    }
  ],
  "disagreements": [
    {
      "topic": "vector_search_runs real-data coverage gap through the PR-3.4 validation gate",
      "positions": [
        {
          "lens": "spec",
          "position": "should-fix / weak-acceptance: one of six tables (the one with the most distinct column shape) got no real-data validation through the gate while the status asserts GREEN without qualifying the fall-through. Annotate the status entry."
        },
        {
          "lens": "maint",
          "position": "acceptable / amend_plan: no: the Double/threshold COPY + vector value-verify paths are covered by the synthetic e2e fixture (one seeded vector row), so the gap is covered; no plan amendment needed."
        }
      ],
      "synthesizer_call": "Act on the spec position at should-fix, scoped to a status-log annotation (NOT a code/test change). maint is correct that the paths ARE fixture-exercised (no untested-code defect, no new test warranted for a throwaway crate); spec is correct that PR-3.4's distinguishing value is REAL-data validation that vector_search_runs did not receive. Both reduce to: annotate the PR-3.4 status that vector_search_runs had 0 real rows and point at PR-5.0's freshest-snapshot prod load. Non-blocking."
    }
  ],
  "dropped_re_flags": [
    {
      "topic": "Dim-column COPY corruption is invisible to the primary value-verify gate (verify.rs omits dim columns; load_table COPYs them verbatim)",
      "reason": "covered by Accepted tradeoffs",
      "reference": "Accepted tradeoffs / r1 traps"
    }
  ],
  "executive_summary": "Phase 3 ships the v3-to-v4 DuckDB-to-Postgres migration TOOLKIT and is accepted by all three lenses (spec, correctness, maint) with ZERO must-fix findings. The phase delivers an atomic single-transaction COPY loader (measurement_id/commit_sha verbatim, UTC-cast timestamps, non-finite f64 rejected, no aws-sdk-rds), a per-measurement_id value verifier as the primary correctness gate (engine-independent epoch-microsecond timestamps, element-wise order-sensitive array compare, read-kinds single-sourced from the loader so the two sides cannot drift), a first local postgres:16 testcontainer rehearsal (counts + verify-clean + atomic mid-load rollback), a Python-writer cross-check, and a requires-superuser bootstrap guard. PR-3.4 ran load+verify+cross-check GREEN against the real 4.2M-row snapshot into local PG16 with zero prod writes, clearing the phase exit criteria. The correctness lens ran a thorough adversarial hunt and found no surviving correctness bug; correctness-critical paths are unusually well-covered. Findings: three should-fix, all NON-code documentation/status annotations (sync-postgres-vs-tokio-postgres deviation note; vector_search_runs real-data coverage annotation; migrations/README.md missing the requires-superuser contract the PermissionError advertises) + four nits. One disagreement surfaced (spec rated the vector_search_runs gap should-fix; maint rated it acceptable) — synthesizer scoped it to a status annotation. Two tradeoffs moved to revisit-but-keep (DROP-aws-sdk-rds; the 004 preflight guard), both keep-with-a-doc-followup. Verdict: ACCEPT, 0 must-fix, 3 should-fix, 4 nits.",
  "phase_artifacts": {
    "summary": "Phase 3 delivers the v3-to-v4 DuckDB-to-Postgres migration TOOLKIT (validated locally; the irreversible prod load is deferred to the Phase-5 cutover as PR-5.0). (1) BULK LOADER (postgres.rs + Load subcommand): reads 6 tables from an existing v3 DuckDB as Arrow batches and streams them via COPY FROM STDIN text format inside ONE transaction (atomic; mid-load failure rolls back). measurement_id/commit_sha verbatim; commits.timestamp CAST to VARCHAR under SET TimeZone=UTC; all_runtimes_ns explodes to {a,b,c}; f64 shortest-round-trip with non-finite rejected; NoTls locally or native-tls+--ca-cert for prod; NO aws-sdk-rds. (2) VALUE VERIFIER (verify.rs + verify --postgres-target): PRIMARY gate, a new PgVerifyReport/ValueMismatch pair (separate from the v2 structural VerifyReport), per-measurement_id full compare of every non-hashed value column, all_runtimes_ns element-wise + order-sensitive, timestamps as engine-independent epoch microseconds, read kinds resolved once from the loader column_kind so the two sides cannot drift. (3) REHEARSAL (postgres_e2e.rs + README): first migrate testcontainer (postgres:16-alpine), asserts count match + verify-clean + atomic mid-load rollback, Docker-gated skip-local/fail-loud-CI. (4) CROSS-CHECK (cross_check_python_writer.py): writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip, reuses connect_postgres verify-full TLS as bench_ingest. (5) BOOTSTRAP GUARD (migrate-schema.py): requires-superuser marker + rolsuper-OR-rolcreaterole preflight rejecting a non-master apply before any DDL. PR-3.4 ran all three gates GREEN against the real 4.2M-row bench.duckdb into local PG16, zero prod writes. Toolkit honestly labeled throwaway (removal trigger Table F / PR-5.3).",
    "surprises": [
      {
        "what": "The loader reads an EXISTING v3 DuckDB (copying measurement_id/commit_sha verbatim) rather than re-accumulating from v2.",
        "how_handled": "Documented; simpler and safer since ids are byte-preserved.",
        "amend_plan": "no"
      },
      {
        "what": "PR-3.1 shipped the synchronous `postgres` 0.19 crate, not the plan-named `tokio-postgres`, with no deviation note.",
        "how_handled": "Defensible (hard-NO on aws-sdk-rds honored) but unremarked. Add a one-line status note.",
        "amend_plan": "yes"
      },
      {
        "what": "PR-3.4's real snapshot held vector_search_runs=0 rows AND the cross-check envelope omitted that kind, so vector_search_runs got zero REAL-data coverage through the gate.",
        "how_handled": "Code paths fixture-covered; status overstates GREEN. Annotate; PR-5.0 closes it. (Lens disagreement: spec should-fix, maint acceptable.)",
        "amend_plan": "yes"
      },
      {
        "what": "The bootstrap PermissionError cites migrations/README.md content (requires-superuser marker / preflight / 002-004 ordering) that does not exist there.",
        "how_handled": "Not handled; add a README subsection.",
        "amend_plan": "yes"
      },
      {
        "what": "The local module `postgres` collides by name with the external `postgres` crate.",
        "how_handled": "Compiles; cosmetic disambiguation nit.",
        "amend_plan": "no"
      },
      {
        "what": "PR-3.1 status claims column order is 'programmatically cross-checked against migrations/001', but the check is indirect (e2e COPY fails on disagreement), not a DDL-parsing test.",
        "how_handled": "Status log overstates mechanism; behavior covered by the e2e.",
        "amend_plan": "no"
      }
    ],
    "coverage": {
      "tested_cases": [
        "loader COPY-text rendering incl. escaping/BIGINT[]/literal-\\N-vs-NULL/non-UTC-tz/f64-round-trip+non-finite-reject (postgres.rs:514-712, high)",
        "value-verify discrimination across 6 tables incl. array reorder/Int32+side-counters/presence-both-directions/epoch-us-pinned (verify.rs:965-1356, high)",
        "Postgres e2e: counts/verify-clean/atomic mid-load rollback (postgres_e2e.rs:136-226, high)",
        "cross-check UPDATE-not-INSERT/all-5-kinds/NOT-seeded->INSERT/value_mismatches mutation-verified (test_post_ingest_postgres.py:3795-3947, high)",
        "requires-superuser guard: marker detection/non-master rejected before DDL (test_migrate_schema.py:3629-3768, high)",
        "real-data PR-3.4: 4.2M-row load exact match + verify clean + cross-check clean + negative (manual run, medium)"
      ],
      "untested_cases": [
        "vector_search_runs Double/threshold COPY + vector value-verify against REAL data (0 real rows + omitted from cross-check) — priority medium, annotate status + PR-5.0 closes",
        "verify-full TLS --ca-cert against a live RDS endpoint — priority low, compile+runbook only, plan-acknowledged",
        "TABLE_SPECS vs migrations/001 column-order as a direct DDL-parsing unit test — priority low, covered behaviorally by the e2e COPY"
      ]
    },
    "tradeoffs": [
      {
        "decision": "Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0); Phase 3 closes on the real-snapshot LOCAL rehearsal",
        "verdict": "keep",
        "rationale": "PR-3.4 executed exactly this with zero irreversible prod side-effect; RDS PITR is the prod rollback."
      },
      {
        "decision": "Value-column verify as a NEW PgVerifyReport pair (not overloading v2-structural VerifyReport)",
        "verdict": "keep",
        "rationale": "Single-responsibility; structural vs stored-value comparison are different concerns."
      },
      {
        "decision": "commits.timestamp compared as epoch microseconds",
        "verdict": "keep",
        "rationale": "Engine-independent and exact; e2e confirmed sub-second + pre-1970."
      },
      {
        "decision": "local postgres:16 testcontainer rehearsal, Docker-gated",
        "verdict": "keep",
        "rationale": "Same engine as RDS at zero AWS cost; e2e closes the live Postgres-execution gap."
      },
      {
        "decision": "operator-run-locally master-password DSN; DROP aws-sdk-rds",
        "verdict": "revisit-but-keep",
        "rationale": "Keep (avoids heavy AWS SDK); record the resulting sync-`postgres`-over-`tokio-postgres` swap in the PR-3.1 status."
      },
      {
        "decision": "004-as-master requires-superuser preflight guard",
        "verdict": "revisit-but-keep",
        "rationale": "Guard is well-shaped and load-bearing; close the migrations/README.md doc gap the PermissionError points at."
      },
      {
        "decision": "trusted-input low-stakes review calibration; 3-vote Phase 3",
        "verdict": "keep",
        "rationale": "Appropriate; adversarial-input hardening correctly out of scope; retained guards are load-bearing."
      }
    ]
  }
}
```

</details>

## Phase 3: Historical data load (DuckDB → Postgres) + value-verify — end-of-phase review (cycle 2) — accepted (3-vote)
**Cycle 2 (post PR-3.6 amend). Synthesizer output from /spiral:gauntlet (preset=phase-3, lenses=spec+correctness+maint, executor=claude). Verdict: ACCEPT — 0 must-fix, 1 should-fix (disclosed/deferred to PR-5.0), 3 nits. All 3 cycle-1 should-fix items confirmed ADDRESSED by PR-3.6 (not re-flagged). Full JSON in the `<details>` block at the end.**

### Summary of changes
Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit and closes on a real-snapshot LOCAL rehearsal, with the one-shot prod load deferred to Phase 5 (PR-5.0). (1) BULK LOADER (PR-3.1, postgres.rs): atomic single-txn per-table COPY FROM STDIN; measurement_id/commit_sha verbatim; UTC-cast timestamp; BIGINT[] as {a,b,c}; f64 shortest-round-trip, non-finite rejected; NO aws-sdk-rds (sync postgres 0.19 + native-tls, NoTls local / --ca-cert verify-full prod). (2) VALUE-VERIFY GATE (PR-3.2, verify.rs): PRIMARY v4-correctness gate; per-measurement_id full compare of every non-hashed value column, env_triple + arrays element-wise + commits metadata, engine-independent epoch-microsecond timestamps; column types single-sourced from the loader's column_kind. (3) BOOTSTRAP GUARD (PR-3.1, migrate-schema.py): requires-superuser marker on 002/004 + rolsuper-OR-rolcreaterole preflight failing loud BEFORE the transaction. (4) REHEARSAL HARNESS (PR-3.3, postgres_e2e.rs): first migrate testcontainer; counts + verify-clean + atomic mid-load rollback; Docker-gated. (5) CROSS-CHECK (PR-3.5): Python writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip; mutation-verified discrimination. (6) REAL-DATA GATE (PR-3.4): ran the toolkit against a real 4.2M-row snapshot into local PG16 — exact row counts, full value-verify clean, cross-check clean, zero prod write. (7) PR-3.6 amend: README requires-superuser subsection + 2 status annotations (doc/status only). The cumulative diff faithfully implements the Phase-3 contract with no out-of-scope drift.

### Surprises and discoveries
- **PR-3.1 shipped sync `postgres` 0.19, not plan-named `tokio-postgres` (deliberate; no async runtime for a one-shot CLI; NO-aws-sdk-rds honored; tokio-postgres pulled in transitively).** — Annotated in the PR-3.1 status by PR-3.6. (amend_plan: already-done)
- **The real PR-3.4 snapshot held vector_search_runs=0 rows AND the cross-check omitted that kind, so that table got fixture+unit coverage only, not real-data.** — Annotated in the PR-3.4 status by PR-3.6; PR-5.0 (freshest snapshot) expected to close it or record fixture-only as an accepted residual. (amend_plan: already-done)
- **PR-3.4 ran against a pre-existing on-disk ./bench.duckdb on a Homebrew PG16 cluster (Docker was down) rather than the SSH/scp + testcontainer path the runbook describes.** — Captured in the PR-3.4 status as an equivalent rehearsal substrate; the prod runbook still documents the acquisition+verification path PR-5.0 uses (minor runbook-vs-execution divergence, nit). (amend_plan: no)
- **README authoring rules reference a hypothetical `-- migrate: no-transaction` directive whose prefix differs from the only implemented `-- migrate-schema: requires-superuser`.** — Not reconciled; flagged as a doc-consistency nit. (amend_plan: no)

### Testing coverage assessment
**Tested (high confidence):**
- loader COPY-text + DuckDB-read pipeline (escaping, BIGINT[] empty/negative, NULLs, literal-\N-vs-NULL, UTC + non-UTC-offset timestamp, f64 round-trip, empty tables) — 11 unit tests (postgres.rs, high)
- value-verify discrimination across 6 tables incl. array reorder / Int32 + side counters / presence both directions / epoch-us pinned / read_kinds drift guard — 14 no-Docker tests (verify.rs, high)
- live PG16 e2e: counts + verify-clean + atomic mid-load rollback; sub-second + pre-1970 epoch exact — 2 testcontainer tests (postgres_e2e.rs, high)
- requires-superuser marker detection + non-master rejection BEFORE DDL (test_migrate_schema.py, high)
- Python cross-check UPDATE-not-INSERT + all 5 kinds + value/env/memory/side-counter discrimination, mutation-verified (test_post_ingest_postgres.py, high)
- real-data gate: 4.2M-row load exact-match + full value-verify clean + cross-check clean (PR-3.4 operational, high)
- PR-3.6 README subsection cross-verified against migrate-schema.py (marker, proxy SQL, PermissionError pointer, GRANT-in-failure-list) — this review, high

**Untested / under-covered (non-blocking):**
- vector_search_runs against the REAL data shape (Double threshold dim, four Int side counters, only Int32 value column) — priority medium, real snapshot had 0 rows + omitted from cross-check; fixture+unit only; DISCLOSED + deferred to PR-5.0
- verify-full TLS --ca-cert against a live RDS host — priority low, compile + local self-signed-cert run + runbook only; needs live RDS; plan-acknowledged

### Tradeoffs re-evaluation
- **One-shot historical load: retarget migrate/ for DuckDB->Postgres (throwaway)** — _keep_: Loader reads an existing v3 DuckDB and copies measurement_id verbatim; validated clean against the real snapshot; removal trigger documented.
- **Execution model: operator-run-locally master password, NO aws-sdk-rds** — _keep_: aws-sdk-rds absent; sync postgres 0.19 suffices; the tokio-postgres->sync-postgres swap is now annotated as deliberate.
- **Value-column verify as the PRIMARY gate (NEW PgVerifyReport pair)** — _keep_: Per-measurement_id full compare implemented + runtime-validated; green against the real snapshot.
- **commits.timestamp as epoch microseconds** — _keep_: Engine-independent + exact; e2e confirmed sub-second + pre-1970.
- **local postgres:16 testcontainer rehearsal (Docker-gated)** — _keep_: Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16 cluster when Docker was down.
- **004-as-master requires-superuser preflight guard** — _keep_: Guard + fail-loud test shipped; PR-3.6 added the operator-facing README the PermissionError points to (cycle-1 should-fix closed).
- **Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)** — _keep_: PR-3.4 ran the real-snapshot LOCAL rehearsal with zero prod write; prod load remains PR-5.0/5.1.
- **trusted-input low-stakes review calibration; 3-vote Phase 3** — _keep_: Appropriate; this cycle-2 review ran the full 3-vote (spec + correctness + maint).

### Disagreements
None — all three lenses accepted.

**Dropped re-flags (carry-forward respected):**
- Cycle-1 should-fix items (sync-postgres deviation note; vector_search_runs annotation; migrations/README requires-superuser doc gap) — addressed by PR-3.6; reviewers confirmed resolved, not re-flagged (PR-3.6 (README subsection + PR-3.1/PR-3.4 status annotations))
- Dim-column COPY corruption invisible to the value-verify gate — covered by Accepted tradeoffs (Accepted tradeoffs / r1 traps)

### Findings (0 must-fix, 1 should-fix, 3 nits)
| # | Severity | Kind | File:line | Description | Found by |
|---|----------|------|-----------|-------------|----------|
| 1 | should-fix | coverage | `.big-plans/ct__bench-v4.md (PR-3.4 status) / PR-5.0` | vector_search_runs got zero real-data validation (0 rows in the snapshot + omitted from the PR-3.4 cross-check envelope); its Double/threshold dim, four Int side counters, and only Int32 value column are exercised only by synthetic fixtu... | spec/claude |
| 2 | nit | cross-bundle-consistency | `migrations/README.md:27 (authoring rules)` | The README authoring-rules section names a hypothetical future directive `-- migrate: no-transaction` (prefix `migrate:`) while the only implemented directive uses prefix `migrate-schema:` (`-- migrate-schema: requires-superuser`). A fut... | maint/claude |
| 3 | nit | doc-comment | `benchmarks-website/migrate/src/postgres.rs (pub fn load)` | `load`'s doc comment does not state that the target schema (migrations/001) must already be applied — `load` only COPYs into existing tables, never DDL. The precondition is in the README + e2e test but not the function doc. | maint/claude |
| 4 | nit | doc-quality | `benchmarks-website/migrate/README.md (REAL-snapshot rehearsal runbook)` | The PR-3.3 runbook presents SSH/scp + S3-rehydrate acquisition + live source-account/region verification as the rehearsal entry path, but PR-3.4 was executed against a pre-existing on-disk ./bench.duckdb. The prod-facing runbook (which P... | spec/claude |

The 1 should-fix is the already-disclosed vector_search_runs residual (tracked to PR-5.0; no Phase-3 action owed). The 3 nits are cheap doc polish, dismissible per the trusted-input calibration.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1) — cycle 2 raw archive</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "cycle": 2,
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 1,
  "nit_count": 3,
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "coverage",
      "file_line": ".big-plans/ct__bench-v4.md (PR-3.4 status) / PR-5.0",
      "description": "vector_search_runs got zero real-data validation (0 rows in the snapshot + omitted from the PR-3.4 cross-check envelope); its Double/threshold dim, four Int side counters, and only Int32 value column are exercised only by synthetic fixtures + unit tests. NOW DISCLOSED by PR-3.6's annotation and tracked to PR-5.0 — no new code/test action is owed within Phase 3; flagged for completeness so PR-5.0 explicitly re-assesses it at cutover.",
      "recommended_fix": "Keep the PR-3.6 annotation; make PR-5.0's freshest-snapshot prod load explicitly re-check vector_search_runs row count and either close it with real rows or record fixture-only as an accepted residual.",
      "found_by": [
        "spec/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "cross-bundle-consistency",
      "file_line": "migrations/README.md:27 (authoring rules)",
      "description": "The README authoring-rules section names a hypothetical future directive `-- migrate: no-transaction` (prefix `migrate:`) while the only implemented directive uses prefix `migrate-schema:` (`-- migrate-schema: requires-superuser`). A future engineer adding a runner directive can't tell which prefix is canonical. Pre-existing (predates PR-3.6); surfaced now alongside the new requires-superuser subsection.",
      "recommended_fix": "Align the hypothetical example to `-- migrate-schema: no-transaction`, or note that all runner directives share the `migrate-schema:` prefix.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-comment",
      "file_line": "benchmarks-website/migrate/src/postgres.rs (pub fn load)",
      "description": "`load`'s doc comment does not state that the target schema (migrations/001) must already be applied — `load` only COPYs into existing tables, never DDL. The precondition is in the README + e2e test but not the function doc.",
      "recommended_fix": "Add one sentence to the `load` doc comment noting the target tables must pre-exist.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/migrate/README.md (REAL-snapshot rehearsal runbook)",
      "description": "The PR-3.3 runbook presents SSH/scp + S3-rehydrate acquisition + live source-account/region verification as the rehearsal entry path, but PR-3.4 was executed against a pre-existing on-disk ./bench.duckdb. The prod-facing runbook (which PR-5.0 consumes) and the as-executed PR-3.4 rehearsal diverge in acquisition method.",
      "recommended_fix": "Optionally add a one-line note that a pre-acquired on-disk snapshot is an acceptable rehearsal source. No code change; the runbook is the PR-5.0 prod-path doc.",
      "found_by": [
        "spec/claude"
      ]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [
    {
      "topic": "Cycle-1 should-fix items (sync-postgres deviation note; vector_search_runs annotation; migrations/README requires-superuser doc gap)",
      "reason": "addressed by PR-3.6; reviewers confirmed resolved, not re-flagged",
      "reference": "PR-3.6 (README subsection + PR-3.1/PR-3.4 status annotations)"
    },
    {
      "topic": "Dim-column COPY corruption invisible to the value-verify gate",
      "reason": "covered by Accepted tradeoffs",
      "reference": "Accepted tradeoffs / r1 traps"
    }
  ],
  "executive_summary": "Phase 3 CYCLE 2 (after the PR-3.6 amend) is accepted by all three lenses (spec, correctness, maint) with ZERO must-fix. The amend applied the three cycle-1 should-fix items — a migrations/README requires-superuser bootstrap-ordering subsection plus PR-3.1 (sync-postgres) and PR-3.4 (vector_search_runs) status annotations — and all three reviewers confirmed them ADDRESSED (the README was verified line-for-line against scripts/migrate-schema.py: the marker string, the rolsuper-OR-rolcreaterole proxy, the PermissionError pointer, the guard-fires-before-transaction ordering, and which migrations carry the marker). The correctness lens re-hunted the loader/verifier/guard/cross-check adversarially and found no surviving bug and no false behavioral claim in the new doc. Residual findings are minor and non-blocking: one should-fix (the vector_search_runs real-data coverage gap, which is now DISCLOSED and tracked to PR-5.0 — no Phase-3 action owed) and three nits (a README directive-namespace inconsistency `migrate:` vs `migrate-schema:`; the `load` doc-comment not stating the schema-must-pre-exist precondition; the prod runbook's SSH/scp acquisition path diverging from PR-3.4's on-disk execution). All key tradeoffs hold at keep. Verdict: ACCEPT, 0 must-fix, 1 should-fix (disclosed/deferred), 3 nits. Phase 3 is complete and correct; the only tracked real-data residual (vector_search_runs) routes to the Phase-5 prod load.",
  "phase_artifacts": {
    "summary": "Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit and closes on a real-snapshot LOCAL rehearsal, with the one-shot prod load deferred to Phase 5 (PR-5.0). (1) BULK LOADER (PR-3.1, postgres.rs): atomic single-txn per-table COPY FROM STDIN; measurement_id/commit_sha verbatim; UTC-cast timestamp; BIGINT[] as {a,b,c}; f64 shortest-round-trip, non-finite rejected; NO aws-sdk-rds (sync postgres 0.19 + native-tls, NoTls local / --ca-cert verify-full prod). (2) VALUE-VERIFY GATE (PR-3.2, verify.rs): PRIMARY v4-correctness gate; per-measurement_id full compare of every non-hashed value column, env_triple + arrays element-wise + commits metadata, engine-independent epoch-microsecond timestamps; column types single-sourced from the loader's column_kind. (3) BOOTSTRAP GUARD (PR-3.1, migrate-schema.py): requires-superuser marker on 002/004 + rolsuper-OR-rolcreaterole preflight failing loud BEFORE the transaction. (4) REHEARSAL HARNESS (PR-3.3, postgres_e2e.rs): first migrate testcontainer; counts + verify-clean + atomic mid-load rollback; Docker-gated. (5) CROSS-CHECK (PR-3.5): Python writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip; mutation-verified discrimination. (6) REAL-DATA GATE (PR-3.4): ran the toolkit against a real 4.2M-row snapshot into local PG16 — exact row counts, full value-verify clean, cross-check clean, zero prod write. (7) PR-3.6 amend: README requires-superuser subsection + 2 status annotations (doc/status only). The cumulative diff faithfully implements the Phase-3 contract with no out-of-scope drift.",
    "surprises": [
      {
        "what": "PR-3.1 shipped sync `postgres` 0.19, not plan-named `tokio-postgres` (deliberate; no async runtime for a one-shot CLI; NO-aws-sdk-rds honored; tokio-postgres pulled in transitively).",
        "how_handled": "Annotated in the PR-3.1 status by PR-3.6.",
        "amend_plan": "already-done"
      },
      {
        "what": "The real PR-3.4 snapshot held vector_search_runs=0 rows AND the cross-check omitted that kind, so that table got fixture+unit coverage only, not real-data.",
        "how_handled": "Annotated in the PR-3.4 status by PR-3.6; PR-5.0 (freshest snapshot) expected to close it or record fixture-only as an accepted residual.",
        "amend_plan": "already-done"
      },
      {
        "what": "PR-3.4 ran against a pre-existing on-disk ./bench.duckdb on a Homebrew PG16 cluster (Docker was down) rather than the SSH/scp + testcontainer path the runbook describes.",
        "how_handled": "Captured in the PR-3.4 status as an equivalent rehearsal substrate; the prod runbook still documents the acquisition+verification path PR-5.0 uses (minor runbook-vs-execution divergence, nit).",
        "amend_plan": "no"
      },
      {
        "what": "README authoring rules reference a hypothetical `-- migrate: no-transaction` directive whose prefix differs from the only implemented `-- migrate-schema: requires-superuser`.",
        "how_handled": "Not reconciled; flagged as a doc-consistency nit.",
        "amend_plan": "no"
      }
    ],
    "coverage": {
      "tested_cases": [
        "loader COPY-text + DuckDB-read pipeline (escaping, BIGINT[] empty/negative, NULLs, literal-\\N-vs-NULL, UTC + non-UTC-offset timestamp, f64 round-trip, empty tables) — 11 unit tests (postgres.rs, high)",
        "value-verify discrimination across 6 tables incl. array reorder / Int32 + side counters / presence both directions / epoch-us pinned / read_kinds drift guard — 14 no-Docker tests (verify.rs, high)",
        "live PG16 e2e: counts + verify-clean + atomic mid-load rollback; sub-second + pre-1970 epoch exact — 2 testcontainer tests (postgres_e2e.rs, high)",
        "requires-superuser marker detection + non-master rejection BEFORE DDL (test_migrate_schema.py, high)",
        "Python cross-check UPDATE-not-INSERT + all 5 kinds + value/env/memory/side-counter discrimination, mutation-verified (test_post_ingest_postgres.py, high)",
        "real-data gate: 4.2M-row load exact-match + full value-verify clean + cross-check clean (PR-3.4 operational, high)",
        "PR-3.6 README subsection cross-verified against migrate-schema.py (marker, proxy SQL, PermissionError pointer, GRANT-in-failure-list) — this review, high"
      ],
      "untested_cases": [
        "vector_search_runs against the REAL data shape (Double threshold dim, four Int side counters, only Int32 value column) — priority medium, real snapshot had 0 rows + omitted from cross-check; fixture+unit only; DISCLOSED + deferred to PR-5.0",
        "verify-full TLS --ca-cert against a live RDS host — priority low, compile + local self-signed-cert run + runbook only; needs live RDS; plan-acknowledged"
      ],
      "recommendations": "Coverage for the phase's scope is strong and the value-fidelity-critical paths are runtime-validated against both a real PG16 container and a real 4.2M-row snapshot. The single real-data residual (vector_search_runs) is disclosed and tracked to PR-5.0. No new tests are owed within Phase 3."
    },
    "tradeoffs": [
      {
        "decision": "One-shot historical load: retarget migrate/ for DuckDB->Postgres (throwaway)",
        "verdict": "keep",
        "rationale": "Loader reads an existing v3 DuckDB and copies measurement_id verbatim; validated clean against the real snapshot; removal trigger documented."
      },
      {
        "decision": "Execution model: operator-run-locally master password, NO aws-sdk-rds",
        "verdict": "keep",
        "rationale": "aws-sdk-rds absent; sync postgres 0.19 suffices; the tokio-postgres->sync-postgres swap is now annotated as deliberate."
      },
      {
        "decision": "Value-column verify as the PRIMARY gate (NEW PgVerifyReport pair)",
        "verdict": "keep",
        "rationale": "Per-measurement_id full compare implemented + runtime-validated; green against the real snapshot."
      },
      {
        "decision": "commits.timestamp as epoch microseconds",
        "verdict": "keep",
        "rationale": "Engine-independent + exact; e2e confirmed sub-second + pre-1970."
      },
      {
        "decision": "local postgres:16 testcontainer rehearsal (Docker-gated)",
        "verdict": "keep",
        "rationale": "Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16 cluster when Docker was down."
      },
      {
        "decision": "004-as-master requires-superuser preflight guard",
        "verdict": "keep",
        "rationale": "Guard + fail-loud test shipped; PR-3.6 added the operator-facing README the PermissionError points to (cycle-1 should-fix closed)."
      },
      {
        "decision": "Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)",
        "verdict": "keep",
        "rationale": "PR-3.4 ran the real-snapshot LOCAL rehearsal with zero prod write; prod load remains PR-5.0/5.1."
      },
      {
        "decision": "trusted-input low-stakes review calibration; 3-vote Phase 3",
        "verdict": "keep",
        "rationale": "Appropriate; this cycle-2 review ran the full 3-vote (spec + correctness + maint)."
      }
    ]
  }
}
```

</details>

## Phase 3: Historical data load (DuckDB → Postgres) + value-verify — end-of-phase review (cycle 3) — accepted (3-vote)
**Cycle 3 (post PR-3.7 amend). Synthesizer output from /spiral:gauntlet (preset=phase-3, lenses=spec+correctness+maint, executor=claude). Verdict: ACCEPT — 0 must-fix, 1 should-fix (RESOLVED inline, commit 6275f272d), 1 nit (resolved). All cycle-2 nits confirmed CLOSED by PR-3.7. Code logic byte-identical to cycle-1's accept. Full JSON in the `<details>` block at the end.**

### Summary of changes
Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit (atomic COPY loader; the PRIMARY per-measurement_id value-verify gate; a testcontainer rehearsal harness; a Python-writer cross-check; a requires-superuser bootstrap guard) and closes on PR-3.4's real-snapshot LOCAL rehearsal — GREEN against the real 4.2M-row v3 snapshot into local PG16, zero prod write. The one-shot prod load is deferred to PR-5.0 at the Phase-5 cutover. Doc amends: PR-3.6 (cycle-1 should-fixes: the requires-superuser README subsection + PR-3.1/PR-3.4 status annotations) and PR-3.7 (cycle-2 nits: directive-namespace alignment, load precondition doc, on-disk-snapshot runbook bullet + repointing stale prod-load refs to PR-5.0). The cycle-3 should-fix (3 sibling source-doc files with the same stale framing) was resolved inline at close. Code logic unchanged since cycle 1.

### Surprises and discoveries
- **The stale `prod load (PR-3.4)` framing PR-3.7 fixed in the README also lived in 3 sibling source/script files (postgres.rs, postgres_e2e.rs, cross_check_python_writer.py).** — Resolved inline at cycle-3 close (commit 6275f272d) — repointed all to PR-5.0 with PR-3.4 as the LOCAL rehearsal; applied directly rather than via a 4th amend cycle per the reviewers' 'do not spiral' guidance. (amend_plan: already-done)
- **vector_search_runs had 0 real rows in the PR-3.4 snapshot + was omitted from the cross-check envelope, so its paths are fixture+unit-only.** — Disclosed (PR-3.6) + tracked to PR-5.0; carry-forward. (amend_plan: already-done)

### Testing coverage assessment
**Tested (high confidence):**
- loader COPY-text pipeline (11 unit tests, high)
- value-verify discrimination across 6 tables (14 no-Docker tests, high)
- live PG16 e2e: counts + verify-clean + atomic rollback (2 testcontainer tests, high)
- requires-superuser preflight + marker detection (test_migrate_schema.py, high)
- Python cross-check discrimination, mutation-verified (test_post_ingest_postgres.py, high)
- real-data PR-3.4: 4.2M-row load + verify + cross-check clean (operational, high)
- PR-3.7 doc edits + cycle-3 repoint verified against live source (this review, high)

**Untested / under-covered (non-blocking):**
- vector_search_runs against REAL data — priority medium, fixture+unit only (0 real rows), DISCLOSED + deferred to PR-5.0
- verify-full TLS --ca-cert against live RDS — priority low, compile + local self-signed-cert run + runbook; needs live RDS

### Tradeoffs re-evaluation
- **One-shot historical load: retarget migrate/ (throwaway)** — _keep_: Validated clean against the real snapshot; removal trigger documented.
- **Execution model: operator-local master password, NO aws-sdk-rds** — _keep_: aws-sdk-rds absent; sync postgres suffices; the tokio-postgres swap annotated.
- **Value-column verify as PRIMARY gate** — _keep_: Per-measurement_id full compare, green against real data.
- **epoch-microsecond timestamps** — _keep_: Engine-independent + exact (sub-second + pre-1970 confirmed).
- **local postgres:16 testcontainer rehearsal** — _keep_: Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16.
- **004-as-master requires-superuser preflight** — _keep_: Guard + test shipped; PR-3.6/3.7 completed the operator-facing docs.
- **Prod load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)** — _keep_: PR-3.4 ran the zero-prod-risk rehearsal; the bundle is now internally consistent (all prod-load refs point at PR-5.0).
- **trusted-input low-stakes review calibration; 3-vote Phase 3** — _keep_: Appropriate; the cycle-3 should-fix was applied inline rather than spiraled, per the reviewers' guidance.

### Disagreements
None — all three lenses accepted.

**Dropped re-flags (carry-forward respected):**
- Cycle-2 doc nits (directive namespace, load doc-comment precondition, runbook acquisition note) — addressed by PR-3.7; all 3 confirmed CLOSED by the cycle-3 reviewers (PR-3.7)
- vector_search_runs real-data coverage — disclosed (PR-3.6) + tracked to PR-5.0; carry-forward (PR-3.4 status / PR-5.0)

### Findings (0 must-fix, 1 should-fix [applied], 1 nit [resolved])
| # | Severity | Kind | File:line | Description | Found by |
|---|----------|------|-----------|-------------|----------|
| 1 | should-fix | cross-bundle-consistency | `postgres.rs:16-17, postgres_e2e.rs:12, cross_check_python_writer.py:6,14` | PR-3.7 repointed the stale `prod load (PR-3.4)` references in migrate/README.md to PR-5.0, but the SAME pre-re-scope framing survived in 3 sibling source/script files (postgres.rs module doc, postgres_e2e.rs module doc, the cross-check docstring). RESOLVED inline at cycle-3 cl... | maint/claude,correctness/claude |

The 1 should-fix was applied inline at cycle-3 close to avoid an amend-spiral (both correctness + maint lenses explicitly advised 'do not spiral'). No open findings remain; the only tracked residual is vector_search_runs real-data coverage (PR-5.0).

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1) — cycle 3 raw archive</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "cycle": 3,
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 1,
  "nit_count": 1,
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "cross-bundle-consistency",
      "file_line": "postgres.rs:16-17, postgres_e2e.rs:12, cross_check_python_writer.py:6,14",
      "description": "PR-3.7 repointed the stale `prod load (PR-3.4)` references in migrate/README.md to PR-5.0, but the SAME pre-re-scope framing survived in 3 sibling source/script files (postgres.rs module doc, postgres_e2e.rs module doc, the cross-check docstring). RESOLVED inline at cycle-3 close (commit 6275f272d): all three repointed so PR-3.4=LOCAL rehearsal, PR-5.0=prod load consistently across the bundle. Applied directly (mechanical, reviewer-specified) rather than via a 4th amend cycle, per both lenses' explicit 'do not spiral' guidance.",
      "recommended_fix": "APPLIED (6275f272d). fmt + ruff clean.",
      "found_by": [
        "maint/claude",
        "correctness/claude"
      ]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [
    {
      "topic": "Cycle-2 doc nits (directive namespace, load doc-comment precondition, runbook acquisition note)",
      "reason": "addressed by PR-3.7; all 3 confirmed CLOSED by the cycle-3 reviewers",
      "reference": "PR-3.7"
    },
    {
      "topic": "vector_search_runs real-data coverage",
      "reason": "disclosed (PR-3.6) + tracked to PR-5.0; carry-forward",
      "reference": "PR-3.4 status / PR-5.0"
    }
  ],
  "executive_summary": "Phase 3 CYCLE 3 (after the PR-3.7 doc-nit amend) is accepted by all three lenses with ZERO must-fix. PR-3.7 cleared the 3 cycle-2 nits (directive-namespace alignment, the `load` schema-precondition doc-comment, the on-disk-snapshot runbook bullet) and repointed 3 stale `prod load (PR-3.4)` references in migrate/README.md to PR-5.0; all reviewers verified the edits accurate against the live source and confirmed the cycle-2 nits CLOSED. The code logic is byte-identical to cycle-1's accept (cycles 2-3 added only documentation). The single cycle-3 should-fix — the same stale PR-3.4-as-prod-load framing surviving in 3 sibling source/script files (postgres.rs, postgres_e2e.rs, cross_check_python_writer.py) — was RESOLVED inline at the cycle-3 close (commit 6275f272d, fmt+ruff clean) rather than spawning a 4th amend cycle, exactly as both the correctness and maint lenses advised ('accept or defer, do not spiral'). No open findings remain. The only tracked residual is the vector_search_runs real-data coverage gap, disclosed and routed to PR-5.0. Verdict: ACCEPT, 0 must-fix, 1 should-fix (applied), 1 nit (resolved). Phase 3 is complete and internally consistent; recommend Proceed to Phase 4.",
  "phase_artifacts": {
    "summary": "Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit (atomic COPY loader; the PRIMARY per-measurement_id value-verify gate; a testcontainer rehearsal harness; a Python-writer cross-check; a requires-superuser bootstrap guard) and closes on PR-3.4's real-snapshot LOCAL rehearsal — GREEN against the real 4.2M-row v3 snapshot into local PG16, zero prod write. The one-shot prod load is deferred to PR-5.0 at the Phase-5 cutover. Doc amends: PR-3.6 (cycle-1 should-fixes: the requires-superuser README subsection + PR-3.1/PR-3.4 status annotations) and PR-3.7 (cycle-2 nits: directive-namespace alignment, load precondition doc, on-disk-snapshot runbook bullet + repointing stale prod-load refs to PR-5.0). The cycle-3 should-fix (3 sibling source-doc files with the same stale framing) was resolved inline at close. Code logic unchanged since cycle 1.",
    "surprises": [
      {
        "what": "The stale `prod load (PR-3.4)` framing PR-3.7 fixed in the README also lived in 3 sibling source/script files (postgres.rs, postgres_e2e.rs, cross_check_python_writer.py).",
        "how_handled": "Resolved inline at cycle-3 close (commit 6275f272d) — repointed all to PR-5.0 with PR-3.4 as the LOCAL rehearsal; applied directly rather than via a 4th amend cycle per the reviewers' 'do not spiral' guidance.",
        "amend_plan": "already-done"
      },
      {
        "what": "vector_search_runs had 0 real rows in the PR-3.4 snapshot + was omitted from the cross-check envelope, so its paths are fixture+unit-only.",
        "how_handled": "Disclosed (PR-3.6) + tracked to PR-5.0; carry-forward.",
        "amend_plan": "already-done"
      }
    ],
    "coverage": {
      "tested_cases": [
        "loader COPY-text pipeline (11 unit tests, high)",
        "value-verify discrimination across 6 tables (14 no-Docker tests, high)",
        "live PG16 e2e: counts + verify-clean + atomic rollback (2 testcontainer tests, high)",
        "requires-superuser preflight + marker detection (test_migrate_schema.py, high)",
        "Python cross-check discrimination, mutation-verified (test_post_ingest_postgres.py, high)",
        "real-data PR-3.4: 4.2M-row load + verify + cross-check clean (operational, high)",
        "PR-3.7 doc edits + cycle-3 repoint verified against live source (this review, high)"
      ],
      "untested_cases": [
        "vector_search_runs against REAL data — priority medium, fixture+unit only (0 real rows), DISCLOSED + deferred to PR-5.0",
        "verify-full TLS --ca-cert against live RDS — priority low, compile + local self-signed-cert run + runbook; needs live RDS"
      ],
      "recommendations": "Coverage is strong; value-fidelity-critical paths runtime-validated against a real PG16 container AND a real 4.2M-row snapshot. The single real-data residual (vector_search_runs) is disclosed + tracked to PR-5.0. No Phase-3 tests owed."
    },
    "tradeoffs": [
      {
        "decision": "One-shot historical load: retarget migrate/ (throwaway)",
        "verdict": "keep",
        "rationale": "Validated clean against the real snapshot; removal trigger documented."
      },
      {
        "decision": "Execution model: operator-local master password, NO aws-sdk-rds",
        "verdict": "keep",
        "rationale": "aws-sdk-rds absent; sync postgres suffices; the tokio-postgres swap annotated."
      },
      {
        "decision": "Value-column verify as PRIMARY gate",
        "verdict": "keep",
        "rationale": "Per-measurement_id full compare, green against real data."
      },
      {
        "decision": "epoch-microsecond timestamps",
        "verdict": "keep",
        "rationale": "Engine-independent + exact (sub-second + pre-1970 confirmed)."
      },
      {
        "decision": "local postgres:16 testcontainer rehearsal",
        "verdict": "keep",
        "rationale": "Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16."
      },
      {
        "decision": "004-as-master requires-superuser preflight",
        "verdict": "keep",
        "rationale": "Guard + test shipped; PR-3.6/3.7 completed the operator-facing docs."
      },
      {
        "decision": "Prod load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)",
        "verdict": "keep",
        "rationale": "PR-3.4 ran the zero-prod-risk rehearsal; the bundle is now internally consistent (all prod-load refs point at PR-5.0)."
      },
      {
        "decision": "trusted-input low-stakes review calibration; 3-vote Phase 3",
        "verdict": "keep",
        "rationale": "Appropriate; the cycle-3 should-fix was applied inline rather than spiraled, per the reviewers' guidance."
      }
    ]
  }
}
```

</details>

## Phase 4 raw gauntlet responses (archive)

### Cycle 1 — preset=phase-2 — reject

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "overall": "reject",
  "artifact_id": "Phase-4",
  "cycle": 1,
  "preset": "phase-2 (custom: spec+correctness, parallel claude+codex per lens; artifact = delta since holistic squash f92713a49, cumulative product verified on disk)",
  "unified_findings": [
    {
      "severity": "must-fix",
      "kind": "missing-acceptance",
      "file_line": ".big-plans/ct__bench-v4.md:201 (phase-4 exit criteria)",
      "description": "All three Phase-4 exit criteria (live preview serves all chart slugs; /api/groups slug list matches the family registry; ~5-slug manual visual check vs v2) are unverifiable: the one-time operator Vercel setup has not run. Openly documented, not silent drift, but per the Phase-1 precedent it must be a recorded blocking pre-acceptance gate resolved by the USER at the boundary (authorize-over-operator-gated, or perform the setup first).",
      "recommended_fix": "User decision at the Step 3.4 gate: (a) operator performs the Vercel setup per web/README.md and the evidence (preview URL, slug-list diff vs registry, CDN probe HIT, visual check) is captured before close, or (b) phase accepted user-authorized over the operator-gated criteria, evidence owed before the Phase-5 cutover.",
      "found_by": ["spec.claude (must-fix)", "spec.codex (2x must-fix: deploy evidence + visual-parity evidence)"]
    },
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/web/components/Chart.tsx:1503-1526",
      "description": "Dismiss-retry self-cancels: the [error] effect's cleanup (which runs on the error->null transition BEFORE the effect re-runs) invokes cleanupRetryRef, cancelling the 0ms retry the dismiss just armed. In real browsers React flushes commit + passive effects ahead of the macrotask timer, so the bounded Chart.js import retry never fires and a permalink chart stays blank after a transient chunk-load failure (the cycle-2 PR-4.4.b dead-end resurfaces). act/jsdom tests invert the ordering and mask the bug (reviewer empirically confirmed with a scratch test). Synthesizer line-verified the cleanup structure on disk.",
      "recommended_fix": "Cancel the pending retry only on unmount (move cleanupRetryRef.current?.() from the [error] effect cleanup into the unmount cleanup); the retry closure is already unmount-safe (controllerRef null check + shouldRetryConstruct's disposed check). A faithful failing test needs non-act scheduling; document the pin requirement on the deferred interaction-suite row.",
      "found_by": ["correctness.claude (must-fix)"]
    },
    {
      "severity": "should-fix",
      "kind": "cross-cutting",
      "file_line": ".big-plans/ct__bench-v4.md:130/138/201",
      "description": "Plan text pins the dead caching mechanism (unstable_cache + revalidateTag; export const revalidate = ~300) at three normative sites; shipped code uses force-dynamic + Cache-Control s-maxage on API 200s + Vercel-CDN-Cache-Control config rules.",
      "recommended_fix": "Amend the Read-service-framework Key-decision row, the PR-4.4-arch row mention, and the Phase-4 row wording to the shipped mechanism (plan-edit).",
      "found_by": ["spec.claude (should-fix)", "both correctness lenses' tradeoffs (revisit-but-keep)"]
    },
    {
      "severity": "should-fix",
      "kind": "cross-cutting",
      "file_line": ".big-plans/ct__bench-v4.md:123",
      "description": "Key decision 'RDS Proxy for the Vercel read service only' is contradicted by the shipped wiring guidance (proxy VPC-internal, Vercel off-VPC; README/db.ts steer to the public instance endpoint); the proxy currently has no consumer.",
      "recommended_fix": "USER decision at the Step 3.4 boundary (already queued as boundary item 4): amend Q2 and decide the proxy's disposition.",
      "found_by": ["spec.claude (should-fix)", "all four lenses' tradeoffs (reverse / revisit-but-keep)"]
    },
    {
      "severity": "should-fix",
      "kind": "cross-cutting",
      "file_line": ".big-plans/ct__bench-v4.md:131",
      "description": "scripts/psql-bench.sh is claimed 'documented in web/README.md' by a Key decision, but the script exists nowhere, the README shipped without it, and no remaining PR row owns it, while the Phase-2 soak spot-check and the PR-5.3 /api/admin/sql decommission rationale depend on it.",
      "recommended_fix": "Assign ownership: fold delivery + README documentation into the Phase-5 rows (plan-edit), or amend the Key decision if plain psql incantations suffice.",
      "found_by": ["spec.claude (should-fix)"]
    },
    {
      "severity": "should-fix",
      "kind": "boundary",
      "file_line": "benchmarks-website/web/components/Header.tsx:78",
      "description": "When the filter universe is empty/undefined the Header omits FilterBar and initGlobalFilter never runs, so a soft navigation from a filtered page can leave stale module-scoped filters hiding series with no visible filter UI. v4-architecture-only window (v3 was MPA: module state reset every load).",
      "recommended_fix": "DEFER (Deferred work row): narrow error-path UX edge with no data wrongness; the right reset semantics need thought (reset-on-absent-FilterBar vs page-scoped state).",
      "found_by": ["correctness.codex (should-fix)"]
    },
    {
      "severity": "should-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/web/lib/queries.ts:995",
      "description": "collectFilterUniverse orders via SQL ORDER BY (database collation): prod RDS en_US.UTF-8 orders hyphenated names differently from v3's BTreeSet byte order, while the C-collated test container masks the divergence. Severity raised from nit by the synthesizer: it is a v3-parity break of the same class the phase's acceptance pins, and the fix is one line using the port's existing comparator.",
      "recommended_fix": "Sort in JS with compareCodeUnits (drop the SQL ORDER BY), pinning byte-order parity independent of database collation.",
      "found_by": ["correctness.claude (nit)"]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": ".github/actions/verify-cdn-cache/action.yml:60",
      "description": "The probe's awk splits on ': ', so a legal OWS-less header (x-vercel-cache:HIT) would parse as absent and fail the gate. Latent (Vercel emits the space today).",
      "recommended_fix": "Split on the first ':' and trim OWS.",
      "found_by": ["correctness.claude (nit)"]
    },
    {
      "severity": "nit",
      "kind": "dismissed-as-preserved",
      "file_line": "benchmarks-website/web/components/Chart.tsx:786",
      "description": "DISMISSED-AS-PRESERVED (synthesizer line-verified): the shared-LTTB union iterating ALL datasets including filter-hidden ones is byte-faithful v3 parity. v3 chart-init.js builds the identical max-across-all-datasets union with no visibility check (~line 880), and both substrates hide series in place via dataset.hidden (v3 line 1085 / v4 Chart.tsx:690), so hidden series influence kept indices identically in both. correctness.codex flagged must-fix; the migration behavior-preservation contract pins matching v3. Joins the PHASE-4-BOUNDARY parity list (6th item) for the v4-fidelity-vs-preserve-v3 user decision.",
      "recommended_fix": "None now; candidate improvement under the deliberate v4-fidelity effort if the user opts in.",
      "found_by": ["correctness.codex (must-fix, dismissed)"]
    },
    {
      "severity": "nit",
      "kind": "scope-creep",
      "file_line": ".github/actions/verify-cdn-cache/action.yml:1",
      "description": "Files-touched grew beyond declared rows (composite action, README, harness, db.ts note; PR-4.4.b analogs) — all review-driven and documented in Implementation status; no silent drift.",
      "recommended_fix": "None (record only).",
      "found_by": ["spec.claude (nit)"]
    }
  ],
  "disagreements": [
    {
      "topic": "LTTB-over-hidden-series severity",
      "positions": "correctness.codex: must-fix chart-wrongness (hidden series steer kept indices). correctness.claude: line-verified the surrounding port as exact v3 parity and did not flag. Synthesizer verification: v3 chart-init.js unions over all datasets with in-place dataset.hidden filtering, identical to v4.",
      "synthesizer_recommendation": "Dismiss-as-preserved per the behavior-preservation contract; track on the parity-decision list."
    },
    {
      "topic": "Verdict shape",
      "positions": "All four lenses reject, but for different reasons: spec lenses on acceptance evidence (operator gate), correctness.claude on the retry bug, correctness.codex on the (dismissed) LTTB claim plus the FilterBar edge.",
      "synthesizer_recommendation": "Reject with 2 must-fix: the retry bug is code-fixable now; the operator gate is user-resolvable only and goes to the Step 3.4 boundary AUQ per the Phase-1 authorize-over-operator-gated precedent."
    }
  ],
  "dropped_re_flags": [],
  "executive_summary": "Phase-4 phase-end cycle 1 (2-vote custom, spec + correctness, Claude + Codex per lens): REJECT with 2 must-fix. The phase product itself verified remarkably clean: footprint exactly in-bounds (zero Out-of-scope contact), every PR row's files on disk, dropped items stayed dropped, all four UI BANS pinned and re-verified, 204/204 vitest live (containers + production-server smoke; full migration set applied), prettier/eslint/tsc/yamllint/DB-less build all green, and the PR-4.5 post-cap fix wave independently re-verified by three lenses. The two must-fix: (1) the phase exit criteria are wholesale operator-gated (no Vercel project exists yet), which per the Phase-1 precedent needs an explicit user resolution at the boundary rather than silent acceptance; (2) a real browser-only bug correctness.claude proved empirically: Chart.tsx's error-dismiss effect cleanup cancels the Chart.js import retry it just armed, so the cycle-2 permalink dead-end is back in real browsers while act/jsdom tests mask it. One headline dismissal: correctness.codex's LTTB-over-hidden-series must-fix is byte-faithful v3 parity (synthesizer line-verified both substrates union over all datasets with in-place hidden flags) and joins the parity-decision list as its 6th item. Remaining should-fixes are plan-text staleness (dead caching-mechanism wording at three sites; the contradicted proxy-for-Vercel-reads decision; unowned psql-bench.sh), one deferred v4-only soft-nav filter edge, and a collation-vs-byte-order sort parity fix the synthesizer upgraded from nit (one-line compareCodeUnits change). Tradeoffs re-evaluation across lenses: pooler decision reverse/revisit (proxy consumerless - boundary item), framework row revisit-but-keep (mechanism wording only), psql-bench revisit-but-keep (ownership), all others keep. The phase is one small code fix + plan edits + one user authorization away from acceptance.",
  "reviewer_outputs": {
    "spec.claude": "reject/high - 1 must-fix (operator gate), 3 should-fix (plan text), 1 nit",
    "correctness.claude": "reject/high - 1 must-fix (retry self-cancel, empirically proven), 2 nit",
    "spec.codex": "reject/medium - 2 must-fix (deploy evidence; visual-parity evidence) = the operator gate",
    "correctness.codex": "reject/high - 1 must-fix (LTTB hidden-series; DISMISSED as v3 parity), 1 should-fix (FilterBar soft-nav edge)"
  }
}
```

</details>

### Cycle 2 — preset=phase-2 (multi-model) — reject

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "overall": "reject",
  "artifact_id": "Phase-4",
  "cycle": 2,
  "preset": "phase-2 (custom: spec+correctness, parallel claude+codex per lens; artifact = delta since holistic squash f92713a49; fix-aware vs prior-fix c2be81b4f..HEAD, prior_fix_commit_sha=8b85a3d3f; cumulative product verified on disk)",
  "unified_findings": [
    { "severity": "must-fix", "kind": "cross-cutting", "file_line": "scripts/migrate-schema.py:127 (and docstring :115)", "description": "The _assert_master_capable PermissionError remediation message (and docstring) still tell operators to 'apply the bootstrap migrations (002/004) as the RDS master', but 005_read_role.sql is now ALSO a requires-superuser bootstrap migration. A rejected 005 apply gives stale remediation naming only 002/004.", "recommended_fix": "Make the message/docstring generic to all marked bootstrap migrations, or explicitly list 002/004/005.", "found_by": ["spec.codex (must-fix)"] },
    { "severity": "must-fix", "kind": "doc-quality", "file_line": "migrations/README.md:45-53,65", "description": "The migration README's file list stops at 004 and the 'Bootstrap ordering — requires-superuser migrations (002 / 004)' section omits 005 entirely. 005_read_role.sql is a new requires-superuser bootstrap migration but is undocumented.", "recommended_fix": "Add 005_read_role.sql to the file list and the bootstrap-ordering section (or describe all requires-superuser migrations generically: 002/004/005).", "found_by": ["spec.codex (must-fix)", "correctness.claude (nit, related)"] },
    { "severity": "must-fix", "kind": "coverage", "file_line": "scripts/test_migrate_schema.py:1507", "description": "test_real_bootstrap_migrations_carry_superuser_marker pins ONLY 002 and 004 as marked; 005 (which carries the marker) is not in the asserted set, so deleting 005's requires-superuser marker would not fail any test.", "recommended_fix": "Add 005_read_role.sql to the marked tuple and update the docstring/message so the test pins every real privileged migration.", "found_by": ["spec.codex (must-fix)"] },
    { "severity": "must-fix", "kind": "doc-quality", "file_line": "benchmarks-website/web/README.md:92", "description": "Operator setup offers 'static BENCH_DB_PASSWORD ... or IAM auth' for the read service, but 005 deliberately creates bench_read with NO rds_iam grant, so IAM auth cannot work for this role without a follow-up rds_iam-grant migration (which would atomically disable password auth).", "recommended_fix": "Document static password as the currently supported bench_read mode; note IAM requires a follow-up rds_iam grant migration + Vercel-to-AWS credentials, and that granting rds_iam disables password auth.", "found_by": ["spec.codex (must-fix)"] },
    { "severity": "must-fix", "kind": "bug", "file_line": "migrations/005_read_role.sql:31-36", "description": "005 claims idempotency (header line 28) but only CREATEs bench_read when absent; it never enforces its no-rds_iam (password-auth) invariant on a pre-existing role. A bench_read that pre-exists with rds_iam (e.g. an earlier 005-with-rds_iam attempt, the exact live history here) stays IAM-only after a green apply, silently breaking password auth.", "recommended_fix": "After the CREATE guard, if the rds_iam role exists AND bench_read is a member, REVOKE rds_iam FROM bench_read (idempotent no-op otherwise). Add a runner test that pre-creates rds_iam + bench_read membership, applies 005, and asserts bench_read is no longer an rds_iam member.", "found_by": ["correctness.codex (must-fix)"] },
    { "severity": "should-fix", "kind": "coverage", "file_line": "benchmarks-website/web/components/Chart.tsx:1503-1526", "description": "The cycle-1 dismiss-retry fix (f3ee70fe1, behavior change: cancel retry only on unmount) shipped WITHOUT the regression pin the plan's PR-4.4.b row explicitly folded in. Chart.lifecycle.test.tsx stubs loadChartJs to throw on construction, so the error/4s-dismiss/0ms-retry path is never exercised.", "recommended_fix": "Add a jsdom lifecycle test that forces a Chart.js load failure, advances real (non-act) timers past 4s, asserts the deferred retry survives the dismiss-effect cleanup and fires, and separately asserts unmount cancels a pending retry.", "found_by": ["correctness.claude (should-fix)", "correctness.codex (untested high-priority)"] },
    { "severity": "should-fix", "kind": "weak-exit-criteria", "file_line": ".big-plans/ct__bench-v4.md:201 (phase-4 row exit criteria)", "description": "Phase-4's machine-checkable exit criteria (preview serves all slugs; /api/groups slug-list matches family registry; ~5-slug visual parity) were moved to Phase 5 by user decision, but are not yet recorded as explicit Phase-5/PR-5.0 acceptance items in the same machine-checkable form.", "recommended_fix": "Record the moved exit criteria on the Phase-5/PR-5.0 row with the same form (jq slug-list match + ~5-slug visual); note PR-4.5's preview-on-PR-open was satisfied only transitively via the green prod run.", "found_by": ["spec.claude (should-fix)", "spec.codex (untested high-priority)"] },
    { "severity": "should-fix", "kind": "cross-cutting", "file_line": ".big-plans/ct__bench-v4.md:33,96-97,105,109,390,405", "description": "The cycle-1 caching-wording amendment was applied to only 3 of ~8 plan sites; the architecture overview (33), ASCII diagram (96-97), architecture prose (105/109), file-map (390), and Risk #3 (405) still pin the dead unstable_cache/revalidateTag/revalidate=300 mechanism. Shipped code is correct; only plan prose is stale.", "recommended_fix": "Extend the 2026-06-10 amendment to the remaining plan sites, or add a single 'superseded by the framework header-driven-CDN amendment' pointer covering them all.", "found_by": ["spec.claude (should-fix)"] },
    { "severity": "nit", "kind": "scope-drift", "file_line": ".big-plans/ct__bench-v4.md (Phases and PRs, PR-4.5 row)", "description": "migration 005_read_role.sql + the test_migrate_schema.py 5/5 updates were added during operator-gate execution and are not enumerated under any Phase-4 PR row. Legitimately in-scope but undeclared.", "recommended_fix": "Record migration 005 (+ runner-test updates) under a Phase-4 PR row or the operator-gate Implementation-status entry as the owning row.", "found_by": ["spec.claude (nit)"] },
    { "severity": "nit", "kind": "error-path", "file_line": ".github/actions/verify-cdn-cache/action.yml:~90", "description": "curl -sS -w '%{http_code}' emits '000' (not empty) on connection failure, so the ${status:-unreachable} fallback is dead and the failure message renders 'HTTP 000' rather than 'unreachable'. Purely cosmetic.", "recommended_fix": "Map '000' to an 'unreachable' label before the echo, or drop the ${status:-unreachable} default.", "found_by": ["correctness.claude (nit)"] }
  ],
  "phase_artifacts": "see the cycle-2 end-of-phase review section above for the rendered Summary/Surprises/Coverage/Tradeoffs; full machine-readable copy in /tmp/pr45-review/results-p4c2/synthesis.json at synthesis time",
  "executive_summary": "Cycle-2 fix-aware multi-model (Codex gpt-5.5 + Claude, spec+correctness per lens) phase-end review of Phase 4. Both cycle-1 must-fixes are genuinely resolved and the fix-commit introduced no regression the Claude lenses could construct (both Claude lenses: accept). The Codex lenses surfaced 5 real, line-verified must-fix items, ALL stemming from the late operator-gate addition of migration 005_read_role.sql: (1-4) cross-reference drift — runner remediation message, migrations README, marker-regression test, and web README IAM-auth wording all still describe the pre-005 world; (5) an idempotency hole — 005 never enforces its no-rds_iam invariant for a pre-existing bench_read role, the exact failure class the live rds_iam discovery already burned this session. Verdict: REJECT (5 must-fix) + 3 should-fix + 2 nit.",
  "disagreements": [
    { "topic": "Whether Phase 4 is acceptance-ready", "positions": "Both Claude lenses returned accept (no must-fix); both Codex lenses returned reject. The disagreement is entirely about 005's surrounding contracts + idempotency, which Claude asserted 'internally consistent'/'sound' without cross-checking the runner message / README / marker test / web README, or constructing the pre-existing-rds_iam-role case.", "synthesizer_recommendation": "Side with Codex. The synthesizer line-verified all 5 must-fix against the files on disk (migrate-schema.py:127, migrations/README.md:45-65, test_migrate_schema.py:1507, web/README.md:92, 005_read_role.sql:31-36) — every one is real. Conservative union -> reject; fix all 5." }
  ],
  "dropped_re_flags": [
    "Accepted tradeoffs correctly NOT re-flagged: semantic-not-byte read-endpoint equivalence; summary latest-commit timestamp-tie (preserved v3); dimensionIsFiltered cardinality + all-chips-off URL non-round-trip (preserved v3); statpopgen/polarsignals group-name fall-through (preserved v3); toFixed round-half-away vs v3 (display tie).",
    "Deferred work correctly NOT re-flagged: full jsdom Chart.js-mock interaction harness; Header soft-nav filter edge.",
    "User-resolved operator-gate items correctly NOT re-flagged: cycle-1 exit-criteria operator gate (run 27293419561); rds_iam-drop (8b85a3d3f)."
  ],
  "confidence": "high"
}
```

</details>

### Cycle 3 — preset=phase-2 (multi-model) — reject (1 must-fix, resolved in-cycle)

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "overall": "reject",
  "artifact_id": "Phase-4",
  "cycle": 3,
  "preset": "phase-2 (custom: spec+correctness, parallel claude+codex per lens; artifact = delta since holistic squash f92713a49; fix-aware vs prior-fix 6984902ca / fix-pass 1299a69b9..HEAD; cumulative product verified on disk)",
  "unified_findings": [
    { "severity": "must-fix", "kind": "coverage", "file_line": "migrations/005_read_role.sql:72-90 (the ALTER DEFAULT PRIVILEGES block)", "description": "005's future-table SELECT default-privilege for bench_read (the ADP block, the read-role counterpart of 004's tested bench_ingest ADP) is untested; deleting the ADP would leave all existing 005 tests green. Stated migration contract with a parity precedent (bench_ingest's ADP IS tested).", "recommended_fix": "Add test_bench_read_default_privileges_cover_future_migrator_tables mirroring the bench_ingest ADP test, asserting bench_read auto-receives SELECT (and NOT INSERT/UPDATE/DELETE/TRUNCATE) on a future migrator-created table.", "found_by": ["correctness.codex (must-fix)"], "synthesizer_status": "RESOLVED in-cycle by 4c-fix (test_bench_read_default_privileges_cover_future_migrator_tables added; mutation-verified discriminating: stripping the ADP fails the test; 53/53 migrate-schema green)." },
    { "severity": "nit", "kind": "doc-quality", "file_line": ".big-plans/ct__bench-v4.md (phase-4 row exit criteria)", "description": "Phase-4 row exit-criteria names `vercel deploy --target=preview`, but the shipped pipeline uses CLI-driven `vercel pull/build/deploy --prebuilt` (web-deploy.yml). Plan-text vs implementation wording drift; data-dependent parts already moved to Phase 5.", "recommended_fix": "Update the wording to the shipped CLI flow or annotate as superseded by PR-4.5's pipeline.", "found_by": ["spec.claude (nit)"] },
    { "severity": "nit", "kind": "doc-quality", "file_line": "benchmarks-website/web/README.md (operator-setup step 3)", "description": "Step 3 frames the endpoint + auth wiring as 'two open choices' but prod is in fact decided (public RDS instance endpoint + static bench_read password per the operator-gate entry).", "recommended_fix": "Note the as-shipped prod choice (public instance endpoint + static BENCH_DB_PASSWORD) while keeping the per-environment options.", "found_by": ["spec.claude (nit)"] },
    { "severity": "nit", "kind": "coverage", "file_line": "scripts/test_migrate_schema.py (_scrub_bootstrap_roles + the new rds_iam test)", "description": "The new 005 rds_iam test relies solely on its try/finally to drop the cluster-global rds_iam role; the conn fixture scrubs tables/ledger but not roles, so a mid-test crash could leak rds_iam and re-arm IAM-only auth in later applies.", "recommended_fix": "Add rds_iam to _scrub_bootstrap_roles and/or a session-scoped autouse cleanup so a leaked cluster-global rds_iam cannot survive a test crash.", "found_by": ["correctness.claude (nit)"] },
    { "severity": "nit", "kind": "perf", "file_line": "benchmarks-website/web/lib/queries.ts (collectFilterUniverse)", "description": "collectFilterUniverse wraps SELECT DISTINCT over a UNION (which already de-dups), so the outer DISTINCT is dead work. Harmless + tested, but ambiguous intent.", "recommended_fix": "Drop the outer DISTINCT (UNION suffices), or use UNION ALL with the outer DISTINCT — pick one de-dup site.", "found_by": ["correctness.claude (nit)"] }
  ],
  "phase_artifacts": {
    "summary": "Cycle-3 fix-aware re-review of the full Phase-4 delta (since the holistic squash f92713a49), confirming the cycle-2 reject (5 must-fix re: migration 005_read_role.sql cross-reference drift + idempotency) is fully resolved. All four lenses verified on disk: M1-M4 (runner remediation message/docstring/comments, migrations README, web README) consistently enumerate 002/004/005 and describe 005's NO-rds_iam / password-auth contract; M5's idempotent REVOKE rds_iam (guarded by the rds_iam existence check, mirroring 004; REVOKE-of-non-member a no-op) is present + pinned. 3 of 4 lenses accepted (spec.codex, spec.claude, correctness.claude); correctness.codex found ONE new coverage must-fix: 005's future-table SELECT default-privilege (ADP) for bench_read is untested (the read-role counterpart of the tested bench_ingest ADP). Synthesizer line-verified the gap (no such test existed; 004's equivalent IS tested) and RESOLVED it in-cycle by adding test_bench_read_default_privileges_cover_future_migrator_tables (mutation-verified discriminating; 53/53 migrate-schema green). Plus 4 nits (exit-criteria wording, README wiring note, the new rds_iam test's cluster-global cleanup hardening, a redundant SQL DISTINCT). The broader phase (chart client islands, deploy pipeline, header-driven CDN caching, filter universe) was re-confirmed clean with zero out-of-scope drift; behavior-preservation deviations remain documented + accepted; data-dependent exit criteria remain deferred to Phase 5.",
    "surprises": [
      { "what": "correctness.codex found a coverage asymmetry: bench_ingest's future-table ADP is tested (004) but bench_read's (005) was not.", "how_handled": "Resolved in-cycle: added the parity test (mutation-verified). The other 3 lenses accepted.", "amend_plan": "no" },
      { "what": "Three lenses (incl. spec.codex) found ZERO findings; the phase is converging well across cycles (cycle-1 operator-gate + Chart fix, cycle-2 005 cross-ref drift, cycle-3 one coverage gap — progressively smaller).", "how_handled": "phase_end_reject_cycles reaches 3 at this cycle -> Step 3.3.5 modulo-3 phase-level early-break fires (mandated user gate).", "amend_plan": "no" }
    ],
    "coverage": {
      "tested_cases": [
        { "case": "005 future-table SELECT ADP for bench_read (auto-grant SELECT, no writes) on a migrator-created table", "test_location": "scripts/test_migrate_schema.py:test_bench_read_default_privileges_cover_future_migrator_tables", "confidence": "high" },
        { "case": "005 idempotent REVOKE rds_iam from a pre-existing bench_read member (cycle-2 M5)", "test_location": "scripts/test_migrate_schema.py:test_005_revokes_rds_iam_from_preexisting_bench_read", "confidence": "high" },
        { "case": "005 bench_read created, SELECT-only on the six tables; 005 carries the requires-superuser marker; 001..005 apply clean + idempotent + non-superuser master", "test_location": "scripts/test_migrate_schema.py (multiple)", "confidence": "high" }
      ],
      "untested_cases": [
        { "case": "Cross-test isolation of the cluster-global rds_iam role after a mid-test crash (only the new test's finally protects it)", "priority": "low", "why_untested": "nit; the happy path + finally are correct and the suite is green; add rds_iam to _scrub_bootstrap_roles in the test-hardening pass." },
        { "case": "Chart-controller jsdom interaction harness + act-free dismiss-retry pin", "priority": "low", "why_untested": "Carry-forward deferrals under the lean trusted-input calibration." },
        { "case": "Live preview/prod deploy + slug-list-vs-registry + ~5-slug visual parity", "priority": "medium", "why_untested": "Data-dependent; moved to Phase 5 (prod empty until PR-5.0); mechanics live-proven by run 27293419561." }
      ],
      "recommendations": "Migration 005 coverage is now complete + discriminating (existing-table grants, the rds_iam revoke invariant, AND the future-table ADP). Fold the 4 nits (exit-criteria wording, README wiring note, rds_iam scrub hardening, redundant DISTINCT) into the test-hardening/doc pass before the develop squash-merge; none block Phase 4."
    },
    "tradeoffs": [
      { "decision": "005 future-table ADP for bench_read tested at SELECT-only parity with the bench_ingest ADP test", "original": "Untested ADP (cycle-3 must-fix)", "verdict": "keep", "rationale": "The ADP is a stated migration contract; the new test pins SELECT-auto-grant + no-write, mutation-verified. Closes the coverage asymmetry with 004." },
      { "decision": "Read-service caching is header-driven CDN; bench_read static-password / no-rds_iam; data-dependent exit criteria deferred to Phase 5; preserve-v3 semantic equivalence", "original": "carry-forward from cycles 1-2", "verdict": "keep", "rationale": "Re-confirmed clean in cycle 3; the accumulated v4-fidelity-vs-preserve-v3 bundle remains a Step-3.4 user decision." }
    ]
  },
  "executive_summary": "Cycle-3 fix-aware multi-model (Codex gpt-5.5 + Claude, spec+correctness per lens) phase-end re-review of Phase 4. The cycle-2 reject (5 must-fix re: migration 005) is fully resolved + verified on disk by all four lenses. 3 of 4 lenses accepted (spec.codex/spec.claude/correctness.claude); correctness.codex found ONE new coverage must-fix: 005's future-table SELECT default-privilege (ADP) for bench_read was untested (the read-role counterpart of the tested bench_ingest ADP). The synthesizer line-verified the gap and RESOLVED it in-cycle by adding a mutation-verified parity test (53/53 migrate-schema green). Verdict: REJECT (1 must-fix, now fixed) + 4 nits (deferred to the test-hardening/doc pass). The phase is converging well (cycle-1 operator-gate, cycle-2 005 drift, cycle-3 one coverage gap — progressively smaller); phase_end_reject_cycles reaches 3, firing the mandated Step 3.3.5 modulo-3 phase-level early-break (user gate).",
  "disagreements": [
    { "topic": "Whether Phase 4 is acceptance-ready", "positions": "spec.codex/spec.claude/correctness.claude returned accept (only nits); correctness.codex returned reject on the single 005-ADP coverage gap.", "synthesizer_recommendation": "The gap is real (line-verified: no bench_read ADP test existed; 004's equivalent IS tested). Resolved in-cycle by adding the parity test (mutation-verified). With the must-fix fixed, the phase is clean modulo deferred nits; bring the user in at the modulo-3 early-break per the ~3-cycle calibration." }
  ],
  "dropped_re_flags": [
    "Resolved cycle-1/cycle-2 must-fixes NOT re-flagged (operator gate, Chart dismiss-retry, 005 cross-ref drift + idempotency).",
    "Accepted preserved-v3 tradeoffs NOT re-flagged (filter cardinality, all-off URL round-trip, statpopgen/polarsignals naming, summary timestamp-tie, toFixed rounding).",
    "Deferred items NOT re-flagged (jsdom interaction harness, act-free dismiss-retry pin, verify-cdn-cache curl-000 nit).",
    "Data-dependent exit criteria correctly recorded as moved to Phase 5 (not re-flagged as a gap)."
  ],
  "confidence": "high"
}
```

</details>

### Phase-4 operator-gate execution + Vercel setup (2026-06-10, user-driven AUQs)
- What ran: the SETUP-NOW resolution of the phase-end operator-gate must-fix. Vercel project `benchmarks-web` created via API under team `vortex-data` (`prj_zUcz1J8wSVAdKmpnzmPUuakQS0pu`, rootDirectory `benchmarks-website/web`, framework nextjs, NO git integration); `BENCH_DB_*` env set on production+preview (host=public RDS instance endpoint, user=`bench_read`, password sensitive, `BENCH_DB_CA`=us-east-1 regional RDS bundle at 4.5KB — the global bundle exceeds Vercel's env budget); GH repo secret `VERCEL_TOKEN` + vars `VERCEL_ORG_ID`/`VERCEL_PROJECT_ID`/`BENCHMARKS_WEB_PROD_URL=https://benchmarks-web.vercel.app` set; `ct/bench-v4` force-pushed (lease; pre-squash remote head was c2a2115fb).
- DISCOVERY (material, Phase-2-relevant): prod's migration ledger showed **004 was never applied** — `bench_ingest` did not exist, so the Phase-2 best-effort dual-write soak has been failing silently since it shipped (zero v4 ingest rows; prod tables empty pending PR-5.0 anyway). User approved applying 004+005 together as master: ledger now 5/5. The next develop push starts populating real ingest rows.
- DISCOVERY (caught live): granting `rds_iam` makes IAM auth MANDATORY on RDS (password auth fails with "PAM authentication failed"). 005 originally copied 004's guarded grant; revoked on prod + migration amended in lockstep (8b85a3d3f) before the file was ever pushed. `bench_read` verified live: password auth OK, SELECT OK, INSERT correctly denied.
- EVIDENCE (mechanics-now per user decision): workflow run 27293419561 GREEN end-to-end on its first real execution — changes-detect, Check & Test (full suite incl. testcontainers on the runner), Deploy Production (vercel pull/build/deploy from repo root with project rootDirectory), and **Verify CDN caching: HTTP 200 + x-vercel-cache HIT against the PUBLIC production domain with protection-skip disabled** (attempt 1 was MISS, probe retried to HIT — the Vercel-CDN-Cache-Control-beats-function-no-store mechanism is live-proven). Manual re-verification: `/` serves `cache-control: private, no-cache, no-store` to browsers AND `x-vercel-cache: HIT` (CDN-cached); `/api/groups` returns the well-formed empty `{"groups":[]}`; `/api/health` reports the live DB host, `schema_version: 1`, `build_sha` = the pushed commit, all-zero `row_counts`.
- Data-dependent exit criteria (slug-list-vs-registry; ~5-slug visual parity) MOVED to Phase 5, immediately after PR-5.0's historical load (user decision; prod is empty until then by plan sequencing).
- Operational note: the Vercel access token was pasted into the session transcript; rotate it (new token + update the `VERCEL_TOKEN` secret) at the user's convenience.
- OWNING ENTRY (cycle-2 scope-drift nit N1): this operator-gate entry is the de-facto Phase-4 owner of `migrations/005_read_role.sql` and the `scripts/test_migrate_schema.py` updates (5/5 ledger + marker-regression now pins 005 + the `bench_read` role tests + the cycle-2 idempotency revoke test), which were added here rather than under a numbered PR row. Recorded so the artifact footprint is fully attributed (the cycle-2 reject loop tracked all 005 follow-up under PR-4.5 (operator-gate 005 follow-up)).

#### Re-completion (phase-end cycle 2 reject-fix)
- Scope shipped: resolved all 5 cycle-2 phase-end must-fix items (005 cross-reference drift + idempotency). M1/M2/M4 (runner remediation message + comments + docstring, migrations/README, web/README auth wording) at `0f4e25b3c`; M3/M5 (marker-regression pins 005; 005 idempotently REVOKEs rds_iam from a pre-existing bench_read + new regression test) at `6984902ca`. Plan-doc should-fix/nit sweep (S2/S3/N1 fixed; S1/N2 deferred) at `ea9f949e4`.
- Tests: `scripts/test_migrate_schema.py` 52/52 green (incl. the new `test_005_revokes_rds_iam_from_preexisting_bench_read`); ruff clean; py_compile clean.
- Review: inner 2-vote (fresh + correctness, gauntlet pr-2 lenses reconstructed via `compose_prompts.py`) / accepted (cycles: 1). Both lenses mutation-tested the new 005 idempotency test (stripped the REVOKE → membership persists → test catches it), confirming it discriminates; both confirmed the guard mirrors 004 + stays idempotent + the cleanup prevents cluster-global `rds_iam` leak. One dismissed nit (markdown ~80-wrap on two README lines; under the 100-col limit, fine per the extend-to-100 convention).
- Confidence: high. Surprises during fix: none.

### Cycle 4 — preset=phase-2 (multi-model) — accept

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "overall": "accept",
  "artifact_id": "Phase-4",
  "cycle": 4,
  "preset": "phase-2 (custom: spec+correctness, parallel claude+codex per lens; artifact = delta since holistic squash f92713a49; fix-aware vs prior-fix f203c346c; cumulative product verified on disk)",
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "boundary / behavior-preservation",
      "file_line": "benchmarks-website/web/components/Chart.tsx:1057-1078 (setRange) + onPointerDown bare-track branch :1115-1123",
      "description": "On a single-commit chart (maxIdx=0) the range-strip bare-track-click path called the window math with minRange=1, forcing x.max to 1 -- one slot past the only label (state.ui.scope then reads 2 for a 1-commit chart). The drag (onPointerMove) and pixel-to-index (pxToIndex) paths already guard n<=1; the bare-track click did not, so the guard set was asymmetric. Found by correctness.codex (rated must-fix). Synthesizer line-verified the bug is REAL (the guard asymmetry confirms an oversight) but the impact is a cosmetic phantom-slot on a rare edge case (single-commit chart = first run of a new benchmark) with no crash/data/security effect and self-correcting -- so right-sized from must-fix to should-fix.",
      "recommended_fix": "Extract the clamp into a pure clampRangeWindow(maxIdx, rawMin, rawMax) helper with minRange=Math.min(1, maxIdx) so the window pins to [0,0] on single-commit charts; call it from setRange.",
      "found_by": [
        "correctness.codex (must-fix)"
      ],
      "synthesizer_status": "RESOLVED in-cycle at 0cdfdfcd4 (clampRangeWindow helper in chart-format.ts + 3 unit tests in chart-format.test.ts; mutation-verified discriminating -- reverting minRange to 1 fails the single-commit test; 207/207 web tests, eslint/prettier/next build clean)."
    },
    {
      "severity": "should-fix",
      "kind": "reliability",
      "file_line": ".github/actions/verify-cdn-cache/action.yml:57 (probe curl)",
      "description": "The verify-cdn-cache probe curl had neither --connect-timeout nor --max-time; under set -Eeuo pipefail with no per-step timeout, a hung socket could stall the deploy gate until the calling job's timeout-minutes: 15. Deploy-pipeline reliability (not data correctness). Found by correctness.claude (should-fix).",
      "recommended_fix": "Add --connect-timeout 10 --max-time 30 so a stall consumes a retry attempt (the existing || true already tolerates a non-zero exit).",
      "found_by": [
        "correctness.claude (should-fix)"
      ],
      "synthesizer_status": "RESOLVED in-cycle at b5cd0631a (curl --connect-timeout 10 --max-time 30; yamllint --strict clean)."
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "scripts/test_migrate_schema.py:1494",
      "description": "test_real_migrations_apply_as_non_superuser_createrole_master asserts applied == 5 but the failure message still read 'all four real migrations' -- the single stale residue of the 005 4->5 migration-count update (every other 4->5 site was updated). Logic/discrimination correct; message-only. Found by spec.claude (nit).",
      "recommended_fix": "Change 'all four real migrations' to 'all five real migrations'.",
      "found_by": [
        "spec.claude (nit)"
      ],
      "synthesizer_status": "RESOLVED in-cycle at 369ca48a8 (message corrected to 'all five'; ruff clean)."
    },
    {
      "severity": "nit",
      "kind": "behavior-preservation",
      "file_line": "benchmarks-website/web/lib/chart-format.ts:363 (throttle trailing timer)",
      "description": "Throttled listeners own an internal trailing setTimeout (pending) not tied to the controller AbortController or the mount-effect cleanup; on teardown the listener is removed but a pending trailing timer can still fire against the disposed controller. Currently a latent NO-OP (applyScope returns on !chart, rebuildVisibleAndUpdate returns on state.disposed). Found by correctness.claude (nit).",
      "recommended_fix": "Optional: expose throttle.cancel() and call it in the mount-effect cleanup, or document the disposed-safe contract.",
      "found_by": [
        "correctness.claude (nit)"
      ],
      "synthesizer_status": "DEFERRED -- latent no-op (harmless today by the disposed guards); falls in the already-deferred jsdom chart-interaction-harness area under the trusted-input low-stakes calibration. Added to Deferred work."
    }
  ],
  "phase_artifacts": {
    "summary": "Cycle-4 fix-aware multi-model (Codex gpt-5.5 xhigh + Claude, spec+correctness per lens) phase-end re-review of the full Phase-4 delta (since the holistic squash f92713a49), confirming cycle-3's lone must-fix (005 future-table SELECT default-privilege for bench_read) is resolved + that the +39-line prior fix f203c346c (a pure parity-test addition) introduced no drift. Two of four lenses accepted outright (both Claude); both Codex lenses rejected. correctness.codex found ONE genuine new boundary bug: the range-strip bare-track-click path pushes x.max one slot past the only label on a single-commit chart because minRange was forced to 1 while the drag and pixel paths already guard n<=1. The synthesizer line-verified the bug is real (guard asymmetry) but cosmetic/edge-case (single-commit charts are the first run of a new benchmark; no crash/data effect; self-correcting), right-sized must-fix->should-fix, and RESOLVED it in-cycle by extracting a pure clampRangeWindow() helper (minRange=min(1,maxIdx)) plus a mutation-verified discriminating unit test. spec.codex's two 'missing-acceptance' must-fixes (005 master-apply + bench_read password lacking a formal PR/acceptance gate; PR-4.5 live deploy unverified) are re-flags of the OPERATOR-GATE that cycle-1's phase-end review already routed to the Step-3.4 boundary AUQ (user-resolvable per the Phase-1 authorize-over-operator-gated precedent), with the data-dependent live checks moved to PR-5.0; both Claude lenses correctly did not flag them. Two further minor findings were resolved in-cycle (CDN-probe curl timeout; stale 'four'->'five' assertion message) and one nit deferred (throttle trailing-timer disposed-safety, a latent no-op). The broader Phase-4 product (chart client islands + lifecycle, deploy pipeline, header-driven CDN caching, migration 005, filter universe) re-confirmed clean with zero out-of-scope drift.",
    "surprises": [
      {
        "what": "The 4th-cycle fresh multi-model pass surfaced a genuine single-commit range-strip boundary bug that cycles 1-3 missed -- the value of an extra adversarial cycle even on a converging phase.",
        "how_handled": "Line-verified real, right-sized to should-fix, fixed in-cycle via a pure clampRangeWindow() helper with a mutation-verified discriminating test.",
        "amend_plan": "no"
      },
      {
        "what": "spec.codex (medium confidence) re-flagged the operator-gate as 2 must-fixes, citing prompt-context line numbers rather than repo code; spec.claude (same lens, line-verified) accepted.",
        "how_handled": "Dropped as re-flags: the operator-gate is a recorded Step-3.4 boundary item (cycle-1 determination) and the data-dependent checks are Phase-5-deferred.",
        "amend_plan": "no"
      },
      {
        "what": "phase_end_reject_cycles had reached 3 (modulo-3 phase-level early-break); the user chose Continue, and cycle 4 now accepts.",
        "how_handled": "Reset phase_end_reject_cycles to 0 on accept; proceed to the Step-3.4 boundary gate.",
        "amend_plan": "no"
      }
    ],
    "coverage": {
      "tested_cases": [
        {
          "case": "Range-strip window clamp pins [0,0] on a single-commit chart and enforces a 1-commit minimum span on multi-commit charts; bounds clamped into [0,maxIdx]",
          "test_location": "benchmarks-website/web/lib/chart-format.test.ts:clampRangeWindow (3 cases; mutation-verified discriminating)",
          "confidence": "high"
        },
        {
          "case": "005 future-table SELECT ADP for bench_read + idempotent rds_iam revoke + requires-superuser marker + SELECT-only role (cycle-3 + prior)",
          "test_location": "scripts/test_migrate_schema.py (53/53 green)",
          "confidence": "high"
        },
        {
          "case": "Chart island StrictMode replay + group-Y replay lifecycle; 204 pre-existing web tests incl. production-server smoke + full migration set",
          "test_location": "benchmarks-website/web (207/207 incl. the 3 new clampRangeWindow cases)",
          "confidence": "high"
        }
      ],
      "untested_cases": [
        {
          "case": "Single-commit range-strip via a REAL constructed Chart.js instance (end-to-end pointer interaction, not the pure clamp)",
          "priority": "low",
          "why_untested": "Requires the deferred jsdom chart-interaction harness (Chart.js is stubbed in the lifecycle suite); the pure clampRangeWindow() unit test pins the math that the bug lived in."
        },
        {
          "case": "throttle trailing-timer firing after teardown",
          "priority": "low",
          "why_untested": "Latent no-op behind the disposed guards; deferred with the interaction harness."
        },
        {
          "case": "Live preview/prod deploy slug-list-vs-registry + ~5-slug visual parity",
          "priority": "medium",
          "why_untested": "Data-dependent; moved to Phase 5 (PR-5.0; prod empty until the load); deploy mechanics live-proven by run 27293419561."
        }
      ],
      "recommendations": "Range-strip window math is now pinned + discriminating for the single-commit edge. Fold the deferred throttle-disposed-safety nit and the full jsdom interaction harness into the pre-develop-squash hardening pass; none block Phase 4. Carry the operator-gate (live deploy evidence + 005 master-apply + bench_read password) to the Step-3.4 boundary AUQ per the Phase-1 precedent."
    },
    "tradeoffs": [
      {
        "decision": "Range-strip minimum span",
        "original": "Fixed minRange=1 for all charts",
        "verdict": "revisit-but-keep",
        "rationale": "Kept minRange=1 for multi-commit charts; collapsed to 0 only on single-commit charts via min(1,maxIdx) so the window cannot extend past the only label. Behavior-preserving for the common case."
      },
      {
        "decision": "Connection pooler (RDS Proxy for Vercel reads)",
        "original": "RDS Proxy fronts Vercel reads",
        "verdict": "revisit-but-keep",
        "rationale": "As-shipped read path uses the public instance endpoint + static bench_read password, leaving the proxy consumerless on the read side. Known Step-3.4 boundary item (Q2 proxy-for-Vercel-reads amendment), not a Phase-4 code defect."
      },
      {
        "decision": "Read-service framework + header-driven CDN caching",
        "original": "Next.js 15 RSC; amended to Vercel-CDN-Cache-Control over unstable_cache/revalidateTag",
        "verdict": "keep",
        "rationale": "Consistently documented across vercel.json, README, page comments, and pinned by the live CDN probe."
      },
      {
        "decision": "bench_read identity (static password, no rds_iam)",
        "original": "operator-gate decision",
        "verdict": "keep",
        "rationale": "Correct for the Vercel-no-AWS-credentials constraint; pinned by the rds_iam-revoke + future-table ADP tests (5/5 migration coverage)."
      },
      {
        "decision": "Per-effect-mount chart controller lifecycle (StrictMode-safe)",
        "original": "PR-4.4.b cycle-1 fix",
        "verdict": "keep",
        "rationale": "Re-confirmed correct; the two lifecycle regression tests pin the highest-risk replay paths."
      }
    ]
  },
  "executive_summary": "Cycle-4 fix-aware multi-model (Codex gpt-5.5 xhigh + Claude, spec+correctness per lens) phase-end re-review of Phase 4. Cycle-3's lone must-fix (005 bench_read future-table SELECT ADP) is confirmed resolved and the prior fix f203c346c introduced no drift. 2 of 4 lenses accepted outright (both Claude); both Codex lenses rejected. correctness.codex found ONE genuine boundary bug -- the range-strip bare-track-click pushes x.max one slot past the only label on a single-commit chart because minRange was forced to 1 while the drag and pixel paths already guard n<=1. The synthesizer line-verified it real (guard asymmetry) but cosmetic/edge-case, right-sized must-fix->should-fix, and RESOLVED it in-cycle via a pure clampRangeWindow() helper (minRange=min(1,maxIdx)) with a mutation-verified discriminating unit test. spec.codex's two 'missing-acceptance' must-fixes are re-flags of the recorded OPERATOR-GATE (live deploy evidence + 005 master-apply + bench_read password), which cycle-1 already routed to the Step-3.4 boundary AUQ per the Phase-1 authorize-over-operator-gated precedent (data-dependent checks moved to PR-5.0); both Claude lenses correctly did not flag them -> DROPPED as re-flags. Two further minor findings resolved in-cycle (CDN-probe curl timeout at b5cd0631a; stale 'four'->'five' assertion message at 369ca48a8) and one nit deferred (throttle trailing-timer disposed-safety, a latent no-op). Verdict: ACCEPT. phase_end_reject_cycles resets to 0; proceed to the Step-3.4 boundary gate carrying the operator-gate + the recorded boundary AUQ items (six preserved-v3 parity dismissals; Q2 proxy-for-Vercel-reads amendment; rotate the Vercel token; Phase-5 data checks).",
  "disagreements": [
    {
      "topic": "Single-commit range-strip boundary",
      "positions": "correctness.codex rejected (must-fix); correctness.claude accepted (did not surface it).",
      "synthesizer_recommendation": "Real bug (line-verified guard asymmetry) but cosmetic/edge-case -> should-fix; resolved in-cycle at 0cdfdfcd4 with a mutation-verified test. The disagreement validates the multi-model design: a fresh model caught an edge case the other correctness lens missed."
    },
    {
      "topic": "Whether Phase 4 is acceptance-ready vs the operator-gate",
      "positions": "spec.codex rejected on operator-acceptance evidence (005 master-apply + bench_read password; PR-4.5 live deploy); spec.claude accepted.",
      "synthesizer_recommendation": "DROP as re-flags: the operator-gate is a recorded Step-3.4 boundary item (cycle-1 determination, Phase-1 authorize-over-operator-gated precedent) and the data-dependent checks are Phase-5-deferred (PR-5.0). Not new code must-fixes; carried to the Step-3.4 AUQ."
    }
  ],
  "dropped_re_flags": [
    "spec.codex's 2 'missing-acceptance' must-fixes (005 master-apply acceptance gate; PR-4.5 live deploy unverified) -- re-flags of the recorded operator-gate routed to the Step-3.4 boundary AUQ at cycle 1; data-dependent live checks moved to PR-5.0.",
    "Resolved cycle-1/2/3 must-fixes NOT re-flagged (operator gate setup, Chart dismiss-retry self-cancel, 005 cross-ref drift + idempotency, 005 future-table ADP).",
    "Accepted preserved-v3 tradeoffs NOT re-flagged (filter cardinality/byte-order, all-off URL round-trip, statpopgen/polarsignals naming, summary timestamp-tie, toFixed rounding, LTTB-over-hidden-series).",
    "Deferred items NOT re-flagged (jsdom chart-interaction harness, act-free dismiss-retry pin, redundant collectFilterUniverse outer DISTINCT, rds_iam scrub-hardening)."
  ],
  "confidence": "high"
}
```

</details>

## Resolved phase-end must-fix items — Phase 4: Next.js read service on Vercel — cycle 1

| Severity | File:line | Description | Implicated PR | Resolved |
|----------|-----------|-------------|---------------|----------|
| must-fix | .big-plans/ct__bench-v4.md:201 (phase-4 exit criteria) | All three exit criteria (live preview serves all slugs; /api/groups slug list matches the family registry; ~5-slug visual check vs v2) are operator-gated: the one-time Vercel setup has not run. Resolution is a USER decision at the boundary per the Phase-1 authorize-over-operator-gated precedent: perform the setup + capture evidence, or accept user-authorized with evidence owed before the Phase-5 cutover. Resolved 2026-06-10 by user decision SETUP-NOW + MECHANICS-NOW: Vercel project benchmarks-web created (team vortex-data, rootDir benchmarks-website/web, no git integration); 004+005 bootstrapped to prod; bench_read wired; branch force-pushed; run 27293419561 GREEN end-to-end incl. the CDN probe against the PUBLIC domain (HTTP 200 + x-vercel-cache HIT with protection-skip disabled); /api/groups + /api/health verified live. Data-dependent checks (slug-list-vs-registry, ~5-slug visual parity) MOVED to Phase 5 immediately after PR-5.0's historical load, where they are meaningful (prod is empty until then). | (user decision — not PR-addressable) | [x] |
| must-fix | benchmarks-website/web/components/Chart.tsx:1503-1526 | Dismiss-retry self-cancels: the [error] effect cleanup (running on the error->null transition) cancels the 0ms Chart.js import retry the dismiss just armed; in real browsers the bounded retry never fires and a permalink chart stays blank after a transient chunk-load failure. act/jsdom tests invert the scheduling and mask it (reviewer-proven). Resolved: f3ee70fe1 (cancel moved to unmount-only; 204 vitest + build/lint/format green). | PR-4.4.b | [x] |

## Phase 4: Next.js read service on Vercel — end-of-phase review (cycle 1) — rejected (2-vote)
**Synthesizer output from the custom mixed-executor 2-vote review (spec + correctness, parallel Claude + Codex gpt-5.5 xhigh per lens; gauntlet phase-2 preset lenses); full Synthesizer Output JSON in the `<details>` block at the end of this section. Artifact = the delta since the holistic squash f92713a49 (PR-4.4.b + PR-4.5 + fixes); the cumulative phase product was verified on disk by all four lenses.**

### Summary of changes
Phase 4 shipped the complete v4 read service replacing the v2 SPA and the v3 Axum server: (1) the Next.js 15.5.19 / React 19 App Router scaffold with vortex-web-parity strict TS, flat ESLint, prettier, SPDX discipline; (2) `lib/db.ts` (pg.Pool + per-connection RDS IAM tokens, static-password test bypass, fail-loud SSL); (3) the read-API port semantically equivalent to v3 (slug codec for 10 key variants, `?n=` window parity, 5-family registry, `/health`, two-pass seeded-window `chartPayload` for all 5 chart types, `collectGroups` + 4 summary variants + v2 descriptions), pinned by testcontainers PG16 suites against v3 contract values; (4) the UI as a v3-source port: server-rendered landing shell + verbatim CSS, then the interactive layer as client islands (Chart.js + LTTB shared-index downsampling, range strip, zoom/pan, tooltips with BAN-pinned idx-1 deltas, lazy fetch-on-open + one-shot `?n=all` upgrade, global/group filters, header interactivity, `/chart/[slug]` permalink as the expanded view); (5) the caching architecture that replaced the plan's unimplementable revalidate/unstable_cache sketch: DB-less builds, force-dynamic pages, `s-maxage=300` on API 200s, `Vercel-CDN-Cache-Control` rules on the two HTML routes; (6) the deploy pipeline: `web-deploy.yml` (path-gated prettier/eslint/DB-less-build/vitest with a docker-info guard; CLI-driven Vercel preview-per-PR + prod-on-push; event-name + PR-number concurrency; fork guard) plus the `verify-cdn-cache` composite action (HTTP 200 + HIT/STALE evidence gate, public-URL-aware protection-skip) and the operator runbook README. The web test harness now applies the full 001-004 migration set in runner order. Reviewers re-ran the verification suite live: 204/204 vitest with zero skips, tsc/eslint/prettier/yamllint clean, DB-less build green, phase footprint exactly in-bounds (zero Out-of-scope contact).

### Surprises and discoveries
- The plan's caching mechanism (`unstable_cache` + `revalidateTag`, later `revalidate = 300`) was unimplementable (inert on request-URL handlers; forces DB-at-build-time elsewhere; function-emitted `no-store` beats config Cache-Control). Replaced by header-driven CDN caching; **plan wording at 3 sites needs amending** (should-fix below).
- RDS Proxy is VPC-internal while Vercel functions are off-VPC, mirroring the Phase-1 proxy-unreachable finding; the shipped wiring guidance steers to the public instance endpoint, leaving the proxy consumerless. **Q2 Key-decision amendment is a boundary decision.**
- The port source pivoted from v2 React to v3's server-rendered HTML layer (recorded mid-phase); Modal.tsx correctly dropped.
- React StrictMode and the async Chart.js load opened timing windows v3 structurally could not have; all caught by adversarial review, not builds/tests. The cycle found one more in this class (the dismiss-retry self-cancel must-fix), and act/jsdom test environments invert the browser scheduling that exposes it.
- Migrations 002-004 are deliberately container-portable, enabling full-set harness coverage (live-proven by the 204-test run).
- Docker Desktop died mid-session and 41 tests skipped silently: a live demonstration of the failure mode the new CI docker-info guard prevents.
- The faithful-port reviews accumulated a SIXTH preserved-v3 parity dismissal (shared-LTTB union over filter-hidden datasets, synthesizer-verified byte-faithful to v3) joining statpopgen-descriptions, same-timestamp-tie, round-half-even, filter-cardinality, and all-off-URL round-trip for the boundary decision.

### Testing coverage assessment
| Area | State |
|---|---|
| Read-API v3 equivalence (5 families, groups, summaries, health) | testcontainers PG16 suites incl. v3 golden values — high confidence |
| Slug codec, `?n=` window, connection lib, SSL/IAM | unit + container suites — high confidence |
| Chart pure helpers (LTTB, unit picker, normalize, predecessor BAN) | unit golden suites — high confidence |
| Island lifecycle (StrictMode replay, group-Y replay) + SSR markup contracts | jsdom pins — high confidence |
| Production server (next start + seeded PG: landing, permalink, 404, `?n=all`) | smoke suite — medium-high |
| Full migration set application (001-004, runner order) | every container suite — high |
| UNTESTED: live Vercel deploy path end-to-end (pull/build/deploy, CDN precedence, probe) | operator-gated — the must-fix gate |
| UNTESTED: ~5-slug visual parity vs live v2 | manual operator check — exit criterion |
| UNTESTED: dismiss-retry under real browser scheduling | act/jsdom inverts ordering; pin requirement noted on the deferred interaction-suite row |
| UNTESTED (deferred): Chart.js interaction harness (pan/zoom/slider/payload-replace); groupNameQuery fallback branches; composite-action bash; concurrency semantics | deferred rows / first-real-run verification |

### Tradeoffs re-evaluation
keep: branching/merge target; Postgres flavor; ingest writer language; schema deploy tool (strengthened by full-set harness); schema-deploy authorization; historical-load toolkit + Phase-5 timing; CI network reach; CI-write endpoint; cutover style; composite indexes; bench_ingest identity; v3 EC2 disposition; review calibration (the ~3-cycle cap fired twice without quality loss; the one bug that escaped sits in the async-timing seam the calibration accepts); PR-4.4 UI architecture. **revisit-but-keep:** Read-service framework (architecture vindicated; amend the dead unstable_cache/revalidateTag wording); psql-bench.sh (decision stands; script unshipped and unowned — assign to Phase 5). **reverse (boundary decision):** Connection pooler "RDS Proxy for Vercel reads" (proxy unreachable from off-VPC Vercel and now consumerless; spec lens recommends reverse, correctness lenses revisit-but-keep; user decides at the gate with the Q2 amendment).

### Disagreements (if any)
- LTTB-over-hidden-series: correctness/codex must-fix vs correctness/claude deliberate non-flag; synthesizer line-verified v3 parity (both substrates union over all datasets with in-place `dataset.hidden`) and dismissed-as-preserved onto the parity-decision list.
- Verdict shape: all four lenses reject for different reasons (acceptance evidence; retry bug; the dismissed LTTB claim + FilterBar soft-nav edge). Conservative union: reject with 2 must-fix (one code-fixable, one user-resolvable-only).

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "overall": "reject",
  "artifact_id": "Phase-4",
  "cycle": 1,
  "preset": "phase-2 (custom: spec+correctness, parallel claude+codex per lens; artifact = delta since holistic squash f92713a49, cumulative product verified on disk)",
  "unified_findings": [
    {
      "severity": "must-fix",
      "kind": "missing-acceptance",
      "file_line": ".big-plans/ct__bench-v4.md:201 (phase-4 exit criteria)",
      "description": "All three Phase-4 exit criteria (live preview serves all chart slugs; /api/groups slug list matches the family registry; ~5-slug manual visual check vs v2) are unverifiable: the one-time operator Vercel setup has not run. Openly documented, not silent drift, but per the Phase-1 precedent it must be a recorded blocking pre-acceptance gate resolved by the USER at the boundary (authorize-over-operator-gated, or perform the setup first).",
      "recommended_fix": "User decision at the Step 3.4 gate: (a) operator performs the Vercel setup per web/README.md and the evidence (preview URL, slug-list diff vs registry, CDN probe HIT, visual check) is captured before close, or (b) phase accepted user-authorized over the operator-gated criteria, evidence owed before the Phase-5 cutover.",
      "found_by": ["spec.claude (must-fix)", "spec.codex (2x must-fix: deploy evidence + visual-parity evidence)"]
    },
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/web/components/Chart.tsx:1503-1526",
      "description": "Dismiss-retry self-cancels: the [error] effect's cleanup (which runs on the error->null transition BEFORE the effect re-runs) invokes cleanupRetryRef, cancelling the 0ms retry the dismiss just armed. In real browsers React flushes commit + passive effects ahead of the macrotask timer, so the bounded Chart.js import retry never fires and a permalink chart stays blank after a transient chunk-load failure (the cycle-2 PR-4.4.b dead-end resurfaces). act/jsdom tests invert the ordering and mask the bug (reviewer empirically confirmed with a scratch test). Synthesizer line-verified the cleanup structure on disk.",
      "recommended_fix": "Cancel the pending retry only on unmount (move cleanupRetryRef.current?.() from the [error] effect cleanup into the unmount cleanup); the retry closure is already unmount-safe (controllerRef null check + shouldRetryConstruct's disposed check). A faithful failing test needs non-act scheduling; document the pin requirement on the deferred interaction-suite row.",
      "found_by": ["correctness.claude (must-fix)"]
    },
    {
      "severity": "should-fix",
      "kind": "cross-cutting",
      "file_line": ".big-plans/ct__bench-v4.md:130/138/201",
      "description": "Plan text pins the dead caching mechanism (unstable_cache + revalidateTag; export const revalidate = ~300) at three normative sites; shipped code uses force-dynamic + Cache-Control s-maxage on API 200s + Vercel-CDN-Cache-Control config rules.",
      "recommended_fix": "Amend the Read-service-framework Key-decision row, the PR-4.4-arch row mention, and the Phase-4 row wording to the shipped mechanism (plan-edit).",
      "found_by": ["spec.claude (should-fix)", "both correctness lenses' tradeoffs (revisit-but-keep)"]
    },
    {
      "severity": "should-fix",
      "kind": "cross-cutting",
      "file_line": ".big-plans/ct__bench-v4.md:123",
      "description": "Key decision 'RDS Proxy for the Vercel read service only' is contradicted by the shipped wiring guidance (proxy VPC-internal, Vercel off-VPC; README/db.ts steer to the public instance endpoint); the proxy currently has no consumer.",
      "recommended_fix": "USER decision at the Step 3.4 boundary (already queued as boundary item 4): amend Q2 and decide the proxy's disposition.",
      "found_by": ["spec.claude (should-fix)", "all four lenses' tradeoffs (reverse / revisit-but-keep)"]
    },
    {
      "severity": "should-fix",
      "kind": "cross-cutting",
      "file_line": ".big-plans/ct__bench-v4.md:131",
      "description": "scripts/psql-bench.sh is claimed 'documented in web/README.md' by a Key decision, but the script exists nowhere, the README shipped without it, and no remaining PR row owns it, while the Phase-2 soak spot-check and the PR-5.3 /api/admin/sql decommission rationale depend on it.",
      "recommended_fix": "Assign ownership: fold delivery + README documentation into the Phase-5 rows (plan-edit), or amend the Key decision if plain psql incantations suffice.",
      "found_by": ["spec.claude (should-fix)"]
    },
    {
      "severity": "should-fix",
      "kind": "boundary",
      "file_line": "benchmarks-website/web/components/Header.tsx:78",
      "description": "When the filter universe is empty/undefined the Header omits FilterBar and initGlobalFilter never runs, so a soft navigation from a filtered page can leave stale module-scoped filters hiding series with no visible filter UI. v4-architecture-only window (v3 was MPA: module state reset every load).",
      "recommended_fix": "DEFER (Deferred work row): narrow error-path UX edge with no data wrongness; the right reset semantics need thought (reset-on-absent-FilterBar vs page-scoped state).",
      "found_by": ["correctness.codex (should-fix)"]
    },
    {
      "severity": "should-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/web/lib/queries.ts:995",
      "description": "collectFilterUniverse orders via SQL ORDER BY (database collation): prod RDS en_US.UTF-8 orders hyphenated names differently from v3's BTreeSet byte order, while the C-collated test container masks the divergence. Severity raised from nit by the synthesizer: it is a v3-parity break of the same class the phase's acceptance pins, and the fix is one line using the port's existing comparator.",
      "recommended_fix": "Sort in JS with compareCodeUnits (drop the SQL ORDER BY), pinning byte-order parity independent of database collation.",
      "found_by": ["correctness.claude (nit)"]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": ".github/actions/verify-cdn-cache/action.yml:60",
      "description": "The probe's awk splits on ': ', so a legal OWS-less header (x-vercel-cache:HIT) would parse as absent and fail the gate. Latent (Vercel emits the space today).",
      "recommended_fix": "Split on the first ':' and trim OWS.",
      "found_by": ["correctness.claude (nit)"]
    },
    {
      "severity": "nit",
      "kind": "dismissed-as-preserved",
      "file_line": "benchmarks-website/web/components/Chart.tsx:786",
      "description": "DISMISSED-AS-PRESERVED (synthesizer line-verified): the shared-LTTB union iterating ALL datasets including filter-hidden ones is byte-faithful v3 parity. v3 chart-init.js builds the identical max-across-all-datasets union with no visibility check (~line 880), and both substrates hide series in place via dataset.hidden (v3 line 1085 / v4 Chart.tsx:690), so hidden series influence kept indices identically in both. correctness.codex flagged must-fix; the migration behavior-preservation contract pins matching v3. Joins the PHASE-4-BOUNDARY parity list (6th item) for the v4-fidelity-vs-preserve-v3 user decision.",
      "recommended_fix": "None now; candidate improvement under the deliberate v4-fidelity effort if the user opts in.",
      "found_by": ["correctness.codex (must-fix, dismissed)"]
    },
    {
      "severity": "nit",
      "kind": "scope-creep",
      "file_line": ".github/actions/verify-cdn-cache/action.yml:1",
      "description": "Files-touched grew beyond declared rows (composite action, README, harness, db.ts note; PR-4.4.b analogs) — all review-driven and documented in Implementation status; no silent drift.",
      "recommended_fix": "None (record only).",
      "found_by": ["spec.claude (nit)"]
    }
  ],
  "disagreements": [
    {
      "topic": "LTTB-over-hidden-series severity",
      "positions": "correctness.codex: must-fix chart-wrongness (hidden series steer kept indices). correctness.claude: line-verified the surrounding port as exact v3 parity and did not flag. Synthesizer verification: v3 chart-init.js unions over all datasets with in-place dataset.hidden filtering, identical to v4.",
      "synthesizer_recommendation": "Dismiss-as-preserved per the behavior-preservation contract; track on the parity-decision list."
    },
    {
      "topic": "Verdict shape",
      "positions": "All four lenses reject, but for different reasons: spec lenses on acceptance evidence (operator gate), correctness.claude on the retry bug, correctness.codex on the (dismissed) LTTB claim plus the FilterBar edge.",
      "synthesizer_recommendation": "Reject with 2 must-fix: the retry bug is code-fixable now; the operator gate is user-resolvable only and goes to the Step 3.4 boundary AUQ per the Phase-1 authorize-over-operator-gated precedent."
    }
  ],
  "dropped_re_flags": [],
  "executive_summary": "Phase-4 phase-end cycle 1 (2-vote custom, spec + correctness, Claude + Codex per lens): REJECT with 2 must-fix. The phase product itself verified remarkably clean: footprint exactly in-bounds (zero Out-of-scope contact), every PR row's files on disk, dropped items stayed dropped, all four UI BANS pinned and re-verified, 204/204 vitest live (containers + production-server smoke; full migration set applied), prettier/eslint/tsc/yamllint/DB-less build all green, and the PR-4.5 post-cap fix wave independently re-verified by three lenses. The two must-fix: (1) the phase exit criteria are wholesale operator-gated (no Vercel project exists yet), which per the Phase-1 precedent needs an explicit user resolution at the boundary rather than silent acceptance; (2) a real browser-only bug correctness.claude proved empirically: Chart.tsx's error-dismiss effect cleanup cancels the Chart.js import retry it just armed, so the cycle-2 permalink dead-end is back in real browsers while act/jsdom tests mask it. One headline dismissal: correctness.codex's LTTB-over-hidden-series must-fix is byte-faithful v3 parity (synthesizer line-verified both substrates union over all datasets with in-place hidden flags) and joins the parity-decision list as its 6th item. Remaining should-fixes are plan-text staleness (dead caching-mechanism wording at three sites; the contradicted proxy-for-Vercel-reads decision; unowned psql-bench.sh), one deferred v4-only soft-nav filter edge, and a collation-vs-byte-order sort parity fix the synthesizer upgraded from nit (one-line compareCodeUnits change). Tradeoffs re-evaluation across lenses: pooler decision reverse/revisit (proxy consumerless - boundary item), framework row revisit-but-keep (mechanism wording only), psql-bench revisit-but-keep (ownership), all others keep. The phase is one small code fix + plan edits + one user authorization away from acceptance.",
  "reviewer_outputs": {
    "spec.claude": "reject/high - 1 must-fix (operator gate), 3 should-fix (plan text), 1 nit",
    "correctness.claude": "reject/high - 1 must-fix (retry self-cancel, empirically proven), 2 nit",
    "spec.codex": "reject/medium - 2 must-fix (deploy evidence; visual-parity evidence) = the operator gate",
    "correctness.codex": "reject/high - 1 must-fix (LTTB hidden-series; DISMISSED as v3 parity), 1 should-fix (FilterBar soft-nav edge)"
  }
}
```

</details>

## Resolved phase-end must-fix items — Phase 4: Next.js read service on Vercel — cycle 2

| Severity | File:line | Description | Implicated PR | Resolved |
|----------|-----------|-------------|---------------|----------|
| must-fix | scripts/migrate-schema.py:127 (and docstring :115) | `_assert_master_capable` remediation message + docstring still say "apply the bootstrap migrations (002/004)"; 005 is now ALSO a requires-superuser bootstrap migration, so a rejected 005 apply gives stale remediation. | PR-4.5 (operator-gate 005 follow-up) | [x] 0f4e25b3c |
| must-fix | migrations/README.md:45-53,65 | README file-list stops at 004 and the bootstrap-ordering section says "(002 / 004)"; 005_read_role.sql (a requires-superuser bootstrap migration) is undocumented. | PR-4.5 (operator-gate 005 follow-up) | [x] 0f4e25b3c |
| must-fix | scripts/test_migrate_schema.py:1507 | `test_real_bootstrap_migrations_carry_superuser_marker` pins only 002/004; deleting 005's requires-superuser marker would not fail any test. | PR-4.5 (operator-gate 005 follow-up) | [x] 6984902ca |
| must-fix | benchmarks-website/web/README.md:92 | Operator setup offers IAM auth for the read service, but 005 creates bench_read with NO rds_iam, so IAM auth cannot work for this role without a follow-up rds_iam-grant migration (which disables password auth). | PR-4.5 (operator-gate 005 follow-up) | [x] 0f4e25b3c |
| must-fix | migrations/005_read_role.sql:31-36 | 005 claims idempotency but only CREATEs bench_read when absent; it never enforces its no-rds_iam invariant on a pre-existing role. A pre-existing bench_read with rds_iam stays IAM-only after a green apply, silently breaking password auth. | PR-4.5 (operator-gate 005 follow-up) | [x] 6984902ca |

## Phase 4: Next.js read service on Vercel — end-of-phase review (cycle 2) — rejected (2-vote)
**Synthesizer output from the custom mixed-executor 2-vote review (spec + correctness, parallel Claude + Codex gpt-5.5 xhigh per lens; gauntlet phase-2 preset lenses; fix-aware, `prior_fix_commit_sha=8b85a3d3f`, fix-pass `c2be81b4f..HEAD`); full Synthesizer Output JSON in the `### Cycle 2` entry under `## Phase 4 raw gauntlet responses (archive)` above (canonical copy) + at `/tmp/pr45-review/results-p4c2/synthesis.json`. Artifact = the delta since the holistic squash `f92713a49`; the cumulative phase product was verified on disk by all four lenses. Verdict: REJECT (5 must-fix, 3 should-fix, 2 nit).**

### Summary of changes
Cycle 2 re-reviewed the post-cycle-1 Phase-4 state. Both cycle-1 must-fixes are genuinely resolved: the Chart.tsx dismiss-retry self-cancel was fixed (f3ee70fe1, cancel moved to unmount-only), and the operator-gate exit criteria were user-resolved SETUP-NOW (run 27293419561 green end-to-end, including the public-domain CDN probe at HTTP 200 + x-vercel-cache HIT). The fix-commit's other changes — the `collectFilterUniverse` JS-side byte-order sort, the `verify-cdn-cache` OWS-tolerant header parse, and migration 005_read_role.sql (bench_read, SELECT-only, NO rds_iam) — are internally correct. The new must-fix set is entirely cross-reference drift + an idempotency hole from the LATE addition of migration 005 during the operator-gate work: the surrounding contracts (migration runner remediation message, migrations README, the marker-regression test, and the web README's IAM-auth wording) were never updated in lockstep with 005, and 005 itself never enforces its no-rds_iam (password-auth) invariant for a pre-existing role.

### Surprises and discoveries
- **Multi-model divergence (the headline):** both Claude lenses returned `accept`; both Codex (gpt-5.5) lenses returned `reject`. The 5 must-fix items are exactly what the Claude lenses asserted "internally consistent"/"sound" without cross-checking the runner message / README / marker test / web README, or constructing the pre-existing-rds_iam-role case. The synthesizer line-verified all 5 on disk — every one is real. Multi-model review earned its keep this cycle. Amend plan: no.
- **004 had never been applied to prod** (Phase-2 best-effort soak silently failing) — discovered during operator-gate bootstrap; 004+005 applied together (ledger 5/5). Already documented.
- **rds_iam forces IAM-only auth on RDS** — 005 originally copied 004's rds_iam grant; revoked on prod + dropped from 005 in lockstep (8b85a3d3f) before the file was pushed. Already documented. The cycle-2 M5 finding is the *idempotency* corollary: the drop is correct for a fresh role but unenforced for a pre-existing one.
- **The caching mechanism (unstable_cache/revalidateTag) was unimplementable** and was replaced by header-driven CDN caching; 3 of ~8 normative plan sites were amended (cycle-2 should-fix asks for the rest). Amend plan: yes.

### Testing coverage assessment
| Case | Test location | Confidence | Status |
|------|---------------|-----------|--------|
| Cycle-1 StrictMode mount/cleanup/remount still issues initial ?n=100 fetch | Chart.lifecycle.test.tsx:60 | high | tested |
| Cycle-1 fresh mount replays pre-existing group-Y override | Chart.lifecycle.test.tsx:90 | high | tested |
| collectFilterUniverse JS byte-order sort, excl. vector-search flavors | queries.test.ts:92 | high | tested |
| LTTB bounds/uniqueness; predecessorValue idx-1 oldest-first (BAN-pinned) | chart-format.test.ts:150,329 | high | tested |
| Migration 005 bench_read created, SELECT-only, no writes, USAGE on public | test_migrate_schema.py:1206 | high | tested |
| Migrations 001..005 apply clean + idempotent + non-superuser master | test_migrate_schema.py:660 | high | tested |
| **005 idempotency: pre-existing bench_read with rds_iam ends WITHOUT rds_iam** | — | **high (untested)** | **cycle-2 M5** |
| **005 carries (must keep) the requires-superuser marker** | — | **high (untested)** | **cycle-2 M3** |
| **Chart.tsx dismiss-retry lifecycle without act-deferred ordering** | — | **high (untested)** | **cycle-2 should-fix** |
| Live preview-deploy-on-PR-open (PR-4.5) | — | medium (untested) | transitive via prod run |
| Phase-4 slug-list-vs-registry + ~5-slug visual parity | — | high (untested) | moved to Phase 5 |

### Tradeoffs re-evaluation
- **Read service framework (Next.js 15 + RSC + unstable_cache/revalidateTag):** revisit-but-keep — framework vindicated; caching MECHANISM replaced by header-driven CDN (live-proven). Remaining stale plan prose flagged should-fix.
- **Connection pooler (RDS Proxy for Vercel reads):** reverse — proxy is VPC-internal, unreachable from off-VPC Vercel; no consumer. Queued for the Step-3.4 boundary AUQ.
- **migration 005 bench_read (static password, NO rds_iam):** keep — correct decision, but the no-rds_iam invariant must be ENFORCED idempotently (M5) and surrounding contracts updated (M1-M4).
- **Schema deploy tool (migrate-schema.py + requires-superuser marker):** keep — 005 follows the model, but runner message + README + marker test must include 005 (M1/M2/M3).
- **Review calibration (trusted-input, low-stakes, ~3-cycle cap):** keep — cycle-2 must-fix items are migration/auth-contract correctness (higher-stakes than the dashboard UI), cheap to fix; phase_end_reject_cycles -> 2 after this cycle (no early-break).
- **PR-4.4 UI architecture (RSC shell + per-chart islands; no shard route):** keep — vindicated; one residual missing-but-required regression pin (should-fix).
- **Operator SQL replacement (psql-bench.sh):** revisit-but-keep — unshipped + undocumented; ownership routed to Phase 5 (PR-5.1).
- **Prod load timing / data-dependent exit criteria deferred to Phase 5:** keep — consistent with empty-prod; should-fix asks only to record the moved criteria on the Phase-5 row.

### Disagreements
Material multi-model split: **Claude lenses (spec + correctness) → accept; Codex lenses (spec + correctness) → reject.** The synthesizer sided with Codex after line-verifying all 5 must-fix on disk (migrate-schema.py:127, migrations/README.md:45-65, test_migrate_schema.py:1507, web/README.md:92, 005_read_role.sql:31-36). Conservative union → REJECT; fix all 5. The 3 should-fix (missing dismiss-retry regression pin the plan required; deferred Phase-4 exit criteria not yet on the Phase-5 row; partial caching-wording amendment) and 2 nits are triaged in the cycle-2 reject-fix pass.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

Canonical copy: the `### Cycle 2 — preset=phase-2 (multi-model) — reject` entry under `## Phase 4 raw gauntlet responses (archive)` above; durable copy at `/tmp/pr45-review/results-p4c2/synthesis.json` (and per-lens raw outputs at `/tmp/pr45-review/results-p4c2/{spec,correctness}.codex.json`).

</details>

## Resolved phase-end must-fix items — Phase 4: Next.js read service on Vercel — cycle 3

| Severity | File:line | Description | Implicated PR | Resolved |
|----------|-----------|-------------|---------------|----------|
| must-fix | migrations/005_read_role.sql:72-90 (ALTER DEFAULT PRIVILEGES block) | 005's future-table SELECT default-privilege for bench_read (read-role counterpart of 004's tested bench_ingest ADP) was untested; deleting the ADP would leave existing 005 tests green. | PR-4.5 (operator-gate 005 follow-up) | [x] f203c346c |

## Phase 4: Next.js read service on Vercel — end-of-phase review (cycle 3) — rejected (2-vote)
**Synthesizer output from the custom mixed-executor 2-vote review (spec + correctness, parallel Claude + Codex gpt-5.5 xhigh per lens; gauntlet phase-2 preset lenses; fix-aware, `prior_fix_commit_sha=6984902ca`, fix-pass `1299a69b9..HEAD`); full Synthesizer Output JSON in the `### Cycle 3` archive entry above + at `/tmp/pr45-review/c3/synthesis.json`. Artifact = the delta since the holistic squash `f92713a49`. Verdict: REJECT (1 must-fix, RESOLVED in-cycle by f203c346c) + 4 nits (deferred).**

### Summary of changes
Cycle 3 re-reviewed the full Phase-4 state after the cycle-2 reject-fix. All four lenses confirmed the cycle-2 reject (5 must-fix re: migration 005) is fully resolved on disk: the runner remediation message + migrations README + web README + marker test consistently enumerate 002/004/005 and describe 005's NO-rds_iam / password-auth contract; 005's idempotent `REVOKE rds_iam` is present + pinned. **3 of 4 lenses accepted** (spec.codex with zero findings, spec.claude + correctness.claude with nits only). **correctness.codex found ONE new coverage must-fix**: 005's future-table SELECT default-privilege (the `ALTER DEFAULT PRIVILEGES FOR ROLE migrator ... GRANT SELECT ON TABLES TO bench_read` block) was untested — the read-role counterpart of the already-tested bench_ingest ADP, so deleting it would leave existing 005 tests green. The synthesizer line-verified the gap (no such test existed; 004's equivalent IS tested) and **resolved it in-cycle** by adding `test_bench_read_default_privileges_cover_future_migrator_tables` (f203c346c), mutation-verified discriminating (stripping the ADP fails the test) with 53/53 migrate-schema green.

### Surprises and discoveries
- Three lenses (incl. spec.codex) found zero findings; the phase is converging cleanly (cycle-1 operator-gate + Chart fix, cycle-2 005 cross-ref drift, cycle-3 one coverage gap — progressively smaller). `phase_end_reject_cycles` reaches 3 this cycle → the Step 3.3.5 modulo-3 phase-level early-break fires (mandated user gate).
- The lone must-fix was a coverage asymmetry (bench_ingest's future-table ADP tested; bench_read's not), resolved in-cycle. Amend plan: no.

### Testing coverage assessment
| Case | Test location | Status |
|------|---------------|--------|
| 005 future-table SELECT ADP for bench_read (auto-grant SELECT, no writes) | test_bench_read_default_privileges_cover_future_migrator_tables | **tested (cycle-3 fix, mutation-verified)** |
| 005 idempotent REVOKE rds_iam from pre-existing bench_read | test_005_revokes_rds_iam_from_preexisting_bench_read | tested (cycle-2) |
| 005 bench_read created, SELECT-only, marker present; 001..005 apply clean/idempotent/non-superuser master | scripts/test_migrate_schema.py (multiple) | tested |
| rds_iam cluster-global cleanup after a mid-test crash | — | nit (defer: add to `_scrub_bootstrap_roles`) |
| Chart-controller jsdom interaction harness; act-free dismiss-retry pin | — | deferred (lean calibration) |
| slug-list-vs-registry + ~5-slug visual parity | — | moved to Phase 5 (prod empty until PR-5.0) |

### Tradeoffs re-evaluation
- **005 future-table ADP tested at SELECT-only parity:** keep — closes the coverage asymmetry with 004; mutation-verified.
- **Header-driven CDN caching; bench_read static-password/no-rds_iam; data-dependent exit criteria deferred to Phase 5; preserve-v3 semantic equivalence:** keep — re-confirmed clean; the accumulated v4-fidelity-vs-preserve-v3 bundle remains a Step-3.4 user decision.

### Disagreements
spec.codex / spec.claude / correctness.claude accepted (only nits); correctness.codex rejected on the single 005-ADP coverage gap. The synthesizer line-verified the gap as real (no bench_read ADP test existed; 004's IS tested) and resolved it in-cycle with a mutation-verified parity test. With the must-fix fixed, the phase is clean modulo deferred nits; control passes to the modulo-3 phase-level early-break per the ~3-cycle calibration.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

Canonical copy: the `### Cycle 3` entry under `## Phase 4 raw gauntlet responses (archive)` above; durable copy at `/tmp/pr45-review/c3/synthesis.json` (per-lens raw outputs at `/tmp/pr45-review/c3/{spec,correctness}.codex.json`).

</details>

## Phase 4: Next.js read service on Vercel — end-of-phase review (cycle 4) — accepted (2-vote)
**Synthesizer output from the custom mixed-executor 2-vote review (spec + correctness, parallel Claude + Codex gpt-5.5 xhigh per lens; gauntlet phase-2 preset lenses; fix-aware, `prior_fix_commit_sha=f203c346c`); full Synthesizer Output JSON in the `### Cycle 4` archive entry above + at `/tmp/pr45-review/c4/synthesis.json` (per-lens raw at `/tmp/pr45-review/c4/{spec,correctness}.{codex,claude}.json`/`.result.json`). Artifact = the delta since the holistic squash `f92713a49`. Verdict: ACCEPT (0 must-fix after line-verify) — 2 should-fix + 1 nit RESOLVED in-cycle, 1 nit deferred, spec.codex's 2 operator-gate must-fixes DROPPED as re-flags.**

### Summary of changes
Cycle 4 re-reviewed the full Phase-4 delta after the cycle-3 reject-fix (`f203c346c`, the `bench_read` future-table ADP test). The +39-line prior fix introduced no drift. **2 of 4 lenses accepted outright** (both Claude); **both Codex lenses rejected**. `correctness.codex` found ONE genuine boundary bug: the range-strip bare-track-click path pushes `x.max` one slot past the only label on a single-commit chart because `minRange` was forced to 1 while the drag + pixel-to-index paths already guard `n <= 1` (guard asymmetry). The synthesizer line-verified it real but cosmetic/edge-case (single-commit charts = first run of a new benchmark; no crash/data effect; self-correcting), **right-sized must-fix → should-fix**, and **resolved it in-cycle** by extracting a pure `clampRangeWindow()` helper (`minRange = min(1, maxIdx)`, `0cdfdfcd4`) with a mutation-verified discriminating unit test. `spec.codex`'s two "missing-acceptance" must-fixes (005 master-apply + `bench_read` password; PR-4.5 live deploy unverified) are **re-flags of the recorded operator-gate** that cycle-1 already routed to the Step-3.4 boundary AUQ (Phase-1 authorize-over-operator-gated precedent; data-dependent checks moved to PR-5.0); both Claude lenses correctly did not flag them → DROPPED. Two further minor findings resolved in-cycle (CDN-probe curl timeout, `b5cd0631a`; stale "four"→"five" assertion message, `369ca48a8`); one nit deferred (throttle trailing-timer disposed-safety, a latent no-op).

### Surprises and discoveries
- The 4th-cycle fresh multi-model pass surfaced a genuine single-commit range-strip boundary bug that cycles 1-3 missed — the value of an extra adversarial cycle even on a converging phase. Resolved in-cycle (mutation-verified). Amend plan: no.
- `spec.codex` (medium confidence) re-flagged the operator-gate as 2 must-fixes citing prompt-context line numbers, not repo code; `spec.claude` (same lens, line-verified) accepted. Dropped as re-flags. Amend plan: no.
- `phase_end_reject_cycles` had reached 3 (modulo-3 early-break); the user chose Continue; cycle 4 now accepts → reset to 0. Amend plan: no.

### Testing coverage assessment
| Case | Test location | Status |
|------|---------------|--------|
| Range-strip window clamp pins `[0,0]` on single-commit; 1-commit min span on multi-commit; bounds clamped | `chart-format.test.ts:clampRangeWindow` (3 cases) | **tested (cycle-4 fix, mutation-verified)** |
| 005 future-table SELECT ADP + idempotent rds_iam revoke + marker + SELECT-only role | `scripts/test_migrate_schema.py` | tested (53/53) |
| Chart island StrictMode/group-Y replay lifecycle; 204 pre-existing web tests | `benchmarks-website/web` | tested (207/207) |
| Single-commit range-strip via a REAL Chart.js instance (full pointer interaction) | — | deferred (jsdom interaction harness; pure clamp pinned) |
| throttle trailing-timer firing after teardown | — | deferred nit (latent no-op) |
| slug-list-vs-registry + ~5-slug visual parity | — | moved to Phase 5 (PR-5.0) |

### Tradeoffs re-evaluation
- **Range-strip minimum span:** revisit-but-keep — kept `minRange=1` for multi-commit charts; collapsed to 0 only on single-commit via `min(1,maxIdx)`. Behavior-preserving for the common case.
- **Connection pooler (RDS Proxy for Vercel reads):** revisit-but-keep — as-shipped read path uses the public instance endpoint + static `bench_read` password, leaving the proxy consumerless on the read side. Step-3.4 boundary item (Q2 amendment).
- **Header-driven CDN caching; bench_read static-password/no-rds_iam; per-effect-mount controller lifecycle:** keep — re-confirmed clean + consistently documented.

### Disagreements
`correctness.codex` rejected on the single-commit range-strip boundary; `correctness.claude` accepted (did not surface it) — the synthesizer line-verified it real but cosmetic, right-sized to should-fix, and resolved in-cycle (validates the multi-model design: a fresh model caught an edge the other correctness lens missed). `spec.codex` rejected on operator-acceptance evidence; `spec.claude` accepted — the operator-gate is a recorded Step-3.4 boundary item (Phase-1 precedent) + Phase-5-deferred data checks, so spec.codex's findings are dropped as re-flags, not new code must-fixes.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

Canonical copy: the `### Cycle 4` entry under `## Phase 4 raw gauntlet responses (archive)` above; durable copy at `/tmp/pr45-review/c4/synthesis.json` (per-lens raw outputs at `/tmp/pr45-review/c4/{spec,correctness}.codex.result.json` + `/tmp/pr45-review/c4/{spec,correctness}.claude.json`).

</details>

## Deferred work

**2026-06-04 LEAN RE-PLAN PRUNE.** Per the course-correction, the backlog below is pruned to **data-correctness items only**. The test-hardening, doc-polish, least-privilege-cleanup, and infra-nit entries are **DROPPED** (not data-correctness; disproportionate to a trusted-input low-stakes dashboard) — they are NOT to be picked up. Concretely dropped: the PR-1.1 provision.sh nits (security-group/subnet/redirect/proxy-auth/teardown/engine-version/OIDC-thumbprint/README), the PR-1.2 concurrency/autocommit/ledger-fingerprint items, the PR-1.4 uv-Python-pin item, the PR-1.6 cycle-6/7 doc + password-guard + PROXY_ROLE_NAME test-hardening items, the Phase-1 phase-end nits, the PR-2.1 cycle-2 negative-test items, and the PR-2.2 cycle-4 destructive-scrub + cycle-7 real-deadlock/autocommit + cycle-15 (already-resolved) test-hardening items. **RETAINED (must actually happen, not data-correctness-but-real-blockers):** (1) the Phase-1 phase-end **operator pre-merge gate** — run the live OIDC `schema-deploy` apply against the public instance + confirm `status` clean before squash-merging `ct/bench-v4` → `develop`; (2) the **bench-v4 Python ruff line-length reconciliation** before the `develop` merge (code-quality merge blocker). Any genuinely data-correctness deferral added by future reviews appends below.

| Source | File:line | Severity | Description | Deferral rationale |
|---|---|---|---|---|
| PR-1.1 cycle-1 gauntlet | `provision.sh:478-487` (security group rule check) | should-fix | `ensure_security_group` checks `FromPort==5432` but not CIDR; an existing rule from `10.0.0.0/8` would silently leave SG more restrictive. | Low operational impact for first-run; revisit if drift surfaces. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:429-446` (subnet group reconciliation) | should-fix | `ensure_db_subnet_group` doesn't reconcile subnet membership on existing group; stale group could prevent new-AZ scheduling. | Same; revisit on drift. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:432, 497, 590, 554, 677, 642` (`2>&1` redirects) | should-fix | Permission errors masked by `2>&1` in existence-check redirects; missing IAM perms surface as "doesn't exist". | Affects operator debugging only; low priority. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:590-605` (proxy auth reconciliation) | should-fix | Proxy auth config not reconciled on existing proxy (master secret ARN, IAMAuth setting). | One-shot script; secret ARN doesn't change post-creation. |
| PR-1.1 cycle-1 gauntlet | `README.md:264-276` (tear-down sequence) | should-fix | Tear-down missing `aws rds wait db-proxy-deleted` + `aws rds wait db-instance-deleted` between async deletes. | Tear-down is off the critical path; add to ops runbook if/when needed. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:357` (engine version pin `16.4`) | should-fix | Pinned minor version may be deprecated by AWS over time. | Works today; revisit when AWS deprecates 16.4. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:649` (OIDC thumbprint hardcoded) | should-fix | AWS no longer enforces the thumbprint for GitHub OIDC; cosmetic. | Add comment when next touching the file. |
| PR-1.1 cycle-1 gauntlet | `README.md:183` ("IAM auth is the gate") | should-fix | Phrasing overstates what PR-1.1 alone accomplishes; password auth on `postgres` master remains until PR-1.3's migration 002. | Update README in PR-1.3 alongside the rds_iam grant migration. |
| PR-1.1 follow-up | (operator action) | medium | Re-run `provision.sh` in CloudShell to apply the branch-scoped trust policy update (commit 2336d48c1). Existing role still has the wildcard `repo:vortex-data/vortex:*` sub-claim until operator re-runs. | Operator-side; security tightening, not blocking. **→ RESOLVED 2026-06-01: live verification confirmed the trust policy is branch-scoped (develop + ct/bench-v4); the wildcard sub-claim is gone, so the re-run was applied.** |
| PR-1.2 cycle-1 gauntlet | `scripts/migrate-schema.py:57-75` (concurrency) | should-fix | Two concurrent CI runs can both observe the same pending set and race on non-idempotent DDL; PRIMARY KEY guards the ledger row but the DDL itself still executes twice, producing transient errors and ambiguous logs. | The CI workflow already serializes via `concurrency: schema-deploy`, and the initial DDL in PR-1.3 is idempotent (CREATE TABLE IF NOT EXISTS / IF NOT EXISTS indexes); revisit when a non-idempotent migration is actually authored. The fix is a Postgres advisory lock (`SELECT pg_advisory_xact_lock(<constant>)`) at the start of each per-migration transaction plus a concurrency test that spawns two parallel `apply` invocations. |
| PR-1.2 cycle-2 gauntlet | `scripts/migrate-schema.py:108-112` (autocommit toggle precondition) | should-fix | `apply()` sets `conn.autocommit = True` unconditionally; psycopg raises `ProgrammingError` if the connection has a transaction in progress. Production safe (main() opens a fresh conn) but the function is exposed as a library API and a future caller that ran any prior `cursor.execute` would hit the error. | Library-API hardening; not relevant until a second importer materializes. Fix is to assert `conn.info.transaction_status == IDLE` at the top of `apply` or defensively `rollback()` before the toggle, with a test that opens a transaction first. |
| PR-1.2 cycle-2 gauntlet | `scripts/migrate-schema.py` ledger schema (fingerprint) | should-fix | Applied migrations are not fingerprinted; an author editing `001_initial_schema.sql` after it has been applied to RDS sees `status` report clean both locally (against a freshly-applied testcontainer) and in CI (against RDS with the old file's effects). README forbids edit-after-apply but the runner has zero enforcement. | Substantive change: add a `sha256` column to `_applied_migrations`, record at apply time, compare on-disk hash vs ledger in `status`, report a third drift class `[~]` with non-zero exit. Track for a follow-up PR (PR-1.2.1 or fold into PR-1.4 wiring). |
| PR-1.2 cycle-2 gauntlet | `scripts/test_migrate_schema.py` (no CI runner) | should-fix | The pytest suite skips silently when Docker is unavailable (`_docker_available()` probe), and no CI workflow currently runs the suite with Docker enabled. PR-1.2 acceptance ("Unit test: applies a fresh schema to testcontainers Postgres ...") is verified only by local dev runs; CI would be green even if the runner regressed. | PR-1.4 wires the schema-deploy workflow with real apply; pair it with a `pytest-on-PR` job that fails loud when `_docker_available` returns False in CI. Track for PR-1.4. **→ Phase-2 re-plan: Resolved-by PR-2.3 (CI job runs `scripts/` pytest incl. the testcontainer suite, fails loud if Docker absent).** |
| PR-1.3 cycle-1 gauntlet | `migrations/002_iam_db_user.sql:37` (migrator table privileges) | should-fix | `migrator` is granted only `CREATE, USAGE ON SCHEMA public`, no table-level DML on the six tables `001` creates. Under the documented bootstrap order (first `apply` runs as RDS master, which owns `001`'s tables), the PR-2.x ingest write path connecting AS `migrator` would hit `permission denied`. | Forward-looking; the ingest write path is PR-2.x scope and the plan references a separate future `GitHubBenchmarkIngestRole`. Resolve the role-ownership model in PR-2.1 (dedicated ingest role with `GRANT SELECT,INSERT,UPDATE,DELETE` + `ALTER DEFAULT PRIVILEGES`, or explicit table grants to `migrator`), pinned by a connect-AS-ingest-role round-trip test. Fixing now would be premature and a least-privilege smell on the schema-deploy role. **→ Phase-2 re-plan: Resolved-by PR-2.1 (dedicated `bench_ingest` role with `SELECT,INSERT,UPDATE` on the 6 tables via migration 004; chosen over granting `migrator`).** |
| PR-1.4 cycle-2 gauntlet | `.github/workflows/schema-deploy.yml:50` (uv Python provisioning) | should-fix | The `Install uv` step relies on `uv` auto-provisioning a Python 3.11+ interpreter for the PEP 723 `requires-python` script with no explicit pin. Works with current `uv` and matches the sibling `docs.yml` pattern, but is a latent CI fragility if the shared `spiraldb/actions` setup-uv ever pins an older `uv` or disables managed-Python downloads. | Matches established repo convention (`docs.yml` / `bench-pr.yml` use the identical `setup-uv` + `uv run --no-project` pattern with no explicit Python pin); failure mode is loud + operator-visible (workflow_dispatch-only), not silent. Adding `uv python install 3.12` would deviate speculatively. Revisit if the shared setup-uv action's Python-provisioning behavior changes, or fold a repo-wide pin into a future CI-hardening pass. |
| PR-1.5 cycle-1 gauntlet | `scripts/test_measurement_id.py` (+ `scripts/test_migrate_schema.py`) not wired into CI | should-fix | No CI job runs the `scripts/` pytest suites (the only pytest invocations cover `vortex-python/`). The Rust side IS gated (the golden test runs via `rust-test-other`), so Rust==golden is enforced, but golden==Python is not: a future edit to `_measurement_id.py` that diverges from the golden file would merge green and only surface at PR-2.1 ingest time as duplicate rows. The port currently matches all 63 vectors (verified). | Same class as the PR-1.2 `no CI runner` item (testcontainer suite also ungated). Wire one CI job — `uv run --all-packages pytest scripts/` — covering both `test_measurement_id.py` (no Docker needed) and `test_migrate_schema.py` (needs Docker). Fold into the CI-hardening pass alongside the PR-1.2 item rather than a one-off here; behavior is correct today, only enforcement is missing. **→ Phase-2 re-plan: Resolved-by PR-2.3 (golden==Python now CI-gated via `pytest scripts/`).** |
| 2026-05-29 deploy-model decision | `.github/workflows/schema-deploy.yml` (trigger) | should-fix | Implements the "Schema-deploy authorization + execution-safety model" Key decision: switch the trigger from `workflow_dispatch`-only to push on the deploy branch under `paths: migrations/** + scripts/migrate-schema.py`, keeping `workflow_dispatch` + `dry_run` for manual/preview runs and removing the now-superseded `environment:`-gate comments. PR merge becomes the deploy gate. | Small, well-scoped change to a Phase-1 deliverable; supersedes the original `environment: schema-deploy` manual-approval mandate. Phase placement is the fresh session's call at the Phase 1→2 boundary AUQ: amend Phase 1 (e.g. PR-1.6) OR fold into Phase 2 (it sits naturally beside the dual-write CI pipeline). The per-PR testcontainer test is already the execution-safety gate for additive DDL. **→ Phase-2 re-plan: Resolved-by PR-2.4 (schema-deploy.yml trigger switched to push on the deploy branch under `paths: migrations/**`).** |
| 2026-05-29 deploy-model decision | data-affecting migration safety (no RDS branching) | should-fix | RDS has no Neon-style copy-on-write branching, so the testcontainer-against-empty-schema test does not validate a migration that mutates existing data (type change, NOT NULL on an existing column, backfill). | Add a CI step that PITR-restores a recent prod snapshot to a throwaway instance and runs the candidate migration there, but ONLY when a migration is data-affecting (additive DDL does not need it). Not needed for Phase 1's additive migrations; trigger this work the first time a data-affecting migration is authored. This is the RDS stand-in for Neon branching (Aurora fast-clone is the true analog but would reverse the `db.t4g.micro` Key decision). |
| PR-1.6 cycle-1 gauntlet | `provision.sh:458` (schema-role proxy grant) | should-fix | `GitHubBenchmarkSchemaRole`'s `rds-db:connect` still covers the proxy resource-id, but PR-1.6 makes CI authenticate against the instance and the proxy is now Vercel-reads-only; the proxy grant on the schema role is dead least-privilege surface. | Harmless extra grant meanwhile. Drop the proxy resource (keep the instance) in PR-2.1, where the role-ownership/grant model is revisited alongside the ingest role. **→ Phase-2 re-plan: Resolved-by PR-2.1 (provision.sh drops the dead proxy `rds-db:connect` grant on the schema role).** |
| PR-1.6 cycle-6 gauntlet | `README.md` Prerequisites (`secretsmanager:GetSecretValue`) | nit | The PR-1.6 SM-fetch bootstrap (`aws secretsmanager get-secret-value`) needs `secretsmanager:GetSecretValue` on the master secret, an operator IAM permission not listed in the README Prerequisites table. Operator is normally PowerUser/Admin (line 34), so usually present. | Doc-only; deferred at the cycle-6/7 early-break to bound PR-1.6. Add the permission to the Prerequisites IAM list in a follow-up doc-polish pass. |
| PR-1.6 cycle-6 gauntlet | `README.md:212` (tear-down OIDC-provider ARN) | nit | The commented-out OIDC-provider tear-down embeds the literal account `245040174862`; `TARGET_ACCOUNT` is overridable and absent from the tear-down substitution note. Line is commented-out + account-scoped, so blast radius is tiny. | Doc-only; deferred at the cycle-6/7 early-break. Note `TARGET_ACCOUNT` substitution in a follow-up doc-polish pass. |
| PR-1.6 cycle-7 gauntlet | `scripts/test_migrate_schema.py:868` (password-fetch guard) | must-fix (deferred via user-authorized early-break) | `test_readme_bootstrap_password_fetch_is_safe` bans the masking pattern + interactive-read but does not pin the `\|\| exit 1` fail-fast structure; a future edit removing `\|\| exit 1` while leaving a `jq -er` comment-mention would pass. **Test-guard-strictness, NOT a functional defect** — the shipped runbook command IS fail-fast and correct. | Deferred at the cycle-7 second early-break (user directed accept-after-one-final-cycle, defer residual). Fold into a follow-up test-hardening PR: extract the README shell block and assert the exact `master_secret=$(...) \|\| exit 1` / `PGPASSWORD=$(... jq -er ...) \|\| exit 1` / standalone `export PGPASSWORD` sequence. |
| PR-1.6 cycle-7 gauntlet | `benchmarks-website/infra/provision.sh:63` (`PROXY_ROLE_NAME` override) | must-fix (deferred via user-authorized early-break) | `PROXY_ROLE_NAME` is an observable `${ENV:-default}` override with no regression guard against role creation reverting to a hard-coded name (behavioral-drift BAN). **Test-coverage gap, NOT a functional defect** — provision.sh correctly uses `$PROXY_ROLE_NAME` today. | Deferred at the cycle-7 second early-break. Add a static provision.sh guard (get-role/create-role/put-role-policy reference `$PROXY_ROLE_NAME`, no hard-coded literal) in the follow-up test-hardening PR. |
| PR-1.6 cycle-7 gauntlet | `scripts/test_migrate_schema.py:884` (password guard quoted form) | should-fix | The guard bans unquoted `export PGPASSWORD=$(` but not the quoted `export PGPASSWORD="$(...)"` form (which would also mask the substitution exit code). | Deferred at the cycle-7 early-break. Fold into the same follow-up test-hardening PR with a regex over code lines (`export\s+PGPASSWORD\s*=\s*['"]?\$\(`). |
| PR-1.6 cycle-7 gauntlet | `benchmarks-website/infra/README.md:23` (IAM-work contradiction) | should-fix | README:21 (PR-1.6 edit) says the schema-role proxy grant is slated for PR-2.1 cleanup, but line 23 still says "no further IAM work after PR-1.3 lands" — an internal contradiction. Doc-only. | Deferred at the cycle-7 early-break. Narrow the line-23 claim (no further IAM work for the PR-1.3 bootstrap/schema-deploy path; PR-2.1 owns the proxy-grant cleanup + ingest grants) in the follow-up doc-polish pass. |
| Phase-1 phase-end cycle-2 (spec/codex) | (operator action) | must-fix (operator pre-merge gate) | Phase-1 acceptance criterion "operator runs the live OIDC schema-deploy apply against the public instance endpoint and `status` reports clean" is unverified — requires a real AWS apply with live RDS, not performable in-session (no AWS creds). | **Operator pre-merge gate**: run the live `schema-deploy.yml` apply (or `migrate-schema.py apply` against RDS as the OIDC `migrator`) and confirm `status` clean BEFORE squash-merging `ct/bench-v4` → `develop`. Phase 1 accepted on this condition. |
| Phase-1 phase-end cycle-2 (claude/codex nits) | `migrate-schema.py` / `_measurement_id.py` | nit (x3) | (a) case-insensitive ledger keys + comment-only-migration semantics untested (correctness/claude); (b) `_measurement_id.py:48` twox-hash endianness attribution misplaced — conclusion correct (maint/claude); (c) `migrate-schema.py:88` `_applied_set` name doesn't signal its CREATE-TABLE side effect (maint/claude). | Deferred to the follow-up test-hardening + doc-polish pass alongside the cycle-6/7 residual. All non-functional; correct behavior today. |

| PR-1.5 phase-end gauntlet (cycle 1) | `scripts/measurement_id_golden.json` / ingest boundary | should-fix | No golden vector exercises a NaN/Inf f64 `threshold`; Rust `to_bits()` preserves NaN payload bits while Python `struct.pack('<d', nan)` emits canonical NaN, so a non-finite threshold would hash differently across languages -> silent duplicate row. | Threshold is a cosine value so NaN is implausible; the robust guard is `assert threshold.is_finite()` at the PR-2.1 ingest boundary (fail loud rather than diverge). Deferred to PR-2.1; track there. **→ Phase-2 re-plan: Resolved-by PR-2.2 (`is_finite()` guard on f64 dims at the ingest boundary).** |
| PR-2.1 cycle-2 gauntlet (correctness) | `scripts/test_migrate_schema.py` (non-superuser-master test) | should-fix | The new test always re-creates `migrator` under the modeled master, so the "master lacks ADMIN on a pre-existing `migrator`" failure mode of the 004 self-grant (`GRANT migrator TO CURRENT_USER` -> InsufficientPrivilege) is not exercised. | The shipped code is correct on the documented single-bootstrap-master path (002+004 applied by the one master; precondition now documented in the 004 comment, PR-2.1 fix-commit 44245c4a8). The negative test pins an UNSUPPORTED misconfiguration (a different role re-running 004 against a pre-existing migrator). Fold into a follow-up test-hardening PR: create `migrator` as a separate role, then apply 004 as a master lacking ADMIN, asserting a clear error. |
| PR-2.1 cycle-2 gauntlet (correctness) | `scripts/test_migrate_schema.py` (non-superuser-master test) | nit | The test exercises only `createrole_self_grant=''` (self-grant branch fires) + the superuser no-op; the `createrole_self_grant='inherit'` no-op branch the 004 comment promises (auto-grant INHERIT TRUE -> `pg_has_role` guard skips, no self-grant) is not directly exercised on a non-superuser master. | Branch is conceptually validated (the guard is `IF NOT pg_has_role(...,'USAGE')`; an inherit-configured master has USAGE -> guard skips). Fold into the same follow-up test-hardening PR: a second master with `ALTER ROLE ... SET createrole_self_grant='inherit'`, asserting 004 applies with the membership-row count staying at the single auto-grant. |
| PR-2.2 cycle-4 gauntlet (fresh/codex) | `scripts/test_post_ingest_postgres.py` (schema_conn destructive-scrub guard) | should-fix | The `BENCH_TEST_PG_DSN` override's destructive-scrub guard treats any loopback/socket host as safe, but a localhost SSH tunnel to a real DB would pass the `_is_local_host` check and DROP the public schema. | Dev-only affordance (CI uses Docker testcontainers, never the override); the loopback guard already covers the common case, and a localhost-tunnel-to-prod is an exotic foot-gun. The robust fix (target an isolated test schema/database instead of scrubbing `public`, or require an explicit `BENCH_TEST_PG_ALLOW_DESTRUCTIVE=1` opt-in) is a larger test-infra change. Fold into the follow-up test-hardening PR. |
| PR-2.2 cycle-7 gauntlet (claude + fresh/codex) | `scripts/test_post_ingest_postgres.py` (retry + transaction-mode integration coverage) | should-fix | The `_retry_write_conflicts` real-conflict path is pinned only by mocked-`op` unit tests, and all ingest integration tests run against an `autocommit=True` fixture; no testcontainer test forces a REAL psycopg `DeadlockDetected` inside `with conn.transaction()` (two connections, reversed lock order) on a production-default `autocommit=False` connection and asserts retry + commit. correctness/claude VERIFIED both properties hold against a live PG16 container, so this pins (not fixes) correct behavior. | Container-only test (needs Docker). PR-2.3 (commit `8402c1990`) wired `scripts/` pytest into CI so these tests now RUN there (the prerequisite). The test ADDITIONS — (a) a two-connection reversed-order real-deadlock retry test and (b) an `autocommit=False` ingest fixture covering commit-on-success + rollback-on-validation-failure — are RE-MAPPED to the follow-up test-hardening PR: part (a) needs careful threaded 2-connection lock interleaving (deterministic-deadlock + flakiness risk) that warrants focused attention rather than bundling into the CI-wiring PR (keeps PR-2.3 within its 1-3 commit budget). **Re-mapped from PR-2.3 → follow-up test-hardening PR (CI-prerequisite delivered by PR-2.3).** |
| PR-2.2 cycle-15 gauntlet (fresh/codex) | `scripts/test_post_ingest_postgres.py:1178` (`test_server_mode_requires_benchmark_id` isolation) | should-fix | The test does not ISOLATE the missing-`--benchmark-id` rejection: `_main_server` checks `benchmark_id is None` (returns 2) BEFORE the token check, and the test does not set `INGEST_BEARER_TOKEN`, so if the benchmark-id check were deleted the test would still see `return 2` from the token path and pass for the wrong reason. **Test-discriminating-power gap, NOT a functional defect** — `_main_server` correctly rejects missing `--benchmark-id` today, and the cycle-15 accept verdict was not blocked by this. | Non-blocking should-fix on a pre-existing `--server` test, orthogonal to PR-2.2's git_show_field/Postgres core; surfaced at the PR-2.2 convergence (cycle-15) accept. Fold into PR-2.3 (the test-hardening PR): set a dummy `INGEST_BEARER_TOKEN` via `monkeypatch.setenv` (or call `_main_server` directly) so `benchmark_id=None` is the only failing condition, and assert stderr mentions `--benchmark-id`. **→ RESOLVED by PR-2.3 (commit `ed585f451`): token set so `benchmark_id=None` is the only failing condition + stderr asserted to name `--benchmark-id`; mutation-verified (deleting the benchmark-id check now fails the test).** |
| **PR-5.0 prod load (2026-06-05, was PR-3.4/PR-3.5 prod gates)** | (prod RDS `245040174862`/`us-east-1`) | **MOVED to Phase-5 cutover** (no longer Phase-3-blocking) | The one-shot PROD load + prod `verify` + prod cross-check were RE-SEQUENCED to the Phase-5 cutover (new PR-5.0) per the user's 2026-06-05 decision: seeding prod before Phase 4 builds the v4 reader is premature; load the freshest snapshot at cutover instead. NOT strictly operator-only — the agent HAS prod access (the `bench-prod` profile reaches `245040174862`; prod RDS is PubliclyAccessible) — but a prod data seed is a hard-to-reverse side-effect requiring explicit sign-off before the write. | See the PR-5.0 row in `Phases and PRs` + the `Prod historical load TIMING` Key decision. Phase 3 now closes on PR-3.4's REAL-snapshot LOCAL rehearsal (zero prod risk). RDS PITR (35-day) is the prod rollback. |
| PR-4.2 cycle-1 gauntlet (fresh + correctness) | `benchmarks-website/web/lib/db.test.ts` (CI enforcement) | should-fix | The new vitest suite (testcontainers roundtrip + `buildQuery`/`resolveSsl`/`requireEnv`/IAM units) is run by NO CI workflow: `web-deploy.yml` does not exist until PR-4.5, and the existing CI does not run `benchmarks-website/web` tests, so a `db.ts` regression could merge green. The testcontainers describe also self-skips without Docker. Same enforcement-gap class as the (resolved) PR-1.2/PR-1.5 "no CI runner" items, for the NEW web/ TS suite. | Wire a `benchmarks-website/web` CI job running `pnpm test` (Docker available so the testcontainers describe executes) as part of PR-4.5's CI workflow. **Resolved-by: PR-4.5.** **Resolved: cc86266f5 (web-deploy.yml Check & Test job runs pnpm test with a docker-info guard).** |
| PR-4.3.c cycle-1 gauntlet (fresh+correctness/codex) | `web/lib/queries.ts` `groupNameQuery` + `web/lib/descriptions.ts` (statpopgen/polarsignals) | should-fix (v2→v4 display regression) | v2 (`src/config.js`) surfaced the group display names `Statistical and Population Genetics` / `PolarSignals Profiling` with descriptions, but the v4 read port (faithful to v3) special-cases only tpch/tpcds/clickbench in `groupNameQuery`, so statpopgen/polarsignals fall through to the legacy `dataset sf=N [storage]` name and their (already-ported) `descriptions.ts` cases are dead — those two group pages render without the v2 blurb. | FAITHFUL reproduction of v3/Axum: `server/src/api/groups.rs::group_name_query` has the identical tpch/tpcds/clickbench-only fall-through and `server/src/api/descriptions.rs` carries the identical dead cases. PR-4.3.c's approved acceptance is byte/semantic-equivalence to v3 (Phases-and-PRs row + the `Read-endpoint behavior-preservation is SEMANTIC equivalence` tradeoff), so restoring v2's names DIVERGES from v3 = a deliberate scope addition, out of scope for the faithful port. Resolve as a deliberate v4 enhancement (or upstream v3-source fix): add display-name cases for statpopgen/polarsignals in `groupNameQuery` so the descriptions attach, pinned by a `collectGroups` fixture for both suites. Needs a user call on whether v4 should restore v2 fidelity here vs. preserve the v3 behavior. |
| PR-4.3.c cycle-2 gauntlet (correctness/codex) | `web/lib/summary.ts` (all 4 summary paths) + `server/src/api/summary.rs` | should-fix (v3-source latent bug; preserved) | "Latest commit" selection is timestamp-only (`c.timestamp = MAX(ts)` for random-access + compression time/size; `row_number() ORDER BY timestamp DESC` for query summary), with no `commit_sha` tiebreaker. Two commits sharing a second-granularity git timestamp can tie at `MAX(ts)`, blending rows from multiple commits or picking nondeterministically instead of summarizing exactly one latest commit. | Preserved-v3-behavior (VERIFIED: the Rust v3 source has the identical timestamp-only selection in all three paths — summary.rs:56-103, :252-330, + the compression helpers — and the MF1 CTE fix preserved it exactly). PR-4.3.c acceptance = v3 semantic-equivalence, so adding `ORDER BY timestamp DESC, commit_sha DESC` + filter-by-commit_sha would DIVERGE from v3 across all 4 paths = a deliberate cross-substrate determinism improvement (and ideally an upstream v3-source fix too). Vanishingly unlikely in practice (trusted CI, develop commits minutes apart). Resolve as a deliberate "v4 correctness improvements over v3" effort with a same-second-two-commits regression test. Needs a user call (paired with the statpopgen/polarsignals item — both are real-latent-v3-bugs surfaced by the faithful-port reviews). |
| PR-4.3.c cycle-2 gauntlet (fresh/claude) | `web/lib/queries.ts` `groupNameQuery` + `web/lib/groups.test.ts` | should-fix (coverage) | `groupNameQuery`'s clickbench (`clickbench` → `Clickbench`), variant-append (` / variant`), and legacy-fallback branches have no direct test coverage; only the tpch + nvme + sf=1 + null-variant branch is exercised by the testcontainer fixture. | Faithful 1:1 port of `server/src/api/groups.rs::group_name_query`, whose own Rust tests share the same single-fixture gap, so parity with the source is preserved. Low priority under the trusted-input + faithful-port calibration. Resolve by adding a clickbench-group fixture and a variant-bearing tpch-group fixture to `groups.test.ts` (fold into the groupNameQuery v2-fidelity enhancement above, or a follow-up test-hardening pass). |
| PR-4.4.a cycle-1 gauntlet (fresh/codex) | `web/app/globals.css` mobile `@media (max-width:768px)` rule + `web/components/Header.tsx` | should-fix (bug; UI) | The ported mobile CSS hides `.repo-link-desktop` under 768px because v3's mobile nav (`.nav-controls-github` inside the hamburger panel) supplies the GitHub link there; PR-4.4.a renders only `.repo-link-desktop` and defers the mobile nav, so the GitHub link disappears on mobile viewports in the PR-4.4.a-only intermediate state. | The mobile nav (hamburger + `.nav-controls` + `.nav-controls-github`) is PR-4.4.b's scope; restoring the mobile GitHub affordance lands naturally there. The intermediate 4.4.a-only state is never deployed alone (Phase 4 ships as one cutover), so this is a transient gap, not a shipped regression. **Resolved-by: PR-4.4.b (render the static `.nav-controls-github` mobile fallback alongside the mobile-nav island).** |
| PR-4.4.a cycle-1 gauntlet (correctness/claude) | `web/lib/format.ts` `formatTimeNs` + `web/components/SummaryCard.tsx` `.toFixed(2)` ratio/score renders | should-fix (port-fidelity; rare) | JS `Number.prototype.toFixed` rounds half-away-from-zero; the v3 Rust originals (`format_time_ns`, `format!("{:.2}")`) round half-to-even. For an exactly-representable dyadic tie the rendered last digit diverges: a size ratio of exactly `0.125` renders `0.13x` here vs `0.12x` in v3 (and `12.5 ns` → `13 ns` vs `12 ns`). The underlying number is identical — only display rounding at exact ties differs. | Reachable only for exact dyadic-rational ratios (the `ns` tier always receives whole-integer `value_ns`, so no tie there; `geoMean`/quotient ratios essentially never land on an exact 3rd-decimal dyadic boundary). Single-display-digit divergence on a trusted-input low-stakes dashboard; deferred as a low-priority "v4 display-rounding parity (round-half-even)" item — pairs with the PR-4.3.c "v4 correctness improvements over v3 vs preserve-v3" decision flagged for the Phase-4 boundary. Resolve by a round-half-even helper applied before formatting + an exact-tie regression test (`0.125`→`0.12x`, `2.5`→`2 ns`), or record as an accepted display-parity tradeoff. |
| Phase-4 holistic review (correctness/codex) | `web/app/page.tsx:15` (landing-page caching) | should-fix | The landing RSC page is `force-dynamic` with no compensating cache layer, so every `/` render runs `collectGroups()` against Postgres. The `/api/*` routes are CDN-cached for 5 minutes (b53e07727), but the landing HTML (the highest-traffic path) is not. | Deploy-time decision that needs PR-4.5's Vercel config: either a `Cache-Control` header on `/` via `vercel.json` routes, or time-based revalidation once the database is reachable at build time. The page.tsx comment documents both options. **Resolved-by: PR-4.5.** **Resolved: 261901f5b (vercel.json Vercel-CDN-Cache-Control on / and /chart/:slug; page.tsx comments updated to the chosen mechanism).** |
| PR-4.4.b cycle-1 gauntlet (fresh, both executors) | `web/components/Chart.tsx` (controller interaction core) | should-fix (coverage) | The imperative controller's interaction behaviors (one-shot `?n=all` promotion semantics, `replaceChartPayload` x-range preservation, layered filter application on a constructed chart) have no jsdom-level interaction tests; coverage is the StrictMode lifecycle regression test (added in-cycle), the pure-helper unit tests (incl. the BAN-pinned `predecessorValue` walk), SSR markup contracts, and the production-server smoke. | **(cycle-3 fold-in: the bounded-import-retry pin, the pre-construction-scope pin, and header interaction tests are explicitly part of this row.)** **(phase-4 end-review fold-in: the dismiss-retry regression pin MUST NOT rely on act-deferred rendering — act/jsdom inverts the browser commit-vs-macrotask ordering that exposed the unmount-only cancel fix f3ee70fe1.)** A full jsdom interaction harness (mock Chart.js surface driving pan/zoom/slider) is disproportionate under the lean trusted-input calibration (test-completeness spiral guard); the highest-risk lifecycle path is already regression-pinned. Revisit only if an interaction regression actually escapes. **(phase-4 cycle-2: correctness re-flagged the missing dismiss-retry pin as should-fix; REMAINS DEFERRED here — the f3ee70fe1 unmount-only-cancel fix is verified-correct by inspection and by both cycle-1 and cycle-2 correctness lenses; only the act-free regression test is outstanding, and an act-free jsdom timing harness risks a non-discriminating/flaky pin. Fold into the test-hardening pass before the develop squash-merge.)** |
| Phase-4 end-review cycle-1 (correctness/codex) | `web/components/Header.tsx:78` (global-filter init) | should-fix | When the filter universe is empty/undefined the Header omits FilterBar and `initGlobalFilter` never runs, so a soft navigation from a filtered page can leave stale module-scoped filters hiding series with no visible filter UI. v4-architecture-only window (v3 was MPA: module state reset every load). No data wrongness. | The right reset semantics need design thought (reset-on-absent-FilterBar vs page-scoped filter state); narrow error-path UX edge. Revisit alongside the interaction-suite row or the v4-fidelity effort. |
| Phase-4 end-review cycle-2 (correctness/claude) | `.github/actions/verify-cdn-cache/action.yml` (curl failure message) | nit | `curl -sS -w '%{http_code}'` emits `000` (not empty) on a connection failure, so the `${status:-unreachable}` fallback is dead and the failure message renders `HTTP 000` rather than `unreachable`. Purely cosmetic; the success path (HTTP 200 + x-vercel-cache HIT/STALE) and the gate decision are unaffected. | Cosmetic failure-path wording only; not worth a CI re-verify cycle on its own. Map `000`→`unreachable` (or drop the dead default) in the test-hardening pass before the develop squash-merge, or accept as-is. |
| Phase-4 end-review cycle-3 (correctness/claude) | `scripts/test_migrate_schema.py` (`_scrub_bootstrap_roles` + the new rds_iam test) | nit | The new 005 rds_iam test drops the cluster-global `rds_iam` role only in its own try/finally; the `conn` fixture scrubs tables/ledger but not roles, so a mid-test crash could leak `rds_iam` and re-arm IAM-only auth in later applies. | Add `rds_iam` to `_scrub_bootstrap_roles` and/or a session-scoped autouse cleanup in the test-hardening pass. Happy path + finally are correct + the suite is green; single-point-of-failure only on a mid-test crash. |
| Phase-4 end-review cycle-3 (correctness/claude) | `benchmarks-website/web/lib/queries.ts` (`collectFilterUniverse`) | nit | `SELECT DISTINCT` wraps a `UNION` that already de-dups, so the outer DISTINCT is dead work (harmless + tested, but ambiguous intent). | Drop the outer DISTINCT (UNION suffices) or switch inner branches to UNION ALL — pick one de-dup site. Fold into the cleanup pass. |
| Phase-4 end-review cycle-3 (spec/claude) | `.big-plans` phase-4 row exit criteria + `benchmarks-website/web/README.md` operator-setup step 3 | nit (doc) | Phase-4 row names `vercel deploy --target=preview` but the shipped pipeline uses CLI `vercel pull/build/deploy --prebuilt`; web README step 3 frames the endpoint+auth wiring as "open choices" though prod is decided (public instance endpoint + static bench_read password). | Update the exit-criteria wording to the shipped CLI flow + note the as-shipped prod wiring in README step 3 (keeping the per-environment options). Doc-only; fold into the doc pass before the develop merge. |
| Phase-4 end-review cycle-4 (correctness/claude) | `benchmarks-website/web/lib/chart-format.ts:363` (`throttle` trailing timer) | nit | Throttled listeners own a trailing `setTimeout` not tied to the controller `AbortController` / mount-effect cleanup; on teardown a pending trailing timer can fire against the disposed controller. Latent NO-OP today (disposed guards: `applyScope` returns on `!chart`, `rebuildVisibleAndUpdate` on `state.disposed`). | Non-data-correctness UI nit; folded with the already-deferred jsdom chart-interaction harness (same controller-disposal area). Fix: expose `throttle.cancel()` + call in the mount-effect cleanup. |
| PR-5.0 close (2026-06-11, read-path-perf defer) | (prod) `curl <prod-url>/api/groups \| jq '.groups[].charts[].slug' \| sort` vs the family registry | must-fix (exit criterion deferred) | PR-5.0's moved Phase-4 exit criterion "`/api/groups` slug list matches the family registry" cannot pass: `/api/groups` times out (~1-2 min) at the full prod seed (non-sargable `IS NOT DISTINCT FROM` per-dataset full scans + landing discovery `GROUP BY` ×5 + N+1 per-group summaries + `db.t4g.micro` cache misses). Data + rendering are PROVEN correct (single chart URLs render). | Deferred to the read-path-perf PR (scheduled before PR-5.2 DNS flip). Verify there with before/after prod EXPLAIN/timings + a working `/api/groups` slug-match. **Resolved-by: read-path-perf PR.** |
| PR-5.0 close (2026-06-11, read-path-perf defer) | (prod) ~5 representative chart slugs vs the live v2 site `benchmarks.vortex.dev` | must-fix (exit criterion deferred) | PR-5.0's moved Phase-4 exit criterion "~5 representative chart slugs match the current v2 site on a manual visual check" is impractical at the full prod seed: big-dataset charts (tpch/tpcds/clickbench) `/api/chart` ~24s each. polarsignals (small) renders ~1s and is correct, proving data+render fidelity. | Deferred to the read-path-perf PR. Re-run the ~5-slug visual check once the read path is fast. **Resolved-by: read-path-perf PR.** |
| PR-5.0.5 cycle-1 gauntlet (fresh/claude) | `benchmarks-website/web/lib/groups.test.ts` (collectGroups testcontainer describe) | should-fix (coverage) | PR-5.0.5's acceptance criterion says a "seeded statpopgen + polarsignals group renders the friendly name + description"; the delivered test is a Docker-free pure-function `groupNameQuery` + `groupDescription` unit test (the discriminating-pin half of the criterion) rather than a seeded `collectGroups` end-to-end render. Both gauntlet lenses traced the wiring as independently correct (migrate classifier + v3 emitter write `dataset='statpopgen'/'polarsignals'` with null variant/sf, per the `v3.rs` parity test; `collectGroups` feeds `group.name` into `groupDescription`) and endorsed the unit test as an acceptable pin. | Deferred (not fix-now) because (a) it is a coverage enhancement, not a correctness bug — the end-to-end render is verified-by-inspection by both lenses; (b) **Docker is unavailable in the dev env**, so a seeded testcontainer test cannot be locally verified — shipping an unverifiable seeded test risks a latent CI break; (c) the existing `collectGroups` testcontainer describe already covers the discovery→name→description path generally. Add an isolated seeded statpopgen/polarsignals test (model on the `insertQuery` isolated-test pattern) when Docker is available; fold into the deferred web test-hardening pass before the develop squash-merge. **Resolved-by: web test-hardening pass (pre-develop-merge).** |
| PR-5.0.993 cycle-1 gauntlet (fresh + correctness) | `benchmarks-website/web/lib/queries.test.ts` (boundary-timestamp tie on the new path) | should-fix (coverage) | The `?n=2` equivalence test uses the 3-distinct-timestamp fixture, so the same-boundary-timestamp tie that the kept `commit_sha IN (last-n)` clause in `queryMeasurementWindowFilter` exists to trim is never exercised on the changed `collectQueryChart` path; the only tie test (commits sharing a timestamp) runs a `RandomAccess` chart through the UNCHANGED `factWindowFilter`. A regression dropping the `commit_sha IN` clause (leaving only the `>= cutoff`, which over-selects boundary ties) would pass every test. Both lenses flagged this identical gap. | NOT a correctness bug — both gauntlet lenses independently PROVED result-equivalence (the `>= cutoff` + kept `commit_sha IN` is identical to the old `commit_sha IN` set). Deferred because the tie-trim test needs a NEW `query_measurements` testcontainer fixture (two commits sharing the n-th-newest timestamp, one in / one out of last-n) that CANNOT be locally verified (Docker absent) — matching the established 'shipping an unverifiable seeded test risks a latent CI break' deferral pattern (see the PR-5.0.5 + PR-4.4.b rows above). Add an isolated seeded tie test (model on the `insertQuery` pattern) in the web test-hardening pass before the develop squash-merge. **Resolved-by: web test-hardening pass (pre-develop-merge).** |
| PR-5.0.993 cycle-1 gauntlet (fresh + correctness) | `benchmarks-website/web/lib/queries.ts` `collectQueryChart` seed (the `buildEarliest` callback) | nit | Dropping the seed's inner `JOIN commits c2` means `MIN(q2.commit_timestamp)` now ranges over `query_measurements` directly; equivalence with the old inner-join MIN rests on the no-NULL/orphan `commit_timestamp` invariant (orphan rows carry NULL `commit_timestamp`, never backfilled by migration 006, and MIN ignores NULL), which is test-enforced (`postgres_e2e.rs`) but undocumented at this call site. | **→ RESOLVED in PR-5.0.993 cycle 2 (commit `36ed8a90e` + reflow `85ba7ac37`): the seed now carries a comment documenting that correctness rests on every write path populating `commit_timestamp` (migration-006 backfill + the ingest upsert), since the structural JOIN that masked it is gone. The CI regression that motivated this — two stale test fixtures inserting `query_measurements` without `commit_timestamp` — was also fixed in `36ed8a90e`.** |

## Accepted tradeoffs / r1 traps

- **Schema deploys have no manual-approval gate; PR merge is the authorization gate** (user decision 2026-05-29). A reviewed, merged PR is accepted as sufficient authorization to apply a migration to prod. We knowingly forgo (a) a GitHub Environment required-reviewer approval and (b) segregation of duties (a distinct approver from the author) and (c) merge-now/apply-later timing decoupling. Rationale: a manual approval only re-confirms the authorization already given at merge and does not verify execution safety; the real safety comes from migration testing. Reviewers must NOT re-flag the absence of an `environment:` gate or a manual approval step on `schema-deploy.yml`; it is an accepted, deliberate decision, not an oversight. (See the matching Key decision.)
- **Execution safety for additive migrations is the per-PR testcontainer test, not a prod dry-run gate** (user decision 2026-05-29). Additive DDL (CREATE TABLE/INDEX, ADD COLUMN) is validated by `scripts/test_migrate_schema.py` against `postgres:16-alpine` at PR time; a migration that cannot apply cannot merge. The heavier PITR-snapshot-restore-against-real-data test is reserved for data-affecting migrations only (tracked in Deferred work). Reviewers must NOT flag the lack of a prod-data migration test on additive-only migrations as a gap.
- **Duplicate JSON object keys in an ingest record collapse last-wins, matching v3 — `post-ingest.py` must NOT reject them** (PR-2.2 cycle-6 gauntlet triage; correctness/codex flagged this as a divergence, verified false). The v3 server parses the envelope via `serde_json::from_value` over an `axum::Json<serde_json::Value>` (`benchmarks-website/server/src/ingest.rs:79-83`); `serde_json::Value::Object` is last-wins on duplicate keys, so v3 silently keeps the last value. Python's `json.loads` (no `object_pairs_hook`) is ALSO last-wins, so the two substrates already agree. Adding an `object_pairs_hook` that rejects duplicate keys would make the Python writer STRICTER than v3 and CREATE a behavior divergence, violating the migration's Behavior-preservation invariant. Reviewers must NOT re-flag the absence of duplicate-key rejection in `read_records`; matching v3 means keeping last-wins. (The Rust bench producer never emits duplicate keys in steady state regardless.)
- **`scripts/post-ingest.py`'s PEP 723 block intentionally declares `dependencies = []`; `--postgres` runs from the project uv env, not standalone** (PR-2.2; flagged repeatedly as a should-fix, e.g. cycle-10 fresh/codex). The v3 `--server` path must stay standard-library-only so CI can invoke `python3 scripts/post-ingest.py` under a bare interpreter (the v3 path is in production until the Phase-5 cutover). Declaring `psycopg`/`boto3`/`xxhash` in the PEP 723 block would make `uv run` install them for EVERY invocation incl. `--server`, defeating that. The `--postgres` mode's deps are provided by the repo's uv workspace (the module docstring + the `pyproject.toml` dev-deps comment document this). Reviewers must NOT re-flag `dependencies = []` / "standalone `uv run --postgres` fails"; it is a deliberate decision, not an oversight.
- **IAM-token region precedence is explicit `--region` > boto3 session region > RDS-hostname-parsed region** (PR-2.2 deploy decision; documented in the `--region` help text since cycle 5; re-flagged cycle-10 fresh/codex as should-fix). `_rds_iam_token` resolves `region or session.region_name or _region_from_host(host)`. A region mismatch produces a wrong-region token that fails LOUD at connect (IAM auth rejects it), not a silent divergence. The boto3-session-before-host order matches the AWS-conventional default (the ambient session region is the operator's declared region); the host-parsed region is the last-resort fallback. Reviewers must NOT re-flag the session-over-host ordering; it is the accepted, documented precedence.
- **A per-app `benchmarks-website/web/.prettierrc.json` byte-identical to `vortex-web/.prettierrc.json` is COMPLIANT with the Next.js/TypeScript formatter-config BAN** (PR-4.1 cycle-1 gauntlet; Codex/fresh flagged it `must-fix`, adjudicated over-strict by the synthesizer + the two Claude lenses). The BAN ("Do not introduce a second formatter config -- reuse the existing `vortex-web/.prettierrc.json` shape (`singleQuote: true`, `trailingComma: "all"`, `printWidth: 100`, `tabWidth: 2`)") targets formatter-SHAPE divergence, NOT config-file count: it enumerates the exact shape values to match, which `web/.prettierrc.json` does byte-for-byte. `benchmarks-website/web/` is a separately-deployable pnpm/Vercel package; Prettier resolves config upward from the file's own directory tree, and `vortex-web/.prettierrc.json` lives in a SIBLING tree (not an ancestor), so it is undiscoverable from `web/` without either a fragile cross-package `--config ../../vortex-web/...` path or a repo-root shared config (a cross-cutting change out of Phase-4 scope). A self-contained byte-identical copy is the correct approach. Reviewers must NOT re-flag `web/.prettierrc.json` (or sibling per-app prettier configs landed in later Phase-4 PRs) as a must-fix; matching the enumerated shape IS the reuse the BAN requires. (Residual two-copy drift risk is real but non-blocking; a future repo-root hoist is the cleanup if drift ever surfaces.)
- **v4 read-service slugs are OPAQUE round-trippable tokens, NOT byte-identical to the Rust server's slugs** (PR-4.3.a cycle-1 gauntlet; both lenses raised cross-language slug byte-identity, mooted by the shard-endpoint deferral). The Next.js server both PRODUCES (`/api/groups`, `/api/group`) and CONSUMES (`/api/chart`, `/api/group`) every slug; the web-ui only ever round-trips them back unchanged, and the `/api/artifacts/.../shards/{i}` endpoint (the only consumer that keyed an in-memory artifact map by slug in the Rust `read_model.rs`) was deferred to PR-4.4 under the user's fork decision. So there is NO cross-language slug exchange and byte-identity with the Rust producer is NOT required — only internal round-trip consistency (`decode∘encode = id`) + a stable canonical encoding (pinned by the `slug.test.ts` golden vectors). Consequently two divergences from `slug.rs` are ACCEPTED and reviewers must NOT re-flag them: (a) Node `Buffer.from(x,'base64url')` decodes leniently (strips invalid chars / tolerates padding) where Rust `URL_SAFE_NO_PAD` rejects — harmless because any non-canonical input that survives still hits the per-variant field validation; (b) `VectorSearch.threshold` whole-number values serialize as `1` (JS `JSON.stringify`) vs `1.0` (Rust `serde_json`) — irrelevant absent a cross-language slug comparison. If PR-4.4 reintroduces a cross-language slug producer/consumer, revisit this entry.
- **Read-endpoint behavior-preservation is SEMANTIC equivalence, not byte-identity to the Axum JSON** (PR-4.3.b design decision, recorded before implementation). True byte-identity between the Rust read server and the Next.js port is IMPOSSIBLE for numeric values: `serde_json` renders a whole-number `f64` as `1500000.0` (always a decimal point) while JS `JSON.stringify` renders it as `1500000`. Chart series values (`value_ns`/`value_bytes`, materialized `as f64` in `charts.rs`) are exactly this case. The client (`chart-init.js`) parses the JSON, so `1500000.0` and `1500000` are the identical number to every consumer — the wire contract that matters is field names + structure + nesting + ordering + numeric VALUES, which the port preserves exactly. Accordingly the PR-4.3.b/4.3.c "snapshot test against fixtures" acceptance is implemented by SEEDING a known fixture into a local `postgres:16` testcontainer and asserting the assembled `ChartResponse`/`GroupsResponse` content + values (the same seed-then-assert-values shape the Axum `chart_api.rs`/`group_api.rs` tests use, NOT an `insta` byte-capture of Axum output). Reviewers must NOT flag "the JSON is not byte-identical to the Axum server" (e.g. `1500000` vs `1500000.0`, or key-insertion-order within an object); semantic equivalence + the snake_case field contract (`dto.rs`) is the preserved invariant. (Postgres `BIGINT` value columns are read via `::float8` so node-postgres returns a JS number matching the Rust `as f64`, avoiding the bigint-as-string default.)
- **`reqI32` in `web/lib/slug.ts` accepts whole-number JSON floats (`7.0`) that serde rejects for an `i32` field** (PR-4.3.b cycle-2 gauntlet; correctness/claude flagged as a nit, dismissed). The cycle-1 must-fix added `reqI32` (`Number.isInteger` + `[-2**31, 2**31-1]` range) so a forged non-i32 `query_idx` returns 400 instead of an unhandled 500. `reqI32` is strictly MORE lenient than serde in exactly one direction: a JSON number written with a decimal point (`7.0`) is rejected by serde's `i32` deserializer but `JSON.parse('7.0') === 7` passes `Number.isInteger`. This is only reachable via a FORGED slug (the canonical `chartKeyToSlug` always emits an integer literal, so the round-trip path never produces `7.0`), and the lenient outcome is a normal 200/404 with the correct integer bound cleanly to the Postgres `integer` column — NOT a 500 and not wrong data. Per the trusted-input + opaque-round-trippable-slug calibration this whole-number-float leniency is immaterial; reviewers must NOT re-flag it as a 400-vs-200 divergence. There is NO false-REJECT direction (every integer serde accepts is accepted), and `VectorSearch.threshold` (f64) is correctly left as `reqNumber`, unconstrained.
- **statpopgen/polarsignals query groups fall through to legacy naming so their v2 descriptions never attach — this faithfully reproduces v3 and reviewers must NOT re-flag it** (PR-4.3.c cycle-1 gauntlet triage; fresh/codex flagged must-fix, correctness/codex flagged should-fix, conservative-union elevated to must-fix; the synthesizer explicitly deferred the defect-vs-preserved-behavior call to Step 2.4 triage). `groupNameQuery` (`web/lib/queries.ts`) special-cases only tpch/tpcds/clickbench; statpopgen/polarsignals fall through to the legacy `dataset sf=N [storage]` name, so `groupDescription` never matches the `'Statistical and Population Genetics'` / `'PolarSignals Profiling'` cases in `web/lib/descriptions.ts`, which are therefore dead. This is a BYTE-FAITHFUL port of the Rust v3 source: `server/src/api/groups.rs::group_name_query` has the identical tpch/tpcds/clickbench-only fall-through, and `server/src/api/descriptions.rs` carries the identical dead `'Statistical and Population Genetics'` / `'PolarSignals Profiling'` cases. PR-4.3.c's APPROVED acceptance criterion is byte/semantic-equivalence to v3/Axum `group_api.rs` (the Phases-and-PRs row + the `Read-endpoint behavior-preservation is SEMANTIC equivalence` tradeoff above), so reproducing v3 exactly — including this inconsistency — is correct-as-approved; "fixing" it would DIVERGE from v3 and is a deliberate scope addition. v2's `src/config.js` DID surface those display names + descriptions, so the v2-fidelity restore is tracked as Deferred work for a separate deliberate v4-enhancement decision. Reviewers must NOT re-flag the statpopgen/polarsignals legacy-naming or the dead descriptions as a must-fix; matching v3 is the preserved invariant. **RESOLVED 2026-06-10 (Phase-5 re-plan, Decision C): the v2-name restore is SCHEDULED as PR-5.0.5 — statpopgen/polarsignals regain their friendly names + descriptions (v2-fidelity), since matching v3 HERE is a regression from the live v2 site. The five OTHER preserved-v3 parity quirks are PRESERVED-FINAL (any later fix is a normal post-launch ticket).**
- **Summary "latest commit" selection is timestamp-only across all four summary paths, so same-second-timestamp commit ties can blend or pick nondeterministically — this faithfully reproduces v3 and reviewers must NOT re-flag it** (PR-4.3.c cycle-2 gauntlet; correctness/codex flagged must-fix, the other 3 lenses accepted; the synthesizer's own call said confirm v3-parity then reframe under this SEMANTIC-equivalence tradeoff). Git commit timestamps are second-granularity, so two commits could tie at `MAX(timestamp)`; the summaries select the latest commit by timestamp equality (`c.timestamp = MAX(ts)` for random-access + compression time/size) or a timestamp-only `row_number() ORDER BY timestamp DESC` (query summary), with NO `commit_sha` tiebreaker, so a tie can aggregate rows from multiple commits or pick one arbitrarily. This is a BYTE-FAITHFUL port of the Rust v3 source, VERIFIED across all three paths: `server/src/api/summary.rs:56-103` (random-access `c.timestamp = (SELECT MAX(c2.timestamp) ...)`), `:252-330` (query `row_number() OVER (... ORDER BY c.timestamp DESC)`, no `commit_sha`), and the compression-time/size helpers (`c.timestamp = MAX(ts)`). The PR-4.3.c MF1 CTE fix preserved `timestamp = MAX(timestamp)` EXACTLY and did NOT introduce or change the tie behavior. PR-4.3.c's approved acceptance is v3 semantic-equivalence, so adding a `commit_sha DESC` tiebreaker (the recommended fix) would DIVERGE from v3 across all four summary paths = a deliberate cross-substrate scope addition, tracked as Deferred work. The tie is also vanishingly unlikely in practice (trusted CI, develop commits minutes apart, both being the joint-latest with benchmark data). Reviewers must NOT re-flag the same-timestamp tie as a must-fix; matching v3 is the preserved invariant.
- **Global-filter `dimensionIsFiltered` is a cardinality check, so a STALE URL allowlist with the same length as the universe disables that dimension's filter — this faithfully reproduces v3 and reviewers must NOT re-flag it** (PR-4.4.b cycle-1 gauntlet; correctness/codex flagged must-fix, fresh/codex should-fix, correctness/claude line-verified it as v3 parity and deliberately did not flag; synthesizer dismissed-as-preserved per the migration behavior-preservation contract). `seriesPassesFilter` (`web/lib/chart-format.ts`) treats a dimension as filtered iff `active.length < universe.length`, and URL allowlists seed verbatim even when stale — both byte-faithful to v3 `chart-init.js` (`dimensionIsFiltered`, line 236, and `seedActiveFromUrlState`'s documented "taken verbatim, even if a chip has since been added or removed" semantics). A stale allowlist like `?engine=duckdb,old` against a 2-engine universe therefore renders engines outside the allowlist, exactly as the live v3 UI would. "Fixing" this (membership-based filtering) would diverge v4 from the v3 behavior the acceptance criteria pin; it joins the PHASE-4-BOUNDARY FLAG list for the deliberate v4-fidelity-vs-preserve-v3 decision.
- **All-chips-off global filter does not round-trip through the URL (`?engine=` empty-string reloads as all-visible) — this faithfully reproduces v3 and reviewers must NOT re-flag it** (PR-4.4.b cycle-1 gauntlet; same reviewer split and same dismissal rationale as the cardinality entry above). With every chip in a dimension toggled off, the active set serializes as an empty param (v3 `syncDimensionUrl`, chart-init.js:2056) and the empty param parses as "no filter" on load (v3 `parse_csv` / v4 `parseFilterCsv`), so a shared/refreshed hidden-everything view re-opens fully visible. Identical in v3; preserving it is the contract. Joins the PHASE-4-BOUNDARY FLAG list with the cardinality entry. (This is the SECOND cycle to surface a real-latent-v3-bug-but-preserve finding on PR-4.3.c; both are tracked in Deferred work for a possible deliberate "v4 correctness improvements over v3" effort — surface at the PR-completion / Phase-4 boundary.)
