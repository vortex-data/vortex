<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# vortex-bench-server — operations runbook

This is the canonical guide for deploying and operating the v3
benchmarks site (`bench.vortex.dev`) on EC2. It targets a fresh admin
who has SSH access to the box and never seen the system before.

The contents of this directory are everything the EC2 host needs to
build, run, deploy, back up, and inspect the server. There is no
out-of-tree state — every script and unit lives in
`benchmarks-website/ops/` and gets installed onto the host by
[`install.sh`](install.sh).

## TL;DR

- One Rust binary (`vortex-bench-server`), one DuckDB file
  (`/var/lib/vortex-bench/bench.duckdb`).
- A systemd timer polls `origin/develop` every 60s. If commits in the
  range touch website-relevant paths it builds, atomically swaps the
  binary, and restarts the server. Otherwise it fast-forwards the
  working tree and exits.
- A second timer fires hourly, asks the server to write a per-table
  Vortex snapshot (`schema.sql` + one `<table>.vortex` per table),
  `tar czf`s it, and uploads to
  `s3://vortex-benchmark-results-database/v3-backups/<UTC ts>.tar.gz`.
  The vortex DuckDB extension is auto-installed from the DuckDB core
  extension repo on first call. Vortex compresses the BIGINT[] runtime arrays
  and string columns roughly an order of magnitude better than
  gzipped CSV — and dogfoods the project's own format.
- For ad-hoc reads, `inspect.sh` calls a bearer-gated `/api/admin/sql`
  endpoint instead of stopping the server.
- For DB-replacing operations (re-running the v2→v3 migration),
  `migrate.sh` stops the server, snapshots the current DB to
  `bench.prev-<ts>.duckdb`, runs the migration, and starts back up.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ EC2 host (Amazon Linux 2023, ec2-user)                               │
│                                                                      │
│  /home/ec2-user/vortex/         ← git checkout (build context only)  │
│                                                                      │
│  /var/lib/vortex-bench/                                              │
│    bench.duckdb                 ← live DB                            │
│    bench.duckdb.wal                                                  │
│    bench.prev-<ts>.duckdb       ← pre-migration backup, never pruned │
│    bin/                                                              │
│      vortex-bench-server        ← symlink → versioned binary         │
│      vortex-bench-server.<ts>.<pid>                                  │
│                                   ← versioned (PID suffix breaks   │
│                                     same-second collisions), last  │
│                                     $KEEP_BINARIES (3)             │
│    snapshots/<ts>/              ← transient vortex-snapshot landing  │
│    last-deployed-sha            ← stamp file for the deploy timer    │
│    .deploy.lock                 ← flock guard                        │
│    ops -> /home/ec2-user/vortex/benchmarks-website/ops               │
│                                                                      │
│  /etc/vortex-bench.env          ← secrets, mode 0600                 │
│  /etc/sudoers.d/vortex-bench    ← lets ec2-user systemctl restart    │
│                                   the server with no password        │
│  /etc/systemd/system/                                                │
│    vortex-bench-server.service  ← serves :3000                       │
│    vortex-bench-deploy.service  ← oneshot, runs deploy.sh            │
│    vortex-bench-deploy.timer    ← every 60s                          │
│    vortex-bench-backup.service  ← oneshot, runs backup.sh            │
│    vortex-bench-backup.timer    ← hourly                             │
│                                                                      │
│  Logs: journalctl -u vortex-bench-{server,deploy,backup}             │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ aws s3 cp  <ts>.tar.gz
                              ▼
                ┌───────────────────────────────────────┐
                │ s3://vortex-benchmark-results-database/│
                │   v3-backups/                         │
                │     <UTC ts>.tar.gz                   │
                │       <UTC ts>/                       │
                │         schema.sql                    │
                │         <table>.vortex                │
                └───────────────────────────────────────┘
```

## Files in this directory

| Path                                       | Role                                                             |
|--------------------------------------------|------------------------------------------------------------------|
| [`install.sh`](install.sh)                 | One-time bootstrap on a fresh host. Idempotent.                  |
| [`deploy.sh`](deploy.sh)                   | Pull → build (if needed) → atomic restart. Called by timer.      |
| [`migrate.sh`](migrate.sh)                 | Manual: stop, snapshot prev DB, run migrate, restart.            |
| [`backup.sh`](backup.sh)                   | Hourly: trigger `/api/admin/snapshot`, sync to S3, prune local.  |
| [`inspect.sh`](inspect.sh)                 | Read-only SQL via `/api/admin/sql`, no server stop.              |
| [`force-rebuild.sh`](force-rebuild.sh)     | Re-run a deploy of `$DEPLOY_BRANCH` even when origin hasn't moved. |
| [`restart.sh`](restart.sh)                 | Restart the binary in place with visible before/after state.     |
| [`config/vortex-bench.env.example`](config/vortex-bench.env.example) | Template for `/etc/vortex-bench.env`.       |
| [`systemd/`](systemd/)                     | Unit files installed into `/etc/systemd/system/`.                |

## First-time install (on a fresh EC2 host)

This guide walks an admin who has never seen the system before from
"empty box + AWS account" to "site up, hourly backups landing in S3".
There are two parts: cloud-side setup (IAM role, bucket lifecycle) and
host-side setup (`install.sh`, env file, migration). Do them in that
order — the host-side scripts assume the IAM role is already attached.

### Host prereqs

- Amazon Linux 2023 (or any Linux with systemd, sudo, and curl).
- ec2-user has sudo (default on AL2023).
- Rust toolchain installed for the run user — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` if not already.
- `aws` CLI on PATH (Amazon Linux ships with it).
- `git`, `curl`, `jq` (or `python3`), `flock` (`util-linux`), `gcc`/`g++`,
  `cmake`, `pkg-config` (the duckdb-sys build needs these).
- The repo's `origin` remote must be the **HTTPS** URL
  (`https://github.com/vortex-data/vortex.git`), not `git@github.com:…`.
  The deploy timer runs as the unprivileged service user with no SSH
  agent, so SSH-based fetches fail with `Permission denied (publickey)`.
  Public-repo HTTPS reads are unauthenticated and just work.

### AWS setup (do this once, from the AWS console)

The server reads and writes a single S3 prefix —
`s3://vortex-benchmark-results-database/v3-backups/`. Configure two
things in AWS before touching the EC2 box:

**(a) An IAM role for the EC2 instance.** Least-privilege — only what
the runtime actually needs (read/write objects, list backups). Bucket
admin actions (lifecycle, policy) are intentionally not granted; you
manage those separately from the console.

In **IAM → Policies → Create policy**, paste this JSON and name it
`VortexBenchV3Backups`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::vortex-benchmark-results-database"
    },
    {
      "Sid": "ReadWriteV3Backups",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::vortex-benchmark-results-database/v3-backups/*"
    }
  ]
}
```

In **IAM → Roles → Create role**, pick "AWS service" → "EC2", attach
the `VortexBenchV3Backups` policy, name it `VortexBenchServerRole`.

In **EC2 → Instances → bench instance → Actions → Security → Modify
IAM role**, pick `VortexBenchServerRole` and Update. Wait ~15s for the
instance metadata service to refresh.

Verify on the EC2 box:

```bash
aws sts get-caller-identity        # Arn should end in /VortexBenchServerRole/<instance>
echo probe > /tmp/probe.txt
aws s3 cp /tmp/probe.txt s3://vortex-benchmark-results-database/v3-backups/_probe.txt
aws s3 ls s3://vortex-benchmark-results-database/v3-backups/
aws s3 rm s3://vortex-benchmark-results-database/v3-backups/_probe.txt
rm /tmp/probe.txt
```

If any of those four fail with `AccessDenied`, double-check (1) the
policy is actually attached to the role, (2) the instance is using the
new role (`aws sts get-caller-identity` shows the right name), and
(3) there isn't a bucket-level deny in
`S3 → bucket → Permissions → Bucket policy`.

**(b) An S3 lifecycle rule** so hourly snapshots don't accumulate
forever. The runtime role can't manage lifecycle (by design — it's
admin metadata, not runtime data), so do this in the console once:

In **S3 → Buckets → vortex-benchmark-results-database → Management →
Lifecycle rules → Create lifecycle rule**:

- Name: `v3-backups-7d`
- Status: Enabled
- Filter scope: Prefix `v3-backups/`
- Action: "Expire current versions of objects" → **7 days** after creation

Adjust the retention to taste (7 days × 24 hourly snapshots ≈ 170
tarballs). The bucket isn't versioned so you can ignore the
noncurrent-version sections.

### Host setup

```bash
# 1. Clone the repo (anywhere, but the env file's REPO_DIR must point at it).
#    Must be the HTTPS URL — the deploy timer has no SSH agent.
cd ~ && git clone https://github.com/vortex-data/vortex.git
cd vortex
# If you already cloned over SSH, fix the remote in place:
#   git remote set-url origin https://github.com/vortex-data/vortex.git

# 2. Run the installer. It needs sudo for /etc/, /var/lib/, and systemd.
./benchmarks-website/ops/install.sh

# 3. Fill in the env file the installer staged.
sudo $EDITOR /etc/vortex-bench.env
#    Generate the two tokens:
#       openssl rand -hex 32
#    Store INGEST_BEARER_TOKEN in the GitHub Actions Environment used by
#    .github/workflows/<bench>.yml so CI can keep posting.
#    ADMIN_BEARER_TOKEN never leaves the box (used only by ops/* scripts).

# 4. Start the deploy + backup timers. install.sh enables but does NOT
#    start the units when the env file still has empty tokens (so the
#    first run has live secrets). We start ONLY the timers here, not
#    vortex-bench-server itself — there's no binary at
#    /var/lib/vortex-bench/bin/vortex-bench-server until the deploy
#    timer's first fire builds and installs one. Once the binary
#    exists, vortex-bench-server starts automatically via the deploy
#    timer's restart step.
sudo systemctl start vortex-bench-deploy.timer
sudo systemctl start vortex-bench-backup.timer

# 5. Watch the first deploy build the binary and bring the server up
#    (~60-90s for a cold cargo build, then /health responds).
journalctl -fu vortex-bench-deploy.service

# 6. Smoke check (server is up but the DB is empty — schema applied,
#    no rows).
curl -fsS http://127.0.0.1:3000/health | jq
./benchmarks-website/ops/inspect.sh "SELECT COUNT(*) FROM commits;"

# 7. Populate the DB. migrate.sh stops the server, runs the migrator,
#    and restarts it. The deploy timer never does this — populating
#    the DB is a one-time admin action, distinct from deploying code.
/var/lib/vortex-bench/ops/migrate.sh run --output /var/lib/vortex-bench/bench.duckdb

# 8. Verify the backup loop end-to-end. Fire one backup manually and
#    confirm a tarball lands in S3.
sudo systemctl start vortex-bench-backup.service
journalctl -u vortex-bench-backup.service --since '2 min ago' --no-pager
aws s3 ls s3://vortex-benchmark-results-database/v3-backups/ | tail -3

# 9. (Alternative to step 7: preserve an existing $HOME/bench.duckdb
#    instead of re-migrating.)
sudo systemctl stop vortex-bench-server
sudo -u ec2-user mv ~/bench.duckdb /var/lib/vortex-bench/bench.duckdb
sudo systemctl start vortex-bench-server
```

After step 8, the system is fully self-driving: deploys happen
automatically within 60s of merge to develop, snapshots upload
automatically every hour, and the lifecycle rule expires old ones.
You don't need to SSH in for routine operations.

## Day-to-day operations

### "I pushed a website change — when does it ship?"

Within 60s of merge to `develop`. The deploy timer fires every minute,
notices the new SHA, checks whether the diff touches
`benchmarks-website/server/`, `benchmarks-website/migrate/`,
`benchmarks-website/Cargo.toml`, `Cargo.toml`, or `Cargo.lock`. If
yes, it builds, atomically swaps the binary, restarts, and confirms
`/health` is happy.

If the build fails or `/health` doesn't respond within 30s, the symlink
rolls back to the previous binary and the server restarts on the old
version. The stamp file is *not* updated, so the next timer fire
retries — fix the bug, push again.

Watch a deploy live:

```bash
journalctl -fu vortex-bench-deploy.service
```

Force a deploy right now (don't wait for the next tick):

```bash
sudo systemctl start vortex-bench-deploy.service
```

### "Which build is actually running?"

Three equivalent identifiers, in increasing levels of certainty:

```bash
# What the deploy timer last successfully rolled out:
cat /var/lib/vortex-bench/last-deployed-sha

# Which versioned binary the symlink currently points at:
readlink /var/lib/vortex-bench/bin/vortex-bench-server
# → /var/lib/vortex-bench/bin/vortex-bench-server.<UTC ts of build>.<pid>

# What the live process baked in at compile time:
curl -fsS http://127.0.0.1:3000/health | jq '{build_sha, db_path, schema_version}'
```

`build_sha` is the source of truth — it's the git SHA `cargo build`
saw when it produced the running binary. If it disagrees with
`last-deployed-sha`, the running process is stale (e.g. a manual
binary swap, or systemd is still running an older PID).

### "How do I manually restart or redeploy?"

Three knobs, in increasing order of work done:

**(a) Restart the running binary, no rebuild.** Cheapest restart;
useful after editing `/etc/vortex-bench.env` or recovering from a hung
connection. `build_sha` on `/health` will be unchanged afterwards.
`sudo systemctl restart vortex-bench-server` is silent on success;
prefer the wrapper, which prints before/after state so you actually
see the new pid:

```bash
/var/lib/vortex-bench/ops/restart.sh
```

Sample output:

```
BEFORE:
  pid:        12345
  started:    Wed 2026-05-08 14:30:01 UTC
  binary:     /var/lib/vortex-bench/bin/vortex-bench-server.20260508T143000Z.18342
  /health:    {"status":"ok",...,"build_sha":"abc123..."}

running: sudo systemctl restart vortex-bench-server

AFTER:
  pid:        12678        ← different pid proves it restarted
  started:    Wed 2026-05-08 14:35:42 UTC
  binary:     /var/lib/vortex-bench/bin/vortex-bench-server.20260508T143000Z.18342
  /health:    {"status":"ok",...,"build_sha":"abc123..."}

RESTART OK
```

The binary path and `build_sha` don't change (restart doesn't rebuild);
the **pid** and **started** values do. Exit 0 on success, 1 on failure
(with a pointer to `journalctl`) so the script is usable from automation.

**(b) Run a deploy now if origin has moved.** Triggers the same flow
the 60s timer runs. No-op if `origin/$DEPLOY_BRANCH` hasn't moved
since the last successful deploy.

```bash
sudo systemctl start vortex-bench-deploy.service
journalctl -fu vortex-bench-deploy.service
```

**(c) Force-rebuild `origin/$DEPLOY_BRANCH` even if origin hasn't
moved.** Ignores the stamp-file comparison and the path filter and
runs the full build → atomic swap → restart → `/health` check. Use
this when you want to redeploy "whatever's on the branch I'm tracking"
without waiting for a new commit — e.g. after flipping `DEPLOY_BRANCH`
or recovering from wedged build artefacts:

```bash
/var/lib/vortex-bench/ops/force-rebuild.sh
journalctl -fu vortex-bench-deploy.service
```

Under the hood, `force-rebuild.sh` drops a `.force-rebuild` sentinel
under `$STATE_DIR` and triggers `vortex-bench-deploy.service`. The
script consumes the sentinel on its next fire, so the very next
ordinary timer tick is a normal no-op again.

To test a branch that isn't `$DEPLOY_BRANCH`, edit the env file to
point `DEPLOY_BRANCH` at it, then call `force-rebuild.sh` (or wait
60s). The deploy script always builds origin's tip — there is no
"build whatever I have locally checked out" mode by design. Push to
a branch first.

### "A vortex-array PR landed — does the website rebuild?"

No. The path filter ignores anything outside the directories listed
above. The working tree still fast-forwards (so a future website
change builds against the latest deps) but the server keeps running.

If you ever want to force a rebuild against a non-website change, push
a no-op commit that touches `benchmarks-website/server/` (e.g. a
whitespace edit in `README.md`).

### "How do I re-run the v2→v3 migration?"

`migrate.sh` passes its args straight through to `cargo run -p
vortex-bench-migrate --`, so the migrator's CLI is whatever it is on
the current branch. As of writing the invocation is:

```bash
# $VORTEX_BENCH_DB lives in /etc/vortex-bench.env, not the operator
# shell, so source it before invoking. Falls back to the canonical path
# if the env var isn't set.
source /etc/vortex-bench.env
/var/lib/vortex-bench/ops/migrate.sh run --output "${VORTEX_BENCH_DB:-/var/lib/vortex-bench/bench.duckdb}"
```

The script stops the server, snapshots the current DB to
`/var/lib/vortex-bench/bench.prev-<ts>.duckdb` for instant rollback,
runs the migrator, and starts the server back up. Total downtime is
roughly one rebuild cycle.

`migrate.sh` also stops `vortex-bench-deploy.{service,timer}` and
`vortex-bench-backup.{service,timer}` on entry so the autopilot can't
race against the migrator's exclusive DB access. On success the timers
restart automatically; on **failure** they intentionally stay stopped
(the script's stderr instructions print the exact rollback sequence).

If the migrate fails partway, the script leaves the server stopped AND
the autopilot timers stopped. To roll back manually:

```bash
mv /var/lib/vortex-bench/bench.prev-<ts>.duckdb /var/lib/vortex-bench/bench.duckdb
[ -f /var/lib/vortex-bench/bench.prev-<ts>.duckdb.wal ] && \
    mv /var/lib/vortex-bench/bench.prev-<ts>.duckdb.wal \
       /var/lib/vortex-bench/bench.duckdb.wal
sudo systemctl start vortex-bench-server
sudo systemctl start vortex-bench-deploy.timer
sudo systemctl start vortex-bench-backup.timer
```

### "What's in the database right now?"

```bash
./benchmarks-website/ops/inspect.sh "
    SELECT dataset, COUNT(*) AS n
    FROM compression_times
    GROUP BY dataset
    ORDER BY n DESC;
"
```

Server-side validation only allows `SELECT`, `WITH`, `PRAGMA`, `SHOW`,
`DESCRIBE`, and `EXPLAIN`. Anything else is rejected with 403 — a
fat-fingered `UPDATE` or `DROP` cannot run through this path.

For the raw JSON (handier in pipelines):

```bash
./benchmarks-website/ops/inspect.sh -j "SELECT * FROM commits LIMIT 1" | jq
```

### "Where are the backups, and how do I restore?"

Hourly, automatic. List the most recent snapshots:

```bash
aws s3 ls s3://vortex-benchmark-results-database/v3-backups/ | tail -20
```

Each `<ts>.tar.gz` archive contains a single directory `<ts>/` with
a `schema.sql` (verbatim DDL the server applies on boot) and one
`<table>.vortex` per table. Restore on a fresh box:

```bash
sudo systemctl stop vortex-bench-server
cd /tmp
aws s3 cp s3://vortex-benchmark-results-database/v3-backups/<ts>.tar.gz .
tar xzf <ts>.tar.gz                     # extracts ./<ts>/
ts=<ts>                                 # e.g. 20260508T010000Z
sudo -u ec2-user rm -f /var/lib/vortex-bench/bench.duckdb \
                       /var/lib/vortex-bench/bench.duckdb.wal
duckdb /var/lib/vortex-bench/bench.duckdb <<EOF
INSTALL vortex;
LOAD vortex;
.read /tmp/${ts}/schema.sql
INSERT INTO commits             SELECT * FROM read_vortex('/tmp/${ts}/commits.vortex');
INSERT INTO query_measurements  SELECT * FROM read_vortex('/tmp/${ts}/query_measurements.vortex');
INSERT INTO compression_times   SELECT * FROM read_vortex('/tmp/${ts}/compression_times.vortex');
INSERT INTO compression_sizes   SELECT * FROM read_vortex('/tmp/${ts}/compression_sizes.vortex');
INSERT INTO random_access_times SELECT * FROM read_vortex('/tmp/${ts}/random_access_times.vortex');
INSERT INTO vector_search_runs  SELECT * FROM read_vortex('/tmp/${ts}/vector_search_runs.vortex');
EOF
sudo systemctl start vortex-bench-server
```

The `duckdb` CLI version needs to be recent enough that the vortex
core extension is published for it. If `INSTALL vortex` fails, upgrade
the CLI to match (or exceed) the version the server was built against
(`duckdb` crate `1.10502` ≈ DuckDB 1.5.x).

If you want to take an out-of-band snapshot (e.g. before a risky
operation), just call the same endpoint the timer does:

```bash
# Read $ADMIN_BEARER_TOKEN + $ADMIN_URL from the env file rather than
# greping the file in line (the grep below is whitespace-fragile and
# would mangle a token that contained '=').
source /etc/vortex-bench.env
ts=$(date -u +%Y%m%dT%H%M%SZ)
curl -fsS -X POST \
    -H "Authorization: Bearer ${ADMIN_BEARER_TOKEN}" \
    "${ADMIN_URL:-http://127.0.0.1:3001}/api/admin/snapshot?ts=manual-${ts}"
```

The response's `snapshot_dir` JSON field reports the server's
on-disk path. For the timer-driven backup loop, this is always
`${VORTEX_BENCH_SNAPSHOT_DIR}/<ts>` because the same env file feeds
both the server and `backup.sh`. The two MUST stay aligned —
`backup.sh` tars the path it computes from its own env, not the
server-returned path.

### "Token rotation"

`INGEST_BEARER_TOKEN`:

1. Generate a new value: `openssl rand -hex 32`.
2. Update the GitHub Actions Environment secret so CI uses the new value.
3. SSH in, edit `/etc/vortex-bench.env`, then `sudo systemctl restart vortex-bench-server`.

`ADMIN_BEARER_TOKEN`:

1. `openssl rand -hex 32`.
2. Edit `/etc/vortex-bench.env`, restart the server.
3. The next backup timer fire will use the new value (read from the env
   file at script invocation).

The two tokens are independent — rotating one doesn't affect the other.

### "Adding another admin"

There's no separate admin database — being an admin means three things,
each granted independently:

1. **SSH access to the EC2 box.** Append the new admin's SSH public key
   to `/home/ec2-user/.ssh/authorized_keys` (mode 0600 owned by ec2-user)
   on the live host. They'll be able to log in as `ec2-user`, which is
   the same identity systemd runs the service as. Alternatively, enable
   AWS Systems Manager Session Manager for the instance and add the new
   admin's IAM principal to the instance's SSM connect IAM policy —
   that avoids managing SSH keys at all.

2. **AWS console access** for the bits the runtime role can't reach
   (creating IAM roles/policies, editing the bucket lifecycle rule,
   running setup-time admin commands). Grant the new admin an IAM user
   or SSO role that can read/write IAM and the
   `vortex-benchmark-results-database` bucket. The exact scope is your
   call — read-only on IAM is enough to *audit* the setup; full write
   is needed to *change* it.

3. **The bearer tokens**, if they need to call the admin endpoints from
   their laptop or run `inspect.sh` directly. The tokens live in
   `/etc/vortex-bench.env` (mode 0600 owned by ec2-user); once they have
   SSH access they can read it. To revoke an admin's access to the
   tokens specifically, rotate `ADMIN_BEARER_TOKEN` (see above) — every
   admin who knew the old value loses access immediately.

The repo itself is the source of truth for *how* to operate the system
— every script and unit lives in [`benchmarks-website/ops/`](.).
A new admin who can SSH in and read `/etc/vortex-bench.env` has
everything they need to run the existing operations; the runbook above
covers the full surface.

To remove an admin: revoke their SSH key (delete the line from
`authorized_keys`), revoke their AWS console access, and rotate the
admin token. CI's `INGEST_BEARER_TOKEN` is unaffected — it's a separate
token tied to the GitHub Actions Environment, not to any individual.

## Wire APIs the ops scripts depend on

These are the only server endpoints the operator scripts touch. They
also constitute the public admin contract for any future tooling.

The server exposes two listeners. The public listener carries everything
operator-facing and CI-facing; the admin listener stays loopback-only so
`/api/admin/*` cannot reach the public network even when the public bind
opens `0.0.0.0`.

| Method + path                                                    | Bearer        | Listener (env var)                          | Used by                       |
|------------------------------------------------------------------|---------------|---------------------------------------------|-------------------------------|
| `GET /health`                                                    | none          | public (`$SERVER_URL`, `VORTEX_BENCH_BIND`) | `deploy.sh` post-restart probe |
| `POST /api/ingest`                                               | ingest        | public                                      | CI dual-write                 |
| `POST /api/admin/snapshot?ts=<id>`                               | admin         | admin (`$ADMIN_URL`, `VORTEX_BENCH_ADMIN_BIND`) | `backup.sh`                  |
| `POST /api/admin/sql` (body `{"sql": …}`, `?format=json\|table`) | admin         | admin                                       | `inspect.sh`                  |

`POST /api/admin/snapshot` writes `schema.sql` + per-table `.vortex`
files; `ts` must match `[A-Za-z0-9_-]{1,64}` and the directory must not
exist (409 otherwise). `POST /api/admin/sql` allows only
`SELECT`/`WITH`/`PRAGMA`/`SHOW`/`DESCRIBE`/`EXPLAIN` and runs each
statement inside `BEGIN TRANSACTION READ ONLY`.

The admin router is mounted only when `ADMIN_BEARER_TOKEN` is set. With
the env unset (e.g. in local dev) no admin listener is bound at all —
`backup.sh` and `inspect.sh` fail fast against `$ADMIN_URL`, so there's
no silent "backups disabled" mode. Hitting `/api/admin/*` on the
**public** listener always 404s, regardless of whether admin is
configured.

See [`server/src/admin.rs`](../server/src/admin.rs) for the full
contract and the validation rules.

## Failure modes & recovery

### Deploy keeps failing

Symptom: `journalctl -fu vortex-bench-deploy.service` shows repeated
build or `/health` failures, server stays on the old binary.

What's happening: the script's stamp file is only written on success,
so every tick retries the same SHA. Inspect:

```bash
sudo cat /var/lib/vortex-bench/last-deployed-sha
journalctl -u vortex-bench-deploy.service --since '15 min ago'
```

Recovery: fix the bug and push (the timer will pick it up). To stop
the retry loop while you investigate:

```bash
sudo systemctl stop vortex-bench-deploy.timer
# … debug …
sudo systemctl start vortex-bench-deploy.timer
```

### Server is up but `/health` is slow

`/health` runs five `SELECT COUNT(*)`s under the connection mutex. If
ingest is in flight it'll wait. > 1s is normal during the nightly
bench window; > 30s means the connection mutex is stuck.

```bash
journalctl -u vortex-bench-server --since '5 min ago'
sudo systemctl restart vortex-bench-server
```

### Disk filling up under `/var/lib/vortex-bench/`

Likely culprits and the order to check:

```bash
du -sh /var/lib/vortex-bench/* | sort -h
```

- `bench.duckdb` itself growing — expected; ~hundreds of MB after the
  v2 migration.
- `snapshots/` not being cleaned up — `backup.sh` deletes after a
  successful S3 sync. If the IAM role broke, hourly snapshots will pile
  up. `journalctl -u vortex-bench-backup.service` will show the
  upload errors.
- `bin/vortex-bench-server.<ts>*` accumulation — `deploy.sh` keeps the
  most recent `KEEP_BINARIES` (default 3). To prune harder, edit the
  env file and add `KEEP_BINARIES=1`, then trigger a deploy.
- `bench.prev-<ts>.duckdb` from old migrations — these are kept on
  purpose for rollback. Delete by hand once you've verified the
  current DB is good.

### Backup hasn't run

```bash
systemctl list-timers vortex-bench-backup.timer
journalctl -u vortex-bench-backup.service --since '4 hours ago'
```

Run one by hand:

```bash
sudo systemctl start vortex-bench-backup.service
journalctl -fu vortex-bench-backup.service
```

If the script reports `http_status=000` (connection refused) on the
admin URL, the server started without `ADMIN_BEARER_TOKEN` — the admin
listener never bound, so curl can't reach anything. Edit the env file,
restart `vortex-bench-server`, retry. (A 503 from
`require_admin_bearer` is a defensive belt-and-braces branch — in
production the admin router is unmounted when the token is unset, so
that branch is unreachable.)

### Migrating to a new EC2 host

1. Stand the new host up. Run `install.sh`. Fill the env file.
2. On the *old* host, take a final snapshot:
   `sudo systemctl start vortex-bench-backup.service` and wait.
3. On the *new* host, restore from S3 (see "Where are the backups").
4. Cut DNS over.

Total RPO is the gap between the last hourly snapshot and the cutover
moment — bounded by an hour by default, can be tightened by adding
extra `OnCalendar=` lines to the backup timer.

## Local development

You don't need any of this to run the server locally:

```bash
INGEST_BEARER_TOKEN=dev \
ADMIN_BEARER_TOKEN=dev \
VORTEX_BENCH_DB=/tmp/bench.duckdb \
cargo run -p vortex-bench-server
```

The admin endpoints work the same as in production. The hourly timer
and the deploy timer are systemd-only — they have no local equivalent
and don't need one.

## What's intentionally not here

- **Docker.** A previous iteration ran the server under
  `docker compose` with `watchtower` polling GHCR. We removed it: the
  binary is small enough that a build-on-host model is simpler, and
  systemd gives us atomic restarts and rollback for free. The v2 React
  site retains its image-based deploy (separate `Dockerfile` and CI
  workflow); v3 does not.
- **A push-based deploy.** A GitHub Actions workflow could push via
  SSM or SSH on every merge. We chose polling because (a) zero inbound
  surface on the EC2 box, (b) no shared secret to manage in CI, and
  (c) 60s is well under any reasonable expectation for a benchmarks
  site. If the polling becomes unworkable, swap `vortex-bench-deploy.timer`
  for an SSM-triggered ExecStart and the rest of `deploy.sh` doesn't
  change.
- **A dedicated SQL endpoint user.** `/api/admin/sql` is gated by the
  same admin token as `/api/admin/snapshot`. If you want per-operator
  audit, run a reverse proxy that adds a header and log it on the way
  through.
