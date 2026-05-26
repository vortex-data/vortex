<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# vortex-bench-server - operations runbook

This is the canonical guide for deploying and operating the v3 benchmarks site (`bench.vortex.dev`)
on EC2. It targets a fresh admin who has SSH access to the box and never seen the system before.

The contents of this directory are everything the EC2 host needs to build, run, deploy, back up, and
inspect the server. There is no out-of-tree state - every script and unit lives in
`benchmarks-website/ops/` and gets installed onto the host by [`install.sh`](install.sh).

## TL;DR

- One Rust binary (`vortex-bench-server`), one DuckDB file (`/var/lib/vortex-bench/bench.duckdb`).
- A systemd timer polls `origin/develop` every 60s. If commits in the range touch website-relevant
  paths it builds, atomically swaps the binary, and restarts the server. Otherwise it fast-forwards
  the working tree and exits.
- A second timer fires hourly, asks the server to write a per-table Vortex snapshot (`schema.sql` +
  one `<table>.vortex` per table), `tar czf`s it, and uploads to
  `s3://vortex-benchmark-results-database/v3-backups/<UTC ts>.tar.gz`. The vortex DuckDB extension
  is auto-installed from the DuckDB core extension repo on first call. Vortex compresses the
  BIGINT[] runtime arrays and string columns roughly an order of magnitude better than gzipped CSV -
  and dogfoods the project's own format.
- For ad-hoc reads, `inspect.sh` calls a bearer-gated `/api/admin/sql` endpoint instead of stopping
  the server.
- For DB-replacing operations (re-running the v2→v3 migration), `migrate.sh` stops the server,
  snapshots the current DB to `bench.prev-<ts>.duckdb`, runs the migration, and starts back up.

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

| Path                                                                 | Role                                                                        |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [`install.sh`](install.sh)                                           | One-time bootstrap on a fresh host. Idempotent.                             |
| [`deploy.sh`](deploy.sh)                                             | Pull → build (if needed) → atomic restart. Called by timer.                 |
| [`migrate.sh`](migrate.sh)                                           | Manual: stop, snapshot prev DB, run migrate, restart.                       |
| [`backup.sh`](backup.sh)                                             | Hourly: trigger `/api/admin/snapshot`, sync to S3, prune local.             |
| [`inspect.sh`](inspect.sh)                                           | Read-only SQL via `/api/admin/sql`, no server stop.                         |
| [`force-rebuild.sh`](force-rebuild.sh)                               | Re-run a deploy of `$DEPLOY_BRANCH` even when origin hasn't moved.          |
| [`restart.sh`](restart.sh)                                           | Restart the binary in place with visible before/after state.                |
| [`config/vortex-bench.env.example`](config/vortex-bench.env.example) | Template for `/etc/vortex-bench.env`.                                       |
| [`systemd/`](systemd/)                                               | Unit files installed into `/etc/systemd/system/`.                           |
| [`BOOTSTRAP.md`](BOOTSTRAP.md)                                       | Step-by-step bootstrap and recovery runbook (copy-paste, verify-as-you-go). |

**Every runnable command lives in [`BOOTSTRAP.md`](BOOTSTRAP.md).** This file explains *what* the
system is and *why* the moving parts are shaped the way they are. Operators run commands out of
`BOOTSTRAP.md`; the sections below are the conceptual companion you read before or after.

## How the system runs

### The deploy autopilot

`vortex-bench-deploy.timer` fires every 60s. The service it triggers fetches `origin/$DEPLOY_BRANCH`,
compares the tip SHA against `/var/lib/vortex-bench/last-deployed-sha`, and exits early if nothing
moved. When the SHA has moved, the script inspects the diff against the old SHA: it only rebuilds
when the change touches `benchmarks-website/server/`, `benchmarks-website/migrate/`,
`benchmarks-website/Cargo.toml`, the workspace `Cargo.toml`, or `Cargo.lock`. Everything else (e.g. a
vortex-array PR) fast-forwards the working tree so the next website change builds against fresh
dependencies, but the running binary is left alone.

When a rebuild is needed: `cargo build --release` produces a versioned binary at
`bin/vortex-bench-server.<UTC-ts>.<pid>`, the symlink at `bin/vortex-bench-server` swings to it
atomically, the server unit restarts, and `deploy.sh` probes `/health` for up to 30s. On any failure
(build, restart, health check) the symlink rolls back to the previous binary and the server restarts
on the old version. The stamp file is **not** written on a failed deploy, so the next timer fire
retries the same SHA. Fix the bug and push again.

The flock at `/var/lib/vortex-bench/.deploy.lock` serializes deploy / force-rebuild / manual-deploy
attempts. The `force-rebuild.sh` sentinel (`.force-rebuild` under `STATE_DIR`) bypasses the path
filter and the stamp comparison once, then deletes itself.

### Identifying the running build

Three identifiers should always agree on a healthy host:

| Source                                                   | What it represents                                                                       |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `cat /var/lib/vortex-bench/last-deployed-sha`            | What the deploy timer last successfully rolled out.                                      |
| `readlink /var/lib/vortex-bench/bin/vortex-bench-server` | Which versioned binary the symlink points at (its filename embeds the build timestamp). |
| `curl /health` → `.build_sha`                            | What `cargo build` saw at compile time, baked into the running process.                  |

`build_sha` is the source of truth. Disagreement means the running process is stale: a manual binary
swap, a `restart.sh` with no rebuild, or systemd holding an older pid open.

### The three "restart" semantics

Pick the smallest hammer:

- **`restart.sh`** restarts the existing binary. Cheapest. Useful after editing
  `/etc/vortex-bench.env` or recovering from a stuck connection. `build_sha` does not change; `pid`
  and `started` do.
- **Triggering the deploy service** runs the timer's normal flow now instead of waiting up to 60s.
  No-op if `origin/$DEPLOY_BRANCH` has not moved.
- **`force-rebuild.sh`** ignores both the stamp file and the path filter, so it rebuilds whatever's
  on `$DEPLOY_BRANCH` even when origin has not moved. Use this when you flipped `DEPLOY_BRANCH`, are
  recovering from wedged build artifacts, or want to redeploy "whatever is on the branch I'm
  tracking."

There is no "build whatever I have locally checked out" mode. The deploy script always builds
origin's tip, so to test a branch you push it first.

### Migration semantics

The v2 to v3 migration is destructive: it overwrites `/var/lib/vortex-bench/bench.duckdb` from the v2
source. `migrate.sh` treats this as the most dangerous operation in the system:

1. Pause both autopilot timers AND interrupt any in-flight service so a deploy or backup cannot race
   the migrator's exclusive DB access.
2. Stop the server.
3. Copy the current `bench.duckdb` (and any `.wal`) to `bench.prev-<UTC-ts>.duckdb` for instant
   rollback.
4. Run the migrator (positional args pass straight through to the `vortex-bench-migrate` CLI).
5. Start the server, poll `/health` for up to 30s.
6. On success, restart the autopilot timers. On **failure**, intentionally leave the autopilot
   timers stopped and print the rollback command. The prev DB is never auto-deleted; the operator
   removes it once they've verified the migration.

This means "deploy the latest commit" and "rerun the migration" are deliberately distinct actions.
The autopilot never touches the DB.

### The backup loop

`vortex-bench-backup.timer` fires hourly and `vortex-bench-backup.service` runs `backup.sh`. The
script POSTs to the loopback-only `/api/admin/snapshot` endpoint, which writes a per-table Vortex
snapshot (`schema.sql` plus one `<table>.vortex` per table) into
`$VORTEX_BENCH_SNAPSHOT_DIR/<ts>/`. The script then tars and gzips that directory, uploads to
`$S3_BACKUP_PREFIX/<ts>.tar.gz`, and deletes the local copies. The bucket lifecycle rule expires old
objects (default 7 days, 168 hourly snapshots).

Vortex compresses our shape (BIGINT[] runtime arrays + short strings) about an order of magnitude
better than gzipped CSV, and dogfoods the project's own format. The gzip on the tarball mostly
catches `schema.sql` and tar metadata, not the data columns themselves.

`/api/admin/snapshot` requires `ts` to match `[A-Za-z0-9_-]{1,64}` and refuses to overwrite an
existing snapshot directory (409). The same endpoint is used out-of-band before risky operations;
just pick a label the timer can't collide with (e.g. `manual-<ts>`).

### Why two listeners?

The server binds two ports: a public listener (`VORTEX_BENCH_BIND`, typically `0.0.0.0:3000`) carries
`/`, `/api/ingest`, and `/health`. A separate admin listener (`VORTEX_BENCH_ADMIN_BIND`, mandatorily
loopback-only, `127.0.0.1:3001`) carries `/api/admin/*`. The admin listener fails to start on a
non-loopback bind, so `/api/admin/*` cannot reach the public network even when the public bind opens
`0.0.0.0`. Hitting `/api/admin/*` on the public listener always 404s.

The admin router is mounted only when `ADMIN_BEARER_TOKEN` is set. With the env unset (e.g. local
dev) no admin listener is bound at all, and `backup.sh` / `inspect.sh` fail fast. There is no silent
"backups disabled" mode.

### The "three grants" admin model

There is no admin database. Being an admin is three independent things:

1. **SSH access to the EC2 box** as `ec2-user` (the same identity systemd runs the service as).
   Granted by adding the admin's public key to `authorized_keys`, or via AWS Systems Manager Session
   Manager.
2. **AWS console access** for the metadata the runtime role intentionally cannot reach (IAM, bucket
   policy, lifecycle rules). Granted via IAM user or SSO role.
3. **Bearer-token knowledge** for hitting `/api/admin/*` directly. Anyone with SSH access can read
   `/etc/vortex-bench.env`, so this grant follows from grant 1.

Revoking an admin therefore means revoking all three: drop the SSH key, revoke the AWS role, rotate
`ADMIN_BEARER_TOKEN`. CI's `INGEST_BEARER_TOKEN` is unaffected because it lives in GitHub Actions,
not on the host.

## State on disk

Every persistent file the system owns lives under `/var/lib/vortex-bench/` (state) or `/etc/`
(config):

| Path                                                        | Owner                           | Lifetime                                                                                |
| ----------------------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| `/var/lib/vortex-bench/bench.duckdb` (+ `.wal`)             | server                          | Live data; replaced by migrate, restored from S3.                                       |
| `/var/lib/vortex-bench/bench.prev-<ts>.duckdb`              | `migrate.sh`                    | Kept until operator deletes; rollback target.                                            |
| `/var/lib/vortex-bench/bin/vortex-bench-server`             | `deploy.sh`                     | Symlink to current versioned binary.                                                     |
| `/var/lib/vortex-bench/bin/vortex-bench-server.<ts>.<pid>`  | `deploy.sh`                     | Versioned binaries; last `$KEEP_BINARIES` (default 3) kept.                              |
| `/var/lib/vortex-bench/snapshots/<ts>/`                     | `/api/admin/snapshot`           | Transient; `backup.sh` deletes after S3 upload.                                          |
| `/var/lib/vortex-bench/last-deployed-sha`                   | `deploy.sh`                     | Stamp file; only written on success.                                                     |
| `/var/lib/vortex-bench/.deploy.lock`                        | `deploy.sh`                     | flock serialization guard.                                                               |
| `/var/lib/vortex-bench/duckdb-extensions/`                  | DuckDB                          | Writable extension install dir (`ProtectHome` blocks the default DuckDB path).           |
| `/var/lib/vortex-bench/ops`                                 | `install.sh`                    | Symlink to `<repo>/benchmarks-website/ops/`.                                             |
| `/etc/vortex-bench.env`                                     | `install.sh` then operator      | Mode 0600 owned by ec2-user; both server and timers read it.                             |
| `/etc/sudoers.d/vortex-bench`                               | `install.sh`                    | Grants the run user `systemctl restart`/`start`/`stop` on the v3 units only.            |
| `/etc/systemd/system/vortex-bench-*.{service,timer}`        | `install.sh`                    | The five units.                                                                          |

## First-time install and disaster recovery

Removed from this file. Both flows are now in [`BOOTSTRAP.md`](BOOTSTRAP.md): Phases 1 through 7 for
a fresh install, Phases 1 through 6 then 8 for a backup-restore rebuild, Phase 9 for rolling back a
botched migration. Each phase has a verification command so you find out immediately if a step did
not land. Edit `BOOTSTRAP.md` (not this file) when the procedure changes.

## Wire APIs the ops scripts depend on

These are the only server endpoints the operator scripts touch. They also constitute the public
admin contract for any future tooling.

The server exposes two listeners. The public listener carries everything operator-facing and
CI-facing; the admin listener stays loopback-only so `/api/admin/*` cannot reach the public network
even when the public bind opens `0.0.0.0`.

| Method + path                                                    | Bearer | Listener (env var)                              | Used by                        |
| ---------------------------------------------------------------- | ------ | ----------------------------------------------- | ------------------------------ |
| `GET /health`                                                    | none   | public (`$SERVER_URL`, `VORTEX_BENCH_BIND`)     | `deploy.sh` post-restart probe |
| `POST /api/ingest`                                               | ingest | public                                          | CI dual-write                  |
| `POST /api/admin/snapshot?ts=<id>`                               | admin  | admin (`$ADMIN_URL`, `VORTEX_BENCH_ADMIN_BIND`) | `backup.sh`                    |
| `POST /api/admin/sql` (body `{"sql": …}`, `?format=json\|table`) | admin  | admin                                           | `inspect.sh`                   |

`POST /api/admin/snapshot` writes `schema.sql` + per-table `.vortex` files; `ts` must match
`[A-Za-z0-9_-]{1,64}` and the directory must not exist (409 otherwise). `POST /api/admin/sql` allows
only `SELECT`/`WITH`/`PRAGMA`/`SHOW`/`DESCRIBE`/`EXPLAIN` and runs each statement inside
`BEGIN TRANSACTION READ ONLY`.

The admin router is mounted only when `ADMIN_BEARER_TOKEN` is set. With the env unset (e.g. in local
dev) no admin listener is bound at all - `backup.sh` and `inspect.sh` fail fast against
`$ADMIN_URL`, so there's no silent "backups disabled" mode. Hitting `/api/admin/*` on the **public**
listener always 404s, regardless of whether admin is configured.

See [`server/src/admin.rs`](../server/src/admin.rs) for the full contract and the validation rules.

## Failure modes (conceptual)

When something breaks, the symptom usually points at exactly one of these. For the actual diagnostic
and repair commands, see the symptom table in
[`BOOTSTRAP.md`](BOOTSTRAP.md#what-to-do-if-a-step-fails).

- **Deploys retry the same broken SHA forever.** The stamp file is only written on success, so a
  failing deploy attempts the same SHA on every 60s tick. Fix the bug and push, or pause the timer
  while you investigate.
- **`/health` is slow.** It runs six `SELECT COUNT(*)`s under the connection mutex. Over 1s during a
  benchmark ingest window is normal; over 30s means the mutex is stuck.
- **Disk filling under `/var/lib/vortex-bench/`.** Four culprits in order of likelihood: piled-up
  `bench.prev-*.duckdb` from old migrations, leftover `snapshots/<ts>/` directories (backup uploads
  failing), accumulated versioned binaries (`KEEP_BINARIES` too high), the live `bench.duckdb` itself
  (expected to grow over time).
- **Backups not running.** Either the timer is stopped, the IAM role is broken, or the server
  started without `ADMIN_BEARER_TOKEN` so the admin listener never bound and `curl` to it returns
  `000`.
- **Migrate failed partway.** `migrate.sh` leaves the server and the autopilot timers stopped on
  failure and prints the rollback commands on stderr. The prev DB is on local disk and complete;
  restore it before doing anything else.
- **Migrating to a new EC2 host.** Stand the new host up, take a final snapshot on the old host,
  restore from S3 on the new host, cut DNS. Total RPO is bounded by the backup timer interval (one
  hour by default).

## Local development

You don't need any of this to run the server locally:

```bash
INGEST_BEARER_TOKEN=dev \
ADMIN_BEARER_TOKEN=dev \
VORTEX_BENCH_DB=/tmp/bench.duckdb \
cargo run -p vortex-bench-server
```

The admin endpoints work the same as in production. The hourly timer and the deploy timer are
systemd-only - they have no local equivalent and don't need one.

## What's intentionally not here

- **Docker.** A previous iteration ran the server under `docker compose` with `watchtower` polling
  GHCR. We removed it: the binary is small enough that a build-on-host model is simpler, and systemd
  gives us atomic restarts and rollback for free. The v2 React site retains its image-based deploy
  (separate `Dockerfile` and CI workflow); v3 does not.
- **A push-based deploy.** A GitHub Actions workflow could push via SSM or SSH on every merge. We
  chose polling because (a) zero inbound surface on the EC2 box, (b) no shared secret to manage in
  CI, and (c) 60s is well under any reasonable expectation for a benchmarks site. If the polling
  becomes unworkable, swap `vortex-bench-deploy.timer` for an SSM-triggered ExecStart and the rest
  of `deploy.sh` doesn't change.
- **A dedicated SQL endpoint user.** `/api/admin/sql` is gated by the same admin token as
  `/api/admin/snapshot`. If you want per-operator audit, run a reverse proxy that adds a header and
  log it on the way through.
