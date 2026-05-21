<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# vortex-bench-server bootstrap and recovery walkthrough

A linear, copy-paste runbook for two scenarios:

1. **Fresh install**: empty EC2 host, no DB, no S3 backups yet. Phases 1 through 7.
2. **Disaster recovery**: rebuild the site from S3 backups onto a new host (the old host is gone or
   its DB is unrecoverable). Phases 1 through 6, then phase 8.

[`README.md`](README.md) is the topic-organized reference manual; this file is the recipe you follow
top-to-bottom. Every step has a verification command so you can confirm it landed before moving on.
If a verification fails, the troubleshooting note below it points at the most likely cause.

## Conventions

- `$` lines are shell commands. Lines without `$` are example output.
- Run everything as `ec2-user` on the EC2 host unless a step says otherwise. `sudo` is called
  explicitly where needed.
- The deploy timer cannot fetch over SSH. The repo's `origin` remote MUST be the HTTPS URL
  `https://github.com/vortex-data/vortex.git`. If you already cloned over SSH, fix it in place:
  `git -C ~/vortex remote set-url origin https://github.com/vortex-data/vortex.git`.
- `/var/lib/vortex-bench/ops/` is a directory symlink that `install.sh` creates pointing at
  `<repo>/benchmarks-website/ops/`. Every script under it lives in the repo; the symlink is the
  source-of-truth pointer. Deleting `~/vortex` breaks all five systemd units atomically.

## Phase 1: AWS prerequisites (one-time, from the AWS console)

Skip this entire phase if you are rebuilding into an EC2 instance that already has the
`VortexBenchServerRole` IAM role attached and the bucket lifecycle rule in place. Both survive
instance termination.

### 1.1 Create the IAM policy

In **IAM → Policies → Create policy**, paste:

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

Name it `VortexBenchV3Backups`.

### 1.2 Create the role and attach it

1. **IAM → Roles → Create role → AWS service → EC2**, attach `VortexBenchV3Backups`, name it
   `VortexBenchServerRole`.
2. **EC2 → Instances → bench instance → Actions → Security → Modify IAM role**, pick
   `VortexBenchServerRole`, click Update.
3. Wait about 15 seconds for the instance metadata service to refresh.

### 1.3 Create the S3 lifecycle rule

**S3 → Buckets → vortex-benchmark-results-database → Management → Lifecycle rules → Create lifecycle
rule**:

| Field        | Value                                          |
| ------------ | ---------------------------------------------- |
| Name         | `v3-backups-7d`                                |
| Status       | Enabled                                        |
| Filter scope | Prefix `v3-backups/`                           |
| Action       | Expire current versions, 7 days after creation |

7 days at one snapshot per hour is 168 tarballs. Tune up or down to taste.

### 1.4 Verify

```bash
$ aws sts get-caller-identity
# Arn should end in /VortexBenchServerRole/<instance-id>

$ echo probe > /tmp/probe.txt
$ aws s3 cp /tmp/probe.txt s3://vortex-benchmark-results-database/v3-backups/_probe.txt
$ aws s3 ls s3://vortex-benchmark-results-database/v3-backups/ | grep probe
$ aws s3 rm s3://vortex-benchmark-results-database/v3-backups/_probe.txt
$ rm /tmp/probe.txt
```

All four operations must succeed. If any fails with `AccessDenied`, check (1) the policy is actually
attached to `VortexBenchServerRole`, (2) the instance is using that role per
`aws sts get-caller-identity`, (3) there is no bucket policy denying access.

## Phase 2: Host packages (Amazon Linux 2023)

```bash
$ sudo dnf install -y \
    git curl jq \
    gcc gcc-c++ make cmake pkgconfig \
    util-linux openssl tar gzip
```

`util-linux` provides `flock`, which `deploy.sh` uses as a serialization guard. `gcc`, `gcc-c++`,
`cmake`, and `pkgconfig` are required by the `duckdb-sys` build.

### 2.1 Install the Rust toolchain for `ec2-user`

```bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
$ source $HOME/.cargo/env
$ rustc --version
```

`rustc --version` must succeed. The deploy timer runs `cargo build --release`, which needs this
exact toolchain installed for the user systemd runs the service as (`ec2-user`).

## Phase 3: Clone the repo

```bash
$ cd ~ && git clone https://github.com/vortex-data/vortex.git
$ cd vortex
$ git remote -v
origin  https://github.com/vortex-data/vortex.git (fetch)
origin  https://github.com/vortex-data/vortex.git (push)
```

The HTTPS URL is mandatory. If your `git config` defaults rewrite to SSH, undo that for this
checkout:

```bash
$ git -C ~/vortex remote set-url origin https://github.com/vortex-data/vortex.git
```

## Phase 4: Run the installer

```bash
$ ./benchmarks-website/ops/install.sh
```

This is idempotent. It creates `/var/lib/vortex-bench/` and `/var/log/vortex-bench/` owned by
`ec2-user`, drops a sudoers fragment at `/etc/sudoers.d/vortex-bench`, copies
`/etc/vortex-bench.env` from the template (mode 0600), symlinks `/var/lib/vortex-bench/ops` to the
repo's `ops/`, and installs the systemd units.

Expected tail of output:

```
[install] install complete. Next steps:
[install]   1. Edit /etc/vortex-bench.env (chmod 0600, owned by ec2-user)
[install]      - INGEST_BEARER_TOKEN=...
[install]      - ADMIN_BEARER_TOKEN=...
```

If the installer warns about an SSH `origin` remote, fix it now (see Phase 3) before starting the
timers. The deploy timer will silently fail every minute otherwise.

### 4.1 Verify

```bash
$ ls -ld /var/lib/vortex-bench
drwxr-xr-x. 7 ec2-user ec2-user 4096 ... /var/lib/vortex-bench

$ sudo ls -l /etc/vortex-bench.env
-rw-------. 1 ec2-user ec2-user ... /etc/vortex-bench.env

$ systemctl list-unit-files 'vortex-bench-*' --no-pager
vortex-bench-backup.service        static
vortex-bench-backup.timer          enabled
vortex-bench-deploy.service        static
vortex-bench-deploy.timer          enabled
vortex-bench-server.service        enabled
```

`enabled` for the two timers and the server unit is the expected state. The deploy and backup
service units are `static` because they have no `[Install]` section and are fired by their
respective timers, not enabled directly.

## Phase 5: Fill in the env file and start the timers

### 5.1 Generate the two bearer tokens

```bash
$ openssl rand -hex 32    # this becomes INGEST_BEARER_TOKEN
$ openssl rand -hex 32    # this becomes ADMIN_BEARER_TOKEN
```

Save the `INGEST_BEARER_TOKEN` to the GitHub Actions Environment that the bench CI workflow reads.
The `ADMIN_BEARER_TOKEN` never leaves the box.

### 5.2 Edit `/etc/vortex-bench.env`

```bash
$ sudo $EDITOR /etc/vortex-bench.env
```

Required fields (defaults are correct for the canonical layout):

```
INGEST_BEARER_TOKEN=<the first token from above>
ADMIN_BEARER_TOKEN=<the second token from above>
REPO_DIR=/home/ec2-user/vortex
DEPLOY_BRANCH=develop
S3_BACKUP_PREFIX=s3://vortex-benchmark-results-database/v3-backups
```

The remaining keys (`VORTEX_BENCH_DB`, `VORTEX_BENCH_BIND`, `VORTEX_BENCH_ADMIN_BIND`, `SERVER_URL`,
`ADMIN_URL`, `VORTEX_BENCH_SNAPSHOT_DIR`) already point at the canonical paths.

### 5.3 Start the timers

```bash
$ sudo systemctl start vortex-bench-deploy.timer
$ sudo systemctl start vortex-bench-backup.timer
```

The server unit starts itself once the deploy timer's first fire produces a binary. Do not
`start vortex-bench-server` directly yet, there is nothing for it to exec.

### 5.4 Watch the first deploy build the binary

```bash
$ journalctl -fu vortex-bench-deploy.service
```

The first fire takes about 60 to 90 seconds for a cold `cargo build --release`. Every log line is
prefixed with `[deploy <UTC-ts>]`. Look for these milestones (paraphrased; the literal substrings
to grep are bolded):

- **`building <7-char-sha> (was <prev>)`** -- cargo build starts.
- **`swapped symlink ->`** -- atomic binary swap landed; the next /health probe is imminent.
- **`deploy ok: <7-char-sha> -> live (binary <ts>)`** -- /health passed and the deploy committed
  the stamp. This is the success line.

`Ctrl-C` out of `journalctl` once the `deploy ok:` line appears. If a deploy fails it will exit
with one of the codes documented at the bottom of [`README.md`](README.md#failure-modes-conceptual)
(1 lock contention, 2 git fetch, 3 git rev-parse, 4 cargo build, 5 systemctl restart, 6 /health
failed but rolled back OK, 7 /health failed AND rollback also broken -- server is down).

## Phase 6: Verify the server is up

### 6.1 Public listener

```bash
$ curl -fsS http://127.0.0.1:3000/health | jq
{
  "status": "ok",
  "build_sha": "abc123...",
  "schema_version": "...",
  "db_path": "/var/lib/vortex-bench/bench.duckdb",
  "row_counts": {
    "commits": 0,
    "query_measurements": 0,
    ...
  }
}
```

Empty row counts are expected. The DB is created with an empty schema on first server boot.

### 6.2 Admin listener

```bash
$ /var/lib/vortex-bench/ops/inspect.sh "SELECT COUNT(*) FROM commits;"
```

A `0` is correct (DB is empty). A connection refused means `ADMIN_BEARER_TOKEN` was empty when the
server started: re-check `/etc/vortex-bench.env` and restart the server with
`/var/lib/vortex-bench/ops/restart.sh`.

### 6.3 Build SHA

```bash
$ readlink /var/lib/vortex-bench/bin/vortex-bench-server
/var/lib/vortex-bench/bin/vortex-bench-server.<UTC-ts>.<pid>

$ cat /var/lib/vortex-bench/last-deployed-sha
abc123...
```

The SHA in `last-deployed-sha` must match the `build_sha` in the `/health` JSON.

## Phase 7: Populate the database (fresh install only)

Pick **one** of 7.A or 7.B. Skip this entire phase if you are rebuilding from a backup (Phase 8
supplies the data).

### 7.A Run the v2 to v3 migration

This is the canonical path for a brand-new install. The migrator reads the v2 source (the public
S3 bucket of v2 result JSONs) and writes into the v3 DuckDB file.

```bash
$ source /etc/vortex-bench.env
# Check the migrator's own CLI for the up-to-date flag set. The wrapper passes args verbatim to
# `cargo run -p vortex-bench-migrate -- "$@"`, so the v2 source flag lives in that crate:
$ /var/lib/vortex-bench/ops/migrate.sh run --help
# Typical invocation (substitute the v2 source flag the --help output names):
$ /var/lib/vortex-bench/ops/migrate.sh run --output "$VORTEX_BENCH_DB"
```

`migrate.sh` stops the server, snapshots the current DB to `bench.prev-<ts>.duckdb` for rollback,
runs the migrator, and starts the server back up. The deploy and backup timers are paused for the
duration; they restart automatically on success.

If the migrator fails, the script leaves the server stopped and the timers paused, and prints the
exact rollback command. Follow it. Do not retry the migration without rolling back first or you will
pile new state on top of partially-migrated state.

### 7.B Promote an existing DuckDB file

If you already have a `bench.duckdb` from a previous host or a manual export:

```bash
$ sudo systemctl stop vortex-bench-server
$ cp /path/to/your/bench.duckdb /var/lib/vortex-bench/bench.duckdb
$ sudo systemctl start vortex-bench-server
$ curl -fsS http://127.0.0.1:3000/health | jq '.row_counts'
```

Row counts in `/health` should match the source DB.

### 7.C Verify the data landed

```bash
$ /var/lib/vortex-bench/ops/inspect.sh "
    SELECT 'commits' AS table_name, COUNT(*) AS n FROM commits
    UNION ALL SELECT 'query_measurements', COUNT(*) FROM query_measurements
    UNION ALL SELECT 'compression_times', COUNT(*) FROM compression_times
    UNION ALL SELECT 'compression_sizes', COUNT(*) FROM compression_sizes
    UNION ALL SELECT 'random_access_times', COUNT(*) FROM random_access_times
    UNION ALL SELECT 'vector_search_runs', COUNT(*) FROM vector_search_runs;
"
```

All six tables should have non-zero row counts that match what you expect from the source.

### 7.D Verify the backup loop end-to-end

Fire one snapshot by hand to prove the IAM role, the admin token, and the tarball pipeline all work:

```bash
$ sudo systemctl start vortex-bench-backup.service
$ journalctl -u vortex-bench-backup.service --since '2 min ago' --no-pager
[backup ...] triggering /api/admin/snapshot?ts=20260520T...
[backup ...] compressing /var/lib/vortex-bench/snapshots/... → ....tar.gz
[backup ...] compressed N → M bytes (Kx)
[backup ...] uploading ....tar.gz → s3://vortex-benchmark-results-database/v3-backups/....tar.gz
[backup ...] deleting local copies ...
[backup ...] snapshot 20260520T... ok → ...

$ aws s3 ls s3://vortex-benchmark-results-database/v3-backups/ | tail -3
```

The tarball must appear in the listing. If `aws s3 cp` fails with `AccessDenied`, redo Phase 1.4 to
debug the IAM role.

**At this point the system is fully self-driving.** Deploys land within 60 seconds of a develop
merge, snapshots upload every hour, the lifecycle rule expires old ones. You do not need to SSH back
in for routine operations.

## Phase 8: Disaster recovery: restore the DB from an S3 backup

Use this phase to rebuild onto a fresh host when the old host or DB is unrecoverable. **Do not run
it on a healthy host with a populated DB**, it overwrites the live `bench.duckdb`.

You must have completed Phases 1 through 6 first (the server runs against an empty schema). Skip
Phase 7.

### 8.1 Pick the snapshot you want to restore

```bash
$ aws s3 ls s3://vortex-benchmark-results-database/v3-backups/ | tail -20
2026-05-20 01:00:14   12345678 20260520T010000Z.tar.gz
2026-05-20 02:00:11   12345678 20260520T020000Z.tar.gz
...
```

The lifecycle rule keeps about 168 hourly snapshots (7 days). Pick the most recent known-good one,
or `tail -1` to grab the latest.

### 8.2 Download and extract

```bash
$ ts=20260520T020000Z   # replace with the timestamp from 8.1
$ cd /tmp
$ aws s3 cp "s3://vortex-benchmark-results-database/v3-backups/${ts}.tar.gz" .
$ tar xzf "${ts}.tar.gz"
$ ls /tmp/${ts}/
schema.sql
commits.vortex
query_measurements.vortex
compression_times.vortex
compression_sizes.vortex
random_access_times.vortex
vector_search_runs.vortex
```

Six `.vortex` files plus `schema.sql`. If any file is missing, the snapshot is incomplete, pick an
earlier one.

### 8.3 Stop the server and clear the empty DB

```bash
$ sudo systemctl stop vortex-bench-server
$ rm -f \
    /var/lib/vortex-bench/bench.duckdb \
    /var/lib/vortex-bench/bench.duckdb.wal
```

(`bench.duckdb` is owned by `ec2-user` per the install layout; deleting it does not need sudo.)

### 8.4 Install the duckdb CLI matching the bundled engine

```bash
$ duckdb --version
v1.5.x ...
```

The CLI version must be at least as new as the engine the server bundles (currently `1.5.x`). If
`duckdb` is missing or older:

```bash
$ curl -L https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip -o /tmp/duckdb.zip
$ unzip -j /tmp/duckdb.zip duckdb -d ~/bin
$ export PATH="$HOME/bin:$PATH"
$ duckdb --version
```

### 8.5 Rehydrate the DB from the snapshot

The block below uses `${ts}` from Phase 8.2; the guard re-derives it from `/tmp/` if a fresh shell
lost the variable, so this step is safe to copy-paste into a new terminal.

```bash
$ : "${ts:?ts is not set; \`ts=<the-timestamp>\` or re-source /etc/vortex-bench.env}"
$ [ -d "/tmp/${ts}" ] || { echo "missing /tmp/${ts}; redo 8.2" >&2; exit 1; }
$ duckdb /var/lib/vortex-bench/bench.duckdb <<EOF
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
```

The server's startup applies `CREATE TABLE IF NOT EXISTS` idempotently against this populated DB,
so 8.6 will not error on already-present tables.

If `INSTALL vortex` fails with `extension not found` and `8.4` does not fix it, the upstream
DuckDB community/core extension index is currently the only restore vector for v3 snapshots.
File an issue: a cargo-driven restore fallback (`vortex-bench-migrate restore --input ...`) is
worth building before the next disaster recovery if the extension goes intermittently missing.

If `INSTALL vortex` fails with `extension not found`, the CLI is older than the release that
publishes the vortex core extension. Upgrade per 8.4.

### 8.6 Fix ownership and start the server

```bash
$ sudo chown ec2-user:ec2-user /var/lib/vortex-bench/bench.duckdb
$ sudo systemctl start vortex-bench-server
$ curl -fsS http://127.0.0.1:3000/health | jq '.row_counts'
```

Row counts should match the source. If `/health` reports zeros, the `INSERT INTO` statements
silently no-op'd. The most common cause is `schema.sql` having already been read but the `INSERT`
step erroring, check the CLI output above.

### 8.7 Clean up

```bash
$ rm -rf /tmp/${ts} /tmp/${ts}.tar.gz
```

The hourly backup timer will resume on its own schedule. Watch one fire to confirm the recovered
host is fully online:

```bash
$ sudo systemctl start vortex-bench-backup.service
$ journalctl -u vortex-bench-backup.service --since '2 min ago' --no-pager
```

## Phase 9: Rollback a bad migration (alternative recovery path)

If you ran `migrate.sh` and the result is wrong but the host is otherwise healthy, you do not need
S3, the prev DB is on local disk.

```bash
$ ls -t /var/lib/vortex-bench/bench.prev-*.duckdb | head -3
/var/lib/vortex-bench/bench.prev-20260520T123000Z.duckdb
/var/lib/vortex-bench/bench.prev-20260518T040000Z.duckdb
```

Pick the most recent prev DB (just before the failed migrate ran):

```bash
$ prev=/var/lib/vortex-bench/bench.prev-20260520T123000Z.duckdb
$ sudo systemctl stop vortex-bench-server
$ mv "$prev" /var/lib/vortex-bench/bench.duckdb
$ [ -f "${prev}.wal" ] && mv "${prev}.wal" /var/lib/vortex-bench/bench.duckdb.wal
$ sudo systemctl start vortex-bench-server
$ sudo systemctl start vortex-bench-deploy.timer
$ sudo systemctl start vortex-bench-backup.timer
$ curl -fsS http://127.0.0.1:3000/health | jq '.row_counts'
```

The autopilot timers may have been stopped by the failed `migrate.sh`. Starting them here is
idempotent.

## Steady-state operations

These are the commands you actually use day-to-day. None of them require having read this file from
the top, and none of them touch the data path.

### Inspect

```bash
# Is the site up?
$ curl -fsS http://127.0.0.1:3000/health | jq

# Which build is actually running (three identifiers; all three should agree).
$ cat /var/lib/vortex-bench/last-deployed-sha       # what the deploy timer last rolled out
$ readlink /var/lib/vortex-bench/bin/vortex-bench-server   # what the symlink points at
$ curl -fsS http://127.0.0.1:3000/health | jq '.build_sha' # what the live process baked in

# Tail the next deploy live.
$ journalctl -fu vortex-bench-deploy.service

# Recent deploy attempts (success or failure).
$ journalctl -u vortex-bench-deploy.service --since '15 min ago' --no-pager

# Backup timer status (when did it last fire? when will it fire next?).
$ systemctl list-timers vortex-bench-backup.timer
$ journalctl -u vortex-bench-backup.service --since '4 hours ago' --no-pager

# Disk usage by component (when /var/lib/vortex-bench is filling up).
$ du -sh /var/lib/vortex-bench/* | sort -h
```

### Read the DB (no server stop)

```bash
# Ad-hoc read-only SQL. SELECT/WITH/PRAGMA/SHOW/DESCRIBE/EXPLAIN only.
$ /var/lib/vortex-bench/ops/inspect.sh "SELECT COUNT(*) FROM commits;"

# JSON output (handier in pipelines).
$ /var/lib/vortex-bench/ops/inspect.sh -j "SELECT * FROM commits LIMIT 1" | jq

# Row counts across all six tables.
$ /var/lib/vortex-bench/ops/inspect.sh "
    SELECT 'commits' AS t, COUNT(*) AS n FROM commits
    UNION ALL SELECT 'query_measurements', COUNT(*) FROM query_measurements
    UNION ALL SELECT 'compression_times',   COUNT(*) FROM compression_times
    UNION ALL SELECT 'compression_sizes',   COUNT(*) FROM compression_sizes
    UNION ALL SELECT 'random_access_times', COUNT(*) FROM random_access_times
    UNION ALL SELECT 'vector_search_runs',  COUNT(*) FROM vector_search_runs;
"
```

### Restart / redeploy

Three knobs, in increasing order of work done:

```bash
# (a) Restart the binary in place. No rebuild. Use this after editing
#     /etc/vortex-bench.env or recovering from a stuck connection.
#     Prints before/after pid + binary path + /health JSON (which carries
#     build_sha) so you can confirm the swap.
$ /var/lib/vortex-bench/ops/restart.sh

# (b) Run a deploy now if origin has moved (otherwise a no-op).
$ sudo systemctl start vortex-bench-deploy.service
$ journalctl -fu vortex-bench-deploy.service

# (c) Force-rebuild whatever is on $DEPLOY_BRANCH even when origin
#     has not moved (ignores the stamp file and the path filter).
$ /var/lib/vortex-bench/ops/force-rebuild.sh
$ journalctl -fu vortex-bench-deploy.service
```

### Backup operations

```bash
# Take a snapshot to S3 right now (out-of-band, identical to the timer's fire).
$ sudo systemctl start vortex-bench-backup.service
$ journalctl -fu vortex-bench-backup.service

# List recent backups in S3.
$ aws s3 ls s3://vortex-benchmark-results-database/v3-backups/ | tail -20

# Take a manually-tagged snapshot (e.g. before a risky operation).
# The ts label must match [A-Za-z0-9_-]{1,64}; the directory must not exist.
$ source /etc/vortex-bench.env
$ ts=manual-$(date -u +%Y%m%dT%H%M%SZ)
$ curl -fsS -X POST \
    -H "Authorization: Bearer ${ADMIN_BEARER_TOKEN}" \
    "${ADMIN_URL:-http://127.0.0.1:3001}/api/admin/snapshot?ts=${ts}"
# Then tar + s3 cp by hand (the timer-driven path does this automatically).
```

### Pause and resume the autopilot

```bash
# Stop auto-deploys while debugging or doing a tricky manual change.
$ sudo systemctl stop vortex-bench-deploy.timer
# ... investigate ...
$ sudo systemctl start vortex-bench-deploy.timer

# Stop hourly backups (rarely needed - they are safe under load).
$ sudo systemctl stop vortex-bench-backup.timer
$ sudo systemctl start vortex-bench-backup.timer

# Stop the site entirely (returns 502/connection refused to visitors).
$ sudo systemctl stop vortex-bench-server
$ sudo systemctl start vortex-bench-server
```

### Re-run the v2 to v3 migration

`migrate.sh` stops the server, snapshots the current DB to `bench.prev-<ts>.duckdb`, runs the
migrator, and starts the server back up. The autopilot timers are paused for the duration and
restart on success. On failure they intentionally stay stopped (see Phase 9 for the rollback
recipe).

```bash
$ source /etc/vortex-bench.env
$ /var/lib/vortex-bench/ops/migrate.sh run --output "${VORTEX_BENCH_DB:-/var/lib/vortex-bench/bench.duckdb}"
```

`migrate.sh`'s positional args pass straight through to `cargo run -p vortex-bench-migrate --`, so
the migrator's CLI is whatever the current branch says it is. As of writing it is
`run --output <path>`.

### Token rotation

`INGEST_BEARER_TOKEN`:

```bash
$ openssl rand -hex 32                                     # generate new value
# 1. Update the GitHub Actions Environment secret so CI uses the new value.
# 2. Edit /etc/vortex-bench.env with the new value.
$ sudo $EDITOR /etc/vortex-bench.env
$ /var/lib/vortex-bench/ops/restart.sh                     # picks up the new env
```

`ADMIN_BEARER_TOKEN`:

```bash
$ openssl rand -hex 32                                     # generate new value
$ sudo $EDITOR /etc/vortex-bench.env
$ /var/lib/vortex-bench/ops/restart.sh
# The next backup timer fire reads the env file fresh, so it picks up
# the new value automatically.
```

The two tokens are independent. Rotating one does not affect the other.

### Adding or removing an admin

Being an admin is three independent grants, not a single role:

```bash
# (1) SSH access to the EC2 box.
#     Append the new admin's public key to authorized_keys. They log in
#     as ec2-user (which is also the service identity).
$ sudo -u ec2-user $EDITOR /home/ec2-user/.ssh/authorized_keys
# Or use AWS SSM Session Manager: enable on the instance and add the
# admin's IAM principal to the instance's SSM connect IAM policy.

# (2) AWS console access for IAM/lifecycle/bucket-policy changes
#     (the runtime role intentionally cannot do these).
#     Grant via IAM user or SSO role with read/write on IAM and the
#     vortex-benchmark-results-database bucket.

# (3) Bearer-token knowledge, if they need to hit /api/admin/* from
#     their laptop. /etc/vortex-bench.env is mode 0600 owned by ec2-user;
#     anyone with SSH access can read it.
```

To revoke an admin: delete their key from `authorized_keys`, revoke their AWS console role, and
rotate `ADMIN_BEARER_TOKEN`. CI's `INGEST_BEARER_TOKEN` is unaffected because it lives in GitHub
Actions, not on the host.

### Disk pressure

`/var/lib/vortex-bench/` filling up has four typical causes (see
`du -sh /var/lib/vortex-bench/* | sort -h` to identify which):

```bash
# `bin/vortex-bench-server.<ts>.<pid>` accumulation - deploy.sh keeps the
# last $KEEP_BINARIES (default 3). To prune harder:
$ sudo $EDITOR /etc/vortex-bench.env                       # add KEEP_BINARIES=1
$ /var/lib/vortex-bench/ops/force-rebuild.sh               # next deploy enforces the new cap

# `snapshots/<ts>/` not deleted - backup.sh removes after a successful
# S3 upload, so leftover dirs imply the upload failed. Check:
$ journalctl -u vortex-bench-backup.service --since '4 hours ago'

# `bench.prev-<ts>.duckdb` accumulation from old migrations. These are
# kept on purpose for rollback. Delete by hand once verified:
$ ls -lt /var/lib/vortex-bench/bench.prev-*.duckdb
$ rm /var/lib/vortex-bench/bench.prev-<old-ts>.duckdb{,.wal}

# `bench.duckdb` itself growing - expected, hundreds of MB is normal.
```

## What to do if a step fails

| Symptom                                                                   | Likely cause                                                                   | Fix                                                                               |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `install.sh` exits with `ERROR: <ops_dir> not found. Set REPO_DIR=<repo path>.` | Running from outside the repo root or with a non-default `REPO_DIR`            | `cd ~/vortex && ./benchmarks-website/ops/install.sh`                              |
| `journalctl -u vortex-bench-deploy` shows `Permission denied (publickey)` | `origin` is the SSH remote                                                     | `git -C ~/vortex remote set-url origin https://github.com/vortex-data/vortex.git` |
| `journalctl -u vortex-bench-deploy` shows `cargo: command not found`      | Rust toolchain not installed for `ec2-user`                                    | Re-run Phase 2.1; the timer runs as `ec2-user`, not as you                        |
| First `curl /health` returns connection refused                           | Deploy timer has not produced a binary yet, or the build failed                | `journalctl -fu vortex-bench-deploy.service` and read the most recent failure     |
| `inspect.sh` returns 401 or 503                                           | `ADMIN_BEARER_TOKEN` was empty at server start, the admin listener never bound | Edit `/etc/vortex-bench.env`, `restart.sh`                                        |
| `backup.sh` logs `/api/admin/snapshot returned 000`                       | The server is not running, or the admin port is wrong                          | `systemctl status vortex-bench-server`, check `$ADMIN_URL` in the env file        |
| `backup.sh` logs `aws s3 cp failed`                                       | IAM role missing or wrong                                                      | Re-run Phase 1.4 to debug                                                         |
| `migrate.sh` exits with the rollback instructions                         | The migrator itself errored, the prev DB is intact                             | Follow the printed `mv` lines literally                                           |
| Phase 8.5 `INSTALL vortex` fails                                          | DuckDB CLI is older than the bundled engine                                    | Upgrade the CLI per Phase 8.4                                                     |
| `deploy.sh` exits 4 (`cargo build failed`)                                | Source-tree compile error                                                      | Read the build log in `journalctl -u vortex-bench-deploy.service`; fix and push   |
| `deploy.sh` exits 5 (`systemctl restart failed`)                          | systemd or sudoers issue                                                       | `systemctl status vortex-bench-server`; check the sudoers fragment at `/etc/sudoers.d/vortex-bench` |
| `deploy.sh` exits 6 (`/health failed, rolled back to prior binary`)       | New binary broken; prior binary healthy                                       | Fix the source and push the next commit; the live binary is the prior good one   |
| `deploy.sh` exits 7 (`/health failed AND rollback also broken`)           | Server is DOWN; both new and prior binaries fail /health                       | Pick a known-good binary from `/var/lib/vortex-bench/bin/`, `sudo ln -snT <chosen> /var/lib/vortex-bench/bin/vortex-bench-server`, `sudo systemctl restart vortex-bench-server` |

See [`README.md`](README.md#failure-modes-conceptual) "Failure modes (conceptual)" for the full
reference list. This file covers only the failure modes a bootstrap operator actually hits.
