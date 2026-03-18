# bench-data: Design Document

## Command Reference & State Machines

### `init`

Scaffolds a new dataset directory.

```
                    ┌───────┐
                    │ START │
                    └───┬───┘
                        │
                        ▼
               ┌─────────────────┐   yes   ┌───────────────┐
               │ dir exists?     ├────────► │ ERR: already  │
               └────────┬────────┘          │     exists    │
                        │ no                └───────────────┘
                        ▼
               ┌─────────────────┐
               │ mkdir dir/      │
               │ mkdir dir/data/ │
               │ write template  │
               │ dataset.yaml    │
               └────────┬────────┘
                        │
                        ▼
                    ┌────────┐
                    │  DONE  │
                    └────────┘
```

**Inputs:** name, parent dir (default `.`)
**Outputs:** `{dir}/{name}/dataset.yaml`, `{dir}/{name}/data/`
**Errors:**
- Directory already exists → error (no `--force` to overwrite)

**Notes:** Tries to read `git config user.name`/`user.email` for the author
field. Falls back to empty string if git isn't configured.

---

### `manifest`

Generates `manifest.json` by scanning `data/` and hashing every file.

```
                    ┌───────┐
                    │ START │
                    └───┬───┘
                        │
                        ▼
               ┌─────────────────┐   fail   ┌─────────────────┐
               │ read            ├─────────► │ ERR: parse fail │
               │ dataset.yaml    │           └─────────────────┘
               └────────┬────────┘
                        │ ok
                        ▼
               ┌─────────────────┐   no    ┌─────────────────────┐
               │ data/ exists?   ├────────►│ ERR: no data/ dir   │
               └────────┬────────┘         └─────────────────────┘
                        │ yes
                        ▼
               ┌─────────────────┐
               │ walk data/      │
               │  {fmt}/{tbl}/   │
               │  hash each file │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ write           │
               │ manifest.json   │
               └────────┬────────┘
                        │
                        ▼
                    ┌────────┐
                    │  DONE  │
                    └────────┘
```

**Inputs:** path to dataset directory
**Outputs:** `{path}/manifest.json`
**Errors:**
- `dataset.yaml` missing or unparseable
- `data/` directory missing
- File I/O errors during hashing

**Notes:** Overwrites any existing `manifest.json` without prompting.
Non-regular files (symlinks, directories inside table dirs) are silently
skipped.

---

### `validate`

Checks a dataset directory for correctness before pushing.

```
                    ┌───────┐
                    │ START │
                    └───┬───┘
                        │
                        ▼
               ┌─────────────────┐   no    ┌─────────────────────┐
               │ dataset.yaml    ├────────►│ problem: not found  │──►┐
               │ exists?         │         └─────────────────────┘   │
               └────────┬────────┘                                   │
                        │ yes                                        │
                        ▼                                            │
               ┌─────────────────┐   fail  ┌─────────────────────┐  │
               │ parse           ├────────►│ problem: parse fail │──►┤
               │ dataset.yaml    │         └─────────────────────┘   │
               └────────┬────────┘                                   │
                        │ ok                                         │
                        ▼                                            │
               ┌─────────────────┐                                   │
               │ check: name,    │──── problems? ───────────────────►┤
               │ desc, author,   │                                   │
               │ source.kind,    │                                   │
               │ source.parent   │                                   │
               └────────┬────────┘                                   │
                        │                                            │
                        ▼                                            │
               ┌─────────────────┐   no    ┌─────────────────────┐  │
               │ data/ exists?   ├────────►│ problem: no data/   │──►┤
               └────────┬────────┘         └─────────────────────┘   │
                        │ yes                                        │
                        ▼                                            │
               ┌─────────────────┐   no    ┌─────────────────────┐  │
               │ data/ has files?├────────►│ problem: empty      │──►┤
               └────────┬────────┘         └─────────────────────┘   │
                        │ yes                                        │
                        ▼                                            │
               ┌─────────────────┐                                   │
               │ manifest.json   │                                   │
               │ exists? compare │── stale? ────────────────────────►┤
               │ with current    │                                   │
               └────────┬────────┘                                   │
                        │                                            │
                        ▼                                            │
               ┌────────────────────┐   ┌────────────────────────┐   │
               │ return problems    │◄──┤ collect all problems   │◄──┘
               │ (exit 0 or 1)     │   └────────────────────────┘
               └────────────────────┘
```

**Inputs:** path to dataset directory
**Outputs:** list of problems (exit 0 = pass, exit 1 = fail)
**Validation rules:**
- `name`: non-empty, lowercase alphanumeric + hyphens only
- `description`: non-empty
- `author`: non-empty
- `source.kind`: one of `generator`, `external`, `production`, `derived`
- `source.parent`: required when `kind = "derived"`
- `data/`: must exist and contain at least one regular file
- `manifest.json`: if present, must match current file hashes

---

### `push`

Uploads a dataset to a remote repository.

```
                    ┌───────┐
                    │ START │
                    └───┬───┘
                        │
                        ▼
               ┌─────────────────┐   fail   ┌───────────────────┐
               │ read + validate ├─────────►│ ERR: invalid yaml │
               │ dataset.yaml    │          └───────────────────┘
               └────────┬────────┘
                        │ ok
                        ▼
               ┌─────────────────┐
               │ --force?        │
               └──┬──────────┬───┘
            yes   │          │ no
                  │          ▼
                  │  ┌───────────────┐   yes   ┌───────────────────┐
                  │  │ check catalog │────────►│ prompt:           │
                  │  │ for existing  │         │ "Replace? [y/N]"  │
                  │  └───────┬───────┘         └──┬────────────┬───┘
                  │          │ no                  │ y          │ N
                  │          │                     │            ▼
                  │          │                     │      ┌──────────┐
                  │          │                     │      │ ABORTED  │
                  │◄─────────┘◄────────────────────┘      └──────────┘
                  │
                  ▼
               ┌─────────────────┐   no    ┌──────────────────────┐
               │ data/ exists?   ├────────►│ ERR: no data/ dir    │
               └────────┬────────┘         └──────────────────────┘
                        │ yes
                        ▼
               ┌─────────────────┐  exists  ┌─────────────────────┐
               │ CAS create      ├─────────►│ ERR: another upload │
               │ {name}.uploading│          │ in progress         │
               └────────┬────────┘          └─────────────────────┘
                        │ ok (lock held)
                        ▼
               ┌─────────────────┐
               │ scan data/      │
               │ hash all files  │
               │ build manifest  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐   fail   ┌─────────────────────┐
               │ upload each     ├─────────►│ release lock        │
               │ data file       │          │ ERR: upload failed  │
               └────────┬────────┘          └─────────────────────┘
                        │ ok
                        ▼
               ┌─────────────────┐
               │ upload          │
               │ dataset.yaml    │
               │ manifest.json   │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐   fail   ┌─────────────────────┐
               │ upsert catalog  ├─────────►│ release lock        │
               │ write catalog   │          │ ERR: catalog failed │
               └────────┬────────┘          └─────────────────────┘
                        │ ok
                        ▼
               ┌─────────────────┐
               │ release lock    │
               │ delete .uploading│
               └────────┬────────┘
                        │
                        ▼
                    ┌────────┐
                    │  DONE  │
                    └────────┘
```

**Inputs:** dataset dir, remote URL, `--force` flag
**Outputs:** data uploaded, catalog updated
**Errors:**
- `dataset.yaml` invalid → error before any upload
- Dataset exists + no `--force` + user says N → abort
- Lock file exists → "another upload in progress"
- Upload failure → lock released, partial files left orphaned (cleaned by `gc`)
- Catalog write failure → lock released, files uploaded but not in catalog (cleaned by `gc`)

**Lock semantics:**
- File: `{dataset_name}.uploading` at repo root
- Acquired via `PutMode::Create` (CAS — fails if exists)
- Always released on exit (success or error)
- If backend doesn't support conditional put, falls back to best-effort overwrite

---

### `pull`

Fetches catalog + all manifests + descriptors (no data files).

```
                    ┌───────┐
                    │ START │
                    └───┬───┘
                        │
                        ▼
               ┌─────────────────┐
               │ fetch           │
               │ catalog.json    │──► (empty catalog if not found)
               │ write to local  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ for each dataset│◄──────────────────┐
               │ in catalog      │                    │
               └────────┬────────┘                    │
                        │                             │
                        ▼                             │
               ┌─────────────────┐   match  ┌────────┤
               │ local manifest  ├─────────►│ skip   │
               │ hash matches?   │          │ (log)  │
               └────────┬────────┘          └────────┘
                        │ mismatch or missing
                        ▼
               ┌─────────────────┐   fail
               │ fetch manifest  ├─────────► warn, continue
               │ from remote     │
               └────────┬────────┘
                        │ ok
                        ▼
               ┌─────────────────┐   fail
               │ fetch descriptor├─────────► warn, continue ──────►│
               │ from remote     │                                  │
               └────────┬────────┘                                  │
                        │ ok                                        │
                        ▼                                           │
                  next dataset ─────────────────────────────────────┘
                        │ (all done)
                        ▼
                    ┌────────┐
                    │  DONE  │
                    └────────┘
```

**Inputs:** remote URL, local mirror dir
**Outputs:** local `catalog.json`, per-dataset `manifest.json` + `dataset.yaml`
**Errors:**
- Remote unreachable → hard error
- Individual manifest/descriptor fetch failure → warn + continue

**Notes:** This is intentionally lenient — a single corrupt dataset shouldn't
block pulling the rest. The manifest hash check avoids redundant downloads.

---

### `checkout`

Downloads data files for a specific dataset.

```
                    ┌───────┐
                    │ START │
                    └───┬───┘
                        │
                        ▼
               ┌─────────────────┐   no    ┌──────────────────────┐
               │ local catalog   ├────────►│ ERR: run pull first  │
               │ exists?         │         └──────────────────────┘
               └────────┬────────┘
                        │ yes
                        ▼
               ┌─────────────────┐   no    ┌──────────────────────┐
               │ dataset in      ├────────►│ ERR: not found       │
               │ catalog?        │         └──────────────────────┘
               └────────┬────────┘
                        │ yes
                        ▼
               ┌─────────────────┐   no    ┌──────────────────────┐
               │ local manifest  ├────────►│ ERR: run pull first  │
               │ exists?         │         └──────────────────────┘
               └────────┬────────┘
                        │ yes
                        ▼
               ┌─────────────────┐
               │ for each file   │◄──────────────────┐
               │ in manifest     │                    │
               └────────┬────────┘                    │
                        │                             │
                        ▼                             │
               ┌─────────────────┐   match  ┌────────┤
               │ local file      ├─────────►│ skip   │
               │ exists + hash   │          │ (log)  │
               │ matches?        │          └────────┘
               └────────┬────────┘
                        │ no
                        ▼
               ┌─────────────────┐
               │ download from   │
               │ remote          │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐  mismatch  ┌─────────────────┐
               │ verify hash     ├────────────►│ ERR: corruption │
               └────────┬────────┘             └─────────────────┘
                        │ match
                        ▼
               ┌─────────────────┐
               │ write to local  │─────────────────────────────────►│
               └─────────────────┘                                  │
                        │ (all done)                                │
                        ▼                                           │
                    ┌────────┐                                      │
                    │  DONE  │◄─────────────────────────────────────┘
                    └────────┘
```

**Inputs:** dataset name, remote URL, local mirror dir
**Outputs:** data files in `{mirror}/{dataset_path}/data/`
**Errors:**
- No local catalog → "run pull first"
- Dataset not in catalog → "not found"
- No local manifest → "run pull first"
- Hash mismatch after download → hard error (data corruption in transit or at rest)

**Notes:** Requires `pull` first. Downloads only go through the remote store,
but path lookup uses the local catalog. This means if the remote has been
updated since the last `pull`, `checkout` uses stale metadata.

---

### `list`

Lists datasets from the local catalog.

```
               ┌───────┐
               │ START │
               └───┬───┘
                   │
                   ▼
          ┌─────────────────┐   no    ┌───────────────────────┐
          │ catalog.json    ├────────►│ "Run pull first"      │
          │ exists locally? │         │ (exit 0, not error)   │
          └────────┬────────┘         └───────────────────────┘
                   │ yes
                   ▼
          ┌─────────────────┐   yes   ┌───────────────────────┐
          │ catalog empty?  ├────────►│ "No datasets"         │
          └────────┬────────┘         └───────────────────────┘
                   │ no
                   ▼
          ┌─────────────────┐
          │ print table:    │
          │ NAME    PATH    │
          └────────┬────────┘
                   │
                   ▼
               ┌────────┐
               │  DONE  │
               └────────┘
```

**Inputs:** local mirror dir
**Outputs:** table to stdout
**Errors:** parse failure on catalog.json

**Notes:** Purely local — no network. Shows potentially stale data.

---

### `describe`

Shows details for a specific dataset from local mirror.

```
               ┌───────┐
               │ START │
               └───┬───┘
                   │
                   ▼
          ┌─────────────────┐   fail   ┌───────────────┐
          │ read local      ├─────────►│ ERR: no file  │
          │ catalog.json    │          └───────────────┘
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐   no    ┌───────────────┐
          │ dataset in      ├────────►│ ERR: not found│
          │ catalog?        │         └───────────────┘
          └────────┬────────┘
                   │ yes
                   ▼
          ┌─────────────────┐
          │ if dataset.yaml │
          │ exists: print   │
          │ name, desc,     │
          │ author, tags,   │
          │ source          │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ if manifest.json│
          │ exists: print   │
          │ file count,     │
          │ total size,     │
          │ per-file list   │
          └────────┬────────┘
                   │
                   ▼
               ┌────────┐
               │  DONE  │
               └────────┘
```

**Inputs:** dataset name, local mirror dir
**Outputs:** formatted details to stdout
**Errors:** catalog missing, dataset not in catalog

---

### `delete`

Removes a dataset from the catalog, optionally purging files.

```
               ┌───────┐
               │ START │
               └───┬───┘
                   │
                   ▼
          ┌─────────────────┐
          │ --force?        │
          └──┬──────────┬───┘
        yes  │          │ no
             │          ▼
             │  ┌─────────────────┐
             │  │ show warning    │
             │  │ (varies by      │
             │  │  --purge)       │
             │  │ "Continue? [y/N]│
             │  └──┬──────────┬───┘
             │     │ y        │ N
             │     │          ▼
             │     │    ┌──────────┐
             │     │    │ ABORTED  │
             │◄────┘    └──────────┘
             │
             ▼
          ┌─────────────────┐   no    ┌───────────────┐
          │ dataset in      ├────────►│ ERR: not found│
          │ catalog?        │         └───────────────┘
          └────────┬────────┘
                   │ yes
                   ▼
          ┌─────────────────┐
          │ remove from     │
          │ in-memory       │
          │ catalog         │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ --purge?        │
          └──┬──────────┬───┘
        yes  │          │ no
             ▼          │
    ┌────────────────┐  │
    │ list + delete  │  │
    │ all files in   │  │
    │ dataset dir    │  │
    └────────┬───────┘  │
             │          │
             ▼          │
             ◄──────────┘
             │
             ▼
    ┌────────────────┐
    │ write updated  │
    │ catalog.json   │
    └────────┬───────┘
             │
             ▼
         ┌────────┐
         │  DONE  │
         └────────┘
```

**Inputs:** dataset name, remote URL, `--purge` flag, `--force` flag
**Outputs:** catalog updated, optionally files deleted
**Errors:**
- User says N at confirmation → abort (exit 0)
- Dataset not found → error
- File deletion fails midway → error before catalog write (catalog unchanged, partial files deleted)
- Catalog write fails after purge → impossible (purge happens first)

---

### `gc`

Removes orphaned directories not referenced by the catalog, and cleans
up stale upload locks.

```
               ┌───────┐
               │ START │
               └───┬───┘
                   │
                   ▼
          ┌─────────────────┐
          │ read catalog    │
          │ collect all     │
          │ referenced paths│
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ list top-level  │
          │ objects + dirs  │
          │ in remote       │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ PHASE 1: scan   │
          │ .uploading files│◄───────────────┐
          └────────┬────────┘                │
                   │                         │
                   ▼                         │
          ┌─────────────────┐                │
          │ parse timestamp │                │
          │ from lock body  │                │
          └──┬──────────┬───┘                │
       stale │          │ active             │
             ▼          ▼                    │
    ┌────────────┐  ┌──────────────┐         │
    │ delete     │  │ record name  │         │
    │ lock file  │  │ as locked    │─────────┘
    └────────────┘  └──────────────┘
             │ (all locks processed)
             ▼
          ┌─────────────────┐
          │ PHASE 2: scan   │
          │ orphaned dirs   │◄───────────────┐
          └────────┬────────┘                │
                   │                         │
                   ▼                         │
          ┌─────────────────┐  yes           │
          │ dir name matches├──────► skip ───┘
          │ a locked name?  │
          └────────┬────────┘
                   │ no
                   ▼
          ┌─────────────────┐                │
          │ delete all files│                │
          │ in orphaned dir │────────────────┘
          └────────┬────────┘
                   │ (all done)
                   ▼
          ┌─────────────────┐
          │ return list of  │
          │ removed items   │
          └────────┬────────┘
                   │
                   ▼
               ┌────────┐
               │  DONE  │
               └────────┘
```

**Inputs:** remote URL
**Outputs:** list of removed directories and stale lock files
**Errors:** remote unreachable, file deletion errors

**Lock timestamp format:** `locked at 2024-01-01T00:00:00Z` (RFC 3339).
Default stale threshold: 1 hour.

---

### `verify`

Checks integrity of a dataset's files against the manifest.

```
               ┌───────┐
               │ START │
               └───┬───┘
                   │
                   ▼
          ┌─────────────────┐   no    ┌───────────────┐
          │ dataset in      ├────────►│ ERR: not found│
          │ catalog?        │         └───────────────┘
          └────────┬────────┘
                   │ yes
                   ▼
          ┌─────────────────┐
          │ fetch manifest  │
          │ check hash vs   │── mismatch ──► problem
          │ catalog entry   │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ for each file   │◄─────────────────────┐
          │ in manifest     │                      │
          └────────┬────────┘                      │
                   │                               │
                   ▼                               │
          ┌─────────────────┐                      │
          │ fetch file      │                      │
          │ check hash      │── mismatch ──► problem│
          │ check size      │── mismatch ──► problem│
          │ file missing    │──────────────► problem│
          └────────┬────────┘                      │
                   │ ok                            │
                   ▼                               │
             next file ────────────────────────────┘
                   │ (all done)
                   ▼
          ┌─────────────────┐
          │ return problems │
          │ (exit 0 or 1)   │
          └─────────────────┘
```

**Inputs:** dataset name, remote URL
**Outputs:** list of problems (exit 0 = pass, exit 1 = fail)
**Checks performed:**
- Manifest hash matches what catalog says
- Every file exists in remote
- Every file's SHA-256 matches manifest
- Every file's size matches manifest

---

## User Stories

### Story 1: First-time dataset author

> Alice wants to share a TPC-H SF100 dataset with her team.

```bash
# 1. Scaffold
$ bench-data init tpch-sf100
Initialized dataset at ./tpch-sf100
  1. Edit ./tpch-sf100/dataset.yaml
  2. Add files to ./tpch-sf100/data/{format}/{table}/
  3. Run: bench-data push ./tpch-sf100 --remote <url>

# 2. Edit the descriptor
$ vim tpch-sf100/dataset.yaml

# 3. Generate data and place it
$ mkdir -p tpch-sf100/data/parquet/lineitem
$ cp ~/generated/*.parquet tpch-sf100/data/parquet/lineitem/

# 4. Validate before pushing
$ bench-data validate tpch-sf100/
Validation passed

# 5. Push
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data
Push complete
```

**Flow check:** Clean path, no issues.

---

### Story 2: Consumer downloading data for benchmarks

> Bob needs the TPC-H data Alice published.

```bash
# 1. See what's available
$ bench-data pull --remote s3://vortex-bench-data
Pull complete

$ bench-data list
NAME                           PATH
tpch-sf100                     tpch-sf100-m9d2k4/

# 2. Check details
$ bench-data describe tpch-sf100
Name:        tpch-sf100
Description: TPC-H scale factor 100
Author:      Alice <alice@example.com>
...
Files: 8 total, 24.5 GiB

# 3. Download the actual data
$ bench-data checkout tpch-sf100 --remote s3://vortex-bench-data
Checkout complete

# 4. Use it
$ ls ~/.cache/vortex-bench-data/tpch-sf100-m9d2k4/data/parquet/lineitem/
lineitem_000.parquet  lineitem_001.parquet  ...
```

**Flow check:** Clean path. The two-step pull+checkout is deliberate — pull
is cheap (metadata only), checkout is expensive (data).

---

### Story 3: Updating an existing dataset

> Alice regenerated the data with a bugfix and needs to replace it.

```bash
# Without --force: interactive confirmation
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data
Dataset 'tpch-sf100' already exists at 'tpch-sf100-m9d2k4/'.
Replace it? [y/N] y
Push complete

# With --force: no prompt
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data --force
Push complete

# Clean up old orphaned data
$ bench-data gc --remote s3://vortex-bench-data
Removed: tpch-sf100-m9d2k4/
```

**Flow check:** Works, but the old directory is orphaned until `gc`.
See issue #1 below.

---

### Story 4: Two people push at the same time

> Alice and Charlie both try to push `tpch-sf100` simultaneously.

```bash
# Alice starts first — acquires lock
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data --force
# ... uploading files ...

# Charlie tries — lock exists
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data --force
Error: another upload is in progress for 'tpch-sf100'
       (lock file: tpch-sf100.uploading).
       If this is stale, delete the lock file manually and retry.

# Alice finishes — lock released
# Charlie retries — succeeds
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data --force
Push complete
```

**Flow check:** Lock prevents the race. If Alice's upload crashes and
the lock becomes stale, `bench-data gc` will clean it up automatically
(locks older than 1 hour are treated as stale).

---

### Story 5: Stale lock from a crashed upload

> Alice's upload crashed mid-way. The lock file remains.

```bash
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data --force
Error: another upload is in progress for 'tpch-sf100'

# gc cleans up stale locks (>1 hour old) AND orphaned partial uploads:
$ bench-data gc --remote s3://vortex-bench-data
Removed: tpch-sf100.uploading
Removed: tpch-sf100-a3b4c5/

# Now retry:
$ bench-data push tpch-sf100/ --remote s3://vortex-bench-data --force
Push complete
```

**Flow check:** gc handles both stale locks and orphaned upload directories.
For recent locks (<1 hour), gc skips them and their directories to avoid
interfering with a real in-progress upload.

---

### Story 6: Consumer has stale local data

> Bob pulled last week. Alice pushed a new version. Bob runs checkout.

```bash
# Bob's local catalog still says tpch-sf100-m9d2k4/
# Alice pushed tpch-sf100-x7y8z9/ and updated the catalog.

$ bench-data checkout tpch-sf100 --remote s3://vortex-bench-data
# Downloads from tpch-sf100-m9d2k4/ (stale path from local catalog!)
# Files download fine because the OLD data is still there (not gc'd yet)

# But if Alice already ran gc, the old files are gone:
$ bench-data checkout tpch-sf100 --remote s3://vortex-bench-data
# ERROR: file not found in remote: parquet/lineitem/lineitem_000.parquet.
#        If the dataset was recently updated, run `bench-data pull` to refresh
#        your local catalog, then retry checkout.

# Fix: re-pull first
$ bench-data pull --remote s3://vortex-bench-data
$ bench-data checkout tpch-sf100 --remote s3://vortex-bench-data
Checkout complete
```

**Flow check:** This is by design (pull is cheap, checkout is expensive).
The error message now tells the user to re-pull when files are not found.

---

### Story 7: Deleting a dataset

> Alice wants to remove the clickbench dataset entirely.

```bash
# Just remove from catalog (files remain for gc later):
$ bench-data delete clickbench --remote s3://vortex-bench-data
This will remove 'clickbench' from the catalog. Data files will remain
in remote storage until `gc`.
Continue? [y/N] y
Deleted 'clickbench' from catalog

# Remove from catalog AND delete all files:
$ bench-data delete clickbench --remote s3://vortex-bench-data --purge
This will permanently delete 'clickbench' from the catalog AND remove
all data files from remote storage.
Continue? [y/N] y
Deleted 'clickbench' from catalog

# Non-interactive (CI):
$ bench-data delete clickbench --remote s3://vortex-bench-data --purge --force
Deleted 'clickbench' from catalog
```

**Flow check:** Confirmation prompt now protects against accidental deletion.
`--force` skips the prompt for scripted use.

---

### Story 8: CI pipeline pushing datasets

> A CI job generates datasets nightly and pushes them.

```bash
$ bench-data push generated-data/ --remote s3://vortex-bench-data --force
Push complete
```

**Flow check:** `--force` is required for non-interactive use. Clean path.

---

### Story 9: Verifying a dataset after S3 outage

> After an S3 incident, team wants to verify data integrity.

```bash
$ bench-data verify tpch-sf100 --remote s3://vortex-bench-data
Verification passed

# Or if corruption occurred:
$ bench-data verify tpch-sf100 --remote s3://vortex-bench-data
Verification failed:
  - parquet/lineitem/lineitem_003.parquet: hash mismatch (expected abc..., got def...)
```

**Flow check:** Downloads every file to check hashes. Could be very
slow/expensive for large datasets. See issue #6.

---

## Issues Found

### Issue 1: Old data orphaned on force-push (medium)

When pushing a new version of an existing dataset, the old
`{name}-{rand}/` directory is left in remote storage. The catalog
points to the new path, but old files remain until `gc`.

**Impact:** Storage waste. For large datasets (100+ GB), this could
be significant.

**Options:**
- (a) Auto-delete old files during push (before or after catalog update)
- (b) Document that `gc` should be run after force-push
- (c) Add `--clean` flag to push that deletes old version

**Recommendation:** (a) — delete old files after successful catalog
update. If deletion fails, warn but don't fail the push (the data is
just orphaned, not lost).

---

### Issue 2: `gc` can delete in-progress uploads (high) — FIXED

If user A is mid-push (files uploaded, catalog not yet updated) and
user B runs `gc`, user B sees the new directory as orphaned and deletes
it. User A's push then fails or produces a corrupt dataset.

**Impact:** Data loss during concurrent push + gc.

**Fix (implemented):** `gc` now has two phases:
1. Scan for `.uploading` lock files. Parse their timestamps. If the
   lock is older than the stale threshold (default: 1 hour), delete it.
   Otherwise, record the dataset name as "locked".
2. When removing orphaned directories, skip any whose name prefix
   matches a locked dataset.

This also resolves issue #3: stale locks are automatically cleaned up
by `gc` based on timestamp age, no separate `unlock` command needed.

---

### ~~Issue 3: No CLI command to break a stale lock (medium)~~ — RESOLVED BY #2

Stale locks are now cleaned up automatically by `gc`. The lock file
contains a timestamp (`locked at 2024-01-01T00:00:00Z`), and `gc`
treats locks older than 1 hour as stale.

---

### Issue 4: `delete --purge` has no confirmation (medium) — FIXED

`delete --purge` permanently removes all data files from remote
storage with no confirmation prompt and no `--force` flag. This is
the most destructive operation in the tool.

**Fix (implemented):** `delete` now prompts for confirmation with a
message that varies based on `--purge`. Use `--force` to skip the
prompt (for CI/scripts).

---

### Issue 5: Stale checkout gives confusing errors (low) — FIXED

When `checkout` uses a stale local catalog and the remote files have
been gc'd, the error is a raw "NotFound" per file. The user has to
figure out that they need to re-pull.

**Fix (implemented):** Checkout now catches NotFound and provides a
helpful error: "If the dataset was recently updated, run `bench-data
pull` to refresh your local catalog, then retry checkout."

---

### Issue 6: `verify` downloads every file (informational)

Verification downloads every file to compute hashes. For a 100GB
dataset, this is expensive. This is correct behavior but should be
documented so users aren't surprised by egress costs.

**Possible future improvement:** Support HEAD + ETag verification for
backends that support it, with `--full` flag for hash verification.

---

### Issue 7: `write_catalog` comment says CAS but uses Overwrite (low) — FIXED

The doc comment on `write_catalog` said "Uses conditional put when
possible (CAS)" but the code uses `PutMode::Overwrite`. Two concurrent
`delete` commands or a `push` + `delete` could race on the catalog.

The upload lock prevents push-vs-push races, but push-vs-delete and
delete-vs-delete are unprotected.

**Fix (implemented):** Updated the doc comment to accurately describe
the behavior and document the race window.

---

## Remote Storage Layout

```
s3://bucket/prefix/
├── catalog.json                          # top-level index
├── tpch-sf100.uploading                  # lock file (only during push)
├── tpch-sf100-m9d2k4/                   # dataset directory
│   ├── dataset.yaml                      # human-authored metadata
│   ├── manifest.json                     # auto-generated file index
│   └── parquet/                          # format
│       └── lineitem/                     # table
│           ├── lineitem_000.parquet      # data file
│           └── lineitem_001.parquet
└── clickbench-f7k3j9/
    ├── dataset.yaml
    ├── manifest.json
    └── parquet/
        └── hits/
            └── hits.parquet
```

## Local Mirror Layout (after pull + checkout)

```
~/.cache/vortex-bench-data/
├── catalog.json                          # copy of remote catalog
├── tpch-sf100-m9d2k4/
│   ├── dataset.yaml                      # copy of remote descriptor
│   ├── manifest.json                     # copy of remote manifest
│   └── data/                             # only after checkout
│       └── parquet/
│           └── lineitem/
│               ├── lineitem_000.parquet
│               └── lineitem_001.parquet
└── clickbench-f7k3j9/
    ├── dataset.yaml
    └── manifest.json                     # (no data/ until checkout)
```

## Concurrency Matrix

| Operation A | Operation B | Safe? | Protection |
|---|---|---|---|
| push X | push X | Yes | `.uploading` lock |
| push X | push Y | Yes | Different lock files |
| push X | delete X | **No** | No protection (see issue #7) |
| push X | gc | Yes | gc skips dirs with active `.uploading` lock |
| push X | pull | Yes | Pull reads catalog atomically |
| push X | checkout X | Mostly | Checkout uses local catalog, unaffected by remote push |
| delete X | delete X | **No** | Both modify catalog without CAS |
| gc | gc | Mostly | Both read catalog, then delete — benign double-delete |
| pull | pull | Yes | Overwrites local files idempotently |
| checkout X | checkout X | Yes | File writes are idempotent (hash check) |
