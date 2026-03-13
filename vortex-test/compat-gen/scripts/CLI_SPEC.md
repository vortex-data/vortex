# `compat.py` CLI Specification

## Overview

A single Python CLI that manages the full lifecycle of Vortex backward-compatibility
testing: generating fixture files, publishing them to a fixture store, and validating
that the current reader can still read all historical fixtures.

### Core abstraction: the fixture store

A **fixture store** is a directory tree (local path or S3 bucket) with a fixed layout:

```
<store>/
├── versions.json            # ["0.62.0", "0.63.0", ...]
├── v0.62.0/
│   ├── manifest.json        # {version, generated_at, fixtures: [{name, since}, ...]}
│   ├── primitives.vortex
│   ├── strings.vortex
│   └── ...
└── v0.63.0/
    ├── manifest.json
    └── ...
```

All commands that read or write fixture data take a `--store` argument that can be
either a local path or an S3 URL. The commands work identically regardless of which
backend is used.

| Syntax | Example | Description |
|--------|---------|-------------|
| local path | `/tmp/compat` | Local directory. Created automatically if needed. |
| `s3://bucket` | `s3://vortex-compat-fixtures` | S3 bucket. Public reads via HTTPS; writes require AWS credentials. |

The default store (when `--store` is omitted) is `s3://vortex-compat-fixtures`.

### Design principles

- **Zero external Python dependencies.** Stdlib only (`argparse`, `json`, `urllib`,
  `subprocess`). Heavy lifting stays in the Rust binaries.
- **S3 and local are interchangeable.** Every command works the same way against
  either backend. You can `generate` locally, `upload` to a local store for testing,
  then `upload` the same directory to S3 when ready.
- **The Python layer orchestrates; Rust does the work.** `compat.py` calls
  `compat-gen` (Rust) for fixture generation and `compat-validate` (Rust) for
  validation. Python handles manifest merging, version bookkeeping, and store I/O.
- **Separate generation from publishing.** `generate` produces files into a plain
  local directory. `upload` copies a versioned directory into a store. This lets you
  inspect, tweak, or dry-run before committing to a store.

---

## Subcommands

### `generate`

Build fixture files for a version into a local working directory.

```
compat.py generate --version <VER> [--output <DIR>]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--version <VER>` | yes | — | Semver version tag (e.g. `0.63.0`) |
| `--output <DIR>` | no | `./compat-fixtures/v<VER>` | Local directory for generated files |

#### What it does

1. Runs `cargo run -p vortex-compat --release --bin compat-gen -- --version <VER> --output <DIR>`.
2. Produces `<DIR>/<fixture>.vortex` files + `<DIR>/manifest.json`.
3. This is a **local-only** operation. No store interaction, no manifest merging.
   The manifest written by `compat-gen` has `since` set to `<VER>` for all fixtures
   (the raw/naive manifest).

#### Output

```
$ compat.py generate --version 0.63.0 --output /tmp/fixtures
[1/1] Generating fixtures for v0.63.0...
  wrote primitives.vortex
  wrote strings.vortex
  ...
  wrote manifest.json
done: 9 fixtures for v0.63.0 in /tmp/fixtures
```

---

### `upload`

Publish a generated fixture directory into a store, merging manifests and enforcing
the additive-only contract.

```
compat.py upload --version <VER> --from <DIR> [--store <STORE>] [--dry-run]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--version <VER>` | yes | — | Version being published |
| `--from <DIR>` | yes | — | Local directory containing generated fixtures (output of `generate`) |
| `--store <STORE>` | no | `s3://vortex-compat-fixtures` | Target store (local path or `s3://bucket`) |
| `--dry-run` | no | `false` | Merge manifest and print results, but don't write to the store |

#### What it does

1. **Read previous manifest** — fetches the manifest from the highest existing
   version in the store (if any).
2. **Merge manifest** — updates `<DIR>/manifest.json`:
   - Carries forward `since` values for existing fixtures from the previous manifest.
   - Sets `since` to `<VER>` for any new fixtures.
   - **Enforces additive-only**: errors if any fixture from the previous manifest is
     missing in `<DIR>`.
3. **Copy to store** — copies `<DIR>/*` into `<store>/v<VER>/`.
   - Local store: `cp`/`shutil.copytree`.
   - S3 store: `aws s3 cp --recursive`.
4. **Update `versions.json`** — appends `<VER>` to the store's version index.
   - Local store: direct file write.
   - S3 store: ETag-based optimistic locking with retry.

If `--dry-run` is set, steps 3-4 are skipped and the merged manifest is printed.

#### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success (or dry-run completed) |
| 1 | Additive-only violation (fixture removed) |
| 1 | Store write failed after retries |

---

### `check`

Validate fixtures in a store against the current reader.

```
compat.py check [--store <STORE>] [--versions <V1,V2,...>]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--store <STORE>` | no | `s3://vortex-compat-fixtures` | Store to validate from |
| `--versions <V1,V2,...>` | no | all | Comma-separated list of versions to validate |

#### What it does

Delegates to the Rust binary:

```
cargo run -p vortex-compat --release --bin compat-validate -- \
  {--fixtures-url <HTTPS_URL> | --fixtures-dir <PATH>} \
  [--versions <V1,V2,...>]
```

The Rust binary:
1. Discovers versions from `versions.json` or directory listing.
2. For each version, fetches `manifest.json` and each fixture file.
3. Rebuilds expected arrays from current `build()` methods.
4. Compares stored vs expected using `assert_arrays_eq!`.
5. Reports per-version pass/fail/skip counts.

#### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All fixtures passed |
| non-zero | One or more fixtures failed validation |

---

### `list`

List versions and fixtures in a store.

```
compat.py list [--store <STORE>] [--version <VER>]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--store <STORE>` | no | `s3://vortex-compat-fixtures` | Store to inspect |
| `--version <VER>` | no | — | Show detailed manifest for one version |

#### Output

**Without `--version`** — prints the version list:

```
$ compat.py list
Versions (s3://vortex-compat-fixtures):
  0.62.0
  0.63.0
  0.64.0
```

**With `--version`** — prints that version's manifest:

```
$ compat.py list --version 0.62.0
v0.62.0 (generated 2025-06-01T12:00:00Z):
  primitives.vortex      (since 0.62.0)
  strings.vortex         (since 0.62.0)
  booleans.vortex        (since 0.62.0)
  nullable.vortex        (since 0.62.0)
  struct_nested.vortex   (since 0.62.0)
  chunked.vortex         (since 0.62.0)
  tpch_lineitem.vortex   (since 0.62.0)
  tpch_orders.vortex     (since 0.62.0)
  clickbench_hits_1k.vortex (since 0.62.0)
```

---

## Store abstraction

The store is selected by the `--store` argument and determines how files are
read and written.

### Local store (`/path/to/dir`)

| Operation | Implementation |
|-----------|---------------|
| Read file | `open()` / `os.path.isfile()` |
| Write file | `shutil.copy2()` / `os.makedirs()` |
| List versions | `os.listdir()`, filter `v*` dirs with `manifest.json` |
| Read `versions.json` | `json.load(open(...))` |
| Write `versions.json` | `json.dump()` to file |

### S3 store (`s3://bucket`)

| Operation | Implementation |
|-----------|---------------|
| Read file | `urllib.request.urlopen(https://<bucket>.s3.amazonaws.com/...)` |
| Write file | `aws s3 cp --recursive` |
| List versions | Fetch `versions.json` via HTTPS |
| Read `versions.json` | HTTPS GET |
| Write `versions.json` | `aws s3api put-object` with ETag-based optimistic locking + retry |

The `check` command translates the store into the appropriate Rust CLI flag:
- Local → `--fixtures-dir <path>`
- S3 → `--fixtures-url https://<bucket>.s3.amazonaws.com`

---

## CI integration

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

Manual dispatch with version input:

```yaml
- name: Generate fixtures
  run: >
    python3 vortex-test/compat-gen/scripts/compat.py generate
    --version "${{ inputs.version }}"
    --output /tmp/fixtures

- name: Upload fixtures to S3
  run: >
    python3 vortex-test/compat-gen/scripts/compat.py upload
    --version "${{ inputs.version }}"
    --from /tmp/fixtures
```

### Weekly validation (`.github/workflows/compat-test-weekly.yml`)

Scheduled Monday 6am UTC + manual dispatch:

```yaml
- name: Run compat tests
  run: python3 vortex-test/compat-gen/scripts/compat.py check
```

---

## Implementation notes

### Rust binaries (unchanged)

The Python CLI orchestrates two existing Rust binaries. Their interfaces are stable
and do not change as part of this spec.

**`compat-gen`** — generates fixtures:
```
cargo run -p vortex-compat --release --bin compat-gen -- \
  --version <VER> --output <DIR>
```
Produces: `<DIR>/<fixture>.vortex` files + `<DIR>/manifest.json`.

**`compat-validate`** — validates fixtures:
```
cargo run -p vortex-compat --release --bin compat-validate -- \
  {--fixtures-url <URL> | --fixtures-dir <PATH>} \
  [--versions V1,V2,...]
```

### S3 specifics

- **Bucket:** `vortex-compat-fixtures` (us-east-1, public read)
- **Public URL:** `https://vortex-compat-fixtures.s3.amazonaws.com`
- **Auth:** AWS CLI on PATH + valid credentials (CI uses OIDC role assumption)
- **Optimistic locking:** `versions.json` updates use ETag-based compare-and-swap
  with exponential backoff (up to 5 retries, delays capped at 30s)
- **Upload method:** `aws s3 cp --recursive` for fixture files,
  `aws s3api put-object` for `versions.json`

### Manifest merging rules

Merging happens in `upload`, not `generate`. The `generate` command writes a naive
manifest where all fixtures have `since` set to the current version. The `upload`
command then:

1. Fetches the previous version's manifest from the store (highest version < current).
2. For each fixture in the generated manifest:
   - If it existed in the previous manifest, keep the old `since` value.
   - If it's new, set `since` to the current version.
3. **Additive-only enforcement:** if any fixture from the previous manifest is
   missing in the generated output, abort with an error.
4. Writes the merged manifest into the source directory before copying to the store.

---

## Example workflows

### Local development (no S3, no AWS credentials)

```bash
# Generate fixtures into a working directory
python3 compat.py generate --version 0.63.0 --output /tmp/fixtures

# Publish to a local store
python3 compat.py upload --version 0.63.0 --from /tmp/fixtures --store /tmp/store

# See what's in the store
python3 compat.py list --store /tmp/store

# Validate everything in the store
python3 compat.py check --store /tmp/store
```

### Dry-run before S3 upload

```bash
python3 compat.py generate --version 0.63.0 --output /tmp/fixtures

# Preview what the merged manifest would look like
python3 compat.py upload --version 0.63.0 --from /tmp/fixtures --dry-run

# Looks good — upload for real
python3 compat.py upload --version 0.63.0 --from /tmp/fixtures
```

### Validate specific versions from S3

```bash
python3 compat.py check --versions 0.62.0,0.63.0
```

### Inspect what's in S3

```bash
python3 compat.py list
python3 compat.py list --version 0.62.0
```

### Test locally, then promote to S3

```bash
# Generate
python3 compat.py generate --version 0.63.0 --output /tmp/fixtures

# Test in a local store first
python3 compat.py upload --version 0.63.0 --from /tmp/fixtures --store /tmp/store
python3 compat.py check --store /tmp/store

# All good — upload the same fixtures to S3
python3 compat.py upload --version 0.63.0 --from /tmp/fixtures
```
