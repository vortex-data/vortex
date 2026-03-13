# `compat.py` CLI Specification

## Overview

A single Python CLI that manages the full lifecycle of Vortex backward-compatibility
testing: generating fixture files, storing them (locally or in S3), and validating
that the current reader can still read all historical fixtures.

Replaces the previous split between `upload.py` (Python) and direct `cargo run`
invocations of `compat-gen` / `compat-validate`.

### Design principles

- **Zero external Python dependencies.** Stdlib only (`argparse`, `json`, `urllib`,
  `subprocess`). Heavy lifting stays in the Rust binaries.
- **`--target` is the universal storage selector.** Every subcommand that touches
  fixture storage takes `--target s3` or `--target local:<path>`.
- **The Python layer orchestrates; Rust does the work.** `compat.py` calls
  `compat-gen` (Rust) for fixture generation and `compat-validate` (Rust) for
  validation. Python handles manifest merging, version bookkeeping, and S3 uploads.
- **`--dry-run` everywhere it makes sense.** Any destructive or remote operation
  should be skippable.

---

## Subcommands

### `add-version`

Generate fixtures for a version and store them.

```
compat.py add-version --version <VER> --target <TARGET> [OPTIONS]
```

#### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--version <VER>` | yes | Semver version tag (e.g. `0.63.0`) |
| `--target <TARGET>` | yes | `s3` or `local:<path>` |
| `--skip-build` | no | Skip `cargo run compat-gen`; assume fixtures already exist in the output directory |
| `--dry-run` | no | Generate fixtures and merge manifest locally, but skip S3 upload (S3 target only) |
| `--output <DIR>` | no | Override the build output directory (S3 target only; for local targets the output is always `<path>/v<version>/`) |

#### Workflow

1. **Generate fixtures** — runs `cargo run -p vortex-compat --release --bin compat-gen -- --version <VER> --output <DIR>`. Produces individual `.vortex` files and a `manifest.json`. Skipped if `--skip-build`.
2. **Merge manifest** — fetches the previous version's manifest (from S3 or local) and:
   - Carries forward `since` values for existing fixtures.
   - Sets `since` to `<VER>` for any new fixtures.
   - **Enforces additive-only**: errors if any fixture from the previous manifest is missing.
3. **Store** — depends on target:
   - `local:<path>`: writes fixtures to `<path>/v<VER>/`, updates `<path>/versions.json`.
   - `s3`: uploads fixtures to `s3://vortex-compat-fixtures/v<VER>/`, updates `versions.json` with ETag-based optimistic locking. Skipped if `--dry-run`.
4. **Report** — prints the final manifest to stderr.

#### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Additive-only violation (fixture removed) |
| 1 | S3 upload failed after retries |
| non-zero | `cargo run` / `aws` CLI failure (propagated) |

---

### `check`

Validate stored fixtures against the current reader.

```
compat.py check --target <TARGET> [OPTIONS]
```

#### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--target <TARGET>` | yes | `s3` or `local:<path>` |
| `--versions <V1,V2,...>` | no | Comma-separated list of versions to validate. Default: all discovered versions. |

#### Workflow

Delegates entirely to the Rust binary:

```
cargo run -p vortex-compat --release --bin compat-validate -- \
  {--fixtures-url <URL> | --fixtures-dir <PATH>} \
  [--versions <V1,V2,...>]
```

The Rust binary:
1. Discovers versions from `versions.json` (S3) or directory listing (local).
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

List known versions and their fixture inventories.

```
compat.py list --target <TARGET> [OPTIONS]
```

#### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--target <TARGET>` | yes | `s3` or `local:<path>` |
| `--version <VER>` | no | Show detailed manifest for a specific version instead of the version list |

#### Behavior

**Without `--version`** — prints the version list:

```
Versions (from s3):
  0.62.0
  0.63.0
  0.64.0
```

**With `--version <VER>`** — fetches and prints that version's manifest:

```
v0.63.0 (generated 2025-06-01T12:00:00Z):
  primitives.vortex    (since 0.62.0)
  strings.vortex       (since 0.62.0)
  my_new_fixture.vortex (since 0.63.0)
```

#### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Version not found / target unreachable |

---

## Target types

The `--target` flag accepts two forms:

| Syntax | Description |
|--------|-------------|
| `s3` | Use the shared S3 bucket `vortex-compat-fixtures`. Public reads via HTTPS; writes require AWS credentials. |
| `local:<path>` | Use a local directory. Creates `<path>/v<VER>/` subdirectories and a `<path>/versions.json` index automatically. No AWS credentials needed. |

### Directory layout (both targets)

```
<root>/
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

---

## CI integration

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

Manual dispatch with version input:

```yaml
- name: Generate and upload fixtures
  run: >
    python3 vortex-test/compat-gen/scripts/compat.py add-version
    --version "${{ inputs.version }}"
    --target s3
```

### Weekly validation (`.github/workflows/compat-test-weekly.yml`)

Scheduled Monday 6am UTC + manual dispatch:

```yaml
- name: Run compat tests
  run: >
    python3 vortex-test/compat-gen/scripts/compat.py check
    --target s3
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

1. Fetch previous version's `manifest.json` (highest version < current).
2. For each fixture in the generated manifest:
   - If it existed in the previous manifest, keep the old `since` value.
   - If it's new, set `since` to the current version.
3. **Additive-only enforcement:** if any fixture from the previous manifest is
   missing in the generated output, abort with an error.

---

## Example workflows

### Local development (no S3)

```bash
# Generate fixtures for the current version
python3 compat.py add-version --version 0.63.0 --target local:/tmp/compat

# See what's stored
python3 compat.py list --target local:/tmp/compat

# Validate
python3 compat.py check --target local:/tmp/compat
```

### Dry-run before S3 upload

```bash
# Generate and merge manifest, but don't upload
python3 compat.py add-version --version 0.63.0 --target s3 --dry-run

# If happy, do it for real
python3 compat.py add-version --version 0.63.0 --target s3
```

### Validate specific versions from S3

```bash
python3 compat.py check --target s3 --versions 0.62.0,0.63.0
```

### Inspect what's in S3

```bash
python3 compat.py list --target s3
python3 compat.py list --target s3 --version 0.62.0
```
