# `vortex-compat` CLI Specification

## Overview

A single Rust binary (`vortex-compat`) that manages the full lifecycle of Vortex
backward-compatibility testing: generating fixture files, publishing them to a
fixture store, and validating that the current reader can still read historical
fixtures.

Consolidates the current `compat-gen`, `compat-validate` binaries, and `compat.py`
orchestration script into one tool.

```
cargo run -p vortex-compat --release -- <COMMAND> [OPTIONS]
```

---

## Core abstraction: the fixture store

A **store** is a directory tree (local path or S3 bucket) with a fixed layout:

```
<store>/
├── versions.json
├── v0.62.0/
│   ├── manifest.json
│   ├── primitives.vortex
│   ├── strings.vortex
│   └── ...
└── v0.63.0/
    ├── manifest.json
    └── ...
```

All commands take `--store` to select the backend:

| Syntax | Example | Description |
|--------|---------|-------------|
| local path | `--store /tmp/compat` | Local directory, created if needed |
| S3 URL | `--store s3://vortex-compat-fixtures` | S3 bucket. Public reads; writes need AWS creds |

Default (when `--store` is omitted): `s3://vortex-compat-fixtures`.

Both backends expose the same operations and the commands work identically
against either.

---

## Subcommands

### `generate`

Build fixtures for a version and write them into a store.

```
vortex-compat generate --version <VER> [--store <STORE>] [--dry-run] [--skip-build]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--version <VER>` | yes | — | Semver tag (e.g. `0.63.0`) |
| `--store <STORE>` | no | `s3://vortex-compat-fixtures` | Target store |
| `--dry-run` | no | false | Generate + merge manifest locally, print result, don't write to store |
| `--skip-build` | no | false | Skip fixture generation; use existing files in a temp/output dir |

#### Workflow

1. **Generate fixtures** into a temp directory.
   Calls `all_fixtures()`, writes each as a `.vortex` file + naive `manifest.json`.
2. **Fetch previous manifest** from the store (highest version < `<VER>`).
3. **Merge manifest**:
   - Carry forward `since` for existing fixtures.
   - Set `since = <VER>` for new fixtures.
   - **Error** if any fixture from the previous manifest is missing (additive-only).
4. **Write to store**: copy `v<VER>/` into the store, update `versions.json`.
   Skipped if `--dry-run`.

#### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Additive-only violation / store write failure |

---

### `check`

Validate fixtures in a store against the current reader.

```
vortex-compat check [--store <STORE>] [--versions <V1,V2,...>]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--store <STORE>` | no | `s3://vortex-compat-fixtures` | Store to validate from |
| `--versions <V1,V2,...>` | no | all | Comma-separated versions to validate |

#### Workflow

1. Discover versions from `versions.json` (or directory listing for local stores).
2. For each version, fetch `manifest.json` and each fixture file.
3. Rebuild expected arrays from current `build()` methods.
4. Compare stored vs expected with `assert_arrays_eq!`.
5. Print per-version pass/fail/skip counts.

This is the existing `validate_all()` logic, now behind a subcommand.

#### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All passed |
| 1 | One or more failures |

---

### `list`

Inspect a store's contents.

```
vortex-compat list [--store <STORE>] [--version <VER>]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--store <STORE>` | no | `s3://vortex-compat-fixtures` | Store to inspect |
| `--version <VER>` | no | — | Show one version's manifest in detail |

#### Output

**All versions:**
```
$ vortex-compat list
Versions (s3://vortex-compat-fixtures):
  0.62.0
  0.63.0
```

**Single version detail:**
```
$ vortex-compat list --version 0.62.0
v0.62.0 (generated 2025-06-01T12:00:00Z):
  primitives.vortex         (since 0.62.0)
  strings.vortex            (since 0.62.0)
  tpch_lineitem.vortex      (since 0.62.0)
  ...
```

---

## Store trait

```rust
/// Abstraction over local filesystem and S3 fixture stores.
trait FixtureStore {
    /// Read a file's bytes. Returns None if not found.
    fn read(&self, key: &str) -> VortexResult<Option<Vec<u8>>>;

    /// Write a file's bytes.
    fn write(&self, key: &str, data: &[u8]) -> VortexResult<()>;

    /// Write a file from a local path (for large fixture files).
    fn write_from_path(&self, key: &str, local_path: &Path) -> VortexResult<()>;

    /// List top-level prefixes matching `v*/manifest.json` (for version discovery).
    fn list_versions(&self) -> VortexResult<Vec<String>>;

    /// Read and parse versions.json.
    fn read_versions_json(&self) -> VortexResult<Vec<String>>;

    /// Write versions.json (with locking for S3).
    fn write_versions_json(&self, versions: &[String]) -> VortexResult<()>;

    /// Display name for user-facing output.
    fn display_name(&self) -> &str;
}
```

### `LocalStore`

| Operation | Implementation |
|-----------|---------------|
| `read` | `std::fs::read()` |
| `write` | `std::fs::write()` with `create_dir_all` |
| `write_from_path` | `std::fs::copy()` |
| `list_versions` | `std::fs::read_dir()`, filter `v*/manifest.json` |
| `read_versions_json` | `std::fs::read()` + `serde_json` |
| `write_versions_json` | `std::fs::write()` |

### `S3Store`

| Operation | Implementation |
|-----------|---------------|
| `read` | `reqwest::blocking::get(https://<bucket>.s3.amazonaws.com/<key>)` |
| `write` | Shell out: `aws s3api put-object` |
| `write_from_path` | Shell out: `aws s3 cp` |
| `list_versions` | Fetch `versions.json` via HTTPS |
| `read_versions_json` | HTTPS GET |
| `write_versions_json` | `aws s3api put-object` with ETag optimistic locking + retry |

S3 reads use public HTTPS (no credentials needed). S3 writes shell out to `aws`
CLI (credentials via environment / IAM role). This avoids pulling in the AWS SDK
as a build dependency.

---

## Binary structure

The crate keeps one binary entry point. The current two binaries (`compat-gen`,
`compat-validate`) are removed.

```toml
[[bin]]
name = "vortex-compat"
path = "src/main.rs"
```

The existing library modules (`fixtures`, `manifest`, `adapter`, `validate`) stay
as-is. The `validate` module's `FixtureSource` enum is replaced by the `FixtureStore`
trait.

### Module layout

```
src/
├── main.rs          # clap CLI with generate/check/list subcommands
├── lib.rs           # pub mod exports
├── store.rs         # FixtureStore trait + LocalStore + S3Store
├── generate.rs      # generate subcommand logic
├── validate.rs      # check subcommand logic (existing, adapted)
├── manifest.rs      # Manifest / FixtureEntry (existing, unchanged)
├── adapter.rs       # read/write vortex files (existing, unchanged)
└── fixtures/
    ├── mod.rs       # Fixture trait + all_fixtures() (existing)
    ├── synthetic.rs
    ├── tpch.rs
    └── clickbench.rs
```

---

## CI integration

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

```yaml
- name: Generate and upload fixtures
  run: >
    cargo run -p vortex-compat --release --
    generate --version "${{ inputs.version }}"
```

### Weekly validation (`.github/workflows/compat-test-weekly.yml`)

```yaml
- name: Run compat tests
  run: cargo run -p vortex-compat --release -- check
```

Both default to the S3 store. No Python, no extra scripts.

---

## Example workflows

### Local development

```bash
# Generate into a local store
cargo run -p vortex-compat --release -- \
  generate --version 0.63.0 --store /tmp/store

# List what's there
cargo run -p vortex-compat --release -- \
  list --store /tmp/store

# Validate
cargo run -p vortex-compat --release -- \
  check --store /tmp/store
```

### Dry-run before S3 upload

```bash
# See what the merged manifest looks like, don't touch S3
cargo run -p vortex-compat --release -- \
  generate --version 0.63.0 --dry-run

# Happy? Do it for real
cargo run -p vortex-compat --release -- \
  generate --version 0.63.0
```

### Check specific versions from S3

```bash
cargo run -p vortex-compat --release -- \
  check --versions 0.62.0,0.63.0
```

### Test locally, then promote to S3

```bash
# Build and test in a local store
cargo run -p vortex-compat --release -- \
  generate --version 0.63.0 --store /tmp/store
cargo run -p vortex-compat --release -- \
  check --store /tmp/store

# Same version to S3
cargo run -p vortex-compat --release -- \
  generate --version 0.63.0
```

---

## Migration

1. Merge `compat-gen` + `compat-validate` + `compat.py` into single `vortex-compat` binary.
2. Replace `FixtureSource` enum with `FixtureStore` trait.
3. Move manifest merging logic from Python into `generate.rs`.
4. Update CI workflows to use `cargo run -p vortex-compat --release -- <subcommand>`.
5. Delete `scripts/compat.py` and `scripts/upload.py` (already deleted).
6. Update README.
