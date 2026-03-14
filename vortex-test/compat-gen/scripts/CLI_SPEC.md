# Vortex Backward-Compatibility Tool

## Architecture

The compat system is split into two layers:

### Rust binary: `vortex-compat`

A thin binary with exactly two commands. It knows how to build fixtures
from current code and compare `.vortex` files against expected arrays.
It has **no** knowledge of versions, S3, manifests, or orchestration.

```
vortex-compat generate --output <DIR>
vortex-compat check --dir <DIR> --mode exact|subset|superset
```

**Why this split?** Only this binary touches Vortex APIs (`Array`, `DType`,
file I/O). When Vortex APIs change across versions, only this small binary
needs to compile against each version. The orchestrator is version-agnostic.

**Output protocol:**
- Progress / diagnostics → stderr
- Structured JSON → stdout (`check` only)

### Python orchestrator: `compat.py`

Handles everything version-agnostic: S3 uploads/downloads, manifest merging,
`versions.json` management, additive-only checks, and git worktree management
for multi-version generation.

```
python compat.py publish  --version 0.63.0 [--git-ref v0.63.0] [--store s3://...] [--dry-run]
python compat.py check    [--versions 0.62.0,0.63.0] [--store s3://...]
python compat.py generate --version 0.63.0 --output ./out [--git-ref v0.63.0]
python compat.py list     [--store s3://...] [--version 0.63.0]
python compat.py validate-manifest [--store s3://...]
```

---

## Store layout

A **store** is a directory tree (local path or S3 bucket):

```
<store>/
├── versions.json          # ["0.62.0", "0.63.0"]
├── v0.62.0/
│   ├── manifest.json
│   ├── primitives.vortex
│   ├── strings.vortex
│   └── ...
└── v0.63.0/
    ├── manifest.json
    └── ...
```

Store specs: `--store /tmp/compat` (local) or `--store s3://bucket` (S3).
Default: `s3://vortex-compat-fixtures`.

---

## Rust binary details

### `generate --output <DIR>`

1. **Setup** — run each fixture's `setup()` concurrently (async I/O)
2. **Build** — construct arrays in parallel threads (CPU-bound)
3. **Write** — serialize `.vortex` files + `fixtures.json` to `<DIR>`

The `fixtures.json` lists all generated files:
```json
{
  "fixtures": [
    {"name": "primitives.vortex", "description": "..."},
    ...
  ]
}
```

### `check --dir <DIR> --mode <MODE>`

Reads each `.vortex` file, rebuilds expected arrays from current fixture code,
and compares them. JSON result to stdout:

```json
{
  "passed": ["primitives.vortex", "strings.vortex"],
  "failed": [{"name": "foo.vortex", "error": "mismatch at row 42"}],
  "skipped": ["old_fixture.vortex"]
}
```

**Modes:**

| Mode | Extra files in dir | Missing fixtures |
|------|--------------------|------------------|
| `subset` (default) | Skipped | Error |
| `exact` | Error | Error |
| `superset` | Error | Skipped |

Use `subset` when checking old versions (they may have fixtures since removed
from code). Use `exact` for current version. Use `superset` when checking a
partial directory.

---

## Python orchestrator details

### `publish --version <VER> [--git-ref <REF>]`

1. Generate fixtures (from current tree, or from a git worktree at `<REF>`)
2. Fetch previous manifest, merge (carry forward `since`, enforce additive-only)
3. Upload `.vortex` files + `manifest.json` to store
4. Update `versions.json`

### `check [--versions <V1,V2,...>]`

1. List versions from store
2. For each: download fixtures, run `vortex-compat check`, aggregate results
3. Exit 1 if any failures

### `list [--version <VER>]`

Without `--version`: print all version numbers.
With `--version`: print that version's `manifest.json`.

### `validate-manifest`

Walk all versions in order and verify no fixtures were removed between
consecutive versions (additive-only property).

---

## Git worktree workflow

The `--git-ref` flag on `publish` and `generate` automates this workflow:

1. `git worktree add <tmpdir> <ref>` — check out the target commit
2. `cargo build -p vortex-compat --release` — build in the worktree
3. `<worktree>/target/release/vortex-compat generate --output <dir>` — generate
4. `git worktree remove <tmpdir>` — clean up

This means you can publish fixtures for any historical release without
manually switching branches:

```bash
# Publish v0.62.0 fixtures from the v0.62.0 tag
python compat.py publish --version 0.62.0 --git-ref v0.62.0
```

---

## CI integration

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

```yaml
- name: Generate and publish fixtures
  run: >
    python vortex-test/compat-gen/scripts/compat.py
    publish --version "${{ inputs.version }}"
```

### Weekly validation (`.github/workflows/compat-test-weekly.yml`)

```yaml
- name: Run compat tests
  run: >
    python vortex-test/compat-gen/scripts/compat.py check
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `VORTEX_COMPAT_BIN` | Path to pre-built `vortex-compat` binary. Skips `cargo run`. |

---

## Module layout

```
vortex-test/compat-gen/
├── Cargo.toml
├── scripts/
│   ├── CLI_SPEC.md          # this file
│   └── compat.py            # Python orchestrator
└── src/
    ├── main.rs              # clap CLI: generate + check
    ├── lib.rs               # pub mod exports
    ├── generate.rs          # fixture generation (setup → build → write)
    ├── check.rs             # fixture checking (read → compare → JSON)
    ├── adapter.rs           # read/write vortex files
    └── fixtures/
        ├── mod.rs           # Fixture trait + all_fixtures()
        ├── synthetic.rs
        ├── tpch.rs
        └── clickbench.rs
```
