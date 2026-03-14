# Compat Store Format

This document describes the storage layout and metadata format for Vortex
backward-compatibility fixtures. It is the source of truth for how the
orchestrator (`compat.py`) and CI workflows interact with the fixture store.

## Design Goals

1. **Provenance.** Every set of fixture files is traceable to the exact git
   commit that produced them.
2. **Safe re-publishing.** Re-publishing fixtures for a release version (e.g.
   to add a new fixture or fix a generation bug) creates a new directory rather
   than silently overwriting the old one. The old directory is retained as
   orphaned storage and can be garbage-collected later.
3. **Simplicity.** The root index is a flat `version → hash` map. No nested
   objects, no history arrays, no extra indirection.

## Root index: `versions.json`

```json
{
  "schema_version": 1,
  "versions": {
    "0.62.0": "1234abc",
    "0.63.0": "a1b2c3d"
  }
}
```

| Field | Description |
|-------|-------------|
| `schema_version` | Integer. Allows the format to evolve without breaking old tooling. Currently `1`. |
| `versions` | Map of Vortex release version → short commit hash (first 7 characters of the full SHA that produced the fixtures). |

The combination of version and hash forms the directory name: `v{version}-{hash}`.

### Re-publishing

When fixtures for an existing version are re-published from a new commit:

1. The new fixtures are uploaded to a new directory (e.g. `v0.63.0-b2c3d4e/`).
2. The `versions` map is updated: `"0.63.0": "b2c3d4e"`.
3. The old directory (`v0.63.0-a1b2c3d/`) is **not deleted**. It becomes
   orphaned — invisible to `check` but still accessible for manual debugging.

If someone attempts to publish from the same commit that is already recorded,
the publish is rejected (the fixtures already exist).

## Store layout

```
store/
├── versions.json
├── v0.62.0-1234abc/
│   ├── manifest.json
│   ├── primitives.vortex
│   ├── strings.vortex
│   └── ...
├── v0.63.0-a1b2c3d/              ← current for 0.63.0
│   ├── manifest.json
│   ├── primitives.vortex
│   └── ...
└── v0.63.0-f9e8d7c/              ← orphaned (previous publish of 0.63.0)
    ├── manifest.json
    └── ...
```

The store is a flat list of versioned directories. Each directory name is
`v{version}-{short_hash}`. The `check` command resolves which directory to
use by looking up the hash in `versions.json`.

## Per-version manifest: `v{version}-{hash}/manifest.json`

```json
{
  "version": "0.63.0",
  "commit": "a1b2c3d4e5f67890abcdef1234567890abcdef12",
  "generated_at": "2026-03-14T08:00:00Z",
  "dirty": false,
  "rust_version": "1.82.0",
  "fixtures": [
    {
      "name": "primitives.vortex",
      "description": "All primitive types with boundary values",
      "since": "0.62.0",
      "sha256": "deadbeefcafebabe...",
      "size_bytes": 4096
    }
  ]
}
```

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | The Vortex release version (e.g. `"0.63.0"`). |
| `commit` | string | Full 40-character git commit SHA that produced these fixtures. |
| `generated_at` | string | ISO 8601 timestamp of when the fixtures were generated. |
| `dirty` | bool | Whether the git worktree had uncommitted changes at generation time. If `true`, the `commit` alone does not fully reproduce the output. |
| `rust_version` | string | The `rustc` version used to compile the generator binary. Useful for diagnosing floating-point codegen or optimization differences. |

### Per-fixture fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Fixture filename (e.g. `"primitives.vortex"`). |
| `description` | string | Human-readable description of what the fixture tests. |
| `since` | string | First Vortex version that introduced this fixture. Carried forward across versions by manifest merging. |
| `sha256` | string | Hex-encoded SHA-256 hash of the `.vortex` file bytes. Used to detect oracle drift (see below). |
| `size_bytes` | integer | File size in bytes. Useful for detecting compression regressions. |

### Oracle drift detection via `sha256`

The compat test has two moving parts:

1. **`build()`** generates expected arrays in memory (the oracle).
2. **`check()`** reads old `.vortex` files and compares against `build()`.

If a test fails, you need to know which side changed. The `sha256` field
enables this: at check time, re-run `build()` → write to a temp buffer →
hash it. If the hash matches the manifest, `build()` still produces the same
output and the failure is a real reader regression. If the hash differs,
`build()` output has drifted (e.g. a compressor tweak changed encoding
selection) and the failure may be a false alarm.

## Workflows

### Publish

```
detect version from git tag  →  "0.63.0"
detect commit                →  "a1b2c3d4e5f6..."
short hash                   →  "a1b2c3d"
                                    │
generate fixtures                   │
write to v0.63.0-a1b2c3d/          │
compute sha256 per file             │
write manifest.json                 │
                                    │
upload directory to store           │
update versions.json:               │
  "0.63.0": "a1b2c3d"              │
```

### Check

```
read versions.json
for each version:
  resolve directory: v{version}-{hash}/
  download manifest.json + *.vortex
  run vortex-compat check
  report pass/fail
```

### Check a specific historical publish

```
compat.py check --versions 0.63.0 --hash f9e8d7c
  → reads from v0.63.0-f9e8d7c/ instead of current
```

This enables bisecting: is the failure from a reader change or from
something in the latest publish?
