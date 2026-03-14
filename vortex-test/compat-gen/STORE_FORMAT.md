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
      "size_bytes": 4096,
      "expected_encodings": [
        "array:vortex.primitive",
        "layout:vortex.flat",
        "layout:vortex.struct"
      ]
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
| `sha256` | string | Hex-encoded SHA-256 hash of the `.vortex` file bytes. Verifies the file on disk hasn't been corrupted or tampered with since publish. Also used to skip re-uploads when re-publishing if the file hasn't changed. |
| `size_bytes` | integer | File size in bytes. Useful for spotting compression regressions across versions. |
| `expected_encodings` | string[] | Encodings this fixture is designed to exercise (see below). |

## Expected encodings

Each fixture declares a list of encodings it is designed to test. These are
the encodings the fixture author intentionally targets — not an exhaustive
list of every encoding that appears in the file (the compressor may introduce
additional intermediate encodings).

Encoding IDs use a `type:id` format:

| Prefix | Source | Example |
|--------|--------|---------|
| `array:` | Array encoding (compression layer) | `array:vortex.dict`, `array:vortex.fsst`, `array:vortex.primitive` |
| `layout:` | Layout encoding (storage layer) | `layout:vortex.flat`, `layout:vortex.chunked`, `layout:vortex.struct` |

### How they're declared

Each fixture declares its expected encodings via the `Fixture` trait:

```rust
pub enum ExpectedEncoding {
    Array(ArrayId),
    Layout(LayoutEncodingId),
}

pub trait Fixture: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn expected_encodings(&self) -> Vec<ExpectedEncoding>;
    fn build(&self, tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>>;
}
```

### How they're verified

**At generate time (publish):** After writing the `.vortex` file, the tool
walks the layout tree via `footer().layout().depth_first_traversal()` and
collects all encoding IDs present in the file. It then asserts that every
encoding in `expected_encodings()` appears somewhere in the file. If a
declared encoding is missing — e.g. the compressor chose a different strategy
than expected — the publish fails. This catches the case where a fixture
claims to test dict encoding but the file doesn't actually contain one.

**At check time (validation):** After opening the old file, the tool walks
the layout tree the same way and verifies the declared encodings are still
present. This catches the case where a reader or format change silently
drops or substitutes an encoding.

The check is a **subset assertion**: every declared encoding must appear in
the file, but the file may contain additional encodings not in the list.
The fixture author only declares what the fixture is *designed* to exercise,
not every internal encoding the compressor happens to use.

### Why this matters

- **Coverage tracking.** "Which fixtures exercise FSST?" → grep manifests
  for `array:vortex.fsst`. "Do we have any fixture testing dict encoding?"
  → search for `array:vortex.dict`. Find gaps without running code.
- **Intent documentation.** The encoding list tells you *why* a fixture
  exists, distinct from its description. `strings.vortex` exists to test
  varbin/FSST, not just "strings in general."
- **Encoding removal safety.** Before removing an encoding, search the
  manifests to see which fixtures depend on it.

## Fixture evolution

Fixtures are immutable. The `build()` output for a given fixture name must
never change — old files written with the old `build()` must remain valid
and readable forever.

- **Adding coverage** (new column, new edge case, new dtype) → create a new
  fixture file. The old fixture keeps testing the old schema.
- **New encoding or structural pattern** → create a new fixture file that
  declares the new encoding in `expected_encodings()`.

Never modify an existing fixture's `build()` to change its schema, values,
or structure. The fixture contract is append-only at both the fixture list
level and the individual fixture level.

## Workflows

### Publish

```
detect version from git tag  →  "0.63.0"
detect commit                →  "a1b2c3d4e5f6..."
short hash                   →  "a1b2c3d"
                                    │
generate fixtures                   │
verify expected_encodings present   │
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
  run vortex-compat check:
    - read file, compare values against build()
    - walk layout tree, verify expected_encodings present
  report pass/fail
```

### Check a specific historical publish

```
compat.py check --versions 0.63.0 --hash f9e8d7c
  → reads from v0.63.0-f9e8d7c/ instead of current
```

This enables bisecting: is the failure from a reader change or from
something in the latest publish?
