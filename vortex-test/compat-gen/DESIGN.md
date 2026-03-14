# Vortex Backward-Compatibility Testing

## The Problem

Vortex is a columnar file format. Users write `.vortex` files with one version
of the library and expect to read them with any future version. If a code
change silently breaks the ability to decode old files, we ship data loss.

We need a system that catches this before it merges.

## The Solution

We maintain a library of `.vortex` fixture files, one set per released
version, stored in S3. A test reads every old fixture with the current reader
and compares the decoded values against a known-good oracle. If any fixture
from any version decodes to the wrong values, the test fails.

## How Fixtures Work

A fixture is a small `.vortex` file with known contents. The expected
contents are defined by a Rust function — `build()` — that deterministically
constructs the arrays. The same `build()` code is used at both ends:

- **At publish time:** `build()` produces arrays → the writer serializes them
  to a `.vortex` file → the file is uploaded to S3.
- **At check time:** `build()` produces the same arrays → the reader decodes
  the old file → the two are compared value-by-value.

If the reader is correct, the values match. If a code change breaks
decoding, they don't.

### The oracle question

"Isn't comparing against `build()` circular? What if `build()` itself
changes?"

No. `build()` is the specification — it defines what the fixture *should*
contain. The contract is that `build()` for a given fixture name is immutable
once defined. It must never change its output. If someone modifies it, the
check fails loudly against every old version, which is the correct signal
that the oracle moved.

### Fixture evolution

Because `build()` is immutable, you cannot add a column to an existing
fixture or change its schema. If you want to test a new type, encoding, or
structural pattern, you create a **new fixture file** with a new name. The
old fixture continues testing the old schema.

This is intentional: old files have the old schema, and we want to keep
testing that the reader handles it correctly.

## What Each Fixture Declares

Every fixture implements a trait:

```rust
pub trait Fixture: Send + Sync {
    /// Filename, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// Encodings this fixture is designed to exercise.
    fn expected_encodings(&self) -> Vec<ExpectedEncoding>;

    /// Optional async setup (download external data, etc).
    fn setup(&self, _tmp_dir: &Path) -> VortexResult<()> { Ok(()) }

    /// Build the expected arrays. Must be deterministic.
    fn build(&self, tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>>;
}
```

Besides name, description, and build, each fixture declares which
**encodings** it is designed to exercise:

```rust
pub enum ExpectedEncoding {
    Array(ArrayId),           // e.g. "vortex.dict", "vortex.fsst"
    Layout(LayoutEncodingId), // e.g. "vortex.chunked", "vortex.flat"
}
```

These are serialized in the manifest as `"array:vortex.dict"`,
`"layout:vortex.flat"`, etc.

### Why declare encodings?

The compressor chooses which encodings to use. A fixture designed to test
dict encoding might not actually get dict-encoded if the compressor's
heuristics change. By declaring intent, we can verify at both publish and
check time that the expected encodings actually appear in the file.

The verification is a **subset check**: every declared encoding must appear
somewhere in the file's layout tree, but the file may contain additional
encodings the fixture doesn't care about. We walk the tree via
`footer().layout().depth_first_traversal()` to collect what's present.

This also gives you coverage tracking. "Which fixtures exercise FSST?" →
grep the manifests for `array:vortex.fsst`. "Is it safe to remove dict
encoding?" → check which fixtures depend on `array:vortex.dict`.

## Architecture

The system is split into two layers:

### Rust binary (`vortex-compat`)

A thin binary with two commands. It links against Vortex and handles only:

- `generate --output <DIR>` — runs `build()` for all fixtures, writes
  `.vortex` files and a `fixtures.json` listing.
- `check --dir <DIR> --mode subset|exact|superset` — reads `.vortex` files,
  rebuilds expected arrays, compares them, outputs JSON results to stdout.

It has no knowledge of versions, S3, manifests, or orchestration.

Generation runs in three phases:
1. **Setup** — each fixture's `setup()` runs concurrently (for downloading
   external data like TPC-H or ClickBench datasets).
2. **Build** — `build()` runs in parallel threads. All must succeed before
   any files are written.
3. **Write** — each fixture's arrays are serialized via the adapter module.
   After writing, the layout tree is walked to verify declared encodings are
   present.

### Python orchestrator (`compat.py`)

Handles everything version-agnostic:

- **`publish`** — generates fixtures, computes sha256/size per file, merges
  the manifest with the previous version (carrying forward `since` values,
  enforcing additive-only), uploads to S3, updates `versions.json`.
- **`check`** — downloads fixtures for each version, invokes the Rust binary,
  aggregates results.
- **`list`** — inspects store contents.
- **`validate-manifest`** — verifies no fixtures were removed between
  consecutive versions.
- **`generate`** — local-only generation (no S3).

The orchestrator also manages **git worktrees** for publishing from old tags:

```bash
python compat.py publish --git-ref v0.62.0
```

This checks out the tag in a temporary worktree, builds the Rust binary
against that version's code, generates fixtures, then publishes them. You
can publish fixtures for any historical release without switching branches.

## Store Format

Fixtures live in an S3 bucket (or a local directory for development).

### `versions.json`

```json
{
  "schema_version": 1,
  "versions": {
    "0.62.0": "1234abc",
    "0.63.0": "a1b2c3d"
  }
}
```

A flat map from release version to the short commit hash (first 7 characters)
that produced the fixtures. The `check` command looks up the hash to find the
right directory.

### Directory layout

```
store/
├── versions.json
├── v0.62.0-1234abc/
│   ├── manifest.json
│   ├── primitives.vortex
│   └── ...
└── v0.63.0-a1b2c3d/
    ├── manifest.json
    └── ...
```

Each directory is named `v{version}-{short_hash}`. This means re-publishing
the same version from a different commit creates a **new** directory instead
of silently overwriting the old one. The old directory becomes orphaned —
invisible to `check` but still there for debugging. Publishing from the same
commit that's already recorded is rejected.

### Per-version manifest

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
      "sha256": "deadbeef...",
      "size_bytes": 4096,
      "expected_encodings": ["array:vortex.primitive", "layout:vortex.flat"]
    }
  ]
}
```

The manifest records provenance (which commit, which rustc, was the tree
dirty) and per-fixture metadata. The `sha256` is an integrity check — it
verifies the file hasn't been corrupted since upload, and lets the publish
step skip re-uploading unchanged files. The `since` field tracks which
version introduced each fixture and is carried forward automatically by
manifest merging.

## CI

Two workflows:

**Fixture upload** (manual dispatch) — generates fixtures at HEAD (or a
specified `--git-ref`), publishes to S3. Runs with AWS credentials via the
`GitHubBenchmarkRole`.

**Weekly validation** (every Monday 06:00 UTC + manual) — checks all
published versions against the current reader. Exits 1 if any fixture fails.

## Known Limitations

This system tests one thing well: "can the current reader decode old files to
the correct logical values?" It does **not** test:

- **Predicate pushdown.** Files are read via `scan()` with no filter. If
  statistics or zone maps are corrupted, queries silently scan everything
  instead of pruning — and this tool won't notice.
- **Column projection.** `read_all()` reads every column. Bugs that only
  manifest during selective column reads are invisible.
- **Type coverage gaps.** No fixtures yet for decimal, temporal, list, binary,
  or extension types, nor for degenerate cases like empty files, all-null
  columns, or float specials (NaN, Inf).
- **Forward compatibility.** We don't test that old readers can handle new
  files.

These are intentional scope boundaries, not oversights. Predicate pushdown
and column projection testing would require significant changes to the check
harness. Expanding type coverage is straightforward and should happen as new
fixtures are added over time.
