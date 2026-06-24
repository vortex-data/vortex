# vortex-sqllogictest

This crate uses [`sqllogictest-rs`](https://github.com/risinglightdb/sqllogictest-rs) to run
`.slt`-based tests against both DataFusion and DuckDB, each preconfigured to read Vortex files.

Every `.slt` file is turned into one independent test per engine, driven by `sqllogictest`'s
`libtest-mimic` harness. Tests run in parallel; within a single file the records execute
sequentially for one engine. Each test is named `slt::<engine>::<relative-path>`, for example
`slt::datafusion::integers.slt` or `slt::duckdb::duckdb/explain.slt`.

## Running tests

Some tests use TPC-H data at scale factor 0.1. Generate it first, then run the suite with
`cargo nextest`:

```shell
./vortex-sqllogictest/slt/tpch/generate_data.sh
cargo nextest run -p vortex-sqllogictest
# the built-in cargo test harness also works:
cargo test -p vortex-sqllogictest --test sqllogictests
```

The generated data lives under `slt/tpch/data/` (git-ignored). If it is missing, the TPC-H tests
are reported as **ignored** rather than failing, so the rest of the suite still runs. The TPC-H
`.slt` files load their tables through paths relative to the crate root, so run the tests via
`cargo nextest`/`cargo test`, which set the working directory accordingly.

Because the harness is `libtest-mimic`-based, the standard test flags work, including
`cargo nextest`, filtering, and listing:

```shell
# Run only DuckDB tests:
cargo nextest run -p vortex-sqllogictest -E 'test(/slt::duckdb::/)'
# Run a single file on both engines (substring filter):
cargo nextest run -p vortex-sqllogictest -E 'test(strings)'
# List every generated test without running:
cargo nextest list -p vortex-sqllogictest
```

## Scratch directory and `${WORK_DIR}`

Tests reference a per-test working directory through the `${WORK_DIR}` substitution variable. The
runner sets `WORK_DIR` to a constant, git-ignored scratch directory **inside this crate** —
`scratch/<test-name>/` — rather than an OS tempdir. The path is deterministic (named after the test,
not random), so it is easy to inspect, and each test gets its own directory so concurrent tests
never collide. The directory is recreated empty before each test and removed afterwards — whether
the test passed, failed, or panicked (cleanup errors are logged, not fatal).

Query output is passed through a normalization step that rewrites the scratch path back to the
`${WORK_DIR}` token. This keeps expected output stable across machines and runs, and is what lets
`--complete` (below) write portable expected values instead of machine-specific paths.

## Selecting which engine runs a test

There are two complementary mechanisms:

- **Per-file, by directory.** A file under a `datafusion/` directory runs **only** on DataFusion;
  a file under a `duckdb/` directory runs **only** on DuckDB. Anything else runs on **both**. This
  is how engine-specific features (e.g. DuckDB `EXPLAIN` plans) are kept isolated.
- **Per-record, by label.** Use `onlyif <label>` / `skipif <label>` on an individual record to
  include or exclude it for one engine. The available labels are `datafusion` and `duckdb`:

  ```text
  onlyif duckdb
  query T
  SELECT string_agg(str, ',') FROM '${WORK_DIR}/strings.vortex' WHERE prefix(str, 'He');
  ----
  Hello,Hey
  ```

## Regex assertions (DuckDB only)

For volatile output such as `EXPLAIN` plans, the DuckDB validator supports regex directives,
inspired by DuckDB's own `.test` files. When the expected block is a single line beginning with
one of these markers, the actual output (rows joined by newlines) is matched against the pattern
(`.` matches newlines):

- `<REGEX>:<pattern>` — passes when the pattern matches.
- `<!REGEX>:<pattern>` — passes when the pattern does **not** match.

```text
query TT
EXPLAIN (FORMAT json) SELECT strlen(str) FROM '${WORK_DIR}/pe-pushdown.vortex';
----
<REGEX>:SELECT projections
```

These markers are only honored by the DuckDB validator, which is why regex-based plan assertions
live under `slt/duckdb/`. A malformed pattern fails the assertion (it does not panic the run).

## Regenerating expected output (`--complete`)

Passing `--complete` rewrites each `.slt` file **in place** so its expected output matches what the
engine currently produces, instead of comparing against it. This is useful after an intentional
change to query results or plan formatting.

```shell
# Complete every file (generate TPC-H data first if you want its result files updated):
cargo test -p vortex-sqllogictest --test sqllogictests -- --complete
# Complete only the files whose name matches a substring:
cargo test -p vortex-sqllogictest --test sqllogictests -- --complete strings
```

Notes and caveats:

- **It encodes whatever the engine outputs today, bugs included.** Always review the diff before
  committing; a completion is not a substitute for knowing the correct answer.
- Each file is completed from a **single reference engine**: DuckDB for files under `slt/duckdb/`,
  DataFusion for everything else (including files that also run on DuckDB). If DuckDB then diverges
  from a shared file's DataFusion output, split the differing records out with
  `onlyif`/`skipif`.
- Scratch paths in output are normalized to `${WORK_DIR}` before being written, so completed files
  stay portable.
- `--complete` is intercepted before the test harness, so pass it after `--` (it is not a
  `cargo nextest` flag).

## Writing a new test

Tests must account for differences between the engines. The general pattern that works for basic
cases is a view over a file, since DuckDB and DataFusion don't share syntax for creating a table
backed by external storage.

`${WORK_DIR}` is a special variable pointing to a per-test working directory (the crate scratch
directory described above). It is only available when substitution is enabled via
`control substitution on` (see `slt/setup.slt.no`, included by most tests).

Here is a simple test that can be reused:

```text
query I
COPY (values (1, 2), (3, 4)) TO '${WORK_DIR}/test.vortex';
----
2

statement ok
CREATE VIEW foo AS SELECT * FROM '${WORK_DIR}/test.vortex';

query II
SELECT * FROM foo;
----
1 2
3 4

statement ok
DROP VIEW IF EXISTS foo;
```

Files ending in `.slt.no` are include fragments (pulled in via `include`), not standalone tests;
the runner only discovers `.slt` files.

## SLT Syntax

We generally use the default `slt` syntax as described in the
[SQLite wiki](https://sqlite.org/sqllogictest/doc/trunk/about.wiki). and the underlying crate's
[SLT Cookbook](https://github.com/risinglightdb/sqllogictest-rs#slt-test-file-format-cookbook). The
one difference is that we use the same column types as `datafusion-sqllogictest`'s, so when
specifying expected query result column types, we support the following identifiers:

- 'B' for boolean
- 'D' for datetime
- 'I' for integer
- 'P' for timestamp
- 'R' for float
- 'T' for text
- '?' for anything else
