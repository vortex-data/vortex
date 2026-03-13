# vortex-sqllogictest

This crate uses `sqllogictest-rs` to run `slt` based tests on both DF and DuckDB, both preconfigured to work with Vortex.

Different test files might run in parallel, but within the same file, the file will run for each query engine sequentially before the next one starts.

## Running tests

In order to run the tests, you first need to generate TPC-H data (scale factor 0.1), the commands are:

```shell
./vortex-sqllogictest/slt/tpch/generate_data.sh
cargo test -p vortex-sqllogictest --test sqllogictests
```

Note that `nextest` isn't currently supported, but might be in the future.

## Writing a new test

Currently, tests must account for the differences between the engines, the general pattern that works for basic things is using views over files, as DuckDB as and DataFusion don't seem to have a shared syntax to create a table backed by an external storage format.

`$__TEST_DIR__` is a special variable used to point to a tempdir, its only available if substitution is enabled, by using `control substitution on`.

Here is a simple test that can be reused:

```text
query I
COPY (values (1, 2), (3, 4)) TO '$__TEST_DIR__/test.vortex';
----
2

statement ok
CREATE VIEW foo AS SELECT * FROM '$__TEST_DIR__/test.vortex';

query II
SELECT * FROM foo;
----
1 2
3 4

statement ok
DROP VIEW IF EXISTS foo;
```

## SLT Syntax

We generally use the default `slt` syntax as described in the [SQLite wiki](https://sqlite.org/sqllogictest/doc/trunk/about.wiki). The one difference is that we use the same column types as `datafusion-sqllogictest`'s, so when specifying expected query result column types, we support the following identifiers:

- 'B' for boolean
- 'D' for datetime
- 'I' for integer
- 'P' for timestamp
- 'R' for float
- 'T' for text
- '?' for anything else
