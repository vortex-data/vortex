# GitHub Archive benchmark

A deeply nested real-world dataset from the [GitHub Archive](https://www.gharchive.org/):
GitHub event records whose `payload`, `repo`, `actor`, and `org` fields are nested structs.
The queries in [`gharchive.sql`](./gharchive.sql) (numbered from Q0 in file order) filter
and aggregate on those nested fields, making this the main struct-field-pushdown workload.

The harness lives in [`src/realnest/gharchive.rs`](../src/realnest/gharchive.rs).

## Running locally

```bash
cargo run --bin datafusion-bench --profile release_debug -- --benchmark gharchive
```
