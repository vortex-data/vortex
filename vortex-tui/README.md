# `vx`

A small, helpful CLI tool for exploring and analyzing Vortex files.

* `browse`: Browse the structure of your Vortex file with a rich TUI
* `tree`: Print the encoding tree of a Vortex file
* `inspect`: Inspect Vortex file footer and metadata
* `convert`: Convert a Parquet file to a Vortex file
* `query`: Run a SQL query against a Vortex file and print JSON
* `segments`: Print the segment layout of a Vortex file as JSON
* `coverage`: Inventory session registrations and warn about missing expected fast paths

## Examples

Using the `tree` subcommand to print the encoding tree for a file:

```
$ vx tree ./vortex-bench/data/tpch/1/vortex_compressed/nation.vortex

root: vortex.struct(0x04)({n_nationkey=i64, n_name=utf8, n_regionkey=i64, n_comment=utf8?}, len=25) nbytes=3.04 kB (100.00%)
  metadata: StructMetadata { validity: NonNullable }
  n_nationkey: $vortex.primitive(0x03)(i64, len=25) nbytes=201 B (6.62%)
    metadata: PrimitiveMetadata { validity: NonNullable }
    buffer (align=8): 200 B
  n_name: $vortex.varbinview(0x06)(utf8, len=25) nbytes=461 B (15.18%)
    metadata: VarBinViewMetadata { validity: NonNullable, buffer_lens: [27] }
    views: $vortex.primitive(0x03)(u8, len=400) nbytes=401 B (13.20%)
      metadata: PrimitiveMetadata { validity: NonNullable }
      buffer (align=1): 400 B
    bytes_0: $vortex.primitive(0x03)(u8, len=27) nbytes=28 B (0.92%)
      metadata: PrimitiveMetadata { validity: NonNullable }
      buffer (align=1): 27 B
  n_regionkey: $vortex.dict(0x14)(i64, len=25) nbytes=83 B (2.73%)
    metadata: DictMetadata { codes_ptype: U8, values_len: 5 }
    values: $vortex.primitive(0x03)(i64, len=5) nbytes=41 B (1.35%)
      metadata: PrimitiveMetadata { validity: NonNullable }
      buffer (align=8): 40 B
    codes: $vortex.primitive(0x03)(u8, len=25) nbytes=26 B (0.86%)
      metadata: PrimitiveMetadata { validity: NonNullable }
      buffer (align=1): 25 B
  n_comment: $vortex.varbinview(0x06)(utf8?, len=25) nbytes=2.29 kB (75.44%)
    metadata: VarBinViewMetadata { validity: AllValid, buffer_lens: [1857] }
    views: $vortex.primitive(0x03)(u8, len=400) nbytes=401 B (13.20%)
      metadata: PrimitiveMetadata { validity: NonNullable }
      buffer (align=1): 400 B
    bytes_0: $vortex.primitive(0x03)(u8, len=1857) nbytes=1.86 kB (61.18%)
      metadata: PrimitiveMetadata { validity: NonNullable }
      buffer (align=1): 1.86 kB
```

Opening an interactive TUI to browse the sample file:

```
vx browse ./vortex-bench/data/tpch/1/vortex_compressed/nation.vortex
```

### Inspecting File Footer

The `inspect` subcommand allows you to examine the internal structure of a Vortex file at different levels:

```bash
# Inspect just the EOF marker (8 bytes at end of file)
vx inspect ./data/sample.vortex eof

# Inspect the postscript (includes EOF and postscript segments)
vx inspect ./data/sample.vortex postscript

# Inspect the full footer structure (default - includes all segments)
vx inspect ./data/sample.vortex footer

# Or simply run without specifying mode to get full footer info
vx inspect ./data/sample.vortex
```

Example output:
```
File: "./data/sample.vortex"
Size: 1048576 bytes

=== EOF Marker ===
Version: 1 (current: 1)
Postscript size: 256 bytes
Magic bytes: "VTXF" (VALID)

=== Postscript ===
  DType: offset=1047808, length=512, alignment=1
  Layout: offset=1048000, length=320, alignment=1
  Statistics: <not present>
  Footer: offset=1048320, length=256, alignment=1

=== Footer Segments ===
Total segments: 42
Total data size: 1047808 bytes

Segment details:
  [0] offset=0, length=4096, alignment=8
  [1] offset=4096, length=8192, alignment=8
  ...
```

### Converting Files

Convert a Parquet file to Vortex format:

```bash
# Basic conversion
vx convert input.parquet --out output.vortex

# With compression enabled
vx convert input.parquet --out output.vortex --compress
```

## Browser (WASM)

The TUI browser can also run in a web browser via WebAssembly. Users can drag-and-drop a `.vtx`
file onto a web page and interactively explore it using the same Layout and Segments tabs.

### Quick start

```bash
# From the vortex-tui directory:
make serve
```

This builds the WASM package and starts a local server at `http://localhost:8000`.

Requires `wasm-pack` (`cargo install wasm-pack`).

### How it works

- **Native** builds use `crossterm` for terminal I/O and `CurrentThreadRuntime` (smol-based) for
  async execution. The `native` feature (enabled by default) pulls in DataFusion, crossterm, and
  other native-only dependencies.
- **WASM** builds use [ratzilla](https://github.com/ratatui/ratzilla) (the official ratatui web
  backend) for rendering and `WasmRuntime` for async execution. Building with
  `--no-default-features` excludes all native-only dependencies.
- The Query tab (DataFusion SQL) is only available in native builds.

## Development

TODO:

* [ ] `cat` to print a Vortex file as JSON to stdout
* [ ] `compress` to ingest JSON/CSV/other formats that are Arrow-compatible

## Diagnosing kernel coverage

`vx coverage` inspects the active session after initialization. It lists in-memory encoding
plugins (without counting historical wire IDs twice), layout and function IDs, parent execution
kernels, session parent reductions, exposed static reduction rules, and ordinary/grouped aggregate
kernels. Custom sessions passed to `launch_from` are inspected in the same way.

```bash
cargo run -p vortex-tui --bin vx -- coverage
vx coverage --json

# Capture the current declarations, then review/edit the required set for each encoding.
vx coverage --baseline > coverage-expectations.json
vx coverage --expectations coverage-expectations.json
vx coverage --expectations coverage-expectations.json --strict --json
```

The default report is an inventory, with no implied minimum kernel set. Supplying expectations
adds warnings; `--strict` returns a failure after printing the report if any expectations are
unmet. Invalid JSON, unknown manifest fields, and unknown kinds are always errors. This minimal
manifest requires two compute paths, an aggregate kernel, and the flat layout:

```json
{
  "layouts": ["vortex.flat"],
  "encodings": {
    "vortex.runend": [
      {"kind": "compute", "operation": "vortex.filter"},
      {"kind": "compute", "operation": "vortex.dict"},
      {"kind": "aggregate", "operation": "vortex.sum"}
    ]
  }
}
```

Kinds are `compute`, `session_reduce`, `static_reduce`, `static_parent_reduce`, `aggregate`,
and `grouped_aggregate`. Compute and session-reduce operations are parent array or scalar-function
IDs (`vortex.dict` is the take parent). Static-rule operations are exact debug descriptions from
`--baseline`, including their matcher types. They are diagnostic names, so a Rust rule or module
rename can require updating expectations. `--baseline` also enables `require_all_encodings`, which
warns when a registered encoding has no entry in the manifest. Set it to `false` (the default for
hand-written manifests) to validate a selected subset. An empty per-encoding list requires that
encoding or kernel owner to exist, without requiring any particular kernel.

Aggregate operation `"*"` means an encoding-wide aggregate candidate; encoding `"*"` holds
encoding-agnostic grouped candidates. These are literal declarations, not expectation wildcards:
an encoding-wide candidate does not satisfy an expectation for a function-specific registration.
Aggregate vtable shortcuts and canonical fallbacks are outside this registry report. Likewise,
opaque reduction hooks are reported as unknown, never inferred to be missing or supported. Static
rules are discovered through registered array plugins; execution-only arrays without a plugin
may appear as kernel owners but do not expose their static rules through this inventory.

Registration coverage and runtime traces answer different questions. This command identifies
missing declarations and registration regressions; it does not prove a candidate accepts a dtype,
nullability, child position, or input shape. It also does not measure file layout usage, time, bytes,
or rows. For a specific execution path, the existing `vortex_array::test_harness::trace` harness
can show applied and declined reductions and parent kernels with `TraceResolution::Attempts`.
That capture is thread-local and feature-gated, so it is not a whole-query profiler. A future
runtime report should correlate candidate attempts, successes, decode fallbacks, rows, bytes,
and timing with the encoding/function pairs reported here, including aggregate and layout events.
