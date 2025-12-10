# `vx`

A small, helpful CLI tool for exploring and analyzing Vortex files.

* `browse`: Browse the structure of your Vortex file with a rich TUI
* `tree`: Print the encoding tree of a Vortex file
* `inspect`: Inspect Vortex file footer and metadata
* `convert`: Convert a Parquet file to a Vortex file

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

## Development

TODO:

* [ ] `cat` to print a Vortex file as JSON to stdout
* [ ] `compress` to ingest JSON/CSV/other formats that are Arrow-compatible
