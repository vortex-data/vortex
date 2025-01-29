# `vx` Vortex CLI

A small, helpful CLI tool for exploring and analyzing Vortex files.

* `browse`: Browse the structure of your Vortex file with a rich TUI
* `tree`: print the file contents as JSON


## Examples

Using the `tree` subcommand to print the encoding tree for a file:

```
$ vx tree ./bench-vortex/data/tpch/1/vortex_compressed/nation.vortex

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
vx browse ./bench-vortex/data/tpch/1/vortex_compressed/nation.vortex
```

## Development

TODO:

- [ ] `cat` to print a Vortex file as JSON to stdout
- [ ] `compress` to ingest JSON/CSV/other formats that are Arrow-compatible
