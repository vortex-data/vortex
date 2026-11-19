<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Protobuf as a Vortex extension type

This branch (`ct/protobuf-ext`) is an experiment. It stores many protobuf messages of one schema
as a Vortex column, with one Vortex column per protobuf field, so that query engines such as
DataFusion read only the fields that a query uses. Nothing here is stable or merged. The work
also fixed several general scan gaps on the way, and some of them can stand on their own.

[`HANDOFF.md`](HANDOFF.md) is the prompt for an agent that continues this work. This file is the
reference that the prompt points to.

## Contents

1. [Summary](#summary)
2. [Background: protobuf and columns](#background-protobuf-and-columns)
3. [Architecture](#architecture)
4. [The commits](#the-commits)
5. [The protobuf crate](#the-protobuf-crate)
6. [The extension layout](#the-extension-layout)
7. [Scan changes](#scan-changes)
8. [DataFusion changes](#datafusion-changes)
9. [Format and edition changes](#format-and-edition-changes)
10. [Measurements](#measurements)
11. [Decisions to review](#decisions-to-review)
12. [Known gaps](#known-gaps)
13. [How to split this branch](#how-to-split-this-branch)
14. [How to build and test](#how-to-build-and-test)
15. [Related work](#related-work)

## Summary

- `vortex.protobuf` is an extension dtype. Its metadata is a canonical protobuf descriptor set plus
  the root message name, a mapping version and mapping options. `ExtVTable::storage_dtype` derives
  the storage dtype (a tree of `Struct`, `List`, primitives, `Utf8` and `Binary`) from the metadata.
- `vortex.ext` is a new layout. For an extension type that opts in, the writer splits the storage
  struct into per-field columns, and both scan paths rewrite `ext_storage(root)` to the storage
  root. Before this layout, every extension column was one segment per block.
- A fast decoder and encoder read and write the protobuf wire format straight into and out of
  Vortex column builders, with protobuf parse semantics.
- Mapping version 2 covers the rest of the protobuf spec, except protobuf editions syntax: kept
  unknown fields, recursion to a depth limit, wrappers, timestamps, JSON well-known types, enum
  names and oneof cases.
- The scans now read one field of a nullable struct instead of every field, write zone maps
  beneath nullable structs, read list layouts on the planner, and push field access on list
  elements into the element columns.
- DataFusion pushes `msg['a']['b']` into the scan through `ext_storage`, pushes the convertible
  parts of expressions that do not convert as a whole (partial projection pushdown), and has three
  protobuf SQL functions.

All of it is built from abstractions that already existed: `ExtVTable` with storage declared from
metadata, scalar functions, `ext_storage`, array builders, the struct and list layouts, the Arrow
registry and the pluggable `ExpressionConverter`. The new abstractions are the extension layout,
the planner's list source, and partial projection pushdown.

## Background: protobuf and columns

A protobuf schema (`.proto`) compiles to a descriptor, which is itself a protobuf message, so it
can be stored as metadata. A serialized message is a flat list of records. Each record is a tag
(a varint that holds the field number and a wire type) and a value. A nested message is a
length-delimited record whose payload is another message. The bytes hold no field names and no
index, so reading one field means walking every record before it.

The protobuf data model maps onto Vortex dtypes. A message is a struct, a repeated field is a
list, and a map is a list of key and value structs. A oneof is a group of nullable fields. Google's
Dremel paper (2010) first stored nested protobuf records in columns, and Parquet copied that
design, so the columnar form is well tested.

Wire rules that the codec must respect:

- Fields can come in any order.
- A singular field that appears twice keeps the last value, and two copies of a nested message
  merge. So two encoded messages joined end to end parse as the merge of the two.
- Setting a oneof member clears the member set before it.
- Repeated scalars can be packed or unpacked, and a parser accepts both.
- A parser skips field numbers that it does not know and keeps their bytes.
- Presence: message fields, oneof members and fields marked `optional` track whether they are
  set. Plain proto3 scalars do not, so "not set" and "set to 0" are the same value.

## Architecture

```text
protobuf bytes ──ProtobufDecoder──► ExtensionArray(vortex.protobuf, storage: Struct{...})
                                         │
                       file writer: TableStrategy
                                         │ opted-in extension with struct storage
                                         ▼
                       vortex.ext layout ─► vortex.struct layout ─► one column per field
                                                                  (lists: vortex.list when enabled)
query: SELECT msg['a']['b'] (DataFusion)
   └─ converter: get_item("b", get_item("a", ext_storage($.msg)))
        └─ vortex.ext reader / ExtPack plan: ext_storage(root) → storage root
             └─ struct reader / Pack plan: reads only column a.b (+ validity of nullable parents)
```

Field access on a protobuf value is always written as `get_item` over `ext_storage`. That fits
the opaque extension type plan on `ct/ext-types-research`: no claim is needed, because
`ext_storage` is the explicit way to read storage, and no `ext_wrap` expression exists. The layout
reader and the planner wrap storage back into the extension type themselves when a query reads
whole values.

## The commits

The branch is eight commits on `develop` at `f5b77122a`, then the notes commit. Every commit is
signed off.

| Commit | Title | Main crates |
| --- | --- | --- |
| `5e90a2898` | Add an experimental extension layout that decomposes struct storage | vortex-layout, vortex-file, vortex-edition |
| `ba36f90425c01a28a57d957a71c1ae7fcb8511dd` | Add an experimental protobuf extension type | vortex-protobuf (new), vortex-datafusion |
| `91dbd9bf9` | Read single fields of nullable structs and prune beneath them | vortex-scan, vortex-layout |
| `e11a0b7d6` | Decode and encode protobuf columns from the wire format | vortex-protobuf |
| `65801f4f7` | Push field reads under unsupported expressions into Vortex scans | vortex-datafusion |
| `7ac0dbbd0` | Keep NaN counts beneath nullable structs out of a new aggregate | vortex-layout, vortex-edition |
| `d318b6ea0` | Read single fields of list elements on both scans | vortex-layout, vortex-scan |
| `0cb27a3cf` | Add DataFusion functions for protobuf columns | vortex-datafusion, vortex-protobuf |

`7ac0dbbd0` corrects `91dbd9bf9`. The message of `91dbd9bf9` says that fields beneath a nullable
struct record "whether a zone holds no NaN in place of the NaN count" and that `vortex.all_non_nan`
joins the unstable edition. Both were undone in `7ac0dbbd0`: those fields record the NaN count, and
the unstable edition has no new aggregate. Fold the two together when you split the branch.

## The protobuf crate

`vortex-protobuf/` is a new workspace crate. The crate docs in `src/lib.rs` hold the full mapping
table, the options and the round-trip rules; read them first.

### Files

| File | Role |
| --- | --- |
| `src/lib.rs` | Crate docs, public exports, `initialize(session)` |
| `src/dtype.rs` | `Protobuf`, the `ExtVTable` (ID `vortex.protobuf`) |
| `src/metadata.rs` | `ProtobufMetadata`: mapping version, root name, canonical descriptor bytes, options, the parsed `MessageDescriptor` |
| `src/canonical.rs` | Builds the canonical descriptor set: only reachable messages and enums, sorted, with options and source info dropped except the ones that change the wire format |
| `src/pool.rs` | Process-wide cache of parsed descriptor pools, keyed by canonical bytes (64 entries, least recently used) |
| `src/options.rs` | `MappingOptions`, `WrapperMapping`, `TimestampMapping`, `JsonMapping`, `MAX_RECURSION_DEPTH` |
| `src/mapping.rs` | The storage plan: one tree built from the descriptor and the options, which the dtype, decoder, encoder and helpers all follow. `UNKNOWN_FIELDS` (`$unknown`) |
| `src/wire.rs` | Wire-format helpers shared by the codec |
| `src/decode/` | `ProtobufDecoder`, `decode_messages`; `message.rs` is the per-message parser, `builders.rs` the truncatable column builders, `well_known.rs` the dedicated forms |
| `src/encode/` | `encode_messages` (columns to wire bytes), `scalar.rs` and `value.rs` for one value, `function.rs` for the `vortex.protobuf.encode` scalar function |
| `src/well_known/` | Timestamps, wrappers and the proto3 JSON mapping for `Struct` and `Value` |
| `src/enum_name.rs` | `EnumName`, the `vortex.protobuf.enum_name` scalar function |
| `src/field.rs` | `protobuf_field`, `protobuf_enum_name`, `protobuf_oneof_case` expression helpers |
| `src/value.rs` | `ProtobufValue`, the native scalar value, with `to_message` and `encode_to_vec` |
| `src/tests/` | Tests; `oracle.rs` is the old prost-reflect codec, kept as a test oracle |
| `benches/protobuf.rs` | divan benchmark on a clickstream schema (`benches/clickstream/`) |

### Metadata and identity

- `ProtobufMetadata::try_new(&MessageDescriptor)` and `try_new_with_options` build the metadata.
  Serialization is a small `#[derive(prost::Message)]` struct in Rust, with no `.proto` file.
- `Eq` and `Hash` use only the mapping version, the root name, the canonical bytes and the options.
  Two metadata values of one schema share one pool through the cache, so their descriptors compare
  equal too.
- Reads require that the stored storage dtype equals the declared one. So the mapping from
  descriptor to dtype is part of the file format. Version 1 must keep giving exactly the dtype it
  gave at `ba36f90425c01a28a57d957a71c1ae7fcb8511dd`, and a test pins that.
- `validate_metadata` rejects unknown versions, unknown option values, version 1 metadata with
  options, malformed map entries, and recursive schemas without `max_recursion_depth`. A storage
  tree is capped at 100,000 fields for every version, because a schema without recursion can still
  expand exponentially through shared message types.

### Storage mapping

Version 1 (all options at their defaults):

| Protobuf | Vortex storage |
| --- | --- |
| message | `Struct`, fields in declaration order, protobuf names |
| field with presence (message, oneof member, `optional`, proto2 singular) | nullable |
| plain proto3 scalar | non-nullable, absent means the default |
| `repeated T` | non-nullable `List<T>` with non-nullable elements |
| `map<K, V>` | non-nullable `List<Struct{key, value}>`, sorted by key, last duplicate wins |
| `double`, `float` | `f64`, `f32` |
| `int32`, `sint32`, `sfixed32` / 64-bit forms | `i32` / `i64` |
| `uint32`, `fixed32` / 64-bit forms | `u32` / `u64` |
| `bool`, `string`, `bytes` | `Bool`, `Utf8`, `Binary` |
| `enum` | `i32` (names through `protobuf_enum_name`) |
| groups, well-known types | ordinary messages |

Version 2 adds options. Each option is stored in the metadata, so the dtype stays a pure function
of the metadata:

| Option | Effect |
| --- | --- |
| `preserve_unknown_fields` | Each message level gets a `Binary` field `$unknown` with the raw bytes of unknown fields. Protobuf names cannot contain `$`. The encoder writes them after the known fields |
| `max_recursion_depth` | A recursive message is expanded to this depth (at most 100). Deeper occurrences are stored as `Binary` holding the encoded message |
| `wrappers: Primitive` | `Int32Value` and the other wrappers become nullable primitives |
| `timestamps: Nanoseconds` | `Timestamp` becomes the Vortex timestamp type in nanoseconds; values outside the `i64` range fail |
| `json: Text` | `Struct`, `Value` and `ListValue` become the `vortex.json` extension (proto3 JSON text) |

### Codec semantics

- The decoder compiles a plan from the descriptor once. Field numbers up to 1024 use a direct
  table, larger ones a binary search. Each message is one pass with `prost::encoding`.
- Columns fill lazily: a column that a row does not set catches up with one bulk append when it is
  next set or at the end, so the cost follows the values present, not rows times fields.
- The builders can remove rows. That gives last-value-wins for singular fields, oneof replacement
  (including whole nested messages), and rollback of a message that fails to decode, so a failed
  `append_message` leaves the decoder unchanged.
- Strings are checked with `simdutf8`. Malformed input returns an error and never panics; fuzz-like
  tests check this.
- The encoder has a size pass and a write pass, writes fields in number order, packs packed
  fields, and omits implicit-presence defaults but writes `-0.0` and NaN. The old prost-reflect
  codec lost `-0.0`.
- The old prost-reflect codec is now only a test oracle (`src/tests/oracle.rs`). It differs from
  protobuf parse rules in a few places, which `src/tests/semantics.rs` documents.

### Helpers and functions

- `protobuf_field("a.b", msg)` builds `get_item("b", get_item("a", ext_storage(msg)))`.
- `protobuf_enum_name(...)` wraps a field in the `vortex.protobuf.enum_name` function; open enums
  give null for unknown numbers.
- `protobuf_oneof_case(...)` builds a `case_when` over `is_not_null` of the members.
- `protobuf_encode(msg)` builds the `vortex.protobuf.encode` function (`Encode`), which returns the
  wire bytes.
- `initialize(session)` registers the dtype, the functions, `vortex_json` (which also enables the
  JSON edition), the unstable edition membership of `vortex.protobuf`, and opts the type into the
  extension layout with `LayoutSession::decompose_extension`.

### Tests

The crate has 346 unit tests. The main suites:

| Suite | Covers |
| --- | --- |
| `differential.rs` | Seeded random messages on several schemas; the decoder against the oracle; non-canonical bytes (shuffled, duplicated, split, packed and unpacked, overlong varints, unknown fields) against canonical bytes; concatenation as merge; encode round trips |
| `malformed.rs` | 21 malformed inputs with expected errors, every prefix of a message, random corruption, rollback |
| `semantics.rs` | One hand-written byte case per parse rule |
| `untrusted.rs` | Metadata that must be rejected without panics |
| `unknown.rs`, `recursion.rs`, `well_known.rs`, `enums_and_oneofs.rs` | The version 2 options and helpers |
| `file.rs` | Files written with the default strategy: layout snapshot, round trips, field reads, filters, list element reads, both scans |
| `metadata.rs`, `mapping.rs`, `arrow.rs`, `codec.rs`, `encode_function.rs`, `editions.rs`, `protobuf_editions.rs` | Metadata, mapping snapshots, Arrow round trip, codec basics, the encode function, edition membership, the editions-syntax gap |

## The extension layout

Code: `vortex-layout/src/layouts/extension/` (`mod.rs`, `writer.rs`, `reader.rs`, `plan.rs`,
`expr.rs`, `tests.rs`).

- ID `vortex.ext`, one child slot `storage` (`LayoutChildType::Transparent`), empty metadata. The
  layout dtype is the extension dtype.
- Writer: `TableStrategy::with_extension_layout(ids)` lists the extension types to split. For such
  a type whose storage is a struct, or a list when list decomposition is on, `ExtensionStrategy`
  maps each chunk to its storage (constants stay constant) and writes it with a copy of the same
  `TableStrategy`, so field overrides address storage fields. Other extension types keep the leaf
  path, so existing files do not change.
- Opt-in: `LayoutSession::decompose_extension(id)`. `WriteStrategyBuilder::from_session` uses the
  list only when the enabled editions include `vortex.ext`. The opt-in exists because splitting
  removes statistics computed over whole extension values, such as the spatial bounding boxes.
- Rewrite (`expr.rs`, `rebind_to_storage`): `ext_storage(root)` becomes the storage root, and
  `is_null(root)` and `is_not_null(root)` move to the storage, because extension validity is
  storage validity. Any other use of the root does not rebind.
- Reader (`ExtensionReader`): a rebound expression goes to the storage reader for projection,
  filters, pruning and splits. Otherwise it reads the whole storage, wraps it, and evaluates the
  original expression. Pruning that cannot rebind keeps every row. Rebound expressions are cached
  by `ExactBoundExpr` so that the struct reader's caches stay warm.
- Plan (`ExtPack`, ID `vortex.plan.ext_pack`): wraps each storage batch, and its `reduce_parent`
  rule rewrites `Eval(expr, ExtPack(storage))` to `Eval(rebound, storage)`.

## Scan changes

These are general improvements. They help every nested dataset, not only protobuf.

### Nullable structs on the planner (`91dbd9bf9`)

- `ExpressionPackRule` and `with_pruned_fields` gave up on nullable `Pack`s, so the planner read
  every field of a nullable struct for a one-field projection. Now `get_item` over a nullable Pack
  reads the field and the struct validity. `lift_masks` (`vortex-scan/src/plan/builtins/pack/mask.rs`)
  moves nested field reads below the masks.
- The same early exit caused "Plan vortex.plan.list_pack has no pipeline implementation" for any
  read beside a list-layout child of a nullable struct.
- Code: `vortex-scan/src/plan/builtins/pack/`, `vortex-layout/src/layouts/struct_/unmask.rs`.

### Zone maps beneath nullable structs (`91dbd9bf9`, `7ac0dbbd0`)

- Fields of a nullable struct are stored unmasked, so they hold values in rows where the struct is
  null. The zoned writer wrote no zone map for them (the `TODO(joe)` in
  `vortex-layout/src/layouts/zoned/writer.rs`).
- Now such fields record only statistics that stay correct under masking: min, max, bounded min
  and max, and the NaN count. Hidden values can only widen a bound or raise the NaN count, so
  pruning stays conservative. Null counts are not recorded, because hidden values make them wrong
  (issue #9205: `is_null(s.a)` would prune zones that hold matching rows).
- `holds_under_masking` (vortex-scan) decides which statistics lift past a mask. A statistic is
  lifted only when every function above it cannot fail, because a falsifier can feed a bound into a
  fallible cast.
- This applies to every writer, in every edition.

### Lists (`d318b6ea0`)

- Classic reader (`vortex-layout/src/layouts/list/expr/`, `reader/`): `ElementProjection` moves
  each `list_transform(root, lambda)` body onto the elements child, so `items[*].sku` reads the
  `sku` column plus offsets and validity. Several lambdas share one elements read. Lambdas that can
  fail use a compacted path that reads only elements of selected, valid lists.
- Struct reader: `expand_struct_root` keeps a chain of one-argument functions that are strict and
  cannot fail inside the mask of a nullable struct, so the pushdown reaches lists inside nullable
  messages.
- Planner (`vortex-scan/src/plan/builtins/list_pack/`): `ListPack::compile` builds a `ListSource`.
  It reads offsets and validity of the selected lists, computes the element range of selected,
  valid lists, then runs a small inner scan of the elements whose reads become its own reads.
  `ExpressionListPackRule` handles validity-only, length-only and lambda pushdown. Element
  segments are not announced for prefetch.
- `vortex.list` is not in the unstable edition (see below). Tests include it explicitly.

## DataFusion changes

### `get_field` through extensions (`ba36f90425c01a28a57d957a71c1ae7fcb8511dd`)

`vortex-datafusion/src/convert/exprs/mod.rs`, the `GetFieldFunc` branch: at each step of a field
path, if the Arrow field names an extension that the session imports, and the session exports that
extension as its storage struct, the converter inserts `ext_storage` before `get_item`. An
extension that the session does not know stays in DataFusion.

### Partial projection pushdown (`65801f4f7`, `0cb27a3cf`)

`vortex-datafusion/src/convert/exprs/partial/`:

- If every projection expression converts, nothing changes.
- Otherwise the scan evaluates the largest convertible subexpressions that read columns, as
  `__vortex_expr_N` columns, and DataFusion evaluates the rest over the scan output. Identical
  subtrees are read once.
- A fallible subexpression is pushed only where DataFusion evaluates it on every row (a whole
  projection expression with no residual filter). Elsewhere only infallible subexpressions are
  pushed.
- A top-level conversion error leaves that expression to DataFusion instead of failing the query.
- `ExpressionConverter` gained two public provided methods, `split_filtered_projection` and
  `no_pushdown_filtered_projection`, so the opener's residual-filter path uses a converter's
  overrides.
- This overlaps with open PRs #11431 and #11432 (danking), which embed DataFusion fragments in
  Vortex expression trees. The two designs must be reconciled.

### Protobuf SQL functions (`0cb27a3cf`)

`vortex-datafusion/src/protobuf.rs`, feature `unstable_protobuf`: `protobuf_enum_name(msg, 'path')`,
`protobuf_oneof_case(msg, 'path')` and `protobuf_encode(msg)`. `register_functions(registry)`
registers them. Each converts to the vortex-protobuf expression
(`vortex-datafusion/src/convert/exprs/protobuf.rs`); evaluated in DataFusion, it imports the storage
and runs the same expression.

### Semantics to know

Nested field access under a null parent returns null when pushed down. DataFusion's own
`get_field` returns the stored child value. The converter already made this choice for fully
converted projections; partial pushdown applies it in more places.

## Format and edition changes

| Change | Who it affects |
| --- | --- |
| `vortex.ext` in the unstable edition | Writers with unstable enabled and a type opted in |
| Zone maps (bounds, NaN count) beneath nullable structs | Every writer, every edition |
| `vortex.protobuf` dtype in the unstable edition (added by `initialize`) | Sessions that initialize vortex-protobuf |
| `vortex.list` | Not in unstable. PR #11595 removed it after an audit, and vortex-bench enables unstable for every suite |
| `docs/developer-guide/benchmarking.md` | One stale sentence about the unstable edition's members was corrected |
| `scripts/codspeed_shards.py` | vortex-protobuf is excluded from CodSpeed, like vortex-btrblocks, because its benchmarks are 100,000-message local runs |

## Measurements

From `cargo bench -p vortex-protobuf` (bench profile, one thread, files in memory), on 100,000
synthetic clickstream messages (282 bytes on average, 28.5 MB, 30 leaf fields), on a shared VM
with 4 CPUs. The data compresses easily, so real data will compress less.

| Measure | Result |
| --- | --- |
| Decode | 209 MB/s; `DynamicMessage::decode` alone: 78 MB/s (2.7x) |
| Encode | 234 MB/s; `DynamicMessage` encode: 46 MB/s (5.1x) |
| Vortex file, protobuf column split into fields | 2.85 MB (0.10x of wire bytes) |
| Wire bytes with zstd level 3 | 4.44 MB |
| Vortex file, wire bytes in a binary column | 19.1 MB |
| Read leaf `timestamp_ms` / whole column | 233 us / 7.5 ms, 94 kB / 2,822 kB read |
| Read leaf `geo.city` | 518 us, 63 kB of needed segments |

Not measured: `items[*].sku` with the list layout, DataFusion SQL end to end, real data.

## Decisions to review

1. Zone maps beneath nullable structs for every writer. This is a format change outside the
   experiment. Open PR #11160 (normalize child validity beneath nullable structs) is the other
   approach. If it lands, hidden values become defaults and bounds widen.
2. Null under a null parent for pushed-down nested field access (see above).
3. Partial pushdown versus #11431 and #11432.
4. The two new public methods on `ExpressionConverter`.
5. vortex-protobuf excluded from CodSpeed.
6. `vortex.list` stays out of unstable, so repeated protobuf fields split into element columns only
   when a session includes `vortex.list` itself.
7. The 100,000-field cap also applies to version 1 metadata.
8. Unknown fields are written after known fields. If a newer schema adds a member to an existing
   oneof, or a number to a closed enum, the known value and the unknown one can swap which one
   wins. C++ protobuf has the same limit.
9. `initialize` enables the JSON edition as a side effect, through `vortex_json::initialize`.
10. The layout ID `vortex.ext` is the same string as the array encoding ID of `ExtensionArray`
    (`vortex-array/src/arrays/extension/vtable/mod.rs`). Layout and array IDs live in separate
    registries and edition lists, so nothing breaks, but a distinct name such as
    `vortex.extension` would be clearer before the format is frozen.

## Known gaps

- Protobuf editions syntax (`edition = "2023"`): protox 0.9.1 cannot parse it, prost-reflect
  0.16.5 rejects it, and prost-types 0.14 has no `edition` field.
- Unknown fields inside map entries and inside the dedicated forms of well-known types are dropped,
  even with `preserve_unknown_fields`. Bytes at a recursion cut are not validated on decode.
- Classic list reader: no element zone maps, so element filters do not prune;
  `register_splits` uses `FieldMask::All` for elements; functions with several arguments over a
  nullable field (for example `list_contains(list_transform(...), lit)`) push only the inner
  `list_transform`.
- Planner lists: element segments are not announced for prefetch, and a split collects all its
  elements before it emits a batch.
- Beneath nullable structs, `is_null` and `is_not_null` on fields never prune, because null counts
  are not recorded. Exact statistics need the struct validity to travel with each chunk through
  repartitioning (the `TODO(joe)` in the zoned writer). File-level statistics
  (`vortex-layout/src/layouts/file_stats.rs`) still record nothing beneath nullable structs.
- The classic scan reads the projected columns in zones that the filter pruned.
- `vortex-bench/src/lib.rs` has a comment saying the planner cannot read list layouts. It is now
  out of date.
- Not run: the MSRV check for vortex-protobuf (CI uses `cargo hack --rust-version`), the
  vortex-datafusion integration test targets, and the Python, DuckDB and sqllogictest suites, which
  the zone map change could affect.

## How to split this branch

The branch mixes general scan work with the experiment. A suggested stack, from most general to
most specific, each a separate PR on `develop`:

1. Planner: read single fields of nullable structs, and the `list_pack` failure it fixes. From
   `91dbd9bf9` (`vortex-scan/src/plan/builtins/pack/`, the struct layout changes, their tests).
2. Zone maps beneath nullable structs. From `91dbd9bf9` and `7ac0dbbd0` (zoned writer,
   `holds_under_masking`, `lift_masks` statistics part, the vortex-file and v2 pruning tests). This
   is a format change, so it needs its own discussion and an answer on #11160.
3. List element pushdown on the classic reader, with the struct reader rule for strict functions.
   From `d318b6ea0` (`vortex-layout/src/layouts/list/`, `struct_/reader.rs`).
4. Planner support for list layouts (`ListPack` compile and `ExpressionListPackRule`). From
   `d318b6ea0` (`vortex-scan/src/plan/builtins/list_pack/`, `mask.rs` strict-function lifting,
   `vortex-layout/src/scan/v2/tests/lists.rs`).
5. Partial projection pushdown in vortex-datafusion. From `65801f4f7` and the `partial/` and opener
   parts of `0cb27a3cf`. Reconcile with #11431 and #11432 first.
6. DataFusion `get_field` through extension storage. From `ba36f90425c01a28a57d957a71c1ae7fcb8511dd` (converter only). Useful for
   any struct-storage extension type.
7. The extension layout `vortex.ext`. From `5e90a2898`.
8. The vortex-protobuf crate, the protobuf DataFusion functions and the benchmarks. The rest.

Items 1 to 4 do not depend on protobuf; their tests use plain structs and lists, except some
protobuf file tests, which move with item 8. When splitting, a "benchmarked change" (see AGENTS.md)
needs its benchmark landed first.

## How to build and test

The machine used had 4 CPUs and about 60 GB of disk. Use the `ci` profile and scope every command
to crates; the full workspace does not fit.

```bash
CARGO_INCREMENTAL=0 cargo test --profile ci -p vortex-protobuf
CARGO_INCREMENTAL=0 cargo test --profile ci -p vortex-layout -p vortex-scan -p vortex-file -p vortex-edition
CARGO_INCREMENTAL=0 cargo test --profile ci -p vortex-datafusion --lib --all-features
CARGO_INCREMENTAL=0 cargo test --profile ci -p vortex --lib
cargo bench -p vortex-protobuf --bench protobuf
```

The last full run passed: vortex-protobuf 346, vortex-layout 849, vortex-scan 307, vortex-file 235
plus integration tests, vortex-edition 17, vortex-datafusion `--lib` 1,100 with all features,
vortex `--lib` 27. Scoped clippy with `-D warnings`, the pinned nightly fmt, rustdoc with CI's
flags, typos, REUSE, `check-editions` and the CodSpeed count tests also passed.

## Related work

- `ct/ext-types-research`, `research/extension-types/`: the opaque extension type plan. This work
  follows it: field access goes through `ext_storage`, and there is no `ext_wrap`.
- #11431 and #11432: mixed DataFusion and Vortex expression trees.
- #11160: normalize child validity beneath nullable structs.
- #11595: removed `vortex.list` from the unstable edition. #11534 enabled unstable in benchmarks.
- #9205: wrong `is_null` pruning beneath nullable structs.
