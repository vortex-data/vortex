# JSONPath Queries for Vortex Encoding Trees

## Design Document

**Status:** Proposal
**Date:** 2025-01-24

---

## 1. Overview

This document proposes a JSONPath-based query system for exploring and filtering Vortex encoding trees. The goal is to enable users to quickly find arrays matching specific criteria (encoding type, size, dtype, compression ratio, etc.) within complex nested file layouts.

### Goals

- Natural navigation: `$.struct._.user_id.chunked` reads like a path
- Powerful filtering: `$..flat[?(@.nbytes > 1000000)]` finds large uncompressed arrays
- Tool compatibility: Works with `jq`, Python `jsonpath-ng`, JavaScript, etc.
- Zero ambiguity: Field names cannot conflict with encoding names

### Non-Goals

- Sibling comparisons (use `jq` post-processing)
- Mutations (read-only queries)
- Streaming evaluation (full tree in memory)

---

## 2. Design Decisions

### 2.1 Encoding Names

| Decision | Choice |
|----------|--------|
| Separator | Underscore `_` (not dot) |
| Default mode | Short names: `struct`, `flat`, `dict` |
| Strict mode | Full names: `vortex_struct`, `vortex_flat` |

**Rationale:** Dots conflict with JSONPath navigation. Short names are cleaner for queries; full names available via `--strict` flag.

| Full Name | Short (default) | Strict Mode |
|-----------|-----------------|-------------|
| `vortex.struct` | `struct` | `vortex_struct` |
| `vortex.flat` | `flat` | `vortex_flat` |
| `vortex.dict` | `dict` | `vortex_dict` |
| `vortex.chunked` | `chunked` | `vortex_chunked` |
| `vortex.list` | `list` | `vortex_list` |
| `vortex.bitpacking` | `bitpacking` | `vortex_bitpacking` |
| `vortex.fsst` | `fsst` | `vortex_fsst` |
| `vortex.alp` | `alp` | `vortex_alp` |

### 2.2 Structure Rules

| Rule | Description |
|------|-------------|
| **Encoding = key** | Encoding name is the JSON key itself (no separate `enc` property) |
| **`_` for struct only** | Struct fields go under `_` to avoid name conflicts |
| **Fixed children direct** | `dict.codes`, `list.offsets`, `chunked.[0]` need no wrapper |
| **Metadata as properties** | `dtype`, `rows`, `nbytes` are sibling properties |

### 2.3 Why `_` Only for Struct?

| Encoding | Children | Conflict possible? |
|----------|----------|-------------------|
| `struct` | User-defined field names | ✅ Yes - field could be named `flat` |
| `chunked` | `[0]`, `[1]`, `[2]`... | ❌ No - brackets not encoding names |
| `dict` | `codes`, `values` | ❌ No - fixed names |
| `list` | `offsets`, `elements` | ❌ No - fixed names |
| `runend` | `ends`, `values` | ❌ No - fixed names |
| `sparse` | `indices`, `values` | ❌ No - fixed names |

Only `struct` has user-defined child names that could collide with encoding names.

---

## 3. JSON Tree Format

### 3.1 Node Structure

```
{
  "<encoding>": {
    "dtype": "<dtype_string>",
    "rows": <row_count>,
    "nbytes": <total_bytes>,
    "segments": [<segment_ids>],      // optional

    // Children (structure depends on encoding):
    "_": { ... },                      // struct fields
    "[0]": { ... },                    // chunked chunks
    "codes": { ... },                  // dict codes
    "values": { ... },                 // dict values
    ...
  }
}
```

### 3.2 Complete Example

```json
{
  "struct": {
    "dtype": "{user_id: i64, name: utf8, orders: list<{product: utf8, price: f64}>}",
    "rows": 100000,
    "nbytes": 15000000,

    "_": {
      "user_id": {
        "chunked": {
          "dtype": "i64",
          "rows": 100000,
          "nbytes": 800000,

          "[0]": {
            "bitpacking": {
              "dtype": "i64",
              "rows": 50000,
              "nbytes": 400000,
              "segments": [0]
            }
          },
          "[1]": {
            "bitpacking": {
              "dtype": "i64",
              "rows": 50000,
              "nbytes": 400000,
              "segments": [1]
            }
          }
        }
      },

      "name": {
        "dict": {
          "dtype": "utf8",
          "rows": 100000,
          "nbytes": 200000,

          "codes": {
            "bitpacking": {
              "dtype": "u16",
              "rows": 100000,
              "nbytes": 50000,
              "segments": [2]
            }
          },
          "values": {
            "fsst": {
              "dtype": "utf8",
              "rows": 847,
              "nbytes": 25000,
              "segments": [3]
            }
          }
        }
      },

      "orders": {
        "list": {
          "dtype": "list<{product: utf8, price: f64}>",
          "rows": 100000,
          "nbytes": 14000000,

          "offsets": {
            "flat": {
              "dtype": "u64",
              "rows": 100001,
              "nbytes": 800008,
              "segments": [4]
            }
          },
          "elements": {
            "struct": {
              "dtype": "{product: utf8, price: f64}",
              "rows": 500000,
              "nbytes": 13200000,

              "_": {
                "product": {
                  "fsst": {
                    "dtype": "utf8",
                    "rows": 500000,
                    "nbytes": 9200000,
                    "segments": [5, 6]
                  }
                },
                "price": {
                  "alp": {
                    "dtype": "f64",
                    "rows": 500000,
                    "nbytes": 4000000,
                    "segments": [7]
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### 3.3 Field Name Conflicts

With `_` wrapper, field names cannot conflict with encoding names:

```json
{
  "struct": {
    "_": {
      "user_id": { "chunked": { ... } },
      "flat": { "dict": { ... } },
      "dict": { "flat": { ... } }
    }
  }
}
```

- `$.struct._.flat` → field named "flat"
- `$..flat` → all flat encodings (keys named "flat")

No ambiguity!

---

## 4. Metadata Properties

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `dtype` | string | Logical data type | `"i64"`, `"utf8?"`, `"list<i32>"` |
| `rows` | int | Logical row count | `100000` |
| `nbytes` | int | Total bytes (subtree) | `4500000` |
| `segments` | int[] | Segment IDs | `[0, 1, 2]` |

### Future Properties

| Property | Type | Description |
|----------|------|-------------|
| `nbytes_direct` | int | Bytes owned directly (not children) |
| `compression_ratio` | float | `uncompressed_size / nbytes` |
| `depth` | int | Tree depth (root = 0) |
| `path` | string | JSONPath to this node |

---

## 5. JSONPath Queries

### 5.1 Navigation

```bash
# Field access (struct fields under _)
$.struct._.user_id
$.struct._.name
$.struct._.flat                    # field named "flat" - no conflict!

# Encoding chain
$.struct._.user_id.chunked.[0].bitpacking
$.struct._.name.dict.values.fsst

# Fixed children (no _ needed)
$.struct._.name.dict.codes
$.struct._.name.dict.values
$.struct._.orders.list.offsets
$.struct._.orders.list.elements

# Nested struct
$.struct._.orders.list.elements.struct._.price
```

### 5.2 Recursive Descent

```bash
# All encodings of a type
$..flat                            # all flat encodings
$..dict                            # all dict encodings
$..struct                          # all structs (including nested)
$..bitpacking                      # all bitpacked arrays

# All struct fields
$..struct._.*

# All chunks
$..chunked.*
```

### 5.3 Filters

```bash
# By size
$..flat[?(@.nbytes > 1000000)]           # flat arrays > 1MB
$..*[?(@.nbytes > 10000000)]             # any node > 10MB

# By dtype
$..*[?(@.dtype == "utf8")]               # string columns
$..*[?(@.dtype == "utf8?")]              # nullable strings
$..*[?(@.dtype =~ "^(i|u)(8|16|32|64)")] # integer columns

# By rows
$..*[?(@.rows > 100000)]                 # large arrays
$..dict[?(@.values.rows < 1000)]         # dicts with few unique values

# Combined
$..flat[?(@.dtype == "utf8" && @.nbytes > 100000)]  # large uncompressed strings
```

### 5.4 Common Patterns

```bash
# "Why is my file so big?"
$..*[?(@.nbytes > 10000000)]

# "What compression is used for timestamps?"
$.struct._.*[?(@.dtype =~ "timestamp")]

# "Find compression opportunities" (large flat arrays)
$..flat[?(@.rows > 100000)]

# "Find nested structs in lists"
$..list.elements.struct

# "Show all leaf encodings"
$..flat
$..bitpacking
$..fsst
$..alp
```

---

## 6. CLI Interface

### 6.1 Command Structure

```bash
vx tree layout <file> [OPTIONS]

OPTIONS:
    -m, --match <JSONPATH>    Filter nodes matching JSONPath expression
    -o, --output <FORMAT>     Output format: tree (default), json, paths, table
    -v, --verbose             Include all metadata properties
    -c, --context <N>         Show N ancestor levels for matches
    -s, --stats               Show aggregate statistics for matches
        --strict              Use full encoding names (vortex_struct)
        --no-color            Disable colored output
```

### 6.2 Usage Examples

```bash
# Default tree view
vx tree layout data.vtx

# Find large flat arrays
vx tree layout data.vtx -m '$..flat[?(@.nbytes > 1000000)]'

# Output as JSON for piping to jq
vx tree layout data.vtx -m '$..dict' -o json

# Show paths only (for scripting)
vx tree layout data.vtx -m '$..flat' -o paths

# Table format with stats
vx tree layout data.vtx -m '$..*' -o table --stats

# Show matches with parent context
vx tree layout data.vtx -m '$..flat' -c 2

# Strict mode with full encoding names
vx tree layout data.vtx --strict -m '$..vortex_flat'
```

### 6.3 Output Formats

#### Tree (default)
```
struct {user_id: i64, name: utf8} rows=100000 nbytes=15.0MB
├── user_id: chunked i64 rows=100000 nbytes=800KB
│   ├── [0]: bitpacking i64 rows=50000 nbytes=400KB
│   └── [1]: bitpacking i64 rows=50000 nbytes=400KB
└── name: dict utf8 rows=100000 nbytes=200KB
    ├── codes: bitpacking u16 rows=100000 nbytes=50KB
    └── values: fsst utf8 rows=847 nbytes=25KB [MATCH]
```

#### JSON
```json
{
  "matches": [...],
  "count": 3,
  "total_nbytes": 4542350
}
```

#### Paths
```
$.struct._.user_id.chunked.[0].bitpacking
$.struct._.user_id.chunked.[1].bitpacking
$.struct._.name.dict.values.fsst
```

#### Table
```
PATH                                      ENCODING     DTYPE  ROWS    NBYTES
$.struct._.user_id.chunked.[0]           bitpacking   i64    50000   400KB
$.struct._.user_id.chunked.[1]           bitpacking   i64    50000   400KB
$.struct._.name.dict.values              fsst         utf8   847     25KB

Total: 3 matches, 825KB
```

---

## 7. JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://vortex.dev/schemas/layout-tree-v1.json",
  "title": "Vortex Layout Tree",
  "description": "JSON representation of a Vortex encoding tree for JSONPath queries",

  "$defs": {
    "metadata": {
      "type": "object",
      "properties": {
        "dtype": {
          "type": "string",
          "description": "Logical data type"
        },
        "rows": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of logical rows"
        },
        "nbytes": {
          "type": "integer",
          "minimum": 0,
          "description": "Total bytes including descendants"
        },
        "segments": {
          "type": "array",
          "items": { "type": "integer", "minimum": 0 },
          "description": "Segment IDs"
        }
      },
      "required": ["dtype", "rows", "nbytes"]
    },

    "encoding_node": {
      "allOf": [
        { "$ref": "#/$defs/metadata" },
        {
          "type": "object",
          "additionalProperties": {
            "oneOf": [
              { "$ref": "#/$defs/encoding_wrapper" },
              { "$ref": "#/$defs/struct_fields" },
              { "type": ["string", "number", "array"] }
            ]
          }
        }
      ]
    },

    "struct_fields": {
      "type": "object",
      "description": "Struct fields under _ key",
      "additionalProperties": {
        "$ref": "#/$defs/encoding_wrapper"
      }
    },

    "encoding_wrapper": {
      "type": "object",
      "minProperties": 1,
      "maxProperties": 1,
      "additionalProperties": {
        "$ref": "#/$defs/encoding_node"
      }
    }
  },

  "type": "object",
  "minProperties": 1,
  "maxProperties": 1,
  "additionalProperties": {
    "$ref": "#/$defs/encoding_node"
  }
}
```

---

## 8. Encoding Reference

### 8.1 Encoding Types

| Encoding | Description | Children |
|----------|-------------|----------|
| `struct` | Named fields | `_`: field names |
| `chunked` | Row chunks | `[0]`, `[1]`, ... |
| `list` | Variable-length lists | `offsets`, `elements` |
| `dict` | Dictionary encoding | `codes`, `values` |
| `runend` | Run-end encoding | `ends`, `values` |
| `sparse` | Sparse array | `indices`, `values` |
| `flat` | Uncompressed | None (leaf) |
| `bitpacking` | Bit-packed integers | None (leaf) |
| `fastlanes` | FastLanes compression | None (leaf) |
| `fsst` | String compression | None (leaf) |
| `alp` | Float compression | None (leaf) |
| `roaring` | Bitmap compression | None (leaf) |
| `constant` | Single value | None (leaf) |

### 8.2 Data Types

| Category | Types |
|----------|-------|
| Integers | `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` |
| Floats | `f32`, `f64` |
| Boolean | `bool` |
| Strings | `utf8`, `binary` |
| Nullable | Suffix with `?`: `i64?`, `utf8?` |
| List | `list<element_type>` |
| Struct | `{field: type, ...}` |

---

## 9. Comparison to Alternatives

### 9.1 Query Languages

| Feature | JSONPath | XPath | jq |
|---------|----------|-------|-----|
| Recursive descent | `$..` | `//` | `..` |
| Filters | `[?(@.x)]` | `[@x]` | `select(.x)` |
| Regex | `=~` (some) | `matches()` | `test()` |
| Learning curve | Low | Medium | High |

### 9.2 Why JSONPath?

- **Familiar**: Used in Kubernetes, REST APIs, etc.
- **JSON native**: Output is already JSON
- **Tool ecosystem**: `jq`, Python libraries, Rust crates
- **Simple**: Covers 90% of use cases

---

## 10. Implementation Plan

### Phase 1: JSON Format
- [ ] Implement `layout_to_json()` with new structure
- [ ] Add `nbytes` calculation (sum of segment sizes)
- [ ] Add `_` wrapper for struct fields
- [ ] Add short encoding names (default)

### Phase 2: JSONPath Integration
- [ ] Add `serde_json_path` dependency
- [ ] Implement `--match` flag
- [ ] Filter and highlight matching nodes

### Phase 3: Output Formats
- [ ] `--output tree` (default, with highlighting)
- [ ] `--output json` (matches as JSON)
- [ ] `--output paths` (JSONPath strings)
- [ ] `--output table` (tabular format)

### Phase 4: Enhancements
- [ ] `--strict` flag for full encoding names
- [ ] `--context N` for ancestor display
- [ ] `--stats` for aggregations
- [ ] Per-namespace prefix configuration

---

## 11. References

- [JSONPath Specification (RFC 9535)](https://www.rfc-editor.org/rfc/rfc9535)
- [serde_json_path crate](https://docs.rs/serde_json_path)
- [jq Manual](https://stedolan.github.io/jq/manual/)
