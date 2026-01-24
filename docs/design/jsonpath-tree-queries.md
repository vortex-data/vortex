# JSONPath Queries for Vortex Encoding Trees

## Design Document

**Status:** Proposal
**Author:** Claude
**Date:** 2025-01-24

---

## 1. Overview

This document proposes a JSONPath-based query system for exploring and filtering Vortex encoding trees. The goal is to enable users to quickly find arrays matching specific criteria (encoding type, size, dtype, compression ratio, etc.) within complex nested file layouts.

### Goals

- Natural navigation: `$.struct.user_id.chunked` reads like a path
- Powerful filtering: `$..flat[?(@._nbytes > 1000000)]` finds large uncompressed arrays
- Tool compatibility: Works with `jq`, Python `jsonpath-ng`, JavaScript, etc.
- Zero learning curve: JSONPath is widely known from Kubernetes, REST APIs, etc.

### Non-Goals

- Sibling comparisons (use `jq` post-processing)
- Mutations (read-only queries)
- Streaming evaluation (full tree in memory)

---

## 2. JSON Tree Format

### 2.1 Design Principles

| Principle | Implementation |
|-----------|----------------|
| Encoding as key | `{ "flat": { ... } }` not `{ "encoding": "flat" }` |
| Metadata prefix | Underscore: `_rows`, `_nbytes`, `_dtype` |
| Field names as keys | `{ "user_id": { ... } }` for struct fields |
| Chunk indices as keys | `{ "[0]": { ... }, "[1]": { ... } }` for chunks |
| Single root | Tree always has one root encoding node |

### 2.2 Node Structure

Every node in the tree has this shape:

```
{
  "<encoding>": {
    "_encoding": "<full_encoding_name>",
    "_dtype": "<dtype_string>",
    "_rows": <row_count>,
    "_nbytes": <total_bytes>,
    "_nbytes_direct": <direct_bytes>,
    "_metadata_bytes": <metadata_size>,
    "_segments": [<segment_ids>],
    "_depth": <tree_depth>,
    "_path": "<json_path_to_here>",

    // Children as named keys:
    "<field_or_child_name>": { "<child_encoding>": { ... } },
    ...
  }
}
```

### 2.3 Complete Example

```json
{
  "struct": {
    "_encoding": "vortex.struct",
    "_dtype": "{user_id: i64, name: utf8, orders: list<{product: utf8, price: f64}>}",
    "_rows": 1000000,
    "_nbytes": 48576000,
    "_nbytes_direct": 0,
    "_metadata_bytes": 128,
    "_segments": [],
    "_depth": 0,
    "_path": "$.struct",

    "user_id": {
      "chunked": {
        "_encoding": "vortex.chunked",
        "_dtype": "i64",
        "_rows": 1000000,
        "_nbytes": 8000000,
        "_nbytes_direct": 0,
        "_metadata_bytes": 64,
        "_segments": [],
        "_depth": 1,
        "_path": "$.struct.user_id.chunked",

        "[0]": {
          "fastlanes": {
            "_encoding": "vortex.fastlanes",
            "_dtype": "i64",
            "_rows": 500000,
            "_nbytes": 2000000,
            "_nbytes_direct": 2000000,
            "_metadata_bytes": 32,
            "_segments": [0],
            "_depth": 2,
            "_path": "$.struct.user_id.chunked.[0].fastlanes"
          }
        },
        "[1]": {
          "fastlanes": {
            "_encoding": "vortex.fastlanes",
            "_dtype": "i64",
            "_rows": 500000,
            "_nbytes": 2000000,
            "_nbytes_direct": 2000000,
            "_metadata_bytes": 32,
            "_segments": [1],
            "_depth": 2,
            "_path": "$.struct.user_id.chunked.[1].fastlanes"
          }
        }
      }
    },

    "name": {
      "chunked": {
        "_encoding": "vortex.chunked",
        "_dtype": "utf8",
        "_rows": 1000000,
        "_nbytes": 4500000,
        "_nbytes_direct": 0,
        "_metadata_bytes": 64,
        "_segments": [],
        "_depth": 1,
        "_path": "$.struct.name.chunked",

        "[0]": {
          "dict": {
            "_encoding": "vortex.dict",
            "_dtype": "utf8",
            "_rows": 1000000,
            "_nbytes": 4500000,
            "_nbytes_direct": 0,
            "_metadata_bytes": 48,
            "_segments": [],
            "_depth": 2,
            "_path": "$.struct.name.chunked.[0].dict",

            "codes": {
              "bitpacking": {
                "_encoding": "vortex.bitpacking",
                "_dtype": "u32",
                "_rows": 1000000,
                "_nbytes": 500000,
                "_nbytes_direct": 500000,
                "_metadata_bytes": 24,
                "_segments": [2],
                "_depth": 3,
                "_path": "$.struct.name.chunked.[0].dict.codes.bitpacking"
              }
            },
            "values": {
              "flat": {
                "_encoding": "vortex.flat",
                "_dtype": "utf8",
                "_rows": 847,
                "_nbytes": 42350,
                "_nbytes_direct": 42350,
                "_metadata_bytes": 32,
                "_segments": [3],
                "_depth": 3,
                "_path": "$.struct.name.chunked.[0].dict.values.flat"
              }
            }
          }
        }
      }
    },

    "orders": {
      "chunked": {
        "_encoding": "vortex.chunked",
        "_dtype": "list<{product: utf8, price: f64}>",
        "_rows": 1000000,
        "_nbytes": 36000000,
        "_nbytes_direct": 0,
        "_metadata_bytes": 64,
        "_segments": [],
        "_depth": 1,
        "_path": "$.struct.orders.chunked",

        "[0]": {
          "list": {
            "_encoding": "vortex.list",
            "_dtype": "list<{product: utf8, price: f64}>",
            "_rows": 1000000,
            "_nbytes": 36000000,
            "_nbytes_direct": 0,
            "_metadata_bytes": 48,
            "_segments": [],
            "_depth": 2,
            "_path": "$.struct.orders.chunked.[0].list",

            "offsets": {
              "flat": {
                "_encoding": "vortex.flat",
                "_dtype": "u64",
                "_rows": 1000001,
                "_nbytes": 8000008,
                "_nbytes_direct": 8000008,
                "_metadata_bytes": 24,
                "_segments": [4],
                "_depth": 3,
                "_path": "$.struct.orders.chunked.[0].list.offsets.flat"
              }
            },
            "elements": {
              "struct": {
                "_encoding": "vortex.struct",
                "_dtype": "{product: utf8, price: f64}",
                "_rows": 5000000,
                "_nbytes": 28000000,
                "_nbytes_direct": 0,
                "_metadata_bytes": 64,
                "_segments": [],
                "_depth": 3,
                "_path": "$.struct.orders.chunked.[0].list.elements.struct",

                "product": {
                  "fsst": {
                    "_encoding": "vortex.fsst",
                    "_dtype": "utf8",
                    "_rows": 5000000,
                    "_nbytes": 12000000,
                    "_nbytes_direct": 12000000,
                    "_metadata_bytes": 2048,
                    "_segments": [5, 6],
                    "_depth": 4,
                    "_path": "$.struct.orders.chunked.[0].list.elements.struct.product.fsst"
                  }
                },
                "price": {
                  "alp": {
                    "_encoding": "vortex.alp",
                    "_dtype": "f64",
                    "_rows": 5000000,
                    "_nbytes": 8000000,
                    "_nbytes_direct": 8000000,
                    "_metadata_bytes": 64,
                    "_segments": [7],
                    "_depth": 4,
                    "_path": "$.struct.orders.chunked.[0].list.elements.struct.price.alp"
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

---

## 3. JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://vortex.dev/schemas/layout-tree.json",
  "title": "Vortex Layout Tree",
  "description": "JSON representation of a Vortex file's encoding tree for JSONPath queries",

  "$defs": {
    "encoding_node": {
      "type": "object",
      "description": "A node representing a single encoding in the tree",
      "properties": {
        "_encoding": {
          "type": "string",
          "description": "Full encoding name (e.g., 'vortex.flat', 'vortex.dict')",
          "pattern": "^vortex\\.[a-z_]+$"
        },
        "_dtype": {
          "type": "string",
          "description": "Logical data type (e.g., 'i64', 'utf8', '{field: type}')"
        },
        "_rows": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of logical rows in this array"
        },
        "_nbytes": {
          "type": "integer",
          "minimum": 0,
          "description": "Total bytes including all descendants"
        },
        "_nbytes_direct": {
          "type": "integer",
          "minimum": 0,
          "description": "Bytes directly owned by this node (segments)"
        },
        "_metadata_bytes": {
          "type": "integer",
          "minimum": 0,
          "description": "Size of encoding metadata in bytes"
        },
        "_segments": {
          "type": "array",
          "items": { "type": "integer", "minimum": 0 },
          "description": "Segment IDs referenced by this node"
        },
        "_depth": {
          "type": "integer",
          "minimum": 0,
          "description": "Depth in tree (root = 0)"
        },
        "_path": {
          "type": "string",
          "description": "JSONPath to this node from root",
          "pattern": "^\\$.*"
        }
      },
      "required": ["_encoding", "_dtype", "_rows", "_nbytes"],
      "additionalProperties": {
        "$ref": "#/$defs/child_wrapper"
      }
    },

    "child_wrapper": {
      "type": "object",
      "description": "Wrapper containing a single encoding node",
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

## 4. Metadata Properties Reference

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `_encoding` | string | Full encoding identifier | `"vortex.dict"` |
| `_dtype` | string | Logical data type | `"utf8"`, `"i64?"`, `"list<i32>"` |
| `_rows` | int | Logical row count | `1000000` |
| `_nbytes` | int | Total bytes (subtree) | `4500000` |
| `_nbytes_direct` | int | Direct segment bytes | `500000` |
| `_metadata_bytes` | int | Metadata size | `48` |
| `_segments` | int[] | Segment IDs | `[0, 1, 2]` |
| `_depth` | int | Tree depth (root=0) | `3` |
| `_path` | string | JSONPath to node | `"$.struct.name.chunked"` |

### Computed Properties (Future)

| Property | Type | Description |
|----------|------|-------------|
| `_compression_ratio` | float | `uncompressed_size / _nbytes` |
| `_is_leaf` | bool | No children |
| `_child_count` | int | Number of children |
| `_bytes_per_row` | float | `_nbytes / _rows` |

---

## 5. Query Examples

### 5.1 Navigation Queries

```bash
# Get specific field
$.struct.user_id

# Get all chunks of a field
$.struct.name.chunked.*

# Get first chunk only
$.struct.name.chunked.[0]

# Nested field access
$.struct.orders..list.elements.struct.price
```

### 5.2 Recursive Descent (Find Anywhere)

```bash
# All flat encodings anywhere in tree
$..flat

# All dict encodings
$..dict

# All list encodings
$..list

# All struct encodings (including nested)
$..struct
```

### 5.3 Filter by Encoding Properties

```bash
# Large uncompressed arrays (>1MB)
$..flat[?(@._nbytes > 1000000)]

# Small dictionary value sets (<1000 unique)
$..dict.values.*[?(@._rows < 1000)]

# Nullable columns
$..*[?(@._dtype =~ "\\?$")]

# Integer columns
$..*[?(@._dtype =~ "^(i|u)(8|16|32|64)")]

# String columns (utf8 or binary)
$..*[?(@._dtype =~ "^(utf8|binary)")]

# Deep nodes (depth > 3)
$..*[?(@._depth > 3)]

# Leaf nodes (no children beyond metadata)
$..flat
$..bitpacking
$..fastlanes
$..fsst
$..alp
```

### 5.4 Compression Analysis Queries

```bash
# Uncompressed strings (flat utf8) - compression candidates
$..flat[?(@._dtype == "utf8" || @._dtype == "utf8?")]

# Large flat arrays that could use bitpacking
$..flat[?(@._dtype =~ "^(i|u)" && @._nbytes > 100000)]

# Dict with high cardinality (values nearly as many as rows)
# Needs computed property or post-processing

# Find FSST-encoded columns (already compressed strings)
$..fsst

# Find ALP-encoded columns (compressed floats)
$..alp
```

### 5.5 Structure Analysis Queries

```bash
# Nested lists (list inside list)
$..list..list

# Structs inside lists (repeated nested records)
$..list..struct

# All fields of nested structs in lists
$..list..struct.*

# Chunked layouts (columnar organization)
$..chunked

# Multi-level chunking
$..chunked..chunked
```

### 5.6 Path-Specific Queries

```bash
# Specific column encoding chain
$.struct.user_id..*

# All encodings under "orders" field
$.struct.orders..*

# Price field anywhere in structure
$..*[?(@._path =~ "price")]
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
        --no-color            Disable colored output
```

### 6.2 Usage Examples

```bash
# Default tree view
vx tree layout data.vtx

# Find large flat arrays
vx tree layout data.vtx -m '$..flat[?(@._nbytes > 1000000)]'

# Output as JSON for piping to jq
vx tree layout data.vtx -m '$..dict' -o json

# Show paths only (for scripting)
vx tree layout data.vtx -m '$..flat' -o paths

# Table format with stats
vx tree layout data.vtx -m '$..*' -o table --stats

# Show matches with 2 levels of parent context
vx tree layout data.vtx -m '$..flat' -c 2

# Verbose output with all properties
vx tree layout data.vtx -v
```

### 6.3 Output Formats

#### Tree (default)
```
vortex.struct {user_id: i64, name: utf8} rows=1000000 nbytes=48.5MB
├── user_id: vortex.chunked i64 rows=1000000 nbytes=8.0MB
│   ├── [0]: vortex.fastlanes i64 rows=500000 nbytes=2.0MB [MATCH]
│   └── [1]: vortex.fastlanes i64 rows=500000 nbytes=2.0MB [MATCH]
└── name: vortex.chunked utf8 rows=1000000 nbytes=4.5MB
    └── [0]: vortex.dict utf8 rows=1000000 nbytes=4.5MB
        ├── codes: vortex.bitpacking u32 rows=1000000 nbytes=500KB
        └── values: vortex.flat utf8 rows=847 nbytes=42KB [MATCH]
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
$.struct.user_id.chunked.[0].fastlanes
$.struct.user_id.chunked.[1].fastlanes
$.struct.name.chunked.[0].dict.values.flat
```

#### Table
```
PATH                                          ENCODING    DTYPE  ROWS      NBYTES
$.struct.user_id.chunked.[0].fastlanes       fastlanes   i64    500000    2.0MB
$.struct.user_id.chunked.[1].fastlanes       fastlanes   i64    500000    2.0MB
$.struct.name.chunked.[0].dict.values.flat   flat        utf8   847       42KB

Total: 3 matches, 4.04MB
```

---

## 7. Implementation Plan

### Phase 1: JSON Format
- [ ] Modify `layout_to_json()` in `vortex-tui/src/tree.rs`
- [ ] Add `_nbytes` calculation (sum of segment sizes)
- [ ] Restructure output to use encoding names as keys
- [ ] Add `_path` property generation

### Phase 2: JSONPath Integration
- [ ] Add `serde_json_path` dependency
- [ ] Implement `--match` flag parsing
- [ ] Filter and highlight matching nodes

### Phase 3: Output Formats
- [ ] Implement `--output` flag with tree/json/paths/table
- [ ] Add `--context` for showing ancestors
- [ ] Add `--stats` for aggregations

### Phase 4: Advanced Features
- [ ] Computed properties (`_compression_ratio`, `_is_leaf`)
- [ ] Regex support in filters (`=~`)
- [ ] Multiple match expressions (OR)
- [ ] Exclude expressions (`--exclude`)

---

## 8. Alternatives Considered

### 8.1 XPath
- **Pro:** More powerful (axes, functions)
- **Con:** XML-centric syntax, less familiar
- **Decision:** JSONPath is more natural for JSON output

### 8.2 jq
- **Pro:** Extremely powerful
- **Con:** Steep learning curve, full language
- **Decision:** JSONPath for simple queries, pipe to `jq` for complex ones

### 8.3 Custom DSL
- **Pro:** Optimized for Vortex concepts
- **Con:** Learning curve, maintenance burden
- **Decision:** Leverage existing JSONPath ecosystem

### 8.4 SQL over Tree
- **Pro:** Familiar syntax
- **Con:** Trees don't map well to tables
- **Decision:** JSONPath better for hierarchical queries

---

## 9. References

- [JSONPath Specification (RFC 9535)](https://www.rfc-editor.org/rfc/rfc9535)
- [serde_json_path crate](https://docs.rs/serde_json_path)
- [jq Manual](https://stedolan.github.io/jq/manual/)
- [XPath 3.1 Specification](https://www.w3.org/TR/xpath-31/)

---

## Appendix A: Encoding Types

| Encoding | Description | Typical Children |
|----------|-------------|------------------|
| `flat` | Uncompressed Arrow-compatible | None |
| `struct` | Named fields | Field names |
| `list` | Variable-length lists | `offsets`, `elements` |
| `chunked` | Row-chunked layout | `[0]`, `[1]`, ... |
| `dict` | Dictionary encoding | `codes`, `values` |
| `bitpacking` | Bit-packed integers | None |
| `fastlanes` | FastLanes compression | None |
| `fsst` | String compression | None |
| `alp` | Float compression | None |
| `runend` | Run-end encoding | `ends`, `values` |
| `roaring` | Bitmap encoding | None |
| `constant` | Single repeated value | None |
| `sparse` | Sparse representation | `indices`, `values` |

---

## Appendix B: Common Query Patterns

### "Why is my file so big?"
```bash
vx tree layout f.vtx -m '$..*[?(@._nbytes > 10000000)]' -o table --stats
```

### "What compression is used for timestamps?"
```bash
vx tree layout f.vtx -m '$..*[?(@._path =~ "(time|date|ts|created)")]' -o table
```

### "Find compression opportunities"
```bash
# Large uncompressed strings
vx tree layout f.vtx -m '$..flat[?(@._dtype =~ "utf8" && @._nbytes > 100000)]'

# Large uncompressed integers
vx tree layout f.vtx -m '$..flat[?(@._dtype =~ "^(i|u)" && @._nbytes > 100000)]'
```

### "Validate structure"
```bash
# Find unexpected nesting
vx tree layout f.vtx -m '$..*[?(@._depth > 5)]'

# Find missing compression
vx tree layout f.vtx -m '$..flat[?(@._rows > 10000)]' -o table
```
