<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Vortex Go bindings

Go bindings for the [Vortex](https://github.com/vortex-data/vortex) columnar file
format. They wrap the Vortex C FFI (`vortex-ffi`) through cgo and exchange data
with the rest of the Go ecosystem via the [Apache Arrow C Data
Interface](https://arrow.apache.org/docs/format/CDataInterface.html), using
[`arrow-go`](https://github.com/apache/arrow-go).

The scope mirrors the C++, Java and Python bindings: **reading and writing
Vortex arrays, Arrow conversion, plus building the expression trees used for
projection and predicate pushdown.**

## Building

The bindings link against `libvortex_ffi`. Build it once from the repository
root:

```sh
cargo build --release -p vortex-ffi    # produces target/release/libvortex_ffi.{so,dylib,a}
```

By default cgo finds the library and header at their in-tree locations
(`../../target/{debug,ci,release}` and `../../vortex-ffi/cinclude`, relative to
this package), with an rpath baked in so tests and binaries run without extra
configuration. To point elsewhere — for example when consuming the package
outside this repository — set the usual cgo environment variables:

```sh
export CGO_CFLAGS="-I/path/to/vortex-ffi/cinclude"
export CGO_LDFLAGS="-L/path/to/lib -lvortex_ffi -Wl,-rpath,/path/to/lib"
```

Then, from this directory:

```sh
go build ./...
go test ./...
```

A `Makefile` is provided that builds `vortex-ffi` (release) and runs the Go
tests: `make test`.

## Usage

### Writing Vortex arrays converted from Arrow data

```go
package main

import (
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	vortex "github.com/vortex-data/vortex/lang/go"
)

func main() {
	session := vortex.NewSession()
	defer session.Close()

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)

	rb := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	rb.Field(0).(*array.Int64Builder).AppendValues([]int64{1, 2, 3}, nil)
	rb.Field(1).(*array.StringBuilder).AppendValues([]string{"a", "b", "c"}, nil)
	rec := rb.NewRecordBatch()
	rb.Release()
	defer rec.Release()

	vxArr, err := vortex.FromArrow(rec)
	if err != nil {
		panic(err)
	}
	defer vxArr.Close()

	if err := vortex.Write(session, "people.vortex", vxArr); err != nil {
		panic(err)
	}
}
```

### Reading Arrow data with projection and filter

```go
session := vortex.NewSession()
defer session.Close()

f, err := vortex.Open(session, "people.vortex")
if err != nil {
	panic(err)
}
defer f.Close()

// keep only the "name" column, for rows where id >= 2
lit, _ := vortex.Lit(int64(2))
rdr, err := f.Scan(&vortex.ScanOptions{
	Projection: vortex.Root().Select("name"),
	Filter:     vortex.Column("id").Gte(lit),
})
if err != nil {
	panic(err)
}
defer rdr.Release()

for rdr.Next() {
	batch := rdr.RecordBatch()
	// ... process batch (an arrow.RecordBatch) ...
}
if err := rdr.Err(); err != nil {
	panic(err)
}
```

`File.Scan` returns an `array.RecordReader` (from `arrow-go`) that concatenates
the scan's internal partitions. `f.Scan(nil)` reads everything.

### Expressions

Build expression trees with the package-level constructors and the methods on
`*vortex.Expr`:

| Vortex / SQL          | Go                                            |
|-----------------------|-----------------------------------------------|
| the row itself        | `vortex.Root()`                               |
| `col`                 | `vortex.Column("col")`                        |
| `struct.field`        | `e.GetItem("field")`                          |
| project `{a, b}`      | `vortex.Root().Select("a", "b")`              |
| `a == b`, `a >= b`, … | `a.Eq(b)`, `a.Gte(b)`, …                      |
| `a + b`, `a * b`, …   | `a.Add(b)`, `a.Mul(b)`, …                     |
| `a AND b`, `a OR b`   | `vortex.And(a, b)`, `vortex.Or(a, b)`         |
| `NOT a`, `a IS NULL`  | `a.Not()`, `a.IsNull()`                       |
| `list CONTAINS v`     | `list.ListContains(v)`                        |
| literal               | `vortex.Lit(v)` (bool, ints, floats, string, []byte) |

`ScanOptions.Projection` must produce a struct — use `Root()` or `.Select(...)`
(a projection that reduces to a single scalar column is not supported when
scanning to Arrow). `ScanOptions.Filter` is an ordinary boolean expression over
the columns; filter columns need not be in the projection.

## Memory and threading

* `Session`, `File`, `Array` and the `array.RecordReader` returned by `Scan`
  hold native resources. Call `Close()` (or `Release()` for the reader) when
  done; a finalizer is registered as a backstop. `Expr` values are cleaned up by
  a finalizer only.
* A `Session` may be shared between goroutines, but a given `File`, scan reader,
  or write call is not safe for concurrent use — serialise access, or use one
  per goroutine.

## Notes / limitations

* `vortex-ffi`'s data-source API takes a path, a comma-separated list of paths,
  or a glob. Object-store URLs (`s3://`, `gs://`, …) resolve to whatever
  credentials are present in the environment; per-call credentials are not
  exposed here.
* These bindings deliberately cover only Vortex array writing, Arrow conversion,
  Arrow scanning and expression building. Lower-level Vortex scalar and dtype
  manipulation is available through the C FFI directly
  (`vortex-ffi/cinclude/vortex.h`).
