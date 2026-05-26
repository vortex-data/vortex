// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package vortex_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	vortex "github.com/vortex-data/vortex/lang/go"
)

// Example writes a small Arrow table to a Vortex file and reads it back with a
// projection and a predicate.
func Example() {
	session := vortex.NewSession()
	defer session.Close()

	dir, _ := os.MkdirTemp("", "vortex-example")
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "people.vortex")

	// Build a 3-row Arrow record: { id: int64, name: utf8 }.
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "name", Type: arrow.BinaryTypes.String},
	}, nil)
	rb := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	rb.Field(0).(*array.Int64Builder).AppendValues([]int64{1, 2, 3}, nil)
	rb.Field(1).(*array.StringBuilder).AppendValues([]string{"ann", "bob", "cleo"}, nil)
	rec := rb.NewRecordBatch()
	rb.Release()
	defer rec.Release()

	vxArr, err := vortex.FromArrow(rec)
	if err != nil {
		panic(err)
	}
	defer vxArr.Close()

	if err := vortex.Write(session, path, vxArr); err != nil {
		panic(err)
	}

	// Read back the "name" column for rows with id >= 2.
	f, err := vortex.Open(session, path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

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
		names := batch.Column(0)
		for i := 0; i < int(batch.NumRows()); i++ {
			fmt.Println(names.ValueStr(i))
		}
	}
	if err := rdr.Err(); err != nil {
		panic(err)
	}
	// Output:
	// bob
	// cleo
}
