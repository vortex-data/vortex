// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package vortex_test

import (
	"path/filepath"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	vortex "github.com/vortex-data/vortex/lang/go"
)

// sampleSchema is { a: int64 (non-null), b: string (nullable) }.
func sampleSchema() *arrow.Schema {
	return arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "b", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
}

// sampleRecord builds a 5-row record: a = [1..5], b = ["v1", null, "v3", null, "v5"].
func sampleRecord(t *testing.T) arrow.RecordBatch {
	t.Helper()
	rb := array.NewRecordBuilder(memory.DefaultAllocator, sampleSchema())
	defer rb.Release()
	rb.Field(0).(*array.Int64Builder).AppendValues([]int64{1, 2, 3, 4, 5}, nil)
	rb.Field(1).(*array.StringBuilder).AppendValues(
		[]string{"v1", "", "v3", "", "v5"},
		[]bool{true, false, true, false, true},
	)
	return rb.NewRecordBatch()
}

func writeSample(t *testing.T, session *vortex.Session) string {
	t.Helper()
	rec := sampleRecord(t)
	defer rec.Release()

	arr, err := vortex.FromArrow(rec)
	if err != nil {
		t.Fatalf("FromArrow: %v", err)
	}
	defer arr.Close()

	path := filepath.Join(t.TempDir(), "sample.vortex")
	if err := vortex.Write(session, path, arr); err != nil {
		t.Fatalf("Write: %v", err)
	}
	return path
}

func collect(t *testing.T, r array.RecordReader) []arrow.RecordBatch {
	t.Helper()
	var out []arrow.RecordBatch
	for r.Next() {
		rec := r.RecordBatch()
		rec.Retain()
		out = append(out, rec)
	}
	if err := r.Err(); err != nil {
		t.Fatalf("reader error: %v", err)
	}
	return out
}

func TestRoundTrip(t *testing.T) {
	session := vortex.NewSession()
	defer session.Close()

	path := writeSample(t, session)

	f, err := vortex.Open(session, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	if n, exact, known := f.RowCount(); !known || !exact || n != 5 {
		t.Fatalf("RowCount = (%d, exact=%v, known=%v); want (5, true, true)", n, exact, known)
	}

	schema, err := f.ArrowSchema()
	if err != nil {
		t.Fatalf("ArrowSchema: %v", err)
	}
	if got, want := schema.NumFields(), 2; got != want {
		t.Fatalf("schema fields = %d; want %d", got, want)
	}
	if schema.Field(0).Name != "a" || schema.Field(1).Name != "b" {
		t.Fatalf("schema fields = %v; want [a b]", []string{schema.Field(0).Name, schema.Field(1).Name})
	}

	rdr, err := f.Scan(nil)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	defer rdr.Release()

	batches := collect(t, rdr)
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()

	var total int64
	for _, b := range batches {
		total += b.NumRows()
	}
	if total != 5 {
		t.Fatalf("scanned %d rows; want 5", total)
	}

	// Concatenate columns and check values.
	got := flattenInt64(t, batches, 0)
	if want := []int64{1, 2, 3, 4, 5}; !equalInt64(got, want) {
		t.Fatalf("column a = %v; want %v", got, want)
	}
	gotStr, gotValid := flattenString(t, batches, 1)
	wantStr := []string{"v1", "", "v3", "", "v5"}
	wantValid := []bool{true, false, true, false, true}
	for i := range wantStr {
		if gotValid[i] != wantValid[i] || (gotValid[i] && gotStr[i] != wantStr[i]) {
			t.Fatalf("column b row %d = (%q, valid=%v); want (%q, valid=%v)",
				i, gotStr[i], gotValid[i], wantStr[i], wantValid[i])
		}
	}
}

func TestFromArrowArray(t *testing.T) {
	builder := array.NewInt64Builder(memory.DefaultAllocator)
	defer builder.Release()
	builder.AppendValues([]int64{10, 20, 30}, nil)
	arrowArr := builder.NewArray()
	defer arrowArr.Release()

	arr, err := vortex.FromArrow(arrowArr)
	if err != nil {
		t.Fatalf("FromArrow: %v", err)
	}
	defer arr.Close()

	if got, want := arr.Len(), 3; got != want {
		t.Fatalf("array len = %d; want %d", got, want)
	}
}

func TestScanProjection(t *testing.T) {
	session := vortex.NewSession()
	defer session.Close()
	path := writeSample(t, session)

	f, err := vortex.Open(session, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	rdr, err := f.Scan(&vortex.ScanOptions{Projection: vortex.Root().Select("a")})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	defer rdr.Release()

	if got, want := rdr.Schema().NumFields(), 1; got != want {
		t.Fatalf("projected schema fields = %d; want %d", got, want)
	}
	if rdr.Schema().Field(0).Name != "a" {
		t.Fatalf("projected field = %q; want %q", rdr.Schema().Field(0).Name, "a")
	}

	batches := collect(t, rdr)
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()
	got := flattenInt64(t, batches, 0)
	if want := []int64{1, 2, 3, 4, 5}; !equalInt64(got, want) {
		t.Fatalf("projected column a = %v; want %v", got, want)
	}
}

func TestScanFilter(t *testing.T) {
	session := vortex.NewSession()
	defer session.Close()
	path := writeSample(t, session)

	f, err := vortex.Open(session, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	lit, err := vortex.Lit(int64(3))
	if err != nil {
		t.Fatalf("Lit: %v", err)
	}
	rdr, err := f.Scan(&vortex.ScanOptions{Filter: vortex.Column("a").Gte(lit)})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	defer rdr.Release()

	batches := collect(t, rdr)
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()
	got := flattenInt64(t, batches, 0)
	if want := []int64{3, 4, 5}; !equalInt64(got, want) {
		t.Fatalf("filtered column a = %v; want %v", got, want)
	}
}

func TestScanSelection(t *testing.T) {
	session := vortex.NewSession()
	defer session.Close()
	path := writeSample(t, session)

	f, err := vortex.Open(session, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	rdr, err := f.Scan(&vortex.ScanOptions{
		SelectionIndices: []uint64{0, 2, 4},
		SelectionMode:    vortex.SelectInclude,
	})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	defer rdr.Release()

	batches := collect(t, rdr)
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()
	got := flattenInt64(t, batches, 0)
	if want := []int64{1, 3, 5}; !equalInt64(got, want) {
		t.Fatalf("selected column a = %v; want %v", got, want)
	}
}

func TestScanComplexFilter(t *testing.T) {
	session := vortex.NewSession()
	defer session.Close()
	path := writeSample(t, session)

	f, err := vortex.Open(session, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	// (a >= 2 AND a <= 4) AND NOT (a == 3)  =>  {2, 4}
	lo, _ := vortex.Lit(int64(2))
	hi, _ := vortex.Lit(int64(4))
	three, _ := vortex.Lit(int64(3))
	pred := vortex.And(
		vortex.Column("a").Gte(lo),
		vortex.Column("a").Lte(hi),
		vortex.Column("a").Eq(three).Not(),
	)
	rdr, err := f.Scan(&vortex.ScanOptions{Filter: pred})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	defer rdr.Release()

	batches := collect(t, rdr)
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()
	got := flattenInt64(t, batches, 0)
	if want := []int64{2, 4}; !equalInt64(got, want) {
		t.Fatalf("complex-filter column a = %v; want %v", got, want)
	}
}

func TestScanRowRangeAndLimit(t *testing.T) {
	session := vortex.NewSession()
	defer session.Close()
	path := writeSample(t, session)

	f, err := vortex.Open(session, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	rdr, err := f.Scan(&vortex.ScanOptions{RowRangeBegin: 1, RowRangeEnd: 4, Limit: 2})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	defer rdr.Release()

	batches := collect(t, rdr)
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()
	got := flattenInt64(t, batches, 0)
	if want := []int64{2, 3}; !equalInt64(got, want) {
		t.Fatalf("row a = %v; want %v (rows [1,4) limited to 2)", got, want)
	}
}

func TestExprErrors(t *testing.T) {
	if _, err := vortex.Lit(struct{}{}); err == nil {
		t.Fatalf("Lit(struct{}{}) should error")
	}
	if e := vortex.And(); e != nil {
		t.Fatalf("And() with no operands should return nil")
	}
}

// --- helpers ---

func flattenInt64(t *testing.T, batches []arrow.RecordBatch, col int) []int64 {
	t.Helper()
	var out []int64
	for _, b := range batches {
		a := b.Column(col).(*array.Int64)
		for i := 0; i < a.Len(); i++ {
			out = append(out, a.Value(i))
		}
	}
	return out
}

func flattenString(t *testing.T, batches []arrow.RecordBatch, col int) ([]string, []bool) {
	t.Helper()
	var vals []string
	var valid []bool
	for _, b := range batches {
		a := b.Column(col)
		sa, ok := a.(*array.String)
		if !ok {
			// Vortex may materialise utf8 as a StringView; handle both.
			sv := a.(*array.StringView)
			for i := 0; i < sv.Len(); i++ {
				valid = append(valid, sv.IsValid(i))
				if sv.IsValid(i) {
					vals = append(vals, sv.Value(i))
				} else {
					vals = append(vals, "")
				}
			}
			continue
		}
		for i := 0; i < sa.Len(); i++ {
			valid = append(valid, sa.IsValid(i))
			if sa.IsValid(i) {
				vals = append(vals, sa.Value(i))
			} else {
				vals = append(vals, "")
			}
		}
	}
	return vals, valid
}

func equalInt64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
