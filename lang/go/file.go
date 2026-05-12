// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package vortex

/*
#include <stdlib.h>
#include "vortex.h"
*/
import "C"

import (
	"errors"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/arrio"
	"github.com/apache/arrow-go/v18/arrow/cdata"
)

// File is a handle to one or more Vortex files opened through a [Session].
//
// The path passed to [Open] may name a single file, a comma-separated list of
// files, or a glob (e.g. "data/*.vortex"). Opening is cheap: only the first
// matched file is read eagerly, to determine the schema; the rest is deferred to
// [File.Scan], which may be called more than once.
//
// Native resources are released by [File.Close] or by a finalizer when the File
// becomes unreachable.
type File struct {
	session *Session
	ds      *C.vx_data_source
}

// Open opens the Vortex file(s) referenced by path through session.
//
// path is a local path, a comma-separated list of paths, or a glob. Object-store
// URLs (s3://, gs://, …) resolve to whatever credentials are available in the
// environment.
func Open(session *Session, path string) (*File, error) {
	if session == nil || session.ptr == nil {
		return nil, &Error{Message: "nil session"}
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	opts := C.vx_data_source_options{paths: cpath}
	var cerr *C.vx_error
	ds := C.vx_data_source_new(session.ptr, &opts, &cerr)
	if err := consumeError(cerr); err != nil {
		return nil, err
	}
	if ds == nil {
		return nil, &Error{Message: "failed to open " + path}
	}
	runtime.KeepAlive(session)
	f := &File{session: session, ds: ds}
	runtime.SetFinalizer(f, (*File).free)
	return f, nil
}

func (f *File) free() {
	if f != nil && f.ds != nil {
		C.vx_data_source_free(f.ds)
		f.ds = nil
	}
}

// Close releases the native data source. Safe to call more than once; further
// use of the File or readers derived from a still-running scan is undefined.
func (f *File) Close() {
	runtime.SetFinalizer(f, nil)
	f.free()
}

// RowCount returns the number of rows in the file(s). exact reports whether the
// count is authoritative; known reports whether any estimate is available.
func (f *File) RowCount() (count int64, exact bool, known bool) {
	var est C.vx_estimate
	C.vx_data_source_get_row_count(f.ds, &est)
	runtime.KeepAlive(f)
	switch est._type {
	case C.VX_ESTIMATE_EXACT:
		return int64(est.estimate), true, true
	case C.VX_ESTIMATE_INEXACT:
		return int64(est.estimate), false, true
	default:
		return 0, false, false
	}
}

// ArrowSchema returns the Arrow schema of the file (and of unprojected scans).
func (f *File) ArrowSchema() (*arrow.Schema, error) {
	dt := C.vx_data_source_dtype(f.ds)
	runtime.KeepAlive(f)
	if dt == nil {
		return nil, &Error{Message: "data source has no dtype"}
	}
	return dtypeToArrowSchema(dt)
}

func dtypeToArrowSchema(dt *C.vx_dtype) (*arrow.Schema, error) {
	var cSchema C.struct_ArrowSchema
	var cerr *C.vx_error
	if C.vx_dtype_to_arrow_schema(dt, (*C.FFI_ArrowSchema)(unsafe.Pointer(&cSchema)), &cerr) != 0 {
		return nil, consumeError(cerr)
	}
	// ImportCArrowSchema takes ownership of cSchema and releases it.
	return cdata.ImportCArrowSchema((*cdata.CArrowSchema)(unsafe.Pointer(&cSchema)))
}

// SelectionMode controls how [ScanOptions.SelectionIndices] are interpreted.
type SelectionMode int

// Selection modes.
const (
	// SelectAll ignores SelectionIndices (the default).
	SelectAll SelectionMode = C.VX_SELECTION_INCLUDE_ALL
	// SelectInclude keeps only rows at the given indices.
	SelectInclude SelectionMode = C.VX_SELECTION_INCLUDE_RANGE
	// SelectExclude drops rows at the given indices.
	SelectExclude SelectionMode = C.VX_SELECTION_EXCLUDE_RANGE
)

// ScanOptions configures a scan. The zero value reads every row and column.
type ScanOptions struct {
	// Projection selects which columns / computed values to return. Nil returns
	// all columns. Only columns referenced by the expression are read.
	Projection *Expr
	// Filter keeps only rows matching the predicate. Nil applies no filter.
	// Filter columns need not appear in the projection.
	Filter *Expr
	// RowRangeBegin / RowRangeEnd restrict the scan to the half-open row range
	// [RowRangeBegin, RowRangeEnd). Leaving both zero means no row-range limit.
	RowRangeBegin uint64
	RowRangeEnd   uint64
	// SelectionIndices, together with SelectionMode, includes or excludes rows
	// by index (applied after RowRange). Indices must be sorted ascending and
	// non-null. Nil means no index selection.
	SelectionIndices []uint64
	SelectionMode    SelectionMode
	// Limit caps the number of rows returned (after filtering). Zero means no
	// limit.
	Limit uint64
	// Ordered, when true, preserves the original row order across partitions.
	Ordered bool
}

// Scan runs a scan over the file and returns the result as an Arrow record
// reader. opts may be nil to read everything.
//
// The returned reader concatenates the scan's internal partitions; call Release
// on it when finished (or rely on the finalizer). The reader keeps the File and
// Session alive for as long as it is in use.
func (f *File) Scan(opts *ScanOptions) (array.RecordReader, error) {
	if f == nil || f.ds == nil {
		return nil, &Error{Message: "scan on closed file"}
	}

	var copts C.vx_scan_options
	var cidx unsafe.Pointer
	if opts != nil {
		copts.projection = exprPtr(opts.Projection)
		copts.filter = exprPtr(opts.Filter)
		copts.row_range_begin = C.uint64_t(opts.RowRangeBegin)
		copts.row_range_end = C.uint64_t(opts.RowRangeEnd)
		copts.limit = C.uint64_t(opts.Limit)
		copts.ordered = C.bool(opts.Ordered)
		if n := len(opts.SelectionIndices); n > 0 && opts.SelectionMode != SelectAll {
			// vx_data_source_scan copies the indices, so a short-lived C buffer
			// suffices (and keeps Go pointers out of the struct we pass to C).
			cidx = C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.uint64_t(0))))
			dst := unsafe.Slice((*C.uint64_t)(cidx), n)
			for i, v := range opts.SelectionIndices {
				dst[i] = C.uint64_t(v)
			}
			copts.selection.idx = (*C.uint64_t)(cidx)
			copts.selection.idx_len = C.size_t(n)
			copts.selection.include = C.vx_scan_selection_include(opts.SelectionMode)
		}
	}
	if cidx != nil {
		defer C.free(cidx)
	}

	var cerr *C.vx_error
	scan := C.vx_data_source_scan(f.ds, &copts, nil, &cerr)
	runtime.KeepAlive(f)
	runtime.KeepAlive(opts) // keeps the projection/filter Exprs alive past the call
	if err := consumeError(cerr); err != nil {
		return nil, err
	}
	if scan == nil {
		return nil, &Error{Message: "failed to start scan"}
	}

	// Output schema must be fetched before the first partition is pulled.
	dt := C.vx_scan_dtype(scan, &cerr)
	if err := consumeError(cerr); err != nil {
		C.vx_scan_free(scan)
		return nil, err
	}
	schema, err := dtypeToArrowSchema(dt)
	if err != nil {
		C.vx_scan_free(scan)
		return nil, err
	}

	r := &scanReader{schema: schema, file: f, scan: scan}
	r.refCount.Store(1)
	runtime.SetFinalizer(r, (*scanReader).finalize)
	return r, nil
}

// scanReader implements array.RecordReader over the partitions of a vx_scan.
type scanReader struct {
	refCount    atomic.Int64
	schema      *arrow.Schema
	file        *File        // keeps the data source (and via it the session) alive
	scan        *C.vx_scan   // owned; freed when refCount reaches zero
	cur         arrio.Reader // current partition reader, or nil
	rec         arrow.RecordBatch
	err         error
	done        bool // no more partitions / fatal error
	cleanupOnce sync.Once
}

var _ array.RecordReader = (*scanReader)(nil)

func (r *scanReader) Retain() { r.refCount.Add(1) }

func (r *scanReader) Release() {
	if r.refCount.Add(-1) != 0 {
		return
	}
	r.cleanup()
}

func (r *scanReader) finalize() { r.cleanup() }

func (r *scanReader) cleanup() {
	r.cleanupOnce.Do(func() {
		if r.rec != nil {
			r.rec.Release()
			r.rec = nil
		}
		r.cur = nil // its finalizer releases the underlying C stream
		if r.scan != nil {
			C.vx_scan_free(r.scan)
			r.scan = nil
		}
		runtime.KeepAlive(r.file)
	})
}

func (r *scanReader) Schema() *arrow.Schema          { return r.schema }
func (r *scanReader) Err() error                     { return r.err }
func (r *scanReader) Record() arrow.RecordBatch      { return r.rec }
func (r *scanReader) RecordBatch() arrow.RecordBatch { return r.rec }

func (r *scanReader) Next() bool {
	if r.done {
		return false
	}
	if r.rec != nil {
		r.rec.Release()
		r.rec = nil
	}
	for {
		if r.cur == nil {
			rdr, ok, err := r.nextPartition()
			if err != nil {
				r.err = err
				r.done = true
				return false
			}
			if !ok {
				r.done = true
				return false
			}
			r.cur = rdr
		}
		rec, err := r.cur.Read()
		if errors.Is(err, io.EOF) {
			r.cur = nil
			continue
		}
		if err != nil {
			r.err = err
			r.done = true
			return false
		}
		rec.Retain()
		r.rec = rec
		return true
	}
}

// nextPartition advances to the next scan partition, returning a reader over its
// record batches. ok is false when the scan is exhausted.
func (r *scanReader) nextPartition() (rdr arrio.Reader, ok bool, err error) {
	var cerr *C.vx_error
	part := C.vx_scan_next_partition(r.scan, &cerr)
	if e := consumeError(cerr); e != nil {
		return nil, false, e
	}
	if part == nil {
		return nil, false, nil // exhausted
	}
	// vx_partition_scan_arrow consumes the partition (even on error) and writes
	// an owned Arrow array stream into the supplied struct. We allocate the
	// struct on the Go heap and hand it (and ownership of its release callback)
	// to the imported reader, which releases it on finalization.
	stream := new(C.struct_ArrowArrayStream)
	if rc := C.vx_partition_scan_arrow(
		r.file.session.ptr, part, (*C.FFI_ArrowArrayStream)(unsafe.Pointer(stream)), &cerr,
	); rc != 0 {
		return nil, false, consumeError(cerr)
	}
	runtime.KeepAlive(r.file)
	return cdata.ImportCArrayStream((*cdata.CArrowArrayStream)(unsafe.Pointer(stream)), nil), true, nil
}
