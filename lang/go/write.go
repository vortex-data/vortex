// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package vortex

/*
#include <stdlib.h>
#include "vortex.h"
*/
import "C"

import (
	"runtime"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/cdata"
)

// WriteArrow writes every record batch produced by reader to a Vortex file at
// path. The file's schema is taken from reader.Schema(); all batches must share
// that schema.
//
// path is a local filesystem path. The reader is consumed: WriteArrow calls
// Next/RecordBatch on it until exhausted. WriteArrow does not close or release
// reader; callers retain that responsibility.
func WriteArrow(session *Session, path string, reader array.RecordReader) error {
	if session == nil || session.ptr == nil {
		return &Error{Message: "nil session"}
	}
	if reader == nil {
		return &Error{Message: "nil reader"}
	}

	// Derive the Vortex file dtype from the Arrow schema.
	var schemaC C.struct_ArrowSchema
	cdata.ExportArrowSchema(reader.Schema(), (*cdata.CArrowSchema)(unsafe.Pointer(&schemaC)))
	var cerr *C.vx_error
	dtype := C.vx_dtype_from_arrow_schema((*C.FFI_ArrowSchema)(unsafe.Pointer(&schemaC)), &cerr)
	if err := consumeError(cerr); err != nil {
		return err
	}
	if dtype == nil {
		return &Error{Message: "failed to derive dtype from schema"}
	}
	defer C.vx_dtype_free(dtype)

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	sink := C.vx_array_sink_open_file(session.ptr, cpath, dtype, &cerr)
	if err := consumeError(cerr); err != nil {
		return err
	}
	if sink == nil {
		return &Error{Message: "failed to open sink for " + path}
	}
	// On any early return, still close the sink so the writer task is joined.
	closed := false
	closeSink := func() error {
		if closed {
			return nil
		}
		closed = true
		var e *C.vx_error
		C.vx_array_sink_close(sink, &e)
		return consumeError(e)
	}
	defer func() { _ = closeSink() }()

	for reader.Next() {
		if err := pushBatch(sink, reader); err != nil {
			return err
		}
	}
	if err := reader.Err(); err != nil {
		return err
	}
	return closeSink()
}

func pushBatch(sink *C.vx_array_sink, reader array.RecordReader) error {
	rec := reader.RecordBatch()

	var arrayC C.struct_ArrowArray
	var schemaC C.struct_ArrowSchema
	cdata.ExportArrowRecordBatch(rec,
		(*cdata.CArrowArray)(unsafe.Pointer(&arrayC)),
		(*cdata.CArrowSchema)(unsafe.Pointer(&schemaC)))

	var cerr *C.vx_error
	vxArr := C.vx_array_from_arrow(
		(*C.FFI_ArrowArray)(unsafe.Pointer(&arrayC)),
		(*C.FFI_ArrowSchema)(unsafe.Pointer(&schemaC)),
		C.bool(false), &cerr)
	if err := consumeError(cerr); err != nil {
		return err
	}
	if vxArr == nil {
		return &Error{Message: "failed to import record batch"}
	}
	defer C.vx_array_free(vxArr)

	C.vx_array_sink_push(sink, vxArr, &cerr)
	runtime.KeepAlive(reader)
	return consumeError(cerr)
}
