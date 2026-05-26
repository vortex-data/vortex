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

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/cdata"
)

// Write writes one or more Vortex arrays to a Vortex file at path.
//
// The file dtype is taken from the first array; all subsequent arrays must have
// the same dtype. path is a local filesystem path. Write does not close arrays;
// callers retain that responsibility.
func Write(session *Session, path string, arrays ...*Array) error {
	if session == nil || session.ptr == nil {
		return &Error{Message: "nil session"}
	}
	if len(arrays) == 0 {
		return &Error{Message: "no arrays to write"}
	}
	for _, arr := range arrays {
		if arr == nil || arr.ptr == nil {
			return &Error{Message: "nil array"}
		}
	}

	dtype := C.vx_array_dtype(arrays[0].ptr)
	sink, closeSink, err := openArraySink(session, path, dtype)
	runtime.KeepAlive(arrays[0])
	if err != nil {
		return err
	}
	defer func() { _ = closeSink() }()

	for _, arr := range arrays {
		if err := pushArray(sink, arr); err != nil {
			return err
		}
	}
	return closeSink()
}

// WriteArrow converts each Arrow record batch produced by reader into a Vortex
// array and writes those arrays to path.
//
// The file's schema is taken from reader.Schema(); all batches must share that
// schema. WriteArrow does not close or release reader; callers retain that
// responsibility.
//
// Deprecated: use [FromArrow] and [Write] so the write path receives Vortex
// arrays explicitly.
func WriteArrow(session *Session, path string, reader array.RecordReader) error {
	if session == nil || session.ptr == nil {
		return &Error{Message: "nil session"}
	}
	if reader == nil {
		return &Error{Message: "nil reader"}
	}

	dtype, err := dtypeFromArrowSchema(reader.Schema())
	if err != nil {
		return err
	}
	defer C.vx_dtype_free(dtype)

	sink, closeSink, err := openArraySink(session, path, dtype)
	if err != nil {
		return err
	}
	defer func() { _ = closeSink() }()

	for reader.Next() {
		arr, err := FromArrow(reader.RecordBatch())
		if err != nil {
			return err
		}
		if err := pushArray(sink, arr); err != nil {
			arr.Close()
			return err
		}
		arr.Close()
	}
	if err := reader.Err(); err != nil {
		return err
	}
	return closeSink()
}

func dtypeFromArrowSchema(schema *arrow.Schema) (*C.vx_dtype, error) {
	var schemaC C.struct_ArrowSchema
	cdata.ExportArrowSchema(schema, (*cdata.CArrowSchema)(unsafe.Pointer(&schemaC)))
	var cerr *C.vx_error
	dtype := C.vx_dtype_from_arrow_schema((*C.FFI_ArrowSchema)(unsafe.Pointer(&schemaC)), &cerr)
	if err := consumeError(cerr); err != nil {
		return nil, err
	}
	if dtype == nil {
		return nil, &Error{Message: "failed to derive dtype from schema"}
	}
	return dtype, nil
}

func openArraySink(session *Session, path string, dtype *C.vx_dtype) (*C.vx_array_sink, func() error, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var cerr *C.vx_error
	sink := C.vx_array_sink_open_file(session.ptr, cpath, dtype, &cerr)
	if err := consumeError(cerr); err != nil {
		return nil, nil, err
	}
	if sink == nil {
		return nil, nil, &Error{Message: "failed to open sink for " + path}
	}

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
	runtime.KeepAlive(session)
	return sink, closeSink, nil
}

func pushArray(sink *C.vx_array_sink, arr *Array) error {
	var cerr *C.vx_error
	C.vx_array_sink_push(sink, arr.ptr, &cerr)
	runtime.KeepAlive(arr)
	return consumeError(cerr)
}
