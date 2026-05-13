// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package vortex

/*
#include <stdlib.h>
#include "vortex.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/cdata"
)

// Array is an owned native Vortex array.
//
// Create one with [FromArrow]. Native resources are released by [Array.Close]
// or, as a backstop, by a finalizer when the Array becomes unreachable.
type Array struct {
	ptr *C.vx_array
}

func newArray(ptr *C.vx_array) *Array {
	if ptr == nil {
		return nil
	}
	arr := &Array{ptr: ptr}
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (a *Array) free() {
	if a != nil && a.ptr != nil {
		C.vx_array_free(a.ptr)
		a.ptr = nil
	}
}

// Close releases the native Vortex array. It is safe to call more than once.
func (a *Array) Close() {
	runtime.SetFinalizer(a, nil)
	a.free()
}

// Len returns the number of rows/elements in the array.
func (a *Array) Len() int {
	if a == nil || a.ptr == nil {
		return 0
	}
	n := C.vx_array_len(a.ptr)
	runtime.KeepAlive(a)
	return int(n)
}

// FromArrow converts an Arrow value into an owned Vortex array.
//
// Supported inputs are [arrow.RecordBatch] and [arrow.Array]. Record batches are
// imported as non-nullable struct arrays whose fields match the Arrow schema.
func FromArrow(value any) (*Array, error) {
	switch v := value.(type) {
	case arrow.RecordBatch:
		return arrayFromArrowRecordBatch(v)
	case arrow.Array:
		return arrayFromArrowArray(v)
	default:
		return nil, fmt.Errorf("vortex: unsupported Arrow value %T", value)
	}
}

func arrayFromArrowRecordBatch(rec arrow.RecordBatch) (*Array, error) {
	if rec == nil {
		return nil, &Error{Message: "nil Arrow record batch"}
	}

	var arrayC C.struct_ArrowArray
	var schemaC C.struct_ArrowSchema
	cdata.ExportArrowRecordBatch(rec,
		(*cdata.CArrowArray)(unsafe.Pointer(&arrayC)),
		(*cdata.CArrowSchema)(unsafe.Pointer(&schemaC)))

	arr, err := importArrowArray(&arrayC, &schemaC, false)
	runtime.KeepAlive(rec)
	return arr, err
}

func arrayFromArrowArray(arr arrow.Array) (*Array, error) {
	if arr == nil {
		return nil, &Error{Message: "nil Arrow array"}
	}

	var arrayC C.struct_ArrowArray
	var schemaC C.struct_ArrowSchema
	cdata.ExportArrowArray(arr,
		(*cdata.CArrowArray)(unsafe.Pointer(&arrayC)),
		(*cdata.CArrowSchema)(unsafe.Pointer(&schemaC)))

	vxArr, err := importArrowArray(&arrayC, &schemaC, arr.NullN() > 0)
	runtime.KeepAlive(arr)
	return vxArr, err
}

func importArrowArray(arrayC *C.struct_ArrowArray, schemaC *C.struct_ArrowSchema, nullable bool) (*Array, error) {
	var cerr *C.vx_error
	ptr := C.vx_array_from_arrow(
		(*C.FFI_ArrowArray)(unsafe.Pointer(arrayC)),
		(*C.FFI_ArrowSchema)(unsafe.Pointer(schemaC)),
		C.bool(nullable), &cerr)
	if err := consumeError(cerr); err != nil {
		return nil, err
	}
	if ptr == nil {
		return nil, &Error{Message: "failed to import Arrow data"}
	}
	return newArray(ptr), nil
}
