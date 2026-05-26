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
)

// BinaryOp identifies a binary operator used by [Binary] and the comparison /
// arithmetic methods on [Expr].
type BinaryOp int

// Binary operators. The comparison operators short-circuit to NULL when either
// side is NULL; And and Or follow Kleene (three-valued) logic.
const (
	OpEq    BinaryOp = C.VX_OPERATOR_EQ
	OpNotEq BinaryOp = C.VX_OPERATOR_NOT_EQ
	OpGt    BinaryOp = C.VX_OPERATOR_GT
	OpGte   BinaryOp = C.VX_OPERATOR_GTE
	OpLt    BinaryOp = C.VX_OPERATOR_LT
	OpLte   BinaryOp = C.VX_OPERATOR_LTE
	OpAnd   BinaryOp = C.VX_OPERATOR_KLEENE_AND
	OpOr    BinaryOp = C.VX_OPERATOR_KLEENE_OR
	OpAdd   BinaryOp = C.VX_OPERATOR_ADD
	OpSub   BinaryOp = C.VX_OPERATOR_SUB
	OpMul   BinaryOp = C.VX_OPERATOR_MUL
	OpDiv   BinaryOp = C.VX_OPERATOR_DIV
)

// Expr is a node in a Vortex expression tree. Expressions are used as scan
// projections (which columns / computed values to return) and predicates (which
// rows to keep).
//
// Build expressions with the package-level constructors ([Root], [Column],
// [Lit], [And], …) and the methods on Expr. Building a derived expression copies
// its inputs on the native side, so an Expr may be reused freely. Native memory
// is released by a finalizer when the Expr becomes unreachable.
type Expr struct {
	ptr *C.vx_expression
}

func newExpr(p *C.vx_expression) *Expr {
	if p == nil {
		return nil
	}
	e := &Expr{ptr: p}
	runtime.SetFinalizer(e, (*Expr).free)
	return e
}

func (e *Expr) free() {
	if e != nil && e.ptr != nil {
		C.vx_expression_free(e.ptr)
		e.ptr = nil
	}
}

func exprPtr(e *Expr) *C.vx_expression {
	if e == nil {
		return nil
	}
	return e.ptr
}

// Root returns the identity expression: applied to a row it yields the row
// itself. It is the starting point for column access, e.g. Root().GetItem("a").
func Root() *Expr { return newExpr(C.vx_expression_root()) }

// Column is shorthand for Root().GetItem(name).
func Column(name string) *Expr { return Root().GetItem(name) }

// GetItem accesses the named field of a struct-typed expression.
func (e *Expr) GetItem(name string) *Expr {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	out := newExpr(C.vx_expression_get_item(cname, exprPtr(e)))
	runtime.KeepAlive(e)
	return out
}

// Select projects a subset of fields out of a struct-typed expression, keeping
// it a struct of just those fields.
func (e *Expr) Select(names ...string) *Expr {
	if len(names) == 0 {
		return newExpr(C.vx_expression_select(nil, 0, exprPtr(e)))
	}
	cstrs := make([]*C.char, len(names))
	for i, n := range names {
		cstrs[i] = C.CString(n)
	}
	defer func() {
		for _, p := range cstrs {
			C.free(unsafe.Pointer(p))
		}
	}()
	out := newExpr(C.vx_expression_select(
		(**C.char)(unsafe.Pointer(&cstrs[0])), C.size_t(len(names)), exprPtr(e)))
	runtime.KeepAlive(e)
	return out
}

// Not returns the logical negation of a boolean expression.
func (e *Expr) Not() *Expr {
	out := newExpr(C.vx_expression_not(exprPtr(e)))
	runtime.KeepAlive(e)
	return out
}

// IsNull returns a boolean expression that is true where the input is NULL.
func (e *Expr) IsNull() *Expr {
	out := newExpr(C.vx_expression_is_null(exprPtr(e)))
	runtime.KeepAlive(e)
	return out
}

// ListContains returns a boolean expression that is true where the list value
// produced by e contains the value produced by value.
func (e *Expr) ListContains(value *Expr) *Expr {
	out := newExpr(C.vx_expression_list_contains(exprPtr(e), exprPtr(value)))
	runtime.KeepAlive(e)
	runtime.KeepAlive(value)
	return out
}

// Binary builds the expression `lhs op rhs`.
func Binary(op BinaryOp, lhs, rhs *Expr) *Expr {
	out := newExpr(C.vx_expression_binary(C.vx_binary_operator(op), exprPtr(lhs), exprPtr(rhs)))
	runtime.KeepAlive(lhs)
	runtime.KeepAlive(rhs)
	return out
}

// And builds the conjunction of the operands. It requires at least one operand
// and returns nil otherwise.
func And(operands ...*Expr) *Expr {
	ptrs := exprPtrSlice(operands)
	if ptrs == nil {
		return nil
	}
	out := newExpr(C.vx_expression_and((**C.vx_expression)(unsafe.Pointer(&ptrs[0])), C.size_t(len(ptrs))))
	runtime.KeepAlive(operands)
	return out
}

// Or builds the disjunction of the operands. It requires at least one operand
// and returns nil otherwise.
func Or(operands ...*Expr) *Expr {
	ptrs := exprPtrSlice(operands)
	if ptrs == nil {
		return nil
	}
	out := newExpr(C.vx_expression_or((**C.vx_expression)(unsafe.Pointer(&ptrs[0])), C.size_t(len(ptrs))))
	runtime.KeepAlive(operands)
	return out
}

func exprPtrSlice(operands []*Expr) []*C.vx_expression {
	if len(operands) == 0 {
		return nil
	}
	ptrs := make([]*C.vx_expression, len(operands))
	for i, o := range operands {
		ptrs[i] = exprPtr(o)
	}
	return ptrs
}

// Comparison and arithmetic helpers.

// Eq builds `e == rhs`.
func (e *Expr) Eq(rhs *Expr) *Expr { return Binary(OpEq, e, rhs) }

// NotEq builds `e != rhs`.
func (e *Expr) NotEq(rhs *Expr) *Expr { return Binary(OpNotEq, e, rhs) }

// Gt builds `e > rhs`.
func (e *Expr) Gt(rhs *Expr) *Expr { return Binary(OpGt, e, rhs) }

// Gte builds `e >= rhs`.
func (e *Expr) Gte(rhs *Expr) *Expr { return Binary(OpGte, e, rhs) }

// Lt builds `e < rhs`.
func (e *Expr) Lt(rhs *Expr) *Expr { return Binary(OpLt, e, rhs) }

// Lte builds `e <= rhs`.
func (e *Expr) Lte(rhs *Expr) *Expr { return Binary(OpLte, e, rhs) }

// And builds the Kleene conjunction `e AND rhs`.
func (e *Expr) And(rhs *Expr) *Expr { return Binary(OpAnd, e, rhs) }

// Or builds the Kleene disjunction `e OR rhs`.
func (e *Expr) Or(rhs *Expr) *Expr { return Binary(OpOr, e, rhs) }

// Add builds `e + rhs` (errors at scan time on overflow).
func (e *Expr) Add(rhs *Expr) *Expr { return Binary(OpAdd, e, rhs) }

// Sub builds `e - rhs`.
func (e *Expr) Sub(rhs *Expr) *Expr { return Binary(OpSub, e, rhs) }

// Mul builds `e * rhs`.
func (e *Expr) Mul(rhs *Expr) *Expr { return Binary(OpMul, e, rhs) }

// Div builds `e / rhs`.
func (e *Expr) Div(rhs *Expr) *Expr { return Binary(OpDiv, e, rhs) }

// Lit builds a non-null literal (constant) expression. Supported value types are
// bool; the sized integer types int8…int64 and uint8…uint64; the platform int
// and uint; float32; float64; string; and []byte. It returns an error for any
// other type, or — for string/[]byte — if the bytes cannot be represented.
func Lit(v any) (*Expr, error) {
	sc, err := newScalar(v)
	if err != nil {
		return nil, err
	}
	defer C.vx_scalar_free(sc)
	var cerr *C.vx_error
	e := newExpr(C.vx_expression_literal(sc, &cerr))
	if err := consumeError(cerr); err != nil {
		return nil, err
	}
	if e == nil {
		return nil, &Error{Message: "failed to build literal expression"}
	}
	return e, nil
}

// newScalar builds an owned *vx_scalar from a Go value. Caller frees with C.vx_scalar_free.
func newScalar(v any) (*C.vx_scalar, error) {
	const notNull = C.bool(false)
	switch x := v.(type) {
	case bool:
		return C.vx_scalar_new_bool(C.bool(x), notNull), nil
	case int8:
		return C.vx_scalar_new_i8(C.int8_t(x), notNull), nil
	case int16:
		return C.vx_scalar_new_i16(C.int16_t(x), notNull), nil
	case int32:
		return C.vx_scalar_new_i32(C.int32_t(x), notNull), nil
	case int64:
		return C.vx_scalar_new_i64(C.int64_t(x), notNull), nil
	case int:
		return C.vx_scalar_new_i64(C.int64_t(x), notNull), nil
	case uint8:
		return C.vx_scalar_new_u8(C.uint8_t(x), notNull), nil
	case uint16:
		return C.vx_scalar_new_u16(C.uint16_t(x), notNull), nil
	case uint32:
		return C.vx_scalar_new_u32(C.uint32_t(x), notNull), nil
	case uint64:
		return C.vx_scalar_new_u64(C.uint64_t(x), notNull), nil
	case uint:
		return C.vx_scalar_new_u64(C.uint64_t(x), notNull), nil
	case float32:
		return C.vx_scalar_new_f32(C.float(x), notNull), nil
	case float64:
		return C.vx_scalar_new_f64(C.double(x), notNull), nil
	case string:
		return newBytesScalar([]byte(x), true)
	case []byte:
		return newBytesScalar(x, false)
	default:
		return nil, fmt.Errorf("vortex: unsupported literal type %T", v)
	}
}

func newBytesScalar(b []byte, utf8 bool) (*C.vx_scalar, error) {
	var cdata unsafe.Pointer
	if len(b) > 0 {
		cdata = C.CBytes(b)
		defer C.free(cdata)
	}
	var cerr *C.vx_error
	var sc *C.vx_scalar
	if utf8 {
		sc = C.vx_scalar_new_utf8((*C.char)(cdata), C.size_t(len(b)), C.bool(false), &cerr)
	} else {
		sc = C.vx_scalar_new_binary((*C.uint8_t)(cdata), C.size_t(len(b)), C.bool(false), &cerr)
	}
	if err := consumeError(cerr); err != nil {
		return nil, err
	}
	if sc == nil {
		return nil, &Error{Message: "failed to build scalar"}
	}
	return sc, nil
}
