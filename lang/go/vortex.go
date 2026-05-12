// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Package vortex provides Go bindings for the Vortex columnar file format.
//
// The bindings wrap the Vortex C FFI ("vortex-ffi") through cgo and exchange
// data with the rest of the Go ecosystem via the Apache Arrow C Data Interface
// (github.com/apache/arrow-go). They are intentionally scoped to reading and
// writing Arrow data, plus building the expression trees used for projection and
// predicate pushdown during reads.
//
// # Linking
//
// These bindings link against libvortex_ffi, which is produced by running
//
//	cargo build --release -p vortex-ffi
//
// from the repository root. By default cgo looks for the library and headers at
// their in-tree locations (../../target/{debug,ci,release} and
// ../../vortex-ffi/cinclude, relative to this package). Set CGO_CFLAGS /
// CGO_LDFLAGS to point elsewhere.
//
// # Thread safety
//
// A [Session] may be shared between goroutines, but the objects derived from it
// (files, scans, readers) are not safe for concurrent use; serialise access to a
// given file or reader, or create one per goroutine.
package vortex

/*
#cgo CFLAGS: -I${SRCDIR}/../../vortex-ffi/cinclude
#cgo linux LDFLAGS: -L${SRCDIR}/../../target/release -L${SRCDIR}/../../target/ci -L${SRCDIR}/../../target/debug -Wl,-rpath,${SRCDIR}/../../target/release -Wl,-rpath,${SRCDIR}/../../target/ci -Wl,-rpath,${SRCDIR}/../../target/debug -lvortex_ffi -lm
#cgo darwin LDFLAGS: -L${SRCDIR}/../../target/release -L${SRCDIR}/../../target/ci -L${SRCDIR}/../../target/debug -Wl,-rpath,${SRCDIR}/../../target/release -Wl,-rpath,${SRCDIR}/../../target/ci -Wl,-rpath,${SRCDIR}/../../target/debug -lvortex_ffi -framework CoreFoundation -framework Security
#include <stdlib.h>
#include "vortex.h"
*/
import "C"

import "runtime"

// LogLevel controls the verbosity of the global Vortex logger.
type LogLevel int

// Log levels accepted by [SetLogLevel].
const (
	LogOff   LogLevel = C.LOG_LEVEL_OFF
	LogError LogLevel = C.LOG_LEVEL_ERROR
	LogWarn  LogLevel = C.LOG_LEVEL_WARN
	LogInfo  LogLevel = C.LOG_LEVEL_INFO
	LogDebug LogLevel = C.LOG_LEVEL_DEBUG
	LogTrace LogLevel = C.LOG_LEVEL_TRACE
)

// SetLogLevel installs (on the first call) a stderr logger at the given level.
// Subsequent calls have no effect.
func SetLogLevel(level LogLevel) {
	C.vx_set_log_level(C.vx_log_level(level))
}

// Error is the error type returned by fallible Vortex operations. It carries the
// message produced by the native library.
type Error struct {
	Message string
}

func (e *Error) Error() string { return "vortex: " + e.Message }

// consumeError converts a (possibly nil) owned *vx_error into a Go error,
// freeing the native error in the process.
func consumeError(cerr *C.vx_error) error {
	if cerr == nil {
		return nil
	}
	s := C.vx_error_get_message(cerr) // borrowed; lifetime tied to cerr
	msg := C.GoStringN(C.vx_string_ptr(s), C.int(C.vx_string_len(s)))
	C.vx_error_free(cerr)
	return &Error{Message: msg}
}

// Session is a handle to a native Vortex session. It owns a current-thread async
// runtime and is the entry point for opening files and writing data.
//
// Create one with [NewSession]. Native resources are released by [Session.Close]
// or, as a backstop, by a finalizer when the Session becomes unreachable.
type Session struct {
	ptr *C.vx_session
}

// NewSession creates a new Vortex session.
func NewSession() *Session {
	s := &Session{ptr: C.vx_session_new()}
	runtime.SetFinalizer(s, (*Session).free)
	return s
}

func (s *Session) free() {
	if s != nil && s.ptr != nil {
		C.vx_session_free(s.ptr)
		s.ptr = nil
	}
}

// Close releases the native session. It is safe to call more than once; further
// use of the Session or of objects derived from it is undefined behaviour.
func (s *Session) Close() {
	runtime.SetFinalizer(s, nil)
	s.free()
}
