// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Thread-local error handling for FFI.
//!
//! This module provides a mechanism to store the last error message in a thread-local
//! variable, which can be retrieved by the C++ side after an FFI call returns an error.
//!
//! # Usage
//!
//! On the Rust side:
//! ```rust,ignore
//! use crate::error::{set_last_error, clear_last_error};
//!
//! pub extern "C" fn some_ffi_function() -> i32 {
//!     clear_last_error();
//!     match do_something() {
//!         Ok(_) => 0,
//!         Err(e) => {
//!             set_last_error(&e.to_string());
//!             -1
//!         }
//!     }
//! }
//! ```
//!
//! On the C++ side:
//! ```cpp
//! int result = some_ffi_function();
//! if (result < 0) {
//!     const char* error = vortex_get_last_error();
//!     if (error) {
//!         std::cerr << "Error: " << error << std::endl;
//!         vortex_free_string(error);
//!     }
//! }
//! ```

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::ptr;

thread_local! {
    /// Thread-local storage for the last error message.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Set the last error message.
///
/// This function stores the error message in thread-local storage,
/// where it can be retrieved by `vortex_get_last_error()`.
pub fn set_last_error(message: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(message.to_string());
    });
}

/// Clear the last error message.
///
/// This should be called at the beginning of each FFI function to ensure
/// that any previous error is cleared.
pub fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Get the last error message.
///
/// Returns the last error message if one was set, or None.
pub fn get_last_error() -> Option<String> {
    LAST_ERROR.with(|cell| cell.borrow().clone())
}

/// Check if there is a last error.
pub fn has_last_error() -> bool {
    LAST_ERROR.with(|cell| cell.borrow().is_some())
}

// =============================================================================
// FFI Exports
// =============================================================================

/// Get the last error message.
///
/// Returns a null-terminated C string with the last error message,
/// or NULL if no error was set. The returned string must be freed
/// by calling `vortex_free_string()`.
///
/// # Safety
/// The returned pointer must be freed with `vortex_free_string()` after use.
/// The pointer is only valid until the next call to any vortex FFI function
/// from the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_get_last_error() -> *mut c_char {
    match get_last_error() {
        Some(msg) => match CString::new(msg) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        None => ptr::null_mut(),
    }
}

/// Check if there is a last error.
///
/// Returns 1 if an error is set, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_has_error() -> i32 {
    if has_last_error() { 1 } else { 0 }
}

/// Clear the last error.
///
/// Call this before starting a new operation if you want to ensure
/// no stale error messages are present.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_clear_error() {
    clear_last_error();
}

/// Free a string returned by vortex FFI functions.
///
/// This function must be called to free strings returned by functions like
/// `vortex_get_last_error()`, `vortex_scanner_column_name()`, etc.
///
/// # Safety
/// The `ptr` must be a valid pointer returned by a vortex FFI function,
/// or NULL (which is safely ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

// =============================================================================
// Macro for FFI error handling
// =============================================================================

/// Macro to wrap FFI function bodies with error handling.
///
/// This macro:
/// 1. Clears the last error at the start
/// 2. Executes the body
/// 3. On error, sets the last error message and returns the error value
///
/// # Example
/// ```rust,ignore
/// ffi_try! {
///     let result = some_operation()?;
///     Ok(result)
/// } or_return -1
/// ```
macro_rules! ffi_try {
    ($body:expr, $error_ret:expr) => {{
        crate::error::clear_last_error();
        match (|| -> vortex::error::VortexResult<_> { $body })() {
            Ok(val) => val,
            Err(e) => {
                crate::error::set_last_error(&e.to_string());
                return $error_ret;
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_set_and_get() {
        clear_last_error();
        assert!(!has_last_error());
        assert!(get_last_error().is_none());

        set_last_error("Test error message");
        assert!(has_last_error());
        assert_eq!(get_last_error(), Some("Test error message".to_string()));

        clear_last_error();
        assert!(!has_last_error());
        assert!(get_last_error().is_none());
    }

    #[test]
    fn test_ffi_get_last_error() {
        clear_last_error();

        // No error set
        let ptr = vortex_get_last_error();
        assert!(ptr.is_null());

        // Set an error
        set_last_error("FFI test error");
        let ptr = vortex_get_last_error();
        assert!(!ptr.is_null());

        // Read the error
        let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(c_str.to_str().unwrap(), "FFI test error");

        // Free the string
        unsafe { vortex_free_string(ptr) };

        // Clear the error
        vortex_clear_error();
        assert_eq!(vortex_has_error(), 0);
    }
}
