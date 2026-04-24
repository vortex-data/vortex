// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASM-compatible fuzz target for array operations.
//!
//! This target is designed to work with wasmfuzz and exports:
//! - `LLVMFuzzerTestOneInput` - the fuzzing entry point
//! - `wasmfuzz_malloc` / `wasmfuzz_free` - memory allocation for wasmfuzz to pass input data

use std::alloc::Layout;
use std::alloc::alloc;
use std::alloc::dealloc;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::run_fuzz_action;

/// Allocate memory for wasmfuzz to pass input data.
///
/// # Safety
///
/// Returns a pointer to allocated memory of at least `size` bytes, or null on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wasmfuzz_malloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = match Layout::from_size_align(size, 8) {
        Ok(layout) => layout,
        Err(_) => return std::ptr::null_mut(),
    };
    unsafe { alloc(layout) }
}

/// Free memory allocated by wasmfuzz_malloc.
///
/// # Safety
///
/// The pointer must have been allocated by wasmfuzz_malloc with the given size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wasmfuzz_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, 8) {
        unsafe { dealloc(ptr, layout) };
    }
}

/// The entry point for wasmfuzz.
///
/// Returns 0 on success, -1 to reject the input from the corpus.
///
/// # Safety
///
/// The caller must ensure that `data` points to a valid memory region of at least
/// `size` bytes. This is guaranteed by wasmfuzz when calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LLVMFuzzerTestOneInput(data: *const u8, size: usize) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, size) };

    let mut u = Unstructured::new(slice);
    let Ok(fuzz_action) = FuzzArrayAction::arbitrary(&mut u) else {
        return -1;
    };

    match run_fuzz_action(fuzz_action) {
        Ok(true) => 0,
        Ok(false) => -1,
        Err(_) => 1,
    }
}

fn main() {}
