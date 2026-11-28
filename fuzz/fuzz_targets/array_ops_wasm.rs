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
use std::ptr;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::error::VortexFuzzResult;
use vortex_fuzz::run_fuzz_action;

const WASMFUZZ_ALIGN: usize = 8;

/// Allocate memory for wasmfuzz to pass input data.
///
/// # Safety
///
/// Returns a pointer to allocated memory of at least `size` bytes, or null on failure or when
/// `size == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wasmfuzz_malloc(size: usize) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(size, WASMFUZZ_ALIGN) else {
        return ptr::null_mut();
    };
    // SAFETY: `size > 0`, so `layout` has non-zero size.
    unsafe { alloc(layout) }
}

/// Free memory allocated by `wasmfuzz_malloc`.
///
/// # Safety
///
/// `ptr` must be null or have been allocated by `wasmfuzz_malloc` with the same `size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wasmfuzz_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, WASMFUZZ_ALIGN) {
        // SAFETY: `ptr` is non-null, `size > 0`, and the caller guarantees `ptr` was allocated
        // by `wasmfuzz_malloc` with this exact `layout`.
        unsafe { dealloc(ptr, layout) };
    }
}

/// Convert a raw `(data, size)` buffer into a byte slice without triggering UB on null/empty inputs.
///
/// # Safety
///
/// When `0 < size <= isize::MAX as usize` and `data` is non-null, `data` must be valid for reads
/// of `size` initialized bytes for `'a` and must not be mutated during `'a`.
unsafe fn input_slice<'a>(data: *const u8, size: usize) -> Option<&'a [u8]> {
    if size == 0 {
        return Some(&[]);
    }
    if data.is_null() || isize::try_from(size).is_err() {
        return None;
    }
    // SAFETY: `size` is in `1..=isize::MAX`, `data` is non-null, `u8` has alignment 1, and the
    // caller guarantees `data` points to `size` valid initialized bytes.
    Some(unsafe { std::slice::from_raw_parts(data, size) })
}

fn handle_fuzz_result(result: VortexFuzzResult<bool>) -> i32 {
    match result {
        Ok(true) => 0,
        Ok(false) => -1,
        Err(e) => vortex_panic!("{e}"),
    }
}

/// The entry point for wasmfuzz.
///
/// Returns 0 to keep the input in the corpus, -1 to reject it, and panics (trapping in WASM)
/// if a fuzz assertion fails.
///
/// # Safety
///
/// When `0 < size <= isize::MAX as usize` and `data` is non-null, the caller must ensure that
/// `data` points to a valid initialized memory region of at least `size` bytes that is not
/// mutated for the duration of the call. This is guaranteed by wasmfuzz when calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LLVMFuzzerTestOneInput(data: *const u8, size: usize) -> i32 {
    // SAFETY: Forwarded from the caller's contract for `LLVMFuzzerTestOneInput`.
    let Some(slice) = (unsafe { input_slice(data, size) }) else {
        return -1;
    };

    let mut u = Unstructured::new(slice);
    let Ok(fuzz_action) = FuzzArrayAction::arbitrary(&mut u) else {
        return -1;
    };

    handle_fuzz_result(run_fuzz_action(fuzz_action))
}

fn main() {}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex_error::vortex_err;
    use vortex_fuzz::error::Backtrace;
    use vortex_fuzz::error::VortexFuzzError;

    use super::LLVMFuzzerTestOneInput;
    use super::WASMFUZZ_ALIGN;
    use super::handle_fuzz_result;
    use super::input_slice;
    use super::wasmfuzz_free;
    use super::wasmfuzz_malloc;

    #[test]
    fn test_wasmfuzz_malloc_and_free_roundtrip() {
        // SAFETY: `wasmfuzz_malloc` accepts any size (returning null for 0 or invalid layouts),
        // and `wasmfuzz_free` is a no-op on null pointers.
        unsafe {
            assert!(wasmfuzz_malloc(0).is_null());
            assert!(wasmfuzz_malloc(usize::MAX).is_null());
            wasmfuzz_free(ptr::null_mut(), 0);
            wasmfuzz_free(ptr::null_mut(), 64);
        }

        let size = 64;
        // SAFETY: `wasmfuzz_malloc` has no preconditions on `size`.
        let ptr = unsafe { wasmfuzz_malloc(size) };
        assert!(!ptr.is_null());
        assert_eq!(ptr.addr() % WASMFUZZ_ALIGN, 0);

        // SAFETY: `ptr` points to `size` writable bytes allocated by `wasmfuzz_malloc(size)` above,
        // is initialized via `write_bytes` before `input_slice`, and is freed once with the same size.
        unsafe {
            ptr.write_bytes(0xAB, size);
            let slice = input_slice(ptr, size);
            assert_eq!(slice, Some([0xABu8; 64].as_slice()));
            wasmfuzz_free(ptr, size);
        }
    }

    #[test]
    fn test_input_slice_guards_empty_null_and_oversized() {
        let buf = [1u8, 2, 3, 4];

        // SAFETY: Zero-sized, null, and oversized (`> isize::MAX`) inputs are rejected without
        // dereferencing `data`; the final call passes a valid initialized slice `buf`.
        unsafe {
            // `wasmfuzz_malloc(0)` returns null; zero-size inputs must produce `&[]` without UB.
            assert_eq!(input_slice(ptr::null(), 0), Some([].as_slice()));
            assert_eq!(input_slice(buf.as_ptr(), 0), Some([].as_slice()));

            // Null pointer or size > `isize::MAX` for non-empty buffers must be rejected.
            assert_eq!(input_slice(ptr::null(), 1), None);
            assert_eq!(input_slice(buf.as_ptr(), usize::MAX), None);

            assert_eq!(input_slice(buf.as_ptr(), buf.len()), Some(buf.as_slice()));
        }
    }

    #[test]
    fn test_llvm_fuzzer_entry_point_handles_null_and_empty_inputs() {
        let byte = 0u8;

        // SAFETY: Each call passes either `size == 0`, a null pointer, or `size > isize::MAX`,
        // all of which `LLVMFuzzerTestOneInput` handles without dereferencing `data`.
        unsafe {
            let empty_rc = LLVMFuzzerTestOneInput(ptr::null(), 0);
            assert!(empty_rc == 0 || empty_rc == -1);

            assert_eq!(LLVMFuzzerTestOneInput(ptr::null(), 16), -1);
            assert_eq!(LLVMFuzzerTestOneInput(&raw const byte, usize::MAX), -1);
        }
    }

    #[test]
    fn test_handle_fuzz_result_maps_ok_values() {
        assert_eq!(handle_fuzz_result(Ok(true)), 0);
        assert_eq!(handle_fuzz_result(Ok(false)), -1);
    }

    #[test]
    #[should_panic(expected = "simulated fuzz failure")]
    fn test_handle_fuzz_result_panics_on_error() {
        let err = VortexFuzzError::VortexError(
            vortex_err!("simulated fuzz failure"),
            Backtrace::capture(),
        );
        handle_fuzz_result(Err(err));
    }
}
