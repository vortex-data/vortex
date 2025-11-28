// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASM-compatible fuzz target for array operations.
//!
//! This target is designed to work with wasmfuzz and exports:
//! - `LLVMFuzzerTestOneInput` - the fuzzing entry point
//! - `wasmfuzz_malloc` / `wasmfuzz_free` - memory allocation for wasmfuzz to pass input data
//!
//! wasmfuzz looks for `wasmfuzz_malloc`/`wasmfuzz_free` or `malloc`/`free` exports.
//! We provide `wasmfuzz_malloc`/`wasmfuzz_free` using Rust's global allocator.

#![allow(clippy::unwrap_used, clippy::result_large_err)]

use std::alloc::Layout;
use std::alloc::alloc;
use std::alloc::dealloc;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::MinMaxResult;
use vortex_array::compute::cast;
use vortex_array::compute::compare;
use vortex_array::compute::fill_null;
use vortex_array::compute::filter;
use vortex_array::compute::mask;
use vortex_array::compute::min_max;
use vortex_array::compute::sum;
use vortex_array::compute::take;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexUnwrap;
use vortex_error::vortex_panic;
use vortex_fuzz::Action;
use vortex_fuzz::CompressorStrategy;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::error::Backtrace;
use vortex_fuzz::error::VortexFuzzError;
use vortex_fuzz::error::VortexFuzzResult;
use vortex_fuzz::sort_canonical_array;
use vortex_scalar::Scalar;

/// Allocate memory for wasmfuzz to pass input data.
///
/// wasmfuzz requires a `wasmfuzz_malloc` or `malloc` export to allocate space for fuzz inputs.
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
/// This function is called by wasmfuzz with fuzzer-generated input data.
/// Returns 0 on success, -1 to reject the input from the corpus.
///
/// # Safety
///
/// The caller must ensure that `data` points to a valid memory region of at least
/// `size` bytes. This is guaranteed by wasmfuzz when calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LLVMFuzzerTestOneInput(data: *const u8, size: usize) -> i32 {
    // Safety: wasmfuzz guarantees data is valid for size bytes
    let slice = unsafe { std::slice::from_raw_parts(data, size) };

    let mut u = Unstructured::new(slice);
    let Ok(fuzz_action) = FuzzArrayAction::arbitrary(&mut u) else {
        return -1; // Reject malformed input
    };

    match run_fuzz_action(fuzz_action) {
        Ok(true) => 0,   // Keep in corpus
        Ok(false) => -1, // Reject from corpus
        Err(_) => {
            // A fuzz error means we found a bug - this will cause wasmfuzz to save the input
            1
        }
    }
}

fn run_fuzz_action(fuzz_action: FuzzArrayAction) -> VortexFuzzResult<bool> {
    let FuzzArrayAction { array, actions } = fuzz_action;
    let mut current_array = array.to_array();

    for (i, (action, expected)) in actions.into_iter().enumerate() {
        match action {
            Action::Compress(strategy) => {
                // Note: CompactCompressor requires zstd which may not be available in WASM
                // For now, only use BtrBlocksCompressor
                current_array = match strategy {
                    CompressorStrategy::Default | CompressorStrategy::Compact => {
                        BtrBlocksCompressor::default()
                            .compress(current_array.to_canonical().as_ref())
                            .vortex_unwrap()
                    }
                };
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Slice(range) => {
                current_array = current_array.slice(range);
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Take(indices) => {
                if indices.is_empty() {
                    return Ok(false); // Reject
                }
                current_array = take(&current_array, &indices).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::SearchSorted(s, side) => {
                let mut sorted = sort_canonical_array(&current_array).vortex_unwrap();

                if !current_array.is_canonical() {
                    sorted = BtrBlocksCompressor::default()
                        .compress(&sorted)
                        .vortex_unwrap();
                }
                assert_search_sorted(sorted, s, side, expected.search(), i)?;
            }
            Action::Filter(mask) => {
                current_array = filter(&current_array, &mask).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Compare(v, op) => {
                let compare_result = compare(
                    &current_array,
                    &ConstantArray::new(v.clone(), current_array.len()).into_array(),
                    op,
                )
                .vortex_unwrap();
                if let Err(e) = assert_array_eq(&expected.array(), &compare_result, i) {
                    vortex_panic!(
                        "Failed to compare {}with {op} {v}\nError: {e}",
                        current_array.display_tree()
                    )
                }
                current_array = compare_result;
            }
            Action::Cast(to) => {
                let cast_result = cast(&current_array, &to).vortex_unwrap();
                if let Err(e) = assert_array_eq(&expected.array(), &cast_result, i) {
                    vortex_panic!(
                        "Failed to cast {} to dtype {to}\nError: {e}",
                        current_array.display_tree()
                    )
                }
                current_array = cast_result;
            }
            Action::Sum => {
                let sum_result = sum(&current_array).vortex_unwrap();
                assert_scalar_eq(&expected.scalar(), &sum_result, i)?;
            }
            Action::MinMax => {
                let min_max_result = min_max(&current_array).vortex_unwrap();
                assert_min_max_eq(&expected.min_max(), &min_max_result, i)?;
            }
            Action::FillNull(fill_value) => {
                current_array = fill_null(&current_array, &fill_value).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Mask(mask_val) => {
                current_array = mask(&current_array, &mask_val).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::ScalarAt(indices) => {
                let expected_scalars = expected.scalar_vec();
                for (j, &idx) in indices.iter().enumerate() {
                    let scalar = current_array.scalar_at(idx);
                    assert_scalar_eq(&expected_scalars[j], &scalar, i)?;
                }
            }
        }
    }
    Ok(true) // Keep in corpus
}

fn assert_search_sorted(
    array: ArrayRef,
    s: Scalar,
    side: SearchSortedSide,
    expected: SearchResult,
    step: usize,
) -> VortexFuzzResult<()> {
    let search_result = array.search_sorted(&s, side);
    if search_result != expected {
        Err(VortexFuzzError::SearchSortedError(
            s,
            expected,
            array.to_array(),
            side,
            search_result,
            step,
            Backtrace::capture(),
        ))
    } else {
        Ok(())
    }
}

fn assert_array_eq(lhs: &ArrayRef, rhs: &ArrayRef, step: usize) -> VortexFuzzResult<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            lhs.clone(),
            rhs.clone(),
            step,
            Backtrace::capture(),
        ));
    }

    if lhs.len() != rhs.len() {
        return Err(VortexFuzzError::LengthMismatch(
            lhs.len(),
            rhs.len(),
            lhs.to_array(),
            rhs.to_array(),
            step,
            Backtrace::capture(),
        ));
    }
    for idx in 0..lhs.len() {
        let l = lhs.scalar_at(idx);
        let r = rhs.scalar_at(idx);

        if l != r {
            return Err(VortexFuzzError::ArrayNotEqual(
                l,
                r,
                idx,
                lhs.clone(),
                rhs.clone(),
                step,
                Backtrace::capture(),
            ));
        }
    }
    Ok(())
}

fn assert_scalar_eq(lhs: &Scalar, rhs: &Scalar, step: usize) -> VortexFuzzResult<()> {
    if lhs != rhs {
        return Err(VortexFuzzError::ScalarMismatch(
            lhs.clone(),
            rhs.clone(),
            step,
            Backtrace::capture(),
        ));
    }
    Ok(())
}

fn assert_min_max_eq(
    lhs: &Option<MinMaxResult>,
    rhs: &Option<MinMaxResult>,
    step: usize,
) -> VortexFuzzResult<()> {
    if lhs != rhs {
        return Err(VortexFuzzError::MinMaxMismatch(
            lhs.clone(),
            rhs.clone(),
            step,
            Backtrace::capture(),
        ));
    }
    Ok(())
}

/// Main function required for WASM binary.
///
/// For wasmfuzz reactor mode, this is called once at initialization.
/// The actual fuzzing happens through `LLVMFuzzerTestOneInput`.
fn main() {
    // Nothing to do - wasmfuzz calls LLVMFuzzerTestOneInput directly
}
