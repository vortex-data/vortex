// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![allow(clippy::unwrap_used, clippy::result_large_err)]

use std::backtrace::Backtrace;

use libfuzzer_sys::{Corpus, fuzz_target};
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{cast, compare, fill_null, filter, min_max, sum, take};
use vortex_array::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::{VortexUnwrap, vortex_panic};
use vortex_fuzz::error::{VortexFuzzError, VortexFuzzResult};
use vortex_fuzz::{Action, CompressorStrategy, FuzzArrayAction, sort_canonical_array};
use vortex_layout::layouts::compact::CompactCompressor;
use vortex_scalar::Scalar;

fuzz_target!(|fuzz_action: FuzzArrayAction| -> Corpus {
    let FuzzArrayAction { array, actions } = fuzz_action;
    let mut current_array = array.to_array();
    for (i, (action, expected)) in actions.into_iter().enumerate() {
        match action {
            Action::Compress(strategy) => {
                current_array = match strategy {
                    CompressorStrategy::Default => BtrBlocksCompressor::default()
                        .compress(current_array.to_canonical().as_ref())
                        .vortex_unwrap(),
                    CompressorStrategy::Compact => CompactCompressor::default()
                        .compress(current_array.to_canonical().as_ref())
                        .vortex_unwrap(),
                };
                assert_array_eq(&expected.array(), &current_array, i).unwrap();
            }
            Action::Slice(range) => {
                current_array = current_array.slice(range);
                assert_array_eq(&expected.array(), &current_array, i).unwrap();
            }
            Action::Take(indices) => {
                if indices.is_empty() {
                    return Corpus::Reject;
                }
                current_array = take(&current_array, &indices).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i).unwrap();
            }
            Action::SearchSorted(s, side) => {
                // TODO(robert): Ideally we'd preserve the encoding perfectly but this is close enough
                let mut sorted = sort_canonical_array(&current_array).vortex_unwrap();

                // If the current array is not in one of these canonical encodings, compress again.
                if !current_array.is_canonical() {
                    sorted = BtrBlocksCompressor::default()
                        .compress(&sorted)
                        .vortex_unwrap();
                }
                assert_search_sorted(sorted, s, side, expected.search(), i).unwrap()
            }
            Action::Filter(mask) => {
                current_array = filter(&current_array, &mask).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i).unwrap();
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
                assert_scalar_eq(&expected.scalar(), &sum_result);
            }
            Action::MinMax => {
                let min_max_result = min_max(&current_array).vortex_unwrap();
                assert_min_max_eq(&expected.min_max(), &min_max_result, i).unwrap();
            }
            Action::FillNull(fill_value) => {
                current_array = fill_null(&current_array, &fill_value).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i).unwrap();
            }
            Action::Mask(mask_val) => {
                use vortex_array::compute::mask;
                current_array = mask(&current_array, &mask_val).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i).unwrap();
            }
        }
    }
    Corpus::Keep
});

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

// TODO(ngates): this is horrific... we should have an array_equals compute function?
fn assert_array_eq(lhs: &ArrayRef, rhs: &ArrayRef, step: usize) -> VortexFuzzResult<()> {
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

fn assert_scalar_eq(lhs: &Scalar, rhs: &Scalar) {
    // Use catch_unwind to handle panics in scalar comparison (e.g., decimal conversion issues)
    assert_eq!(
        lhs, rhs,
        "Scalar mismatch: expected {:?}, got {:?}",
        lhs, rhs
    );
}

fn assert_min_max_eq(
    lhs: &Option<vortex_array::compute::MinMaxResult>,
    rhs: &Option<vortex_array::compute::MinMaxResult>,
    _step: usize,
) -> VortexFuzzResult<()> {
    if lhs != rhs {
        vortex_panic!("MinMax mismatch: expected {:?}, got {:?}", lhs, rhs);
    }
    Ok(())
}
