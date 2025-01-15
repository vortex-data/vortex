#![no_main]

use libfuzzer_sys::{fuzz_target, Corpus};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{
    BoolEncoding, ListEncoding, PrimitiveEncoding, StructEncoding, VarBinEncoding,
    VarBinViewEncoding,
};
use vortex_array::compute::{
    filter, scalar_at, search_sorted, slice, take, SearchResult, SearchSortedSide,
};
use vortex_array::encoding::EncodingRef;
use vortex_array::{ArrayData, IntoCanonical};
use vortex_fuzz::{sort_canonical_array, Action, FuzzArrayAction};
use vortex_sampling_compressor::SamplingCompressor;
use vortex_scalar::Scalar;

fuzz_target!(|fuzz_action: FuzzArrayAction| -> Corpus {
    let FuzzArrayAction { array, actions } = fuzz_action;
    let mut current_array = array.clone();
    for (i, (action, expected)) in actions.into_iter().enumerate() {
        match action {
            Action::Compress(c) => {
                match fuzz_compress(&current_array.into_canonical().unwrap().into(), &c) {
                    Some(compressed_array) => {
                        assert_array_eq(&expected.array(), &compressed_array, i);
                        current_array = compressed_array;
                    }
                    None => return Corpus::Reject,
                }
            }
            Action::Slice(range) => {
                current_array = slice(&current_array, range.start, range.end).unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
            Action::Take(indices) => {
                if indices.is_empty() {
                    return Corpus::Reject;
                }
                current_array = take(&current_array, &indices).unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
            Action::SearchSorted(s, side) => {
                // TODO(robert): Ideally we'd preserve the encoding perfectly but this is close enough
                let mut sorted = sort_canonical_array(&current_array);
                if !HashSet::from([
                    &PrimitiveEncoding as EncodingRef,
                    &VarBinEncoding,
                    &VarBinViewEncoding,
                    &BoolEncoding,
                    &StructEncoding,
                    &ListEncoding,
                ])
                .contains(&current_array.encoding())
                {
                    sorted =
                        fuzz_compress(&sorted, &SamplingCompressor::default()).unwrap_or(sorted);
                }
                assert_search_sorted(sorted, s, side, expected.search(), i)
            }
            Action::Filter(mask) => {
                current_array = filter(&current_array, &mask).unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
        }
    }
    Corpus::Keep
});

fn fuzz_compress(array: &ArrayData, compressor: &SamplingCompressor) -> Option<ArrayData> {
    let compressed_array = compressor.compress(array, None).unwrap();

    compressed_array
        .path()
        .is_some()
        .then(|| compressed_array.into_array())
}

fn assert_search_sorted(
    array: ArrayData,
    s: Scalar,
    side: SearchSortedSide,
    expected: SearchResult,
    step: usize,
) {
    let search_result = search_sorted(&array, s.clone(), side).unwrap();
    assert_eq!(
        search_result,
        expected,
        "Expected to find {s}({}) at {expected} in {} from {side} but instead found it at {search_result} in step {step}",
        s.dtype(),
        array.tree_display()
    );
}

// TODO(ngates): this is horrific... we should have an array_equals compute function?
fn assert_array_eq(lhs: &ArrayData, rhs: &ArrayData, step: usize) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "LHS len {} != RHS len {}, lhs is {} rhs is {} in step {step}",
        lhs.len(),
        rhs.len(),
        lhs.tree_display(),
        rhs.tree_display()
    );
    for idx in 0..lhs.len() {
        let l = scalar_at(lhs, idx).unwrap();
        let r = scalar_at(rhs, idx).unwrap();

        assert_eq!(
            l,
            r,
            "{l} != {r} at index {idx}, lhs is {} rhs is {} in step {step}",
            lhs.tree_display(),
            rhs.tree_display()
        );
    }
}
