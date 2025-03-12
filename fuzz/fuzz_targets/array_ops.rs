#![no_main]

use libfuzzer_sys::{Corpus, fuzz_target};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::{
    BoolEncoding, ConstantArray, ListEncoding, PrimitiveEncoding, StructEncoding, VarBinEncoding,
    VarBinViewEncoding,
};
use vortex_array::compute::{
    SearchResult, SearchSortedSide, compare, filter, scalar_at, search_sorted, slice, take,
};
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexUnwrap;
use vortex_fuzz::{Action, FuzzArrayAction, sort_canonical_array};
use vortex_scalar::Scalar;

fuzz_target!(|fuzz_action: FuzzArrayAction| -> Corpus {
    let FuzzArrayAction { array, actions } = fuzz_action;
    let mut current_array = array.to_array();
    for (i, (action, expected)) in actions.into_iter().enumerate() {
        match action {
            Action::Compress => {
                current_array = BtrBlocksCompressor
                    .compress(current_array.to_canonical().vortex_unwrap().as_ref())
                    .vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
            Action::Slice(range) => {
                current_array = slice(&current_array, range.start, range.end).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
            Action::Take(indices) => {
                if indices.is_empty() {
                    return Corpus::Reject;
                }
                current_array = take(&current_array, &indices).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
            Action::SearchSorted(s, side) => {
                // TODO(robert): Ideally we'd preserve the encoding perfectly but this is close enough
                let mut sorted = sort_canonical_array(&current_array).vortex_unwrap();
                if !HashSet::from([
                    PrimitiveEncoding.id(),
                    VarBinEncoding.id(),
                    VarBinViewEncoding.id(),
                    BoolEncoding.id(),
                    StructEncoding.id(),
                    ListEncoding.id(),
                ])
                .contains(&current_array.encoding())
                {
                    sorted = BtrBlocksCompressor.compress(&sorted).vortex_unwrap();
                }
                assert_search_sorted(sorted, s, side, expected.search(), i)
            }
            Action::Filter(mask) => {
                current_array = filter(&current_array, &mask).vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
            }
            Action::Compare(v, op) => {
                current_array = compare(
                    &current_array,
                    &ConstantArray::new(v, current_array.len()).into_array(),
                    op,
                )
                .vortex_unwrap();
                assert_array_eq(&expected.array(), &current_array, i);
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
) {
    let search_result = search_sorted(&array, s.clone(), side).vortex_unwrap();
    assert_eq!(
        expected,
        search_result,
        "Expected to find {s}({}) at {expected} in {} from {side} but instead found it at {search_result} in step {step}",
        s.dtype(),
        array.tree_display()
    );
}

// TODO(ngates): this is horrific... we should have an array_equals compute function?
fn assert_array_eq(lhs: &dyn Array, rhs: &dyn Array, step: usize) {
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
        let l = scalar_at(lhs, idx).vortex_unwrap();
        let r = scalar_at(rhs, idx).vortex_unwrap();

        assert_eq!(
            l,
            r,
            "{l} != {r} at index {idx}, lhs is {} rhs is {} in step {step}",
            lhs.tree_display(),
            rhs.tree_display()
        );
    }
}
