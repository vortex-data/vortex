// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_scalar::PValue;

use crate::arrays::{BoolArray, PrimitiveArray};
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::search_sorted::rstest_reuse::apply;
use crate::compute::conformance::search_sorted::{search_sorted_conformance, *};
use crate::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray};

#[apply(search_sorted_conformance)]
fn test_search_sorted_primitive(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) {
    let res = array
        .as_primitive_typed()
        .search_sorted(&Some(PValue::from(value)), side);
    assert_eq!(res, expected);
}

#[test]
fn test_mask_primitive_array() {
    test_mask_conformance(PrimitiveArray::from_iter([0, 1, 2, 3, 4]).as_ref());
    test_mask_conformance(
        PrimitiveArray::from_option_iter([Some(0), Some(1), Some(2), Some(3), Some(4)]).as_ref(),
    );
    test_mask_conformance(
        PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllInvalid).as_ref(),
    );
    test_mask_conformance(
        PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
        )
        .as_ref(),
    );
}

#[test]
fn test_filter_primitive_array() {
    // Test various sizes
    test_filter_conformance(PrimitiveArray::from_iter([42i32]).as_ref());
    test_filter_conformance(PrimitiveArray::from_iter([0, 1]).as_ref());
    test_filter_conformance(PrimitiveArray::from_iter([0, 1, 2, 3, 4]).as_ref());
    test_filter_conformance(PrimitiveArray::from_iter([0, 1, 2, 3, 4, 5, 6, 7]).as_ref());

    // Test with validity
    test_filter_conformance(
        PrimitiveArray::from_option_iter([Some(0), Some(1), Some(2), Some(3), Some(4)]).as_ref(),
    );
    test_filter_conformance(
        PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4, 5],
            Validity::Array(
                BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
            ),
        )
        .as_ref(),
    );
}
