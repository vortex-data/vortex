// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::search_sorted::rstest_reuse::apply;
use crate::compute::conformance::search_sorted::search_sorted_conformance;
use crate::compute::conformance::search_sorted::*;
use crate::executor::VortexSessionExecute;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedPrimitiveArray;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

#[apply(search_sorted_conformance)]
fn test_search_sorted_primitive(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) {
    let res = SearchSortedPrimitiveArray::<i32>::new(&array, &mut SESSION.create_execution_ctx())
        .search_sorted(&value, side)
        .vortex_expect("Failed to search sorted");
    assert_eq!(res, expected);
}

#[test]
fn test_mask_primitive_array() {
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllInvalid).into_array(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
        )
        .into_array(),
    );
}

#[test]
fn test_filter_primitive_array() {
    // Test various sizes
    test_filter_conformance(
        &PrimitiveArray::new(buffer![42i32], Validity::NonNullable).into_array(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1], Validity::NonNullable).into_array(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4, 5, 6, 7], Validity::NonNullable).into_array(),
    );

    // Test with validity
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4, 5],
            Validity::Array(
                BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
            ),
        )
        .into_array(),
    );
}
