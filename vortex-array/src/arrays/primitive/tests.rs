// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::search_sorted::rstest_reuse::apply;
use crate::compute::conformance::search_sorted::search_sorted_conformance;
use crate::compute::conformance::search_sorted::*;
use crate::scalar::PValue;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

#[apply(search_sorted_conformance)]
fn test_search_sorted_primitive(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) -> vortex_error::VortexResult<()> {
    let res = array
        .as_primitive_typed()
        .search_sorted(&Some(PValue::from(value)), side)?;
    assert_eq!(res, expected);
    Ok(())
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
