// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;

use super::common::create_basic_fsl;
use super::common::create_empty_fsl;
use super::common::create_large_fsl;
use super::common::create_nullable_fsl;
use super::common::create_single_element_fsl;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builders::ArrayBuilder;
use crate::builders::FixedSizeListBuilder;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

// Conformance tests for common take scenarios.
#[rstest]
#[case::basic(create_basic_fsl())]
#[case::nullable(create_nullable_fsl())]
#[case::large(create_large_fsl())]
#[case::single_element(create_single_element_fsl())]
#[case::empty(create_empty_fsl())]
fn test_take_fsl_conformance(#[case] fsl: FixedSizeListArray) {
    test_take_conformance(&fsl.into_array());
}

// FSL-specific edge case tests that aren't covered by conformance.

#[test]
fn test_take_basic_smoke_test() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let indices = buffer![2u32, 0, 1].into_array();
    let result = fsl.take(indices).unwrap();

    // Expected: [[5,6], [1,2], [3,4]]
    let expected = FixedSizeListArray::new(
        buffer![5i32, 6, 1, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        3,
    );
    assert_arrays_eq!(expected, result);
}

// Parameterized test for FSL-specific degenerate (list_size=0) cases.
#[rstest]
#[case::degenerate_non_null(
    Validity::NonNullable,
    vec![Some(3u32), Some(1), Some(4), Some(0), Some(2)],
    5,
    vec![false; 5]
)]
#[case::degenerate_with_nulls(
    Validity::from_iter([true, false, true, true, false]),
    vec![Some(1u32), Some(3), None, Some(0)],
    4,
    vec![true, false, true, false]
)]
#[case::degenerate_all_null(
    Validity::AllInvalid,
    vec![Some(2u32), Some(0), Some(1)],
    3,
    vec![true, true, true]
)]
fn test_take_degenerate_lists(
    #[case] validity: Validity,
    #[case] indices: Vec<Option<u32>>,
    #[case] expected_len: usize,
    #[case] expected_nulls: Vec<bool>,
) {
    // Create a degenerate FSL array with list_size = 0.
    // This is a specific edge case for FSL where lists have no elements.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, validity, 5);

    test_take_conformance(&fsl.clone().into_array());

    // Also test the specific behavior.
    let indices_array = PrimitiveArray::from_option_iter(indices);
    let result = fsl.take(indices_array.into_array()).unwrap();

    assert_eq!(result.len(), expected_len);
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(result.scalar_at(i).unwrap().is_null(), *expected_null);
    }
}

#[test]
fn test_take_large_list_size() {
    let elements = buffer![0i32..300].into_array();
    let fsl = FixedSizeListArray::new(elements, 100, Validity::NonNullable, 3);

    let indices = buffer![2u16, 0].into_array();
    let result = fsl.take(indices).unwrap();

    // Expected: [[200..300], [0..100]]
    let expected_elems = PrimitiveArray::from_iter((200i32..300).chain(0..100)).into_array();
    let expected = FixedSizeListArray::new(expected_elems, 100, Validity::NonNullable, 2);
    assert_arrays_eq!(expected, result);
}

#[test]
fn test_take_fsl_with_null_indices_preserves_elements() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Indices with nulls: [1, null, 0].
    let indices = PrimitiveArray::from_option_iter([Some(1u32), None, Some(0)]);
    let result = fsl.take(indices.into_array()).unwrap();

    // Expected: [[3,4], null, [1,2]]
    let expected = FixedSizeListArray::new(
        buffer![3i32, 4, 0, 0, 1, 2].into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );
    assert_arrays_eq!(expected, result);
}

// Element index overflow: with u8 indices and list_size=16, data_idx=16 produces element index
// 16*16=256 which overflows u8. The take kernel must widen the element index type.
#[rstest]
#[case::non_nullable(
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(0u32..320).into_array(), 16, Validity::NonNullable, 20,
    ),
    buffer![0u8, 16, 5].into_array(),
    FixedSizeListArray::new(
        PrimitiveArray::from_iter((0u32..16).chain(256..272).chain(80..96)).into_array(),
        16, Validity::NonNullable, 3,
    ),
)]
#[case::nullable(
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(0u32..320).into_array(), 16,
        Validity::from_iter((0..20).map(|i| i != 5)), 20,
    ),
    buffer![0u8, 16, 5].into_array(),
    FixedSizeListArray::new(
        PrimitiveArray::from_iter((0u32..16).chain(256..272).chain(80..96)).into_array(),
        16, Validity::from_iter([true, true, false]), 3,
    ),
)]
fn test_element_index_overflow(
    #[case] fsl: FixedSizeListArray,
    #[case] indices: ArrayRef,
    #[case] expected: FixedSizeListArray,
) {
    let result = fsl.take(indices).unwrap();
    assert_arrays_eq!(result, expected);
}

// Parameterized test for nullable array scenarios that are specific to FSL's implementation.
#[rstest]
#[case::nullable_mixed_elements(
    vec![Some(vec![1i32, 2]), None, Some(vec![5, 6])],
    vec![Some(2u32), Some(1), Some(0)],
    vec![false, true, false]
)]
#[case::nullable_with_null_indices(
    vec![Some(vec![1i32, 2]), None, Some(vec![5, 6])],
    vec![Some(0u32), None, Some(1), Some(2)],
    vec![false, true, true, false]
)]
fn test_take_nullable_arrays_fsl_specific(
    #[case] array_values: Vec<Option<Vec<i32>>>,
    #[case] indices: Vec<Option<u32>>,
    #[case] expected_nulls: Vec<bool>,
) {
    // Build the nullable FSL array.
    let list_size = if let Some(Some(first)) = array_values.first() {
        u32::try_from(first.len()).unwrap()
    } else {
        2
    };

    let mut builder = FixedSizeListBuilder::with_capacity(
        DType::Primitive(PType::I32, Nullability::NonNullable).into(),
        list_size,
        Nullability::Nullable,
        array_values.len(),
    );

    for value in array_values {
        match value {
            Some(list) => {
                let scalars: Vec<Scalar> = list.into_iter().map(|v| v.into()).collect();
                builder
                    .append_value(
                        Scalar::list(
                            DType::Primitive(PType::I32, Nullability::NonNullable),
                            scalars,
                            Nullability::NonNullable,
                        )
                        .as_list(),
                    )
                    .unwrap();
            }
            None => builder.append_null(),
        }
    }

    let fsl = builder.finish();

    // Create indices (with possible nulls).
    let indices_array = PrimitiveArray::from_option_iter(indices.clone());
    let result = fsl.take(indices_array.into_array()).unwrap();

    assert_eq!(result.len(), indices.len());
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(result.scalar_at(i).unwrap().is_null(), *expected_null);
    }
}
