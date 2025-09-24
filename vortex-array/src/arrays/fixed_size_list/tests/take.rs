// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use super::common::{
    create_basic_fsl, create_empty_fsl, create_large_fsl, create_nullable_fsl,
    create_single_element_fsl,
};
use crate::arrays::{FixedSizeListArray, FixedSizeListVTable, PrimitiveArray};
use crate::builders::{ArrayBuilder, FixedSizeListBuilder};
use crate::compute::conformance::take::test_take_conformance;
use crate::compute::take;
use crate::validity::Validity;
use crate::{Array, IntoArray};

// Conformance tests for common take scenarios.
#[rstest]
#[case::basic(create_basic_fsl())]
#[case::nullable(create_nullable_fsl())]
#[case::large(create_large_fsl())]
#[case::single_element(create_single_element_fsl())]
#[case::empty(create_empty_fsl())]
fn test_take_fsl_conformance(#[case] fsl: FixedSizeListArray) {
    test_take_conformance(fsl.as_ref());
}

// FSL-specific edge case tests that aren't covered by conformance.

#[test]
fn test_take_basic_smoke_test() {
    // Basic smoke test to ensure take works for FSL and preserves structure.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let indices = buffer![2u32, 0, 1].into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3, "Wrong number of lists after take");
    assert_eq!(result_fsl.list_size(), 2, "list_size should be preserved");

    // First list should be the original third list [5, 6].
    let first = result_fsl.fixed_size_list_elements_at(0);
    assert_eq!(
        first.scalar_at(0),
        5i32.into(),
        "Wrong value at [2][0] after take"
    );
    assert_eq!(
        first.scalar_at(1),
        6i32.into(),
        "Wrong value at [2][1] after take"
    );

    // Second list should be the original first list [1, 2].
    let second = result_fsl.fixed_size_list_elements_at(1);
    assert_eq!(second.scalar_at(0), 1i32.into());
    assert_eq!(second.scalar_at(1), 2i32.into());

    // Third list should be the original second list [3, 4].
    let third = result_fsl.fixed_size_list_elements_at(2);
    assert_eq!(third.scalar_at(0), 3i32.into());
    assert_eq!(third.scalar_at(1), 4i32.into());
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

    test_take_conformance(fsl.as_ref());

    // Also test the specific behavior.
    let indices_array = PrimitiveArray::from_option_iter(indices);
    let result = take(fsl.as_ref(), indices_array.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), expected_len);
    assert_eq!(result_fsl.list_size(), 0);
    assert_eq!(result_fsl.elements().len(), 0);

    // Check nullability of results.
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(result_fsl.scalar_at(i).is_null(), *expected_null);
    }
}

#[test]
fn test_take_large_list_size() {
    // Test FSL-specific behavior with large list sizes.
    // This tests the performance characteristics specific to FSL's element expansion.
    let elements = buffer![0i32..300].into_array();
    let fsl = FixedSizeListArray::new(elements, 100, Validity::NonNullable, 3);

    let indices = buffer![2u16, 0].into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 2);
    assert_eq!(result_fsl.list_size(), 100);

    // First list should be [200..300].
    let first = result_fsl.fixed_size_list_elements_at(0);
    for i in 0..100i32 {
        assert_eq!(first.scalar_at(i as usize), (200 + i).into());
    }

    // Second list should be [0..100].
    let second = result_fsl.fixed_size_list_elements_at(1);
    for i in 0..100i32 {
        assert_eq!(second.scalar_at(i as usize), i.into());
    }
}

#[test]
fn test_take_fsl_with_null_indices_preserves_elements() {
    // FSL-specific test: verify that null indices don't affect element array indexing.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Create indices with nulls: [1, null, 0].
    let indices = PrimitiveArray::from_option_iter([Some(1u32), None, Some(0)]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);
    assert_eq!(result_fsl.list_size(), 2);

    // First list should be [3, 4].
    assert!(!result_fsl.scalar_at(0).is_null());
    let first = result_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 3i32.into());
    assert_eq!(first.scalar_at(1), 4i32.into());

    // Second list should be null.
    assert!(result_fsl.scalar_at(1).is_null());

    // Third list should be [1, 2].
    assert!(!result_fsl.scalar_at(2).is_null());
    let third = result_fsl.fixed_size_list_elements_at(2);
    assert_eq!(third.scalar_at(0), 1i32.into());
    assert_eq!(third.scalar_at(1), 2i32.into());
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
    let result = take(fsl.as_ref(), indices_array.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), indices.len());

    // Check nullability of results.
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(result_fsl.scalar_at(i).is_null(), *expected_null);
    }
}
