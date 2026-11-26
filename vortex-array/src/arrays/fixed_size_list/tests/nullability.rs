// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_nullable_fsl_with_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 4;
    let list_size = 2;

    // Create FSL with some null lists.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let validity = Validity::from_iter([true, false, true, false]);
    let fsl = FixedSizeListArray::new(elements, list_size, validity, len);

    assert_eq!(fsl.len(), len);
    assert_eq!(fsl.list_size(), list_size);

    // Null-row payload values must be ignored: compare against an array with zeroed null slots.
    let expected = FixedSizeListArray::new(
        buffer![1i32, 2, 0, 0, 5, 6, 0, 0].into_array(),
        list_size,
        Validity::from_iter([true, false, true, false]),
        len,
    );
    assert_arrays_eq!(fsl, expected, &mut ctx);

    // Check individual valid list slices via fixed_size_list_elements_at.
    assert_arrays_eq!(
        fsl.fixed_size_list_elements_at(0)?,
        buffer![1i32, 2],
        &mut ctx
    );
    assert_arrays_eq!(
        fsl.fixed_size_list_elements_at(2)?,
        buffer![5i32, 6],
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_nullable_elements_non_nullable_lists() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 2;
    let list_size = 3;

    // Elements array has nulls but the FSL itself is non-nullable.
    let elements =
        PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(3), Some(4), Some(5), None]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), len);

    // Check dtype - FSL is non-nullable but elements are nullable.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(elem_dtype, 3, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::I32, Nullability::Nullable))
    ));

    assert_arrays_eq!(
        fsl.fixed_size_list_elements_at(0)?,
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]),
        &mut ctx
    );
    assert_arrays_eq!(
        fsl.fixed_size_list_elements_at(1)?,
        PrimitiveArray::from_option_iter([Some(4i32), Some(5), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_nullable_elements_and_nullable_lists() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 3;
    let list_size = 2;

    // Both elements and lists can be null.
    let elements =
        PrimitiveArray::from_option_iter(vec![Some(10u16), None, Some(20), Some(30), None, None]);
    let validity = Validity::from_iter([true, false, true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), len);

    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_option_iter([Some(10u16), None, None, None, None, None]).into_array(),
        list_size,
        Validity::from_iter([true, false, true]),
        len,
    );
    assert_arrays_eq!(fsl, expected, &mut ctx);

    assert_arrays_eq!(
        fsl.fixed_size_list_elements_at(0)?,
        PrimitiveArray::from_option_iter([Some(10u16), None]),
        &mut ctx
    );
    assert_arrays_eq!(
        fsl.fixed_size_list_elements_at(2)?,
        PrimitiveArray::from_option_iter([None::<u16>, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_alternating_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 6;
    let list_size = 1;

    // Alternating null and valid single-element lists.
    let elements = buffer![1u8, 2, 3, 4, 5, 6].into_array();
    let validity = Validity::from_iter([true, false, true, false, true, false]);
    let fsl = FixedSizeListArray::new(elements, list_size, validity, len);

    assert_eq!(fsl.len(), len);

    let expected = FixedSizeListArray::new(
        buffer![1u8, 0, 3, 0, 5, 0].into_array(),
        list_size,
        Validity::from_iter([true, false, true, false, true, false]),
        len,
    );
    assert_arrays_eq!(fsl, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_validity_types() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 4;
    let list_size = 2;

    // Test with different validity buffer configurations.
    let elements = buffer![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_array();

    // Test with AllInvalid.
    {
        let fsl = FixedSizeListArray::new(elements.clone(), list_size, Validity::AllInvalid, len);
        let expected = ConstantArray::new(
            Scalar::null(DType::FixedSizeList(
                Arc::new(PType::F32.into()),
                list_size,
                Nullability::Nullable,
            )),
            len,
        );
        assert_arrays_eq!(fsl, expected, &mut ctx);
    }

    // Test with Array validity.
    {
        let validity_array = BoolArray::from_iter([true, true, false, true]);
        let fsl = FixedSizeListArray::new(
            elements,
            list_size,
            Validity::Array(validity_array.into_array()),
            len,
        );
        let expected = FixedSizeListArray::new(
            buffer![1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0].into_array(),
            list_size,
            Validity::from_iter([true, true, false, true]),
            len,
        );
        assert_arrays_eq!(fsl, expected, &mut ctx);
    }
    Ok(())
}

#[rstest]
#[case::null_list_null_elements([None, None])]
#[case::null_list_non_null_elements([Some(99), Some(100)])]
fn test_mixed_nullability_patterns(
    #[case] null_list_elements: [Option<i16>; 2],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 5;
    let list_size = 2;

    // Complex nullability pattern.
    let elements = PrimitiveArray::from_option_iter(vec![
        Some(1i16), // List 0
        None,
        null_list_elements[0], // List 1 (null list)
        null_list_elements[1],
        Some(5), // List 2
        Some(6),
        Some(7), // List 3
        None,
        None, // List 4
        Some(10),
    ]);
    let validity = Validity::from_iter([true, false, true, true, true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert!(fsl.execute_scalar(1, &mut ctx)?.is_null());

    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_option_iter([
            Some(1i16),
            None,
            None,
            None,
            Some(5),
            Some(6),
            Some(7),
            None,
            None,
            Some(10),
        ])
        .into_array(),
        list_size,
        Validity::from_iter([true, false, true, true, true]),
        len,
    );
    assert_arrays_eq!(fsl, expected, &mut ctx);
    Ok(())
}
