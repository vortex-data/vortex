// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use super::*;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::VTable;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::normalize::NormalizeOptions;
use crate::normalize::Operation;
use crate::validity::Validity;

#[rstest]
#[case(Validity::AllValid, Nullability::Nullable)]
#[case(Validity::from_iter([true, false, true]), Nullability::Nullable)]
fn test_dtype_nullability(#[case] validity: Validity, #[case] expected: Nullability) {
    let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = MaskedArray::try_new(child, validity).unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Primitive(crate::dtype::PType::I32, expected)
    );
}

#[test]
fn test_dtype_nullability_with_nullable_child() {
    // Child can have nullable dtype but no actual nulls.
    // MaskedArray dtype should be determined by validity, not child's dtype.
    let child =
        PrimitiveArray::new(vortex_buffer::buffer![1i32, 2, 3], Validity::AllValid).into_array();

    // Child has nullable dtype.
    assert!(child.dtype().is_nullable());
}

#[test]
fn test_empty_child_with_array_validity() -> VortexResult<()> {
    let child_validity =
        Validity::Array(BoolArray::new(BitBuffer::new_set(0), Validity::NonNullable).into_array());
    let child = PrimitiveArray::new(Buffer::<i32>::empty(), child_validity).into_array();
    let validity =
        Validity::Array(BoolArray::new(BitBuffer::new_set(0), Validity::NonNullable).into_array());

    let mut ctx = array_session().create_execution_ctx();
    assert!(child.all_valid(&mut ctx)?);
    assert!(child.all_invalid(&mut ctx)?);

    let array = MaskedArray::try_new(child, validity)?;

    assert!(array.is_empty());
    Ok(())
}

#[test]
fn test_canonical_dtype_matches_array_dtype() -> VortexResult<()> {
    // The canonical form should have the same nullability as the array's dtype.
    let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = MaskedArray::try_new(child, Validity::AllValid)?;

    let canonical = array
        .clone()
        .into_array()
        .execute::<Canonical>(&mut array_session().create_execution_ctx())?;
    assert_eq!(canonical.dtype(), array.dtype());
    Ok(())
}

#[test]
fn test_masked_child_with_validity() {
    // When validity has nulls, masked_child should apply inverted mask.
    let child = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array();
    let array =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true, false, true])).unwrap();

    // Positions where validity is false should be null in masked_child.
    let mut ctx = array_session().create_execution_ctx();
    let prim = array
        .as_array()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    assert_eq!(prim.valid_count(&mut ctx).unwrap(), 3);
    assert!(
        prim.is_valid(0, &mut array_session().create_execution_ctx())
            .unwrap()
    );
    assert!(
        !prim
            .is_valid(1, &mut array_session().create_execution_ctx())
            .unwrap()
    );
    assert!(
        prim.is_valid(2, &mut array_session().create_execution_ctx())
            .unwrap()
    );
    assert!(
        !prim
            .is_valid(3, &mut array_session().create_execution_ctx())
            .unwrap()
    );
    assert!(
        prim.is_valid(4, &mut array_session().create_execution_ctx())
            .unwrap()
    );
}

#[test]
fn test_masked_child_all_valid() {
    let mut ctx = array_session().create_execution_ctx();
    // When validity is AllValid, masked_child should invert to AllInvalid.
    let child = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let array = MaskedArray::try_new(child, Validity::AllValid).unwrap();

    assert_eq!(array.len(), 3);
    assert_eq!(
        array
            .valid_count(&mut array_session().create_execution_ctx())
            .unwrap(),
        3
    );
    assert_arrays_eq!(
        PrimitiveArray::from_option_iter([10i32, 20, 30].map(Some)),
        array,
        &mut ctx
    );
}

#[rstest]
#[case(Validity::AllValid)]
#[case(Validity::from_iter([true, true, true]))]
#[case(Validity::from_iter([false, false, false]))]
#[case(Validity::from_iter([true, false, true, false]))]
fn test_masked_child_preserves_length(#[case] validity: Validity) {
    let len = match &validity {
        Validity::Array(arr) => arr.len(),
        _ => 3,
    };

    #[expect(clippy::cast_possible_truncation)]
    let child = PrimitiveArray::from_iter(0..len as i32).into_array();
    let array = MaskedArray::try_new(child, validity.clone()).unwrap();

    assert_eq!(array.len(), len);

    let mut ctx = array_session().create_execution_ctx();
    assert!(
        array
            .validity()
            .vortex_expect("masked validity should be derivable")
            .mask_eq(&validity, array.len(), &mut ctx)
            .unwrap(),
    );
}

#[test]
fn test_null_child_merges_validity_lazily() -> VortexResult<()> {
    let child = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array();
    let mask = Validity::from_iter([true, true, false, true]);
    let array = MaskedArray::try_new(child, mask)?;

    // The logical validity is the child's validity ANDed with the mask.
    let mut ctx = array_session().create_execution_ctx();
    let expected = Validity::from_iter([true, false, false, true]);
    assert!(
        array
            .validity()?
            .mask_eq(&expected, array.len(), &mut ctx)?
    );

    assert_arrays_eq!(
        PrimitiveArray::from_option_iter([Some(1i32), None, None, Some(4)]),
        array,
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_null_child_serialize_fails() -> VortexResult<()> {
    let child = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let array = MaskedArray::try_new(child, Validity::from_iter([true, true, false]))?;

    let err = array
        .into_array()
        .serialize(
            &crate::ArrayContext::empty(),
            &array_session(),
            &crate::serde::SerializeOptions::default(),
        )
        .expect_err("serializing a null-carrying child must fail");
    assert!(err.to_string().contains("normalize"), "unexpected: {err}");
    Ok(())
}

#[test]
fn test_normalize_removes_masked_with_null_child() -> VortexResult<()> {
    let child = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let array = MaskedArray::try_new(child, Validity::from_iter([true, true, false]))?.into_array();

    // Even though Masked is an allowed encoding, a null-carrying child forces the mask to be
    // executed into the child, removing the wrapper.
    let allowed = HashSet::from_iter([Masked.id()]);
    let mut ctx = array_session().create_execution_ctx();
    let normalized = array.normalize(&mut NormalizeOptions {
        allowed: &allowed,
        operation: Operation::Execute(&mut ctx),
    })?;

    assert!(!normalized.is::<Masked>());
    assert_arrays_eq!(
        PrimitiveArray::from_option_iter([Some(1i32), None, None]),
        normalized,
        &mut ctx
    );

    // A masked array with a null-free child stays untouched.
    let clean = MaskedArray::try_new(
        PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
        Validity::from_iter([true, false, true]),
    )?
    .into_array();
    let normalized = clean.clone().normalize(&mut NormalizeOptions {
        allowed: &allowed,
        operation: Operation::Execute(&mut ctx),
    })?;
    assert!(crate::ArrayRef::ptr_eq(&clean, &normalized));
    Ok(())
}

#[test]
fn masked_listview_execute_preserves_zctl_true() -> VortexResult<()> {
    // Masking only intersects validity, so the zero-copy-to-list survives
    // execution.
    let elements = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let offsets = PrimitiveArray::from_iter([0i32, 2]).into_array();
    let sizes = PrimitiveArray::from_iter([2i32, 1]).into_array();
    let list_view = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    };

    let masked = MaskedArray::try_new(list_view.into_array(), Validity::from_iter([true, false]))?;
    let canonical = masked
        .into_array()
        .execute::<Canonical>(&mut array_session().create_execution_ctx())?;
    assert!(canonical.into_listview().is_zero_copy_to_list());
    Ok(())
}
