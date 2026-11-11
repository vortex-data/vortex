// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::*;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_repeated_probe_preserves_masked_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = MaskedArray::try_new(
        PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
        Validity::from_iter([true, false, true]),
        &mut ctx,
    )?
    .into_array();
    let expected = [Some(1i32), None, Some(3)];
    let mut probe = array.repeated_probe();
    for index in [1, 2, 0, 1] {
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            Scalar::from(expected[index])
        );
        assert_eq!(
            array.execute_scalar(index, &mut ctx)?,
            Scalar::from(expected[index])
        );
    }
    Ok(())
}

#[rstest]
#[case(Validity::AllValid, Nullability::Nullable)]
#[case(Validity::from_iter([true, false, true]), Nullability::Nullable)]
fn test_dtype_nullability(#[case] validity: Validity, #[case] expected: Nullability) {
    let mut ctx = SESSION.create_execution_ctx();
    let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = MaskedArray::try_new(child, validity, &mut ctx).unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Primitive(crate::dtype::PType::I32, expected)
    );
}

#[test]
fn test_dtype_nullability_with_nullable_child() -> VortexResult<()> {
    let child =
        PrimitiveArray::new(vortex_buffer::buffer![1i32, 2, 3], Validity::AllValid).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let array = MaskedArray::try_new(child, Validity::from_iter([true, false, true]), &mut ctx)?;
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_empty_child_with_array_validity() -> VortexResult<()> {
    let child_validity =
        Validity::Array(BoolArray::new(BitBuffer::new_set(0), Validity::NonNullable).into_array());
    let child = PrimitiveArray::new(Buffer::<i32>::empty(), child_validity).into_array();
    let validity =
        Validity::Array(BoolArray::new(BitBuffer::new_set(0), Validity::NonNullable).into_array());

    let mut ctx = SESSION.create_execution_ctx();
    assert!(child.all_valid(&mut ctx)?);
    assert!(child.all_invalid(&mut ctx)?);

    let array = MaskedArray::try_new(child, validity, &mut ctx)?;

    assert!(array.is_empty());
    Ok(())
}

#[test]
fn test_masked_child_with_validity() -> VortexResult<()> {
    let child = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array();
    let array = MaskedArray::try_new(
        child,
        Validity::from_iter([true, false, true, false, true]),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn test_masked_child_all_valid() {
    let mut ctx = SESSION.create_execution_ctx();
    let child = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let array = MaskedArray::try_new(child, Validity::AllValid, &mut ctx).unwrap();

    assert_eq!(array.len(), 3);
    assert_eq!(
        array
            .valid_count(&mut SESSION.create_execution_ctx())
            .unwrap(),
        3
    );
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_option_iter([10i32, 20, 30].map(Some)),
        &mut ctx
    );
}

#[rstest]
#[case(Validity::AllValid)]
#[case(Validity::from_iter([true, true, true]))]
#[case(Validity::from_iter([false, false, false]))]
#[case(Validity::from_iter([true, false, true, false]))]
fn test_masked_child_preserves_length(#[case] validity: Validity) {
    let mut ctx = SESSION.create_execution_ctx();
    let len = match &validity {
        Validity::Array(arr) => arr.len(),
        _ => 3,
    };

    #[expect(clippy::cast_possible_truncation)]
    let child = PrimitiveArray::from_iter(0..len as i32).into_array();
    let array = MaskedArray::try_new(child, validity.clone(), &mut ctx).unwrap();

    assert_eq!(array.len(), len);

    let mut ctx = SESSION.create_execution_ctx();
    assert!(
        array
            .validity()
            .vortex_expect("masked validity should be derivable")
            .mask_eq(&validity, array.len(), &mut ctx)
            .unwrap(),
    );
}

#[test]
fn masked_listview_execute_preserves_zctl_true() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Masking only intersects validity, so the zero-copy-to-list survives
    // execution.
    let elements = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let offsets = PrimitiveArray::from_iter([0i32, 2]).into_array();
    let sizes = PrimitiveArray::from_iter([2i32, 1]).into_array();
    let list_view = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    };

    let masked = MaskedArray::try_new(
        list_view.into_array(),
        Validity::from_iter([true, false]),
        &mut ctx,
    )?;
    let canonical = masked
        .into_array()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
    assert!(canonical.into_listview().is_zero_copy_to_list());
    Ok(())
}
