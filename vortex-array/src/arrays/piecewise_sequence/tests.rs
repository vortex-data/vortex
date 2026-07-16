// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

fn piecewise_indices(
    starts: impl IntoIterator<Item = u64>,
    lengths: &[u64],
) -> VortexResult<ArrayRef> {
    let len = lengths
        .iter()
        .map(|&length| -> usize { length.as_() })
        .sum();
    let starts = PrimitiveArray::from_iter(starts).into_array();
    let lengths = PrimitiveArray::from_iter(lengths.iter().copied()).into_array();
    let multipliers = ConstantArray::new(1u64, lengths.len()).into_array();
    Ok(PiecewiseSequenceArray::try_new(starts, lengths, multipliers, len)?.into_array())
}

#[test]
fn materializes_piecewise_indices() -> VortexResult<()> {
    let starts = buffer![3u64, 15, 21].into_array();
    let lengths = buffer![3u64, 3, 3].into_array();
    let multipliers = ConstantArray::new(1u64, 3).into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 9)?.into_array();

    let expected = PrimitiveArray::from_iter([3u64, 4, 5, 15, 16, 17, 21, 22, 23]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn materializes_repeated_and_empty_ranges() -> VortexResult<()> {
    let starts = buffer![5u64, 2, 5].into_array();
    let lengths = buffer![2u64, 0, 2].into_array();
    let multipliers = ConstantArray::new(1u64, 3).into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 4)?.into_array();

    let expected = PrimitiveArray::from_iter([5u64, 6, 5, 6]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn materializes_multiplied_ranges() -> VortexResult<()> {
    let starts = buffer![3u64, 15].into_array();
    let lengths = buffer![3u64, 2].into_array();
    let multipliers = buffer![2u64, 4].into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 5)?.into_array();

    let expected = PrimitiveArray::from_iter([3u64, 5, 7, 15, 19]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn supports_constant_lengths() -> VortexResult<()> {
    let starts = buffer![0u64, 10, 20].into_array();
    let lengths = ConstantArray::new(2u64, 3).into_array();
    let multipliers = ConstantArray::new(1u64, 3).into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 6)?.into_array();

    let expected = PrimitiveArray::from_iter([0u64, 1, 10, 11, 20, 21]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn scalar_at_maps_into_piece() -> VortexResult<()> {
    let starts = buffer![3u64, 15, 21].into_array();
    let lengths = buffer![3u64, 3, 3].into_array();
    let multipliers = buffer![1u64, 1, 1].into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 9)?.into_array();
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(array.execute_scalar(0, &mut ctx)?, 3u64.into());
    assert_eq!(array.execute_scalar(4, &mut ctx)?, 16u64.into());
    assert_eq!(array.execute_scalar(8, &mut ctx)?, 23u64.into());
    Ok(())
}

#[test]
fn constructor_defers_range_value_validation() -> VortexResult<()> {
    let starts = buffer![u64::MAX].into_array();
    let lengths = buffer![2u64].into_array();
    let multipliers = buffer![1u64].into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 2)?.into_array();

    let err = array
        .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("PiecewiseSequenceArray range overflows usize"),
        "{err}"
    );
    Ok(())
}

#[test]
fn execution_checks_declared_length() -> VortexResult<()> {
    let starts = buffer![0u64, 3].into_array();
    let lengths = buffer![2u64, 2].into_array();
    let multipliers = ConstantArray::new(1u64, 2).into_array();
    let array = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 3)?.into_array();

    let err = array
        .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("PiecewiseSequenceArray expanded length 4 does not match declared length 3"),
        "{err}"
    );
    Ok(())
}

#[test]
fn serde_roundtrip_preserves_piecewise_indices() -> VortexResult<()> {
    let array = PiecewiseSequenceArray::try_new(
        buffer![3u32, 15, 21].into_array(),
        buffer![2u16, 0, 2].into_array(),
        buffer![2u8, 1, 3].into_array(),
        4,
    )?
    .into_array();
    let dtype = array.dtype().clone();
    let len = array.len();

    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, &array_session(), &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buffer in serialized {
        concat.extend_from_slice(buffer.as_ref());
    }

    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &array_session(),
    )?;

    assert!(decoded.is::<PiecewiseSequence>());
    assert_arrays_eq!(
        decoded,
        PrimitiveArray::from_iter([3u64, 5, 21, 24]).into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn primitive_take_consumes_piecewise_indices() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter(0i32..20).into_array();
    let taken = values.take(piecewise_indices([3, 10], &[2, 3])?)?;

    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_iter([3i32, 4, 10, 11, 12]).into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn primitive_take_handles_non_unit_multiplier() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter(0i32..20).into_array();
    let indices = PiecewiseSequenceArray::try_new(
        buffer![3u64].into_array(),
        buffer![3u64].into_array(),
        buffer![2u64].into_array(),
        3,
    )?
    .into_array();
    let taken = values.take(indices)?;

    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_iter([3i32, 5, 7]).into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn bool_take_consumes_piecewise_indices() -> VortexResult<()> {
    let values = BoolArray::from_iter([true, false, true, true, false, false]).into_array();
    let taken = values.take(piecewise_indices([1, 4], &[2, 2])?)?;

    assert_arrays_eq!(
        taken,
        BoolArray::from_iter([false, true, false, false]).into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn decimal_take_consumes_piecewise_indices() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(19, 2);
    let values = DecimalArray::from_iter([10i128, 20, 30, 40, 50, 60], decimal_dtype).into_array();
    let taken = values.take(piecewise_indices([1, 4], &[2, 1])?)?;

    assert_arrays_eq!(
        taken,
        DecimalArray::from_iter([20i128, 30, 50], decimal_dtype).into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn varbinview_take_consumes_piecewise_indices() -> VortexResult<()> {
    let values = VarBinViewArray::from_iter(
        ["a", "bb", "ccc", "dddd", "eeeee"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let taken = values.take(piecewise_indices([1, 3], &[2, 1])?)?;

    assert_arrays_eq!(
        taken,
        VarBinViewArray::from_iter(
            ["bb", "ccc", "dddd"].map(Some),
            DType::Utf8(Nullability::NonNullable)
        )
        .into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn varbin_take_consumes_piecewise_indices() -> VortexResult<()> {
    let values = VarBinArray::from_iter(
        ["a", "bb", "ccc", "dddd", "eeeee"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let taken = values.take(piecewise_indices([1, 3], &[2, 1])?)?;

    assert_arrays_eq!(
        taken,
        VarBinArray::from_iter(
            ["bb", "ccc", "dddd"].map(Some),
            DType::Utf8(Nullability::NonNullable)
        )
        .into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn fixed_size_list_take_builds_piecewise_element_indices() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter(0i32..12).into_array();
    let array = FixedSizeListArray::new(elements, 3, Validity::NonNullable, 4).into_array();
    let taken = array.take(buffer![1u64, 3].into_array())?;

    let expected_elements = PrimitiveArray::from_iter([3i32, 4, 5, 9, 10, 11]).into_array();
    let expected =
        FixedSizeListArray::new(expected_elements, 3, Validity::NonNullable, 2).into_array();
    assert_arrays_eq!(taken, expected, &mut array_session().create_execution_ctx());
    Ok(())
}
