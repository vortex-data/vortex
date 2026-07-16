// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequential;
use crate::arrays::PiecewiseSequentialArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;

#[test]
fn materializes_piecewise_indices() -> VortexResult<()> {
    let starts = buffer![3u64, 15, 21].into_array();
    let lengths = buffer![3u64, 3, 3].into_array();
    let array = PiecewiseSequentialArray::try_new(starts, lengths, 9)?.into_array();

    let expected = PrimitiveArray::from_iter([3u64, 4, 5, 15, 16, 17, 21, 22, 23]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn materializes_repeated_and_empty_ranges() -> VortexResult<()> {
    let starts = buffer![5u64, 2, 5].into_array();
    let lengths = buffer![2u64, 0, 2].into_array();
    let array = PiecewiseSequentialArray::try_new(starts, lengths, 4)?.into_array();

    let expected = PrimitiveArray::from_iter([5u64, 6, 5, 6]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn supports_constant_lengths() -> VortexResult<()> {
    let starts = buffer![0u64, 10, 20].into_array();
    let lengths = ConstantArray::new(2u64, 3).into_array();
    let array = PiecewiseSequentialArray::try_new(starts, lengths, 6)?.into_array();

    let expected = PrimitiveArray::from_iter([0u64, 1, 10, 11, 20, 21]).into_array();
    assert_arrays_eq!(array, expected, &mut array_session().create_execution_ctx());
    Ok(())
}

#[test]
fn scalar_at_maps_into_piece() -> VortexResult<()> {
    let starts = buffer![3u64, 15, 21].into_array();
    let lengths = buffer![3u64, 3, 3].into_array();
    let array = PiecewiseSequentialArray::try_new(starts, lengths, 9)?.into_array();
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
    let array = PiecewiseSequentialArray::try_new(starts, lengths, 2)?.into_array();

    assert!(
        array
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())
            .is_err()
    );
    Ok(())
}

#[test]
fn execution_checks_declared_length() -> VortexResult<()> {
    let starts = buffer![0u64, 3].into_array();
    let lengths = buffer![2u64, 2].into_array();
    let array = PiecewiseSequentialArray::try_new(starts, lengths, 3)?.into_array();

    assert!(
        array
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())
            .is_err()
    );
    Ok(())
}

#[test]
fn serde_roundtrip_preserves_piecewise_indices() -> VortexResult<()> {
    let array = PiecewiseSequentialArray::try_new(
        buffer![3u32, 15, 21].into_array(),
        buffer![2u16, 0, 2].into_array(),
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

    assert!(decoded.is::<PiecewiseSequential>());
    assert_arrays_eq!(
        decoded,
        PrimitiveArray::from_iter([3u64, 4, 21, 22]).into_array(),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}
