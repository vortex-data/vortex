// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical decimal wire validation and round trips.
//!
//! These tests exercise the canonical reader directly alongside session serialization, so the
//! compatibility plugin cannot hide a vtable validation regression.

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::DecimalMetadata;
use crate::ArrayContext;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::VTable;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_array_serde() -> VortexResult<()> {
    let array = DecimalArray::new(
        buffer![100i64, 200i64, 300i64, 400i64, 500i64],
        DecimalDType::new(10, 2),
        Validity::NonNullable,
    );
    let dtype = array.dtype().clone();

    let array_ctx = ArrayContext::empty();
    let out = array
        .into_array()
        .serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in out {
        concat.extend_from_slice(buf.as_ref());
    }

    let concat = concat.freeze();

    let parts = SerializedArray::try_from(concat)?;
    let decoded = parts.decode(&dtype, 5, &ReadContext::new(array_ctx.to_ids()), &SESSION)?;
    assert!(decoded.is::<Decimal>());
    assert_arrays_eq!(
        decoded,
        DecimalArray::from_iter([100i64, 200, 300, 400, 500], DecimalDType::new(10, 2)),
        &mut SESSION.create_execution_ctx()
    );

    Ok(())
}

#[test]
fn test_nullable_decimal_serde_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = DecimalArray::new(
        buffer![1234567i32, 0i32, -9999999i32],
        DecimalDType::new(7, 3),
        Validity::from_iter([true, false, true]),
    );
    let dtype = array.dtype().clone();
    let len = array.len();

    let array_ctx = ArrayContext::empty();
    let out =
        array
            .clone()
            .into_array()
            .serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in out {
        concat.extend_from_slice(buf.as_ref());
    }

    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &SESSION)?;

    assert_arrays_eq!(decoded, array, &mut ctx);

    Ok(())
}

#[rstest]
#[case::first_unused(6)]
#[case::negative(-1)]
#[case::max(i32::MAX)]
fn test_deserialize_rejects_unknown_values_type(#[case] raw: i32) {
    let metadata = DecimalMetadata { values_type: raw }.encode_to_vec();
    let buffers = [BufferHandle::new_host(ByteBuffer::zeroed(4))];
    let children: Vec<ArrayRef> = Vec::new();

    let result = Decimal.deserialize(
        &DType::Decimal(DecimalDType::new(2, 0), Nullability::NonNullable),
        4,
        &metadata,
        &buffers,
        &children,
        &SESSION,
    );

    let expected = format!("DecimalMetadata values_type must be a known DecimalType, got {raw}");
    assert!(result.is_err_and(|err| err.to_string().contains(&expected)));
}

#[rstest]
#[case::i8(
    DecimalArray::from_iter([1i8, -2, 3], DecimalDType::new(2, 0)),
    DecimalType::I8
)]
#[case::i16(
    DecimalArray::from_iter([1i16, -2, 3], DecimalDType::new(4, 0)),
    DecimalType::I16
)]
#[case::i32(
    DecimalArray::from_iter([1i32, -2, 3], DecimalDType::new(9, 2)),
    DecimalType::I32
)]
#[case::i64(
    DecimalArray::from_iter([1i64, -2, 3], DecimalDType::new(18, 2)),
    DecimalType::I64
)]
#[case::i128(
    DecimalArray::from_iter([1i128, -2, 3], DecimalDType::new(38, 2)),
    DecimalType::I128
)]
#[case::i256(
    DecimalArray::from_iter(
        [i256::from_i128(1), i256::from_i128(-2), i256::from_i128(3)],
        DecimalDType::new(76, 2),
    ),
    DecimalType::I256
)]
fn test_serde_each_values_type(
    #[case] array: DecimalArray,
    #[case] values_type: DecimalType,
) -> VortexResult<()> {
    let dtype = array.dtype().clone();
    let len = array.len();

    let array_ctx = ArrayContext::empty();
    let out =
        array
            .clone()
            .into_array()
            .serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in out {
        concat.extend_from_slice(buf.as_ref());
    }

    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &SESSION)?;

    assert_eq!(decoded.as_::<Decimal>().values_type(), values_type);
    assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());

    Ok(())
}
