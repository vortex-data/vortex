// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical decimal wire validation and round trips.
//!
//! These tests call [`Decimal`] directly as well as through session serialization, so
//! [`DecimalPlugin`] cannot hide a regression in the validation of the vtable.
//!
//! [`DecimalPlugin`]: crate::arrays::decimal::DecimalPlugin

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

/// Serializes through the session, then decodes using the original logical dtype and length.
fn round_trip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();

    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(buffer.as_ref());
    }

    SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}

#[rstest]
#[case::first_unused(6)]
#[case::negative(-1)]
#[case::max(i32::MAX)]
fn test_deserialize_rejects_unknown_values_type(#[case] raw: i32) -> VortexResult<()> {
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

    Ok(())
}

#[rstest]
#[case::i32_nullable(
    DecimalArray::new(
        buffer![1_234_567i32, 0, -9_999_999],
        DecimalDType::new(7, 3),
        Validity::from_iter([true, false, true]),
    ),
    DecimalType::I32
)]
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
    let array = array.into_array();
    let decoded = round_trip(&array)?;

    assert_eq!(decoded.as_::<Decimal>().values_type(), values_type);
    assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());

    Ok(())
}
