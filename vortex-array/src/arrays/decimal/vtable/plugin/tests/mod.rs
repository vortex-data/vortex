// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde of canonical and legacy `vortex.decimal` nodes.
//!
//! Hand-built nodes cover legacy widths without constructing invalid decimal arrays. Their raw
//! slots include nonzero null payloads, and absent validity denotes a non-nullable node.

mod device;

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::DecimalMetadata;
use crate::ArrayContext;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::ArrayDeserialization;
use crate::array::ArrayPlugin;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// An array read from a node, with the address of the buffer of the node to check its reuse.
struct ReadNode {
    array: ArrayRef,
    /// The original host allocation address, used only to check buffer identity.
    stored_ptr: *const u8,
}

/// Reads a node that stores `slots` as `values_type` integers, with the dtype `decimal_dtype`.
fn read_node(
    values_type: DecimalType,
    decimal_dtype: DecimalDType,
    slots: &[i128],
    validity: Option<&[bool]>,
) -> VortexResult<ReadNode> {
    let values = stored_values(values_type, slots);
    let stored_ptr = values.as_host().as_ptr();
    let (nullability, children) = match validity {
        Some(validity) => (
            Nullability::Nullable,
            vec![BoolArray::from_iter(validity.iter().copied()).into_array()],
        ),
        None => (Nullability::NonNullable, Vec::new()),
    };

    let array = deserialize(
        &DType::Decimal(decimal_dtype, nullability),
        slots.len(),
        &metadata(values_type),
        &[values],
        &children,
    )?;

    Ok(ReadNode { array, stored_ptr })
}

/// Deserializes the parts of a node through the [`Decimal`] plugin.
fn deserialize(
    dtype: &DType,
    len: usize,
    metadata: &[u8],
    buffers: &[BufferHandle],
    children: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    let id = ArrayPlugin::id(&Decimal);
    let parts = ArrayDeserialization::new(id, dtype, len, metadata, buffers, &children);

    ArrayPlugin::deserialize(&Decimal, parts, &SESSION)
}

/// Serializes `array` through the session, then decodes it with the dtype `dtype`.
fn serialize_and_decode(array: &ArrayRef, dtype: &DType) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(buffer.as_ref());
    }

    SerializedArray::try_from(bytes.freeze())?.decode(
        dtype,
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}

fn metadata(values_type: DecimalType) -> Vec<u8> {
    DecimalMetadata {
        values_type: values_type as i32,
    }
    .encode_to_vec()
}

/// Returns `slots` as a host buffer of `values_type` integers.
fn stored_values(values_type: DecimalType, slots: &[i128]) -> BufferHandle {
    let bytes = match_each_decimal_value_type!(values_type, |T| {
        slots
            .iter()
            .map(|&slot| <T as BigCast>::from(slot).vortex_expect("a slot fits its stored width"))
            .collect::<Buffer<T>>()
            .into_byte_buffer()
    });

    BufferHandle::new_host(bytes)
}

/// Returns the canonical array of the valid values of a node, with zeros in its null slots.
fn canonical_reference(
    decimal_dtype: DecimalDType,
    slots: &[i128],
    validity: Option<&[bool]>,
) -> VortexResult<DecimalArray> {
    let valid_slots = slots
        .iter()
        .enumerate()
        .map(|(row, &slot)| match validity {
            Some(validity) if !validity[row] => 0,
            _ => slot,
        })
        .collect::<Vec<_>>();
    let validity = match validity {
        Some(validity) => Validity::from_iter(validity.iter().copied()),
        None => Validity::NonNullable,
    };

    let canonical_type = decimal_dtype.canonical_values_type();

    DecimalArray::try_new_handle(
        stored_values(canonical_type, &valid_slots),
        canonical_type,
        decimal_dtype,
        validity,
    )
}

/// Returns the decimal leaf that holds the values of `array`: the child of a `Widen`, or `array`.
fn decimal_leaf(array: &ArrayRef) -> DecimalArray {
    let leaf = match array.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => array.clone(),
    };

    leaf.as_::<Decimal>().into_owned()
}

/// Checks the encoding, the leaf dtype and width, the reuse of the stored buffer, and the values.
///
/// The cases cover each result of a read: a canonical node keeps its buffer, and a legacy node
/// reads as a `Widen` or as the canonical array. The tests of [`WidenArray::from_stored_decimal`]
/// cover the rule that picks the child precision.
///
/// [`WidenArray::from_stored_decimal`]: crate::arrays::WidenArray::from_stored_decimal
#[rstest]
#[case::canonical_reuse(
    DecimalType::I32, DecimalDType::new(9, 2),
    vec![-999_999_999, 0, 999_999_999], None,
    None, true,
)]
#[case::narrow_reuse(
    DecimalType::I16, DecimalDType::new(18, 2),
    vec![-9_999, 0, 9_999], None,
    Some(4), true,
)]
#[case::wider_to_canonical(
    DecimalType::I128, DecimalDType::new(5, 2),
    vec![-99_999, 0, 99_999], None,
    None, false,
)]
#[case::nullable_reuse(
    DecimalType::I16, DecimalDType::new(18, 2),
    vec![-9_999, 32_767, 9_999], Some(vec![true, false, true]),
    Some(4), true,
)]
fn reads_node_at_any_width(
    #[case] values_type: DecimalType,
    #[case] decimal_dtype: DecimalDType,
    #[case] slots: Vec<i128>,
    #[case] validity: Option<Vec<bool>>,
    #[case] child_precision: Option<u8>,
    #[case] reuses_buffer: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let node = read_node(values_type, decimal_dtype, &slots, validity.as_deref())?;

    assert_eq!(node.array.is::<Widen>(), child_precision.is_some());
    let leaf = decimal_leaf(&node.array);
    let leaf_dtype = match child_precision {
        Some(precision) => DecimalDType::new(precision, decimal_dtype.scale()),
        None => decimal_dtype,
    };
    assert_eq!(leaf.decimal_dtype(), leaf_dtype);
    assert_eq!(leaf.values_type(), leaf_dtype.canonical_values_type());

    // A leaf at another width than the stored width is a copy. Two empty buffers can share an
    // address, so only a reuse is checked by address.
    assert_eq!(leaf.values_type() == values_type, reuses_buffer);
    if reuses_buffer {
        assert_eq!(leaf.buffer_handle().as_host().as_ptr(), node.stored_ptr);
    }

    assert_arrays_eq!(
        node.array,
        canonical_reference(decimal_dtype, &slots, validity.as_deref())?,
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::first_unused(6)]
#[case::negative(-1)]
#[case::max(i32::MAX)]
fn unknown_values_type_fails(#[case] raw: i32) -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(2, 0), Nullability::NonNullable);
    let metadata = DecimalMetadata { values_type: raw }.encode_to_vec();
    let values = BufferHandle::new_host(ByteBuffer::zeroed(4));

    let result = deserialize(&dtype, 4, &metadata, &[values], &[]);

    let expected = format!("DecimalMetadata values_type must be a known DecimalType, got {raw}");
    assert!(result.is_err_and(|error| error.to_string().contains(&expected)));

    Ok(())
}

/// Wire length and alignment must be checked before reading the stored integers.
#[rstest]
#[case::short_buffer(18, DecimalType::I32, 4, stored_values(DecimalType::I32, &[1, 2, 3]),
    "byte length of 4 i32 values")]
#[case::overflowing_length(18, DecimalType::I16, usize::MAX / 2 + 1,
    stored_values(DecimalType::I16, &[]), "fits in usize")]
#[case::misaligned_buffer(9, DecimalType::I32, 2,
    BufferHandle::new_host(ByteBuffer::copy_from([0u8; 8])), "alignment")]
fn invalid_buffer_fails(
    #[case] precision: u8,
    #[case] values_type: DecimalType,
    #[case] len: usize,
    #[case] values: BufferHandle,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(precision, 2), Nullability::NonNullable);

    let result = deserialize(&dtype, len, &metadata(values_type), &[values], &[]);

    assert!(result.is_err_and(|error| error.to_string().contains(expected)));

    Ok(())
}

#[rstest]
#[case::extra_child(Nullability::Nullable, 2, "got 2")]
#[case::non_nullable_node(Nullability::NonNullable, 1, "requires a nullable dtype")]
fn invalid_validity_children_fail(
    #[case] nullability: Nullability,
    #[case] child_count: usize,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), nullability);
    let values = stored_values(DecimalType::I32, &[1, 2]);
    let validity = BoolArray::from_iter([true, false]).into_array();

    let result = deserialize(
        &dtype,
        2,
        &metadata(DecimalType::I32),
        &[values],
        &vec![validity; child_count],
    );

    assert!(result.is_err_and(|error| error.to_string().contains(expected)));

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
fn round_trips_each_values_type(
    #[case] array: DecimalArray,
    #[case] values_type: DecimalType,
) -> VortexResult<()> {
    let array = array.into_array();

    let decoded = serialize_and_decode(&array, array.dtype())?;

    assert_eq!(decoded.as_::<Decimal>().values_type(), values_type);
    assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());

    Ok(())
}

/// Writes a canonical array of precision 2 and decodes it at precision 10, as a legacy node.
///
/// This goes through the deserializer, which asserts that the plugin supports the encoding of the
/// result, here a `Widen`.
#[test]
fn decode_legacy_node_through_serialized_array() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = [-99, 0, 99];
    let written = canonical_reference(DecimalDType::new(2, 0), &values, None)?.into_array();
    let read_dtype = DecimalDType::new(10, 0);

    let decoded = serialize_and_decode(
        &written,
        &DType::Decimal(read_dtype, Nullability::NonNullable),
    )?;

    assert!(decoded.is::<Widen>());
    assert_arrays_eq!(
        decoded,
        canonical_reference(read_dtype, &values, None)?,
        &mut ctx
    );

    Ok(())
}
