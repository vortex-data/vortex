// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message as _;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayVTable;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_session::registry::ReadContext;

use super::*;
use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsArraySlotsExt;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;

#[rstest]
#[case::no_lower_parts(DecimalByteParts::try_new(
    buffer![1i32, 2, 3].into_array(), DecimalDType::new(9, 2),
))]
#[case::one_lower_part(DecimalByteParts::try_new_with_lower_parts(
    msp(), vec![lower_part()], DecimalDType::new(38, 2),
))]
#[case::wider_i64_storage(DecimalByteParts::encode(
    &DecimalArray::new(buffer![-99i64, 0, 99], DecimalDType::new(2, 0), Validity::NonNullable),
    &mut array_session().create_execution_ctx(),
))]
#[case::wider_i128_storage(DecimalByteParts::encode(
    &DecimalArray::new(buffer![-99i128, 0, 99], DecimalDType::new(2, 0), Validity::NonNullable),
    &mut array_session().create_execution_ctx(),
))]
#[case::wider_i256_storage(DecimalByteParts::encode(
    &DecimalArray::new(
        buffer![i256::from_i128(-99), i256::ZERO, i256::from_i128(99)],
        DecimalDType::new(2, 0), Validity::NonNullable,
    ),
    &mut array_session().create_execution_ctx(),
))]
#[case::redundant_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    buffer![0i64; 3].into_array(),
    vec![buffer![0u64; 3].into_array(), buffer![0u64; 3].into_array(), lower_part()],
    DecimalDType::new(38, 2),
))]
#[case::narrowed_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    buffer![-1i16, 0, 1].into_array(),
    vec![
        buffer![u8::MAX, 128, 0].into_array(),
        ConstantArray::new(u16::MAX, 3).into_array(),
        buffer![0u32, 1 << 31, u32::MAX].into_array(),
    ],
    DecimalDType::new(76, 2),
))]
#[case::nullable_mixed_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    PrimitiveArray::new(
        buffer![-1i8, 0, 1], Validity::from_iter([true, false, true]),
    ).into_array(),
    vec![
        buffer![u64::MAX, 1 << 63, 0].into_array(),
        buffer![0u8, 128, u8::MAX].into_array(),
        buffer![u32::MAX, 1 << 31, 0].into_array(),
    ],
    DecimalDType::new(76, 2),
))]
fn serde_round_trip(#[case] array: VortexResult<DecimalBytePartsArray>) -> VortexResult<()> {
    let session = session();
    let array = array?;
    let lower_part_count = array.lower_parts().len();
    let lower_part_dtypes: Vec<_> = array
        .lower_parts()
        .iter()
        .map(|part| part.dtype().clone())
        .collect();
    let array = array.into_array();
    let dtype = array.dtype().clone();
    let len = array.len();

    let expected_id = if lower_part_count == 0 {
        decimal_byte_parts_v1_id()
    } else {
        decimal_byte_parts_v2_id()
    };
    assert_eq!(
        session
            .array_serialize(&array)?
            .vortex_expect("byte parts arrays are serializable")
            .serialized_id,
        expected_id
    );

    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, &session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;

    assert_eq!(
        decoded
            .as_opt::<DecimalByteParts>()
            .vortex_expect("byte parts array")
            .lower_parts()
            .iter()
            .map(|part| part.dtype().clone())
            .collect::<Vec<_>>(),
        lower_part_dtypes,
        "lower-part dtypes and order must survive serde"
    );

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(array, decoded, &mut ctx);
    Ok(())
}

#[test]
fn v1_metadata_is_unchanged() -> VortexResult<()> {
    let session = session();
    let array = DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2))?.into_array();
    let serialized = session
        .array_serialize(&array)?
        .vortex_expect("byte parts arrays are serializable");
    assert_eq!(serialized.serialized_id, decimal_byte_parts_v1_id());
    // v1 metadata for an i64 MSP: field 1 = 7, with no lower-part fields emitted.
    assert_eq!(serialized.metadata, [8, 7]);
    Ok(())
}

#[rstest]
#[case::v1_lower_part_count(
    decimal_byte_parts_v1_id(),
    v1_metadata(1),
    vec![msp(), lower_part()],
    "must not carry lower parts"
)]
#[case::v1_extra_child(
    decimal_byte_parts_v1_id(),
    v1_metadata(0),
    vec![msp(), lower_part()],
    "exactly one child"
)]
#[case::v2_missing_child(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![PType::U64 as i32]),
    vec![msp()],
    "expected 2 children, got 1"
)]
#[case::v2_extra_child(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![PType::U64 as i32]),
    vec![msp(), lower_part(), lower_part()],
    "expected 2 children, got 3"
)]
#[case::v2_too_many_lower_parts(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![PType::U64 as i32; MAX_LOWER_PARTS + 1]),
    vec![msp(), lower_part(), lower_part(), lower_part(), lower_part()],
    "lower parts, got 4"
)]
#[case::v2_signed_lower_part(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![PType::I64 as i32]),
    vec![msp(), lower_part()],
    "unsigned integer dtype"
)]
#[case::v2_unknown_ptype(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![i32::MAX]),
    vec![msp(), lower_part()],
    "invalid PType"
)]
fn decoder_rejects_malformed_payloads(
    #[case] serialized_id: ArrayId,
    #[case] metadata: Vec<u8>,
    #[case] children: Vec<ArrayRef>,
    #[case] expected_error: &str,
) {
    let result = deserialize_with(serialized_id, &metadata, children);
    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains(expected_error)),
        "expected {expected_error}, got {result:?}"
    );
}

#[test]
fn serialization_requires_v2_permission() -> VortexResult<()> {
    let session = session();
    let array = DecimalByteParts::try_new_with_lower_parts(
        msp(),
        vec![lower_part()],
        DecimalDType::new(38, 2),
    )?
    .into_array();

    let restricted = ArrayContext::empty().with_allowed_ids(
        [decimal_byte_parts_v1_id(), ArrayVTable::id(&Primitive)]
            .into_iter()
            .collect(),
    );
    let err = array
        .serialize(&restricted, &session, &SerializeOptions::default())
        .expect_err("expected the permitted-encoding check to refuse the v2 format");
    assert!(
        err.to_string().contains("not permitted"),
        "error should name the permitted-encoding check, got: {err}"
    );

    // Permitting the v2 format id is exactly what allows the same array through.
    let permissive = ArrayContext::empty().with_allowed_ids(
        [
            decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
            ArrayVTable::id(&Primitive),
        ]
        .into_iter()
        .collect(),
    );
    array.serialize(&permissive, &session, &SerializeOptions::default())?;
    assert!(
        permissive.to_ids().contains(&decimal_byte_parts_v2_id()),
        "the file's encoding table must carry the v2 format id"
    );

    Ok(())
}

#[test]
fn bare_vtable_refuses_serde() -> VortexResult<()> {
    let session = array_session();
    session.arrays().register(DecimalByteParts);
    let msp = msp();
    let array = DecimalByteParts::try_new(msp.clone(), DecimalDType::new(19, 2))?.into_array();
    let result = session.array_serialize(&array);
    assert!(
        result.as_ref().is_err_and(|err| err
            .to_string()
            .contains("DecimalByteParts serialization requires DecimalBytePartsPlugin")),
        "expected unsupported VTable serialization, got {result:?}"
    );

    let id = VTable::id(&DecimalByteParts);
    let plugin = session
        .arrays()
        .registry()
        .get(&id)
        .vortex_expect("registered");
    let children = vec![msp];
    let result = plugin.deserialize(
        ArrayDeserialization::new(id, array.dtype(), array.len(), &[8, 7], &[], &children),
        &session,
    );
    assert!(
        result.as_ref().is_err_and(|err| err
            .to_string()
            .contains("DecimalByteParts deserialization requires DecimalBytePartsPlugin")),
        "expected unsupported VTable deserialization, got {result:?}"
    );
    Ok(())
}

fn msp() -> ArrayRef {
    buffer![1i64, 2, 3].into_array()
}

fn lower_part() -> ArrayRef {
    buffer![1u64, 2, 3].into_array()
}

/// v1 metadata for an i64 MSP: field 1 = 7, then field 2 only when the count is non-zero, as
/// proto3 omits default values.
fn v1_metadata(lower_part_count: u8) -> Vec<u8> {
    let mut metadata = vec![8, 7];
    if lower_part_count > 0 {
        metadata.extend([16, lower_part_count]);
    }
    metadata
}

fn v2_metadata(lower_part_ptypes: Vec<i32>) -> Vec<u8> {
    DecimalBytePartsV2Metadata {
        msp_ptype: PType::I64 as i32,
        lower_part_ptypes,
    }
    .encode_to_vec()
}

fn session() -> VortexSession {
    let session = array_session();
    crate::initialize(&session);
    session
}

/// Decode a hand-built payload of three rows through the plugin.
fn deserialize_with(
    serialized_id: ArrayId,
    metadata: &[u8],
    children: Vec<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
    DecimalBytePartsPlugin.deserialize(
        ArrayDeserialization::new(serialized_id, &dtype, 3, metadata, &[], &children),
        &array_session(),
    )
}
