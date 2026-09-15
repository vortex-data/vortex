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
use crate::decimal_byte_parts::testing::encode;
use crate::decimal_byte_parts::testing::i128_parts;
use crate::decimal_byte_parts::testing::i256_parts;
use crate::decimal_byte_parts::testing::wide_i128_values;
use crate::decimal_byte_parts::testing::wide_i256_values;

#[rstest]
#[case::v1(0)]
#[case::v2_one_lower_part(1)]
#[case::v2_two_lower_parts(2)]
#[case::v2_three_lower_parts(3)]
fn serde_reuses_children(#[case] lower_part_count: usize) -> VortexResult<()> {
    let array = DecimalByteParts::try_new_with_lower_parts(
        msp(),
        (0..lower_part_count).map(|_| lower_part()).collect(),
        DecimalDType::new(76, 2),
    )?
    .into_array();
    assert_eq!(array.encoding_id(), decimal_byte_parts_v2_id());
    let session = session();
    let serialized = session
        .array_serialize(&array)?
        .vortex_expect("serializable");
    let original_children = array.children();
    assert_eq!(original_children.len(), serialized.children.len());
    for (original, written) in original_children.iter().zip(&serialized.children) {
        assert!(ArrayRef::ptr_eq(original, written));
    }
    let decoded = DecimalBytePartsPlugin.deserialize(
        ArrayDeserialization::new(
            serialized.serialized_id,
            array.dtype(),
            array.len(),
            &serialized.metadata,
            &[],
            &serialized.children,
        ),
        &session,
    )?;
    assert_eq!(decoded.encoding_id(), decimal_byte_parts_v2_id());
    let decoded_children = decoded.children();
    assert_eq!(original_children.len(), decoded_children.len());
    for (original, decoded) in original_children.iter().zip(&decoded_children) {
        assert!(ArrayRef::ptr_eq(original, decoded));
    }
    Ok(())
}

#[rstest]
#[case::wide(msp(), lower_part())]
#[case::numerically_narrow(buffer![0i64; 3].into_array(), lower_part())]
fn v1_serializer_rejects_lower_parts(
    #[case] msp: ArrayRef,
    #[case] lower: ArrayRef,
) -> VortexResult<()> {
    let array =
        DecimalByteParts::try_new_with_lower_parts(msp, vec![lower], DecimalDType::new(38, 2))?;
    assert!(v1::serialize(array.as_view()).is_err());
    Ok(())
}

#[test]
fn frozen_decoder_ignores_unknown_metadata_fields() -> VortexResult<()> {
    let child = msp();
    let children = vec![child.clone()];
    let dtype = DType::Decimal(DecimalDType::new(19, 2), Nullability::NonNullable);
    // Field 3 belongs to v2. The frozen protobuf schema ignores it, as the old decoder did;
    // the recognized lower-part count is still zero and there is still exactly one child.
    let metadata = [8, 7, 26, 1, 0];
    let current = v1::deserialize(ArrayDeserialization::new(
        decimal_byte_parts_v1_id(),
        &dtype,
        3,
        &metadata,
        &[],
        &children,
    ))?;
    assert!(ArrayRef::ptr_eq(&child, current.msp()));
    Ok(())
}

#[rstest]
#[case::no_lower_parts(DecimalByteParts::try_new(
    buffer![1i32, 2, 3].into_array(), DecimalDType::new(9, 2),
))]
#[case::one_lower_part(Ok(i128_parts(wide_i128_values(), Validity::NonNullable)))]
#[case::three_lower_parts(Ok(i256_parts(wide_i256_values(), Validity::NonNullable)))]
#[case::nullable_three_lower_parts(Ok(i256_parts(
    wide_i256_values(),
    Validity::from_iter([true, false, true, true, true, false, true, true, true, true]),
)))]
#[case::wider_i64_storage(encode(&DecimalArray::new(
    buffer![-99i64, 0, 99], DecimalDType::new(2, 0), Validity::NonNullable,
)))]
#[case::wider_i128_storage(encode(&DecimalArray::new(
    buffer![-99i128, 0, 99], DecimalDType::new(2, 0), Validity::NonNullable,
)))]
#[case::wider_i256_storage(encode(&DecimalArray::new(
    buffer![i256::from_i128(-99), i256::ZERO, i256::from_i128(99)],
    DecimalDType::new(2, 0), Validity::NonNullable,
)))]
#[case::redundant_two_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    buffer![0i64; 3].into_array(),
    vec![buffer![0u64; 3].into_array(), lower_part()],
    DecimalDType::new(38, 2),
))]
#[case::redundant_three_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    buffer![0i64; 3].into_array(),
    vec![buffer![0u64; 3].into_array(), buffer![0u64; 3].into_array(), lower_part()],
    DecimalDType::new(38, 2),
))]
#[case::narrowed_one_lower_part(DecimalByteParts::try_new_with_lower_parts(
    buffer![-1i8, 0, 1].into_array(),
    vec![buffer![0u8, 128, u8::MAX].into_array()],
    DecimalDType::new(38, 2),
))]
#[case::narrowed_three_lower_parts(DecimalByteParts::try_new_with_lower_parts(
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
fn test_serde_round_trip(#[case] array: VortexResult<DecimalBytePartsArray>) -> VortexResult<()> {
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

#[rstest]
#[case::missing_lower_part(1, vec![msp()])]
#[case::extra_lower_part(0, vec![msp(), lower_part()])]
#[case::too_many_lower_parts(
    4, vec![msp(), lower_part(), lower_part(), lower_part(), lower_part()],
)]
fn test_deserialize_rejects_child_count_mismatch(
    #[case] lower_part_count: u32,
    #[case] children: Vec<ArrayRef>,
) {
    let serialized_id = if lower_part_count == 0 {
        decimal_byte_parts_v1_id()
    } else {
        decimal_byte_parts_v2_id()
    };
    assert!(plugin_deserialize_with(serialized_id, lower_part_count, children).is_err());
}

#[rstest]
#[case::missing_type(1, vec![], "expected 1 lower-part dtypes, got 0")]
#[case::extra_type(
    1, vec![PType::U64 as i32, PType::U8 as i32],
    "expected 1 lower-part dtypes, got 2",
)]
#[case::signed_type(1, vec![PType::I64 as i32], "unsigned integer dtype")]
#[case::float_type(1, vec![PType::F64 as i32], "unsigned integer dtype")]
#[case::unknown_type(1, vec![i32::MAX], "invalid PType")]
fn test_deserialize_rejects_invalid_lower_part_ptypes(
    #[case] lower_part_count: u32,
    #[case] lower_part_ptypes: Vec<i32>,
    #[case] expected_error: &str,
) {
    let metadata = DecimalBytePartsV2Metadata {
        zeroth_child_ptype: PType::I64 as i32,
        lower_part_count,
        lower_part_ptypes,
    }
    .encode_to_vec();
    let mut children = vec![msp()];
    children.extend((0..lower_part_count).map(|_| lower_part()));
    let serialized_id = if lower_part_count == 0 {
        decimal_byte_parts_v1_id()
    } else {
        decimal_byte_parts_v2_id()
    };
    let dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
    let result = DecimalBytePartsPlugin.deserialize(
        ArrayDeserialization::new(serialized_id, &dtype, 3, &metadata, &[], &children),
        &array_session(),
    );
    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains(expected_error)),
        "expected {expected_error}, got {result:?}"
    );
}

fn plugin_deserialize_with(
    serialized_id: ArrayId,
    lower_part_count: u32,
    children: Vec<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let metadata = DecimalBytePartsV2Metadata {
        zeroth_child_ptype: PType::I64 as i32,
        lower_part_count,
        lower_part_ptypes: (0..lower_part_count).map(|_| PType::U64 as i32).collect(),
    }
    .encode_to_vec();
    let dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
    DecimalBytePartsPlugin.deserialize(
        ArrayDeserialization::new(serialized_id, &dtype, 3, &metadata, &[], &children),
        &array_session(),
    )
}

/// Each serialized ID keeps its own contract: the frozen ID never carries lower parts, and
/// the v2 ID is never written without them.
#[rstest]
#[case::frozen_without_lower_parts(decimal_byte_parts_v1_id(), 0, vec![msp()], true)]
#[case::frozen_with_lower_parts(
    decimal_byte_parts_v1_id(),
    1,
    vec![msp(), lower_part()],
    false
)]
#[case::v2_with_lower_parts(decimal_byte_parts_v2_id(), 1, vec![msp(), lower_part()], true)]
#[case::v2_without_lower_parts(decimal_byte_parts_v2_id(), 0, vec![msp()], false)]
fn plugin_holds_each_id_to_its_contract(
    #[case] serialized_id: ArrayId,
    #[case] lower_part_count: u32,
    #[case] children: Vec<ArrayRef>,
    #[case] accepted: bool,
) {
    let result = plugin_deserialize_with(serialized_id, lower_part_count, children);
    assert_eq!(result.is_ok(), accepted, "{serialized_id}: {result:?}");
}

fn msp() -> ArrayRef {
    buffer![1i64, 2, 3].into_array()
}

fn lower_part() -> ArrayRef {
    buffer![1u64, 2, 3].into_array()
}

fn session() -> VortexSession {
    let session = array_session();
    crate::initialize(&session);
    session
}

#[test]
fn frozen_metadata_is_unchanged() -> VortexResult<()> {
    let session = session();
    let array = DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2))?.into_array();
    let serialized = session
        .array_serialize(&array)?
        .vortex_expect("byte parts arrays are serializable");
    assert_eq!(serialized.serialized_id, decimal_byte_parts_v1_id());
    // Frozen metadata for an i64 MSP: field 1 = 7, with no lower-part fields emitted.
    assert_eq!(serialized.metadata, [8, 7]);
    Ok(())
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
