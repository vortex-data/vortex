// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire translation, metadata validation, and serialized format selection.

mod device;

use std::sync::LazyLock;

use prost::Message as _;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayVTable;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::*;
use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsArraySlotsExt;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;
use crate::decimal_byte_parts::testing::widen_msp;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

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
    let session = VortexSession::empty();
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
        &VortexSession::empty(),
    )
}

/// Casts `values` to `stored_ptype`, as an old writer could store the MSP.
fn store_msp(
    values: PrimitiveArray,
    stored_ptype: PType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullability = values.dtype().nullability();
    Ok(values
        .into_array()
        .cast(DType::Primitive(stored_ptype, nullability))?
        .execute::<PrimitiveArray>(ctx)?
        .into_array())
}

/// Reads a stored MSP with no lower parts through the plugin, as v1 or v2.
///
/// The decimal dtype takes the nullability of `msp`, and the metadata names the stored ptype of
/// `msp`.
fn deserialize_stored_msp(
    msp: ArrayRef,
    decimal_dtype: DecimalDType,
    v2: bool,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
    let stored_ptype = msp.dtype().as_ptype() as i32;
    let (serialized_id, metadata) = if v2 {
        let metadata = DecimalBytePartsV2Metadata {
            msp_ptype: stored_ptype,
            lower_part_ptypes: vec![],
        };
        (decimal_byte_parts_v2_id(), metadata.encode_to_vec())
    } else {
        let metadata = v1::DecimalBytePartsMetadata {
            zeroth_child_ptype: stored_ptype,
            lower_part_count: 0,
        };
        (decimal_byte_parts_v1_id(), metadata.encode_to_vec())
    };

    let len = msp.len();
    let children = vec![msp];
    DecimalBytePartsPlugin.deserialize(
        ArrayDeserialization::new(serialized_id, &dtype, len, &metadata, &[], &children),
        session,
    )
}

/// Writes `array` with the serializers of `session` and reads it back.
fn serialize_and_decode(array: &ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze())?;

    parts.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        session,
    )
}

#[rstest]
#[case::no_lower_parts(DecimalByteParts::try_new(
    buffer![1i32, 2, 3].into_array(), DecimalDType::new(9, 2),
))]
#[case::one_lower_part(DecimalByteParts::try_new_with_lower_parts(
    msp(), vec![lower_part()], DecimalDType::new(38, 2),
))]
#[case::wider_i256_storage_all_null(DecimalByteParts::encode(
    &DecimalArray::new(
        buffer![i256::from_parts(1 << 127, 0); 3],
        DecimalDType::new(76, 0), Validity::AllInvalid,
    ),
    &mut SESSION.create_execution_ctx(),
))]
#[case::redundant_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    buffer![0i64; 3].into_array(),
    vec![buffer![0u64; 3].into_array(), buffer![0u64; 3].into_array(), lower_part()],
    DecimalDType::new(76, 2),
))]
#[case::narrow_msp_with_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    widen_msp(buffer![-1i16, 0, 1].into_array(), DecimalDType::new(76, 2))?,
    vec![
        buffer![u64::from(u8::MAX), 128, 0].into_array(),
        ConstantArray::new(u64::from(u16::MAX), 3).into_array(),
        buffer![0u64, 1 << 31, u64::from(u32::MAX)].into_array(),
    ],
    DecimalDType::new(76, 2),
))]
#[case::nullable_narrow_msp_with_lower_parts(DecimalByteParts::try_new_with_lower_parts(
    widen_msp(
        PrimitiveArray::new(buffer![-1i8, 0, 1], Validity::from_iter([true, false, true]))
            .into_array(),
        DecimalDType::new(76, 2),
    )?,
    vec![
        buffer![u64::MAX, 1 << 63, 0].into_array(),
        buffer![0u64, 128, u64::from(u8::MAX)].into_array(),
        buffer![u64::from(u32::MAX), 1 << 31, 0].into_array(),
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

    let decoded = serialize_and_decode(&array, &session)?;

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

/// A `Widen` MSP writes its narrow child, and the v1 metadata names the narrow ptype in field 1.
#[test]
fn v1_metadata_names_the_narrow_msp_ptype() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(19, 2);
    let values = PrimitiveArray::new(buffer![1i64, 2, 3], Validity::NonNullable);
    let msp = widen_msp(store_msp(values, PType::I8, &mut ctx)?, decimal_dtype)?;
    let array = DecimalByteParts::try_new(msp, decimal_dtype)?.into_array();

    let serialized = session
        .array_serialize(&array)?
        .vortex_expect("byte parts arrays are serializable");
    assert_eq!(serialized.serialized_id, decimal_byte_parts_v1_id());
    assert_eq!(serialized.metadata, [8, 4]);
    assert_eq!(
        serialized.children[0].dtype(),
        &DType::Primitive(PType::I8, Nullability::NonNullable)
    );

    Ok(())
}

/// A take keeps the `Widen` at the top of a narrow MSP, so the taken array still writes its narrow
/// child and reads back.
#[rstest]
#[case::v1(
    DecimalByteParts::try_new(
        widen_msp(buffer![-3i8, 1, 4, -1].into_array(), DecimalDType::new(18, 2))?,
        DecimalDType::new(18, 2),
    ),
    buffer![3u32, 0, 0, 1].into_array(),
)]
#[case::v1_nullable_indices(
    DecimalByteParts::try_new(
        widen_msp(buffer![-3i8, 1, 4, -1].into_array(), DecimalDType::new(18, 2))?,
        DecimalDType::new(18, 2),
    ),
    PrimitiveArray::from_option_iter([Some(2u32), None, Some(0), Some(2)]).into_array(),
)]
#[case::v2_nullable_msp(
    DecimalByteParts::try_new_with_lower_parts(
        widen_msp(
            PrimitiveArray::new(buffer![-1i16, 0, 1], Validity::from_iter([true, false, true]))
                .into_array(),
            DecimalDType::new(38, 2),
        )?,
        vec![buffer![u64::MAX, 1 << 63, 7].into_array()],
        DecimalDType::new(38, 2),
    ),
    buffer![2u32, 0, 2, 1].into_array(),
)]
fn take_keeps_widen_msp_on_top(
    #[case] array: VortexResult<DecimalBytePartsArray>,
    #[case] indices: ArrayRef,
) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let array = array?.into_array();
    let expected = array
        .clone()
        .execute::<DecimalArray>(&mut ctx)?
        .into_array()
        .take(indices.clone())?;

    let taken = array.take(indices)?.optimize_ctx(&session)?;
    assert!(
        taken.is::<DecimalByteParts>(),
        "expected the take to reduce into the encoding, got {}",
        taken.encoding_id()
    );
    let msp = taken.as_::<DecimalByteParts>().msp();
    assert!(
        msp.is::<Widen>(),
        "the MSP slot must have a Widen at the top, got {}",
        msp.encoding_id()
    );

    let decoded = serialize_and_decode(&taken, &session)?;
    assert_arrays_eq!(decoded, expected, &mut ctx);

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
    "v2 expects the msp child plus one child per lower part"
)]
#[case::v2_extra_child(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![PType::U64 as i32]),
    vec![msp(), lower_part(), lower_part()],
    "v2 expects the msp child plus one child per lower part"
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
    "lower part 0 ptype\n  left: i64\n right: u64"
)]
#[case::v2_narrow_lower_part(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![PType::U32 as i32]),
    vec![msp(), lower_part()],
    "lower part 0 ptype\n  left: u32\n right: u64"
)]
#[case::v2_unknown_ptype(
    decimal_byte_parts_v2_id(),
    v2_metadata(vec![i32::MAX]),
    vec![msp(), lower_part()],
    "must have a valid PType"
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

#[rstest]
fn legacy_msp_widths(
    #[values(2, 4, 9, 18)] precision: u8,
    #[values(PType::I8, PType::I16, PType::I32, PType::I64)] stored_ptype: PType,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let dtype = DType::Decimal(DecimalDType::new(precision, 0), nullability);
    let msp = PrimitiveArray::new(
        buffer![-99i64, 0, 99],
        if nullability == Nullability::Nullable {
            Validity::from_iter([true, false, true])
        } else {
            Validity::NonNullable
        },
    );
    let msp = store_msp(msp, stored_ptype, &mut ctx)?;
    let decoded = deserialize_stored_msp(msp, DecimalDType::new(precision, 0), false, &session)?;
    let array = decoded
        .as_opt::<DecimalByteParts>()
        .vortex_expect("byte parts array");
    let logical_ptype = crate::decimal_byte_parts::msp_ptype(DecimalDType::new(precision, 0));
    assert_eq!(
        array.msp().dtype(),
        &DType::Primitive(logical_ptype, nullability)
    );
    let stored_is_narrower = stored_ptype.byte_width() < logical_ptype.byte_width();
    assert_eq!(array.msp().is::<Widen>(), stored_is_narrower);
    let written = session
        .array_serialize(&decoded)?
        .vortex_expect("byte parts arrays are serializable");
    assert_eq!(
        written.children[0].dtype().as_ptype(),
        if stored_is_narrower {
            stored_ptype
        } else {
            logical_ptype
        }
    );
    let expected = DecimalArray::from_option_iter(
        if nullability == Nullability::Nullable {
            vec![Some(-99i64), None, Some(99)]
        } else {
            vec![Some(-99i64), Some(0), Some(99)]
        },
        DecimalDType::new(precision, 0),
    )
    .into_array()
    .cast(dtype)?;
    assert_arrays_eq!(decoded, expected, &mut ctx);

    Ok(())
}

/// A wider legacy MSP narrows only when every valid value is within the precision, not only within
/// the logical integer width. `10^p` fits the logical width in every case.
#[rstest]
#[case::i16_at_p2(PType::I16, 2)]
#[case::i64_at_p2(PType::I64, 2)]
#[case::i32_at_p4(PType::I32, 4)]
#[case::i64_at_p4(PType::I64, 4)]
#[case::i64_at_p9(PType::I64, 9)]
fn wider_legacy_msp_checks_precision(
    #[case] stored_ptype: PType,
    #[case] precision: u8,
    #[values(false, true)] v2: bool,
) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(precision, 0);
    let logical_ptype = crate::decimal_byte_parts::msp_ptype(decimal_dtype);
    let limit = 10i64.pow(u32::from(precision));

    let in_precision = buffer![-(limit - 1), limit - 1];
    let msp = store_msp(
        PrimitiveArray::new(in_precision.clone(), Validity::NonNullable),
        stored_ptype,
        &mut ctx,
    )?;
    let decoded = deserialize_stored_msp(msp, decimal_dtype, v2, &session)?;
    let msp = decoded
        .as_opt::<DecimalByteParts>()
        .vortex_expect("byte parts array")
        .msp();
    assert_eq!(
        msp.dtype(),
        &DType::Primitive(logical_ptype, Nullability::NonNullable)
    );
    assert!(!msp.is::<Widen>());
    let expected = DecimalArray::new(in_precision, decimal_dtype, Validity::NonNullable);
    assert_arrays_eq!(decoded, expected, &mut ctx);

    for outside_precision in [-limit, limit] {
        let msp = store_msp(
            PrimitiveArray::new(buffer![outside_precision], Validity::NonNullable),
            stored_ptype,
            &mut ctx,
        )?;
        let err = deserialize_stored_msp(msp, decimal_dtype, v2, &session)
            .expect_err("a valid value outside the precision must fail");
        assert!(
            err.to_string().contains("must be within the precision of"),
            "{err}"
        );
    }

    Ok(())
}

/// The copy of a wider legacy MSP writes zeros in every null slot, whatever value the slot held.
#[rstest]
#[case::null_inside_width(
    buffer![99i64, 10_000, -99],
    Validity::from_iter([true, false, true]),
    buffer![99i16, 0, -99],
)]
#[case::null_outside_width(
    buffer![99i64, i64::MAX, -99],
    Validity::from_iter([true, false, true]),
    buffer![99i16, 0, -99],
)]
#[case::all_null(
    buffer![10_000i64, i64::MAX, i64::MIN],
    Validity::AllInvalid,
    buffer![0i16, 0, 0],
)]
#[case::empty(buffer![], Validity::AllValid, buffer![])]
fn wider_legacy_msp_zeroes_null_slots(
    #[case] stored: Buffer<i64>,
    #[case] validity: Validity,
    #[case] expected_slots: Buffer<i16>,
    #[values(false, true)] v2: bool,
) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(4, 2);

    let msp = PrimitiveArray::new(stored, validity.clone()).into_array();
    let decoded = deserialize_stored_msp(msp, decimal_dtype, v2, &session)?;
    let msp = decoded
        .as_opt::<DecimalByteParts>()
        .vortex_expect("byte parts array")
        .msp()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(msp.as_slice::<i16>(), expected_slots.as_slice());

    let expected = DecimalArray::new(expected_slots, decimal_dtype, validity);
    assert_arrays_eq!(decoded, expected, &mut ctx);

    Ok(())
}

#[rstest]
fn wider_legacy_msp_rejects_valid_value_outside_width(
    #[values(false, true)] v2: bool,
) -> VortexResult<()> {
    let session = session();
    let msp = buffer![99i64, i64::MAX, -99].into_array();

    let err = deserialize_stored_msp(msp, DecimalDType::new(4, 2), v2, &session)
        .expect_err("a valid value outside the logical width must fail");
    assert!(
        err.to_string().contains("must be within the precision of"),
        "{err}"
    );

    Ok(())
}
