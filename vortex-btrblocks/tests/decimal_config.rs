// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal mode selection, serialized permissions, and compression of wide decimal parts.

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::Scheme;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::schemes::integer::FoRScheme;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_set::HashSet;

static DECIMAL_V1: DecimalScheme = DecimalScheme::v1();
static DECIMAL_V2: DecimalScheme = DecimalScheme::v2();

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

fn decimal_array(wide: bool) -> ArrayRef {
    let base = if wide { 1i128 << 70 } else { 0 };
    DecimalArray::new(
        (0..128i128).map(|i| base + i).collect::<Buffer<i128>>(),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    )
    .into_array()
}

fn assert_decimal_output(
    builder: BtrBlocksCompressorBuilder,
    wide: bool,
    expected_id: Option<ArrayId>,
) -> VortexResult<()> {
    let array = decimal_array(wide);
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder.build().compress(&array, &mut ctx)?;
    if let Some(expected_id) = expected_id {
        assert!(compressed.is::<DecimalByteParts>());
        let serialized = SESSION
            .array_serialize(&compressed)?
            .ok_or_else(|| vortex_err!("expected serializable decimal byte parts"))?;
        assert_eq!(serialized.serialized_id, expected_id);
    } else {
        assert!(compressed.is::<Decimal>());
    }
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}

#[rstest]
#[case::default_edition(None, Some(false))]
#[case::neither(Some(vec![]), None)]
#[case::v1(Some(vec![decimal_byte_parts_v1_id()]), Some(false))]
#[case::v2_only(Some(vec![decimal_byte_parts_v2_id()]), None)]
#[case::both(Some(vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]), Some(true))]
fn decimal_mode_follows_permissions(
    #[case] ids: Option<Vec<ArrayId>>,
    #[case] v2: Option<bool>,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let builder = ids
        .map(|ids| BtrBlocksCompressorBuilder::new(ids.into_iter().collect()))
        .unwrap_or_default();
    let expected = match v2 {
        Some(true) if wide => Some(decimal_byte_parts_v2_id()),
        Some(_) if !wide => Some(decimal_byte_parts_v1_id()),
        _ => None,
    };
    assert_decimal_output(builder, wide, expected)
}

#[rstest]
fn decimal_mode_uses_final_permissions(#[values(false, true)] wide: bool) -> VortexResult<()> {
    let builder =
        BtrBlocksCompressorBuilder::default().allow_encodings([decimal_byte_parts_v2_id()]);
    assert_decimal_output(
        builder.clone(),
        wide,
        Some(if wide {
            decimal_byte_parts_v2_id()
        } else {
            decimal_byte_parts_v1_id()
        }),
    )?;

    assert_decimal_output(
        builder.set_allowed_encodings([decimal_byte_parts_v1_id()]),
        wide,
        (!wide).then(decimal_byte_parts_v1_id),
    )
}

#[rstest]
fn permissions_can_reenable_decimal_v2_after_cuda(
    #[values(false, true)] replace: bool,
) -> VortexResult<()> {
    let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
    let builder = if replace {
        builder.set_allowed_encodings([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()])
    } else {
        builder.allow_encodings([decimal_byte_parts_v2_id()])
    };
    assert_decimal_output(builder, true, Some(decimal_byte_parts_v2_id()))
}

#[rstest]
fn explicit_decimal_modes_only_upgrade(
    #[values(false, true)] allowed_v2: bool,
    #[values(false, true)] initial_v2: bool,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let scheme: &'static dyn Scheme = if initial_v2 { &DECIMAL_V2 } else { &DECIMAL_V1 };
    let mut allowed = HashSet::from([decimal_byte_parts_v1_id()]);
    if allowed_v2 {
        allowed.insert(decimal_byte_parts_v2_id());
    }
    let builder = BtrBlocksCompressorBuilder::empty()
        .allow_encodings(allowed)
        .with_new_scheme(scheme);
    let mode = (!initial_v2 || allowed_v2).then_some(allowed_v2);
    let expected = match mode {
        Some(true) if wide => Some(decimal_byte_parts_v2_id()),
        Some(_) if !wide => Some(decimal_byte_parts_v1_id()),
        _ => None,
    };
    assert_decimal_output(builder, wide, expected)
}

#[test]
fn upgrades_do_not_restore_excluded_decimal() -> VortexResult<()> {
    let allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    let builder =
        BtrBlocksCompressorBuilder::new(allowed).exclude_schemes([DecimalScheme::default().id()]);
    assert_decimal_output(builder, false, None)
}

#[rstest]
fn cuda_never_uses_decimal_v2(
    #[values(false, true)] initial_v2: bool,
    #[values(false, true)] register_later: bool,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    let scheme: &'static dyn Scheme = if initial_v2 { &DECIMAL_V2 } else { &DECIMAL_V1 };
    let mut builder = BtrBlocksCompressorBuilder::empty().allow_encodings(allowed);
    if !register_later {
        builder = builder.with_new_scheme(scheme);
    }
    builder = builder.only_cuda_compatible();
    if register_later {
        builder = builder.with_new_scheme(scheme);
    }
    assert_decimal_output(
        builder,
        wide,
        (!initial_v2 && !wide).then(decimal_byte_parts_v1_id),
    )
}

#[rstest]
#[case::i128(false, 1)]
#[case::i256(true, 3)]
fn wide_decimal_parts_roundtrip(
    #[case] use_i256: bool,
    #[case] lower_part_count: usize,
    #[values(false, true)] negative: bool,
    #[values(false, true)] nullable: bool,
    #[values(false, true)] compress_children: bool,
) -> VortexResult<()> {
    let validity = if nullable {
        Validity::from_iter((0..2048).map(|i| i % 7 != 0))
    } else {
        Validity::NonNullable
    };
    let array = if use_i256 {
        let values = (1..=2048u32)
            .map(|i| {
                let msp = if negative {
                    -i128::from(i)
                } else {
                    i128::from(i)
                };
                i256::from_parts(
                    (u128::from(i) << 64) | u128::from(i * 131 + 17),
                    (msp << 64) | i128::from(i * 3 + 1),
                )
            })
            .collect::<Buffer<i256>>();
        DecimalArray::new(values, DecimalDType::new(76, 2), validity)
    } else {
        let values = (1..=2048i128)
            .map(|i| {
                let msp = if negative { -i } else { i };
                (msp << 70) + i * 131 + 17
            })
            .collect::<Buffer<i128>>();
        DecimalArray::new(values, DecimalDType::new(38, 2), validity)
    }
    .into_array();
    let mut allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    if compress_children {
        allowed.extend(FoRScheme.produced_encodings());
        allowed.extend(BitPackingScheme.produced_encodings());
    }
    let compressor = BtrBlocksCompressorBuilder::empty()
        .allow_encodings(allowed)
        .with_new_scheme(&DECIMAL_V2)
        .with_new_scheme(&FoRScheme)
        .with_new_scheme(&BitPackingScheme)
        .build();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;
    // Every part varies, so without child compression splitting alone cannot save space.
    assert_eq!(compressed.is::<DecimalByteParts>(), compress_children);
    if compress_children {
        let parts = compressed
            .as_opt::<DecimalByteParts>()
            .ok_or_else(|| vortex_err!("expected decimal byte parts"))?;
        assert_eq!(parts.lower_parts().len(), lower_part_count);
        assert!(!parts.msp().is_canonical());
        assert!(parts.lower_parts().iter().all(|part| !part.is_canonical()));
    }

    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in compressed.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(buffer.as_ref());
    }
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;
    assert_arrays_eq!(array, decoded, &mut ctx);
    Ok(())
}
