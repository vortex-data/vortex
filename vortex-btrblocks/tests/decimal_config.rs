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
#[case::unrestricted(None, Some(true))]
#[case::neither(Some(vec![]), None)]
#[case::v1(Some(vec![decimal_byte_parts_v1_id()]), Some(false))]
#[case::v2_only(Some(vec![decimal_byte_parts_v2_id()]), None)]
#[case::both(Some(vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]), Some(true))]
fn decimal_mode_follows_permissions(
    #[case] ids: Option<Vec<ArrayId>>,
    #[case] v2: Option<bool>,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let mut builder = BtrBlocksCompressorBuilder::default();
    if let Some(ids) = ids {
        builder = builder.retain_allowed_encodings(&ids.into_iter().collect());
    }
    let expected = match v2 {
        Some(true) if wide => Some(decimal_byte_parts_v2_id()),
        Some(_) if !wide => Some(decimal_byte_parts_v1_id()),
        _ => None,
    };
    assert_decimal_output(builder, wide, expected)
}

#[rstest]
#[case::unrestricted(None)]
#[case::v1(Some(false))]
#[case::both(Some(true))]
fn explicit_decimal_modes(
    #[case] allowed_v2: Option<bool>,
    #[values(false, true)] initial_v2: bool,
    #[values(false, true)] register_later: bool,
) -> VortexResult<()> {
    let scheme: &'static dyn Scheme = if initial_v2 {
        &DecimalScheme::v2()
    } else {
        &DecimalScheme::v1()
    };
    let mut builder = BtrBlocksCompressorBuilder::empty();
    if !register_later {
        builder = builder.with_new_scheme(scheme);
    }
    if let Some(v2) = allowed_v2 {
        let mut allowed = HashSet::from([decimal_byte_parts_v1_id()]);
        if v2 {
            allowed.insert(decimal_byte_parts_v2_id());
        }
        builder = builder.retain_allowed_encodings(&allowed);
    }
    if register_later {
        builder = builder.with_new_scheme(scheme);
    }
    assert_decimal_output(
        builder,
        true,
        allowed_v2
            .unwrap_or(initial_v2)
            .then(decimal_byte_parts_v2_id),
    )
}

#[rstest]
fn decimal_permissions_intersect(#[values(false, true)] restrictive_first: bool) -> VortexResult<()> {
    let v1 = HashSet::from([decimal_byte_parts_v1_id()]);
    let both = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    let (first, second) = if restrictive_first {
        (&v1, &both)
    } else {
        (&both, &v1)
    };
    let builder = BtrBlocksCompressorBuilder::default()
        .retain_allowed_encodings(first)
        .retain_allowed_encodings(second);
    assert_decimal_output(builder.clone(), false, Some(decimal_byte_parts_v1_id()))?;
    assert_decimal_output(builder, true, None)
}

#[test]
fn permissions_do_not_restore_excluded_decimal() -> VortexResult<()> {
    let allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    let builder = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DecimalScheme::default().id()])
        .retain_allowed_encodings(&allowed);
    assert_decimal_output(builder, false, None)
}

#[rstest]
#[case::unrestricted(None)]
#[case::permissions_first(Some(true))]
#[case::permissions_last(Some(false))]
fn cuda_keeps_decimal_v1(
    #[case] permissions_first: Option<bool>,
    #[values(false, true)] register_later: bool,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    let mut builder = BtrBlocksCompressorBuilder::default();
    if register_later {
        builder = builder.exclude_schemes([DecimalScheme::default().id()]);
    }
    if permissions_first == Some(true) {
        builder = builder.retain_allowed_encodings(&allowed);
    }
    builder = builder.only_cuda_compatible();
    if register_later {
        builder = builder.with_new_scheme(&DecimalScheme::v2());
    }
    if permissions_first.is_some() {
        builder = builder.retain_allowed_encodings(&allowed);
    }
    assert_decimal_output(builder, wide, (!wide).then(decimal_byte_parts_v1_id))
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
        .with_new_scheme(&DecimalScheme::v2())
        .with_new_scheme(&FoRScheme)
        .with_new_scheme(&BitPackingScheme)
        .retain_allowed_encodings(&allowed)
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
