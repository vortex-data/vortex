// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal scheme refinement by allowed serialized IDs, and compression of wide decimal parts.

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
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static DECIMAL_V2: DecimalScheme = DecimalScheme::v2();

/// Registers the decimal and fastlanes encodings, and enables no editions.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

/// Like [`SESSION`], with the latest core edition enabled: it allows decimal v1 but not v2.
static CORE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session().with::<EditionSession>();
    for family in EDITION_FAMILIES {
        session
            .editions()
            .declare_family(family)
            .expect("first-party edition family");
    }
    for declaration in EDITION_DECLARATIONS {
        session
            .register_edition(declaration)
            .expect("first-party edition");
    }
    session
        .enable_edition(CORE_2026_08_3)
        .expect("core edition is registered");
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

/// Which decimal output a builder is expected to produce.
#[derive(Debug, Clone, Copy)]
enum Expected {
    /// No decimal scheme survives, so decimals stay canonical.
    Canonical,
    /// Single-part arrays only: wide decimals stay canonical.
    V1,
    /// Wide decimals split into multi-part arrays.
    V2,
}

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
#[case::no_editions(|| BtrBlocksCompressorBuilder::from_session(&SESSION), Expected::Canonical)]
#[case::core_edition(|| BtrBlocksCompressorBuilder::from_session(&CORE_SESSION), Expected::V1)]
#[case::editions_disabled(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).disable_editions(),
    Expected::V2
)]
#[case::unrestricted(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted(),
    Expected::V2
)]
#[case::cuda(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted().only_cuda_compatible(),
    Expected::V1
)]
fn decimal_scheme_follows_allowed_ids(
    #[case] builder: fn() -> BtrBlocksCompressorBuilder,
    #[case] expected: Expected,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let expected_id = match expected {
        Expected::Canonical => None,
        Expected::V1 | Expected::V2 if !wide => Some(decimal_byte_parts_v1_id()),
        Expected::V1 => None,
        Expected::V2 => Some(decimal_byte_parts_v2_id()),
    };
    assert_decimal_output(builder(), wide, expected_id)
}

#[test]
fn refinement_does_not_restore_excluded_decimal() -> VortexResult<()> {
    let builder = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .exclude_schemes([DECIMAL_V2.id()]);
    assert_decimal_output(builder, false, None)
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
    let mut builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(&DECIMAL_V2);
    if compress_children {
        builder = builder
            .with_new_scheme(&FoRScheme)
            .with_new_scheme(&BitPackingScheme);
    }
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder.build().compress(&array, &mut ctx)?;
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
