// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wide decimals compress into multi-part byte-part arrays only when the compressor may emit the
//! `vortex.decimal_byte_parts.v2` wire format.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

/// 128 ascending decimals. Wide ones exceed a single signed 64-bit part.
fn decimals(wide: bool) -> ArrayRef {
    let base = if wide { 1i128 << 70 } else { 0 };
    DecimalArray::new(
        (0..128i128).map(|i| base + i).collect::<Buffer<i128>>(),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    )
    .into_array()
}

fn serialized_id(array: &ArrayRef) -> VortexResult<ArrayId> {
    let serialized = SESSION
        .array_serialize(array)?
        .ok_or_else(|| vortex_err!("expected a serializable array"))?;
    Ok(serialized.serialized_id)
}

fn v1_only() -> HashSet<ArrayId> {
    HashSet::from([decimal_byte_parts_v1_id()])
}

fn v1_and_v2() -> HashSet<ArrayId> {
    HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()])
}

/// The scheme always needs v1 for single-part arrays; v2 alone does not register it.
#[rstest]
#[case::neither(HashSet::new(), false)]
#[case::v1(v1_only(), true)]
#[case::v2_only(HashSet::from([decimal_byte_parts_v2_id()]), false)]
#[case::both(v1_and_v2(), true)]
fn decimal_scheme_needs_v1(#[case] allowed: HashSet<ArrayId>, #[case] registered: bool) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .retain_allowed_encodings(&allowed)
        .build();
    assert_eq!(compressor.has_scheme(DecimalScheme.id()), registered);
}

/// Narrow decimals always compress under v1. Wide decimals split into a multi-part v2 array when
/// that format is permitted and otherwise stay canonical, including without any allowlist.
#[rstest]
fn wide_decimals_follow_permitted_formats(
    #[values(None, Some(false), Some(true))] permit_v2: Option<bool>,
    #[values(false, true)] wide: bool,
) -> VortexResult<()> {
    let builder = BtrBlocksCompressorBuilder::default();
    let compressor = match permit_v2 {
        None => builder.build(),
        Some(false) => builder.retain_allowed_encodings(&v1_only()).build(),
        Some(true) => builder.retain_allowed_encodings(&v1_and_v2()).build(),
    };
    let array = decimals(wide);
    let mut ctx = SESSION.create_execution_ctx();

    let compressed = compressor.compress(&array, &mut ctx)?;

    let expect_byte_parts = !wide || permit_v2 == Some(true);
    assert_eq!(compressed.is::<DecimalByteParts>(), expect_byte_parts);
    if expect_byte_parts {
        let expected = if wide {
            decimal_byte_parts_v2_id()
        } else {
            decimal_byte_parts_v1_id()
        };
        assert_eq!(serialized_id(&compressed)?, expected);
    }
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}

/// The CUDA preset withholds v2 whether the writer allowlist is applied before or after it, while
/// single-part decimals still compress under v1.
#[rstest]
fn cuda_preset_keeps_wide_decimals_canonical(
    #[values(false, true)] allowlist_first: bool,
) -> VortexResult<()> {
    let builder = BtrBlocksCompressorBuilder::default();
    let builder = if allowlist_first {
        builder
            .retain_allowed_encodings(&v1_and_v2())
            .only_cuda_compatible()
    } else {
        builder
            .only_cuda_compatible()
            .retain_allowed_encodings(&v1_and_v2())
    };
    let compressor = builder.build();
    let mut ctx = SESSION.create_execution_ctx();

    let wide = compressor.compress(&decimals(true), &mut ctx)?;
    assert!(!wide.is::<DecimalByteParts>());

    let narrow = compressor.compress(&decimals(false), &mut ctx)?;
    assert_eq!(serialized_id(&narrow)?, decimal_byte_parts_v1_id());
    Ok(())
}
