// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal format selection and compatibility restrictions on configured schemes.

#![cfg(test)]

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
use vortex_btrblocks::Scheme;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::schemes::integer::FoRScheme;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

static DECIMAL_V1: DecimalScheme = DecimalScheme::new(false);
static DECIMAL_V2: DecimalScheme = DecimalScheme::new(true);

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

fn assert_serialized_id(array: &ArrayRef, expected: ArrayId) -> VortexResult<()> {
    let serialized = SESSION
        .array_serialize(array)?
        .ok_or_else(|| vortex_err!("expected serializable decimal byte parts"))?;
    assert_eq!(serialized.serialized_id, expected);
    Ok(())
}

#[rstest]
#[case::neither(vec![], false)]
#[case::v1(vec![decimal_byte_parts_v1_id()], true)]
#[case::v2_only(vec![decimal_byte_parts_v2_id()], false)]
#[case::both(vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()], true)]
fn supported_ids(#[case] ids: Vec<ArrayId>, #[case] supported: bool) {
    let compressor = BtrBlocksCompressorBuilder::new(&ids.into_iter().collect()).build();
    assert_eq!(
        compressor.has_scheme(DecimalScheme::default().id()),
        supported
    );
}

#[test]
fn v1_writer_rejects_preconfigured_v2_scheme() {
    let scheme = &DECIMAL_V2;
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .retain_allowed_encodings(&HashSet::from([decimal_byte_parts_v1_id()]))
        .build();
    assert!(!compressor.has_scheme(scheme.id()));
}

#[test]
fn decimal_defaults_to_v1() {
    assert_eq!(
        DecimalScheme::default().produced_encodings(),
        vec![decimal_byte_parts_v1_id()],
    );
}

#[rstest]
fn cuda_preset_keeps_later_v1_registration(#[values(false, true)] restrict_ids: bool) {
    let id = DecimalScheme::default().id();
    let mut builder = BtrBlocksCompressorBuilder::empty()
        .only_cuda_compatible()
        .with_new_scheme(&DECIMAL_V1);
    if restrict_ids {
        builder = builder.retain_allowed_encodings(&HashSet::from([
            decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
        ]));
    }
    assert!(builder.build().has_scheme(id));
}

#[rstest]
fn cuda_preset_removes_explicit_v2(#[values(false, true)] wide: bool) -> VortexResult<()> {
    let array = decimal_array(wide);
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&DECIMAL_V2)
        .only_cuda_compatible()
        .build();
    assert!(!compressor.has_scheme(DECIMAL_V2.id()));
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;
    assert!(!compressed.is::<DecimalByteParts>());
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}

#[rstest]
fn decimal_format_follows_configuration(#[values(false, true)] wide: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = decimal_array(wide);
    let builder = BtrBlocksCompressorBuilder::default();
    let default = builder.clone().build();
    let v1 = BtrBlocksCompressorBuilder::new(&HashSet::from([decimal_byte_parts_v1_id()])).build();
    let v2 = BtrBlocksCompressorBuilder::new(&HashSet::from([
        decimal_byte_parts_v1_id(),
        decimal_byte_parts_v2_id(),
    ]))
    .build();
    let default_again = builder.build();

    for (compressor, v2) in [
        (&default, false),
        (&v1, false),
        (&v2, true),
        (&v1, false),
        (&default_again, false),
    ] {
        let compressed = compressor.compress(&array, &mut ctx)?;
        assert_eq!(compressed.is::<DecimalByteParts>(), !wide || v2);
        if compressed.is::<DecimalByteParts>() {
            assert_serialized_id(
                &compressed,
                if wide {
                    decimal_byte_parts_v2_id()
                } else {
                    decimal_byte_parts_v1_id()
                },
            )?;
        }
        assert_arrays_eq!(array, compressed, &mut ctx);
    }
    Ok(())
}

#[rstest]
fn explicit_decimal_format_is_preserved(
    #[values(false, true)] v2: bool,
    #[values(false, true)] restrict_ids: bool,
    #[values(false, true)] replace_default: bool,
) -> VortexResult<()> {
    let array = decimal_array(true);
    let scheme = if v2 { &DECIMAL_V2 } else { &DECIMAL_V1 };
    let mut builder = if replace_default {
        BtrBlocksCompressorBuilder::default().exclude_schemes([scheme.id()])
    } else {
        BtrBlocksCompressorBuilder::empty()
    }
    .with_new_scheme(scheme);
    if restrict_ids {
        builder = builder.retain_allowed_encodings(&HashSet::from([
            decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
        ]));
    }
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder.build().compress(&array, &mut ctx)?;
    assert_eq!(compressed.is::<DecimalByteParts>(), v2);
    if v2 {
        assert_serialized_id(&compressed, decimal_byte_parts_v2_id())?;
    }
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}

#[test]
fn filtering_removes_incompatible_decimal_without_reconfiguration() -> VortexResult<()> {
    let array = decimal_array(true);
    let v1 = HashSet::from([decimal_byte_parts_v1_id()]);
    let both = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    let compressor = BtrBlocksCompressorBuilder::new(&both)
        .retain_allowed_encodings(&v1)
        .build();
    assert!(!compressor.has_scheme(DECIMAL_V1.id()));
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;
    assert!(!compressed.is::<DecimalByteParts>());
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}

#[rstest]
fn varying_decimal_parts_require_child_compression(
    #[values(false, true)] compress_children: bool,
) -> VortexResult<()> {
    let array = DecimalArray::new(
        (1..=2048i128)
            .map(|i| (i << 70) + i * 131 + 17)
            .collect::<Buffer<i128>>(),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    )
    .into_array();
    let mut allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
    if compress_children {
        allowed.extend(FoRScheme.produced_encodings());
        allowed.extend(BitPackingScheme.produced_encodings());
    }
    let compressor = BtrBlocksCompressorBuilder::new(&allowed).build();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;
    // Without integer schemes, distinct parts occupy the same space as canonical decimals.
    assert_eq!(compressed.is::<DecimalByteParts>(), compress_children);
    if compress_children {
        assert_serialized_id(&compressed, decimal_byte_parts_v2_id())?;
    }
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}

#[rstest]
fn cuda_preset_keeps_only_decimal_v1(
    #[values(false, true)] wide: bool,
    #[values(false, true)] v2: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = decimal_array(wide);
    let builder = if v2 {
        BtrBlocksCompressorBuilder::new(&HashSet::from([
            decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
        ]))
    } else {
        BtrBlocksCompressorBuilder::default()
    };
    let compressor = builder.only_cuda_compatible().build();
    assert_eq!(compressor.has_scheme(DECIMAL_V1.id()), !v2);
    let compressed = compressor.compress(&array, &mut ctx)?;
    assert_eq!(compressed.is::<DecimalByteParts>(), !wide && !v2);
    if compressed.is::<DecimalByteParts>() {
        assert_serialized_id(&compressed, decimal_byte_parts_v1_id())?;
    }
    assert_arrays_eq!(array, compressed, &mut ctx);
    Ok(())
}
