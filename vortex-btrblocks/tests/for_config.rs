// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FoR scheme refinement by allowed serialized IDs, and per-chunk references.

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::schemes::integer::FoRScheme;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_fastlanes::for_v1_id;
use vortex_fastlanes::for_v2_id;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static FOR_V1: FoRScheme = FoRScheme::v1();
static BITPACKING: BitPackingScheme = BitPackingScheme;

/// Registers the fastlanes encodings, and enables no editions.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

/// Like [`SESSION`], with the latest core edition enabled: it allows `fastlanes.for` but not
/// `fastlanes.for.v2`.
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
    vortex_fastlanes::initialize(&session);
    session
});

/// Values from a billion that step up by a million every chunk. One reference needs 23 bits and
/// plain BitPacking 30, so FoR is estimated to pay off, and per-chunk references then pack to 7.
fn drifting() -> ArrayRef {
    PrimitiveArray::from_iter(
        (0..8192u32).map(|i| 1_000_000_000 + (i / 1024) * 1_000_000 + i % 100),
    )
    .into_array()
}

/// Values clustered around one base: every chunk has the same minimum, so the references compress
/// to a constant and the array serializes as `fastlanes.for`.
fn clustered() -> ArrayRef {
    PrimitiveArray::from_iter((0..8192u32).map(|i| 1_000_000 + i % 100)).into_array()
}

/// Compresses `array`, round trips it through serialization, and returns the serialized IDs.
fn compress_roundtrip(
    builder: BtrBlocksCompressorBuilder,
    array: &ArrayRef,
) -> VortexResult<Vec<ArrayId>> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder.build().compress(array, &mut ctx)?;
    assert_arrays_eq!(array, compressed, &mut ctx);

    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in compressed.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    let read = SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;
    assert_arrays_eq!(array, read, &mut ctx);
    Ok(array_ctx.to_ids())
}

/// Only FoR and BitPacking, with every serialized ID allowed, so FoR refines to v2.
fn for_only() -> BtrBlocksCompressorBuilder {
    BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FOR_V1)
        .with_new_scheme(&BITPACKING)
}

#[rstest]
#[case::drifting(drifting(), true)]
#[case::clustered(clustered(), false)]
fn v2_serializes_varying_references_as_v2(
    #[case] array: ArrayRef,
    #[case] expect_v2: bool,
) -> VortexResult<()> {
    let ids = compress_roundtrip(for_only(), &array)?;
    assert!(ids.contains(&for_v1_id()) != expect_v2);
    assert_eq!(ids.contains(&for_v2_id()), expect_v2);
    Ok(())
}

/// FoR is estimated as single-reference FoR, which is conservative for v2. Here one reference needs
/// 23 bits, as many as plain BitPacking, so FoR is skipped even though per-chunk references would
/// pack to 7.
#[test]
fn estimate_skips_arrays_only_chunk_references_narrow() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter(
        (0..8192u32).map(|i| 1_000_000 + (i / 1024) * 1_000_000 + i % 100),
    )
    .into_array();
    let ids = compress_roundtrip(for_only(), &array)?;
    assert!(!ids.contains(&for_v1_id()));
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn core_edition_keeps_single_reference() -> VortexResult<()> {
    let ids = compress_roundtrip(
        BtrBlocksCompressorBuilder::from_session(&CORE_SESSION),
        &drifting(),
    )?;
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn nullable_drifting_roundtrip() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter(
        (0..8192i64).map(|i| (i % 7 != 0).then_some(-5_000_000 + (i / 1024) * 1_000_000 + i % 50)),
    )
    .into_array();
    let ids = compress_roundtrip(for_only(), &array)?;
    assert!(ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn cuda_preset_keeps_single_reference() -> VortexResult<()> {
    let ids = compress_roundtrip(for_only().only_cuda_compatible(), &drifting())?;
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}
