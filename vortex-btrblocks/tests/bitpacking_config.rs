// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking scheme refinement by allowed serialized IDs, and per-block bit widths.

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpacked_v1_id;
use vortex_fastlanes::bitpacked_v2_id;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static BITPACKING_V1: BitPackingScheme = BitPackingScheme::v1();

/// Registers the fastlanes encodings, and enables no editions.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    session.arrays().register(Patched);
    vortex_fastlanes::initialize(&session);
    session
});

/// Like [`SESSION`], with the latest core edition enabled: it allows `fastlanes.bitpacked` but not
/// `fastlanes.bitpacked.v2`.
static CORE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<EditionSession>();
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

/// Values whose 1024-value blocks need 1 to 8 bits.
fn drifting() -> ArrayRef {
    PrimitiveArray::from_iter((0..8192u32).map(|i| i % (2 << (i / 1024)))).into_array()
}

/// Values that need 7 bits in every block.
fn uniform() -> ArrayRef {
    PrimitiveArray::from_iter((0..8192u32).map(|i| i % 128)).into_array()
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

/// Only BitPacking, with every serialized ID allowed, so it refines to v2.
fn bitpacking_only() -> BtrBlocksCompressorBuilder {
    BtrBlocksCompressorBuilder::empty().with_new_scheme(&BITPACKING_V1)
}

#[rstest]
#[case::drifting(drifting(), Some(bitpacked_v2_id()))]
#[case::uniform(uniform(), Some(bitpacked_v1_id()))]
#[case::uniform_with_patches(
    PrimitiveArray::from_iter(
        (0..8192u32).map(|i| if i % 1024 == 0 { 1 << 20 } else { i % 128 }),
    )
    .into_array(),
    Some(bitpacked_v1_id())
)]
#[case::native_u8(
    PrimitiveArray::from_iter((128u8..=u8::MAX).cycle().take(8192)).into_array(),
    None
)]
#[case::native_u32(
    PrimitiveArray::from_iter((0..8192u32).map(|i| (1 << 31) + i)).into_array(),
    None
)]
#[case::empty(PrimitiveArray::from_iter(Vec::<u32>::new()).into_array(), None)]
fn v2_selects_wire_format(
    #[case] array: ArrayRef,
    #[case] expected_id: Option<ArrayId>,
) -> VortexResult<()> {
    let ids = compress_roundtrip(bitpacking_only(), &array)?;
    assert_eq!(
        ids.contains(&bitpacked_v1_id()),
        expected_id == Some(bitpacked_v1_id())
    );
    assert_eq!(
        ids.contains(&bitpacked_v2_id()),
        expected_id == Some(bitpacked_v2_id())
    );
    Ok(())
}

#[test]
fn nullable_drifting_roundtrip() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter(
        (0..8192u64).map(|i| (i % 7 != 0).then_some(i % (2 << (i / 1024)))),
    )
    .into_array();
    let ids = compress_roundtrip(bitpacking_only(), &array)?;
    assert!(ids.contains(&bitpacked_v2_id()));
    Ok(())
}

#[test]
fn core_edition_keeps_global_width() -> VortexResult<()> {
    let ids = compress_roundtrip(
        BtrBlocksCompressorBuilder::from_session(&CORE_SESSION),
        &drifting(),
    )?;
    assert!(!ids.contains(&bitpacked_v2_id()));
    Ok(())
}

#[test]
fn cuda_preset_keeps_global_width() -> VortexResult<()> {
    let ids = compress_roundtrip(bitpacking_only().only_cuda_compatible(), &drifting())?;
    assert!(ids.contains(&bitpacked_v1_id()));
    assert!(!ids.contains(&bitpacked_v2_id()));
    Ok(())
}
