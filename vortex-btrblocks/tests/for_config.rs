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
use vortex_btrblocks::CompressionSession;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::schemes::integer::FoRScheme;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_btrblocks::schemes::integer::RunEndScheme;
use vortex_btrblocks::schemes::integer::SequenceScheme;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_edition::declarations::unstable::UNSTABLE;
use vortex_error::VortexResult;
use vortex_fastlanes::for_v1_id;
use vortex_fastlanes::for_v2_id;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static FOR_V1: FoRScheme = FoRScheme::v1();

/// Registers the fastlanes encodings, and enables no editions.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<CompressionSession>();
    vortex_fastlanes::initialize(&session);
    vortex_sequence::initialize(&session);
    session
});

/// Like [`SESSION`], with the latest core edition enabled: it allows `fastlanes.for` but not
/// `fastlanes.for.v2`.
static CORE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<CompressionSession>();
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

/// Core + UNSTABLE editions enabled: allows `fastlanes.for.v2`.
static UNSTABLE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<CompressionSession>();
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
    session
        .enable_edition(UNSTABLE)
        .expect("unstable edition is registered");
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
        .with_new_scheme(&BitPackingScheme)
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

/// With the 8x1024 locality selector, FoR v2 catches drifting arrays whose single-reference width
/// ties plain BitPacking as well as drifting arrays whose global minimum is zero.
#[rstest]
#[case::global_tie(1_000_000u32)]
#[case::min_zero(0u32)]
fn estimate_catches_drifting_arrays_only_chunk_references_narrow(
    #[case] base: u32,
) -> VortexResult<()> {
    let array =
        PrimitiveArray::from_iter((0..8192u32).map(|i| base + (i / 1024) * 1_000_000 + i % 100))
            .into_array();
    let ids = compress_roundtrip(for_only(), &array)?;
    assert!(!ids.contains(&for_v1_id()));
    assert!(ids.contains(&for_v2_id()));
    Ok(())
}

/// When per-chunk references do not beat FoR v1 by the required margin, FoR v2 falls back to
/// FoR v1 and preserves the v1 estimate floor so Dict does not displace it on ties.
#[test]
fn v2_falls_back_to_v1_and_preserves_v1_floor() -> VortexResult<()> {
    // Values in [10_000, 14_000] uniformly spread in every chunk: global min = 10_000,
    // chunk mins are nearly identical so v2 saves < 5%, while v1 needs 12 bits (2.67x).
    let array = PrimitiveArray::from_iter(
        (0..16_384i32).map(|i| 10_000 + (i / 1024) * 3 + ((i * 37) % 3_500)),
    )
    .into_array();
    let builder = for_only().with_new_scheme(&IntDictScheme);
    let ids = compress_roundtrip(builder, &array)?;
    assert!(ids.contains(&for_v1_id()));
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}

/// `RunEnd.ends` is excluded from `FoR.v2` so binary-search `scalar_at` probes do not incur
/// extra blocked-reference indirection.
#[test]
fn runend_ends_excludes_for_v2() -> VortexResult<()> {
    // 12,000 runs of length 16 -> `ends` has 12,000 monotonically increasing u32 values starting
    // at 16, where blocked FoR.v2 would otherwise beat BitPacked.
    let array = PrimitiveArray::from_iter((0..192_000i32).map(|i| (i / 16) % 50)).into_array();
    let builder = for_only().with_new_scheme(&RunEndScheme);
    let ids = compress_roundtrip(builder, &array)?;
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn core_edition_keeps_single_reference() -> VortexResult<()> {
    let ids = compress_roundtrip(
        BtrBlocksCompressorBuilder::from_session(&CORE_SESSION),
        &drifting(),
    )?;
    assert!(ids.contains(&for_v1_id()));
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn unstable_edition_enables_for_v2() -> VortexResult<()> {
    let ids = compress_roundtrip(
        BtrBlocksCompressorBuilder::from_session(&UNSTABLE_SESSION),
        &drifting(),
    )?;
    assert!(ids.contains(&for_v2_id()));
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
    assert!(ids.contains(&for_v1_id()));
    assert!(!ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn v2_handles_wide_global_span_with_narrow_blocks() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter((0..2048i64).map(|i| {
        if i < 1024 {
            -6_000_000_000_000_000_000 + (i % 100)
        } else {
            6_000_000_000_000_000_000 + (i % 100)
        }
    }))
    .into_array();
    let ids = compress_roundtrip(for_only(), &array)?;
    assert!(ids.contains(&for_v2_id()));
    Ok(())
}

#[test]
fn v2_handles_wide_span_inside_block() -> VortexResult<()> {
    // First 8 chunks have narrow local span so sample picks v2; chunk 9 spans > i64::MAX so
    // full-array signed wrapping subtraction produces negative residuals.
    let array = PrimitiveArray::from_iter((0..10_240i64).map(|i| {
        let chunk = i / 1024;
        if chunk == 9 {
            if i % 2 == 0 {
                -6_000_000_000_000_000_000
            } else {
                6_000_000_000_000_000_000
            }
        } else {
            chunk * 1_000_000 + (i % 50)
        }
    }))
    .into_array();
    let _ids = compress_roundtrip(for_only(), &array)?;
    Ok(())
}

#[test]
fn v2_selects_zero_buffer_sequence_references() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter((0..2049i64).map(|i| {
        let block = i / 1024;
        1_000_000_000 + block * 1_000_000
    }))
    .into_array();
    let builder = for_only().with_new_scheme(&SequenceScheme);
    let ids = compress_roundtrip(builder, &array)?;
    assert!(ids.contains(&for_v2_id()));
    Ok(())
}
