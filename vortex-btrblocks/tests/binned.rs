// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compact preset picks `vortex.binned` once its opt-in edition is enabled.

#![cfg(feature = "binned")]
#![allow(clippy::tests_outside_test_module)]

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

fn session(enable_binned: bool) -> VortexResult<VortexSession> {
    let session = vortex_array::array_session().with::<EditionSession>();
    vortex_alp::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_runend::initialize(&session);
    vortex_sequence::initialize(&session);
    vortex_sparse::initialize(&session);
    vortex_zigzag::initialize(&session);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(CORE_2026_08_3)?;
    vortex_binned::initialize(&session);
    if enable_binned {
        session.enable_edition(vortex_binned::editions::BINNED_2026_10)?;
    }
    Ok(session)
}

/// Sensor-like readings with two decimals: a random walk, so neither dictionary nor run
/// encodings fit and entropy coding pays.
fn readings() -> ArrayRef {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut level = 2_000i64;
    PrimitiveArray::from_iter((0..65_536).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        level += (x % 41) as i64 - 20;
        level as f64 / 100.0
    }))
    .into_array()
}

fn uses_binned(array: &ArrayRef) -> bool {
    array.tree_display().to_string().contains("vortex.binned")
}

#[test]
fn compact_picks_binned_only_when_enabled() -> VortexResult<()> {
    let input = readings();
    for enabled in [false, true] {
        let session = session(enabled)?;
        let compressor = BtrBlocksCompressorBuilder::from_session(&session)
            .with_compact()
            .build();
        let mut ctx = session.create_execution_ctx();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert_eq!(uses_binned(&compressed), enabled, "{}", compressed.tree_display());
        assert_arrays_eq!(compressed, input, &mut ctx);
    }
    Ok(())
}
