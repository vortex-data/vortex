// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each integer compression scheme produces the expected encoding.

use std::iter;
use std::sync::LazyLock;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::BtrBlocksCompressor;
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| vortex_array::default_session_builder().build());

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let values: Vec<i32> = iter::repeat_n(42, 100).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_for_compressed() -> VortexResult<()> {
    let values: Vec<i32> = (0..1000).map(|i| 1_000_000 + ((i * 37) % 100)).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<FoR>());
    Ok(())
}

#[test]
fn test_bitpacking_compressed() -> VortexResult<()> {
    let values: Vec<u32> = (0..1000).map(|i| i % 16).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<BitPacked>());
    assert_eq!(
        compressed.statistics().get_as::<u64>(Stat::NullCount),
        Precision::exact(0u64)
    );
    assert_eq!(
        compressed.statistics().get_as::<u32>(Stat::Min),
        Precision::exact(0u32)
    );
    assert_eq!(
        compressed.statistics().get_as::<u32>(Stat::Max),
        Precision::exact(15u32)
    );
    Ok(())
}

#[test]
fn test_sparse_compressed() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    for i in 0..1000 {
        if i % 20 == 0 {
            values.push(2_000_000 + (i * 7) % 1000);
        } else {
            values.push(1_000_000);
        }
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Sparse>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let mut codes = Vec::with_capacity(65_535);
    let numbers: Vec<i32> = [0, 10, 50, 100, 1000, 3000]
        .into_iter()
        .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
        .collect();

    let mut rng = StdRng::seed_from_u64(1u64);
    while codes.len() < 64000 {
        let run_length = rng.next_u32() % 5;
        let value = numbers[rng.next_u32() as usize % numbers.len()];
        for _ in 0..run_length {
            codes.push(value);
        }
    }

    let array = PrimitiveArray::new(Buffer::copy_from(&codes), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Dict>());
    Ok(())
}

#[test]
fn test_runend_compressed() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    for i in 0..100 {
        values.extend(iter::repeat_n((i32::MAX - 50).wrapping_add(i), 10));
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<RunEnd>());
    Ok(())
}

#[test]
fn test_sequence_compressed() -> VortexResult<()> {
    let values: Vec<i32> = (0..1000).map(|i| i * 7).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Sequence>());
    Ok(())
}

#[test]
fn test_rle_compressed() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    for i in 0..1024 {
        // Scramble the per-run value so the data is run-length-dominant but not monotone: this
        // keeps RunEnd the winner instead of Delta (whose residuals would be small on a smooth
        // ramp).
        let v = (i as u32).wrapping_mul(2_654_435_761) as i32;
        values.extend(iter::repeat_n(v, 10));
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    eprintln!("{}", compressed.display_tree());
    assert!(compressed.is::<RunEnd>());
    Ok(())
}

/// A strictly-increasing column with small, irregular steps: not a perfect arithmetic sequence
/// (so Sequence skips), all-unique with no runs (so RunEnd/Dict skip), and a wide absolute range.
/// Delta's residuals are far smaller than the FoR span, so Delta should win and round-trip, and
/// it must appear at most once in the tree.
#[cfg(feature = "unstable_encodings")]
#[test]
fn test_delta_compressed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    use vortex_array::assert_arrays_eq;
    use vortex_fastlanes::Delta;

    let mut rng = StdRng::seed_from_u64(7u64);
    let mut value = 500_000i32;
    let values: Vec<i32> = (0..4096)
        .map(|_| {
            value += 1 + (rng.next_u32() % 6) as i32;
            value
        })
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(
        compressed.is::<Delta>(),
        "expected Delta, got tree:\n{}",
        compressed.display_tree()
    );
    // Delta must appear at most once per tree: no Delta node may be nested under another.
    assert!(
        !has_nested_delta(&compressed, false),
        "Delta was applied more than once in the tree:\n{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);
    Ok(())
}

/// Returns true if any `Delta` array appears below an ancestor `Delta` in the tree.
#[cfg(feature = "unstable_encodings")]
fn has_nested_delta(array: &vortex_array::ArrayRef, under_delta: bool) -> bool {
    use vortex_fastlanes::Delta;

    let is_delta = array.is::<Delta>();
    if is_delta && under_delta {
        return true;
    }
    array
        .children()
        .iter()
        .any(|child| has_nested_delta(child, under_delta || is_delta))
}
