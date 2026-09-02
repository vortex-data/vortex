// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each integer compression scheme produces the expected encoding.

use std::iter;
use std::sync::LazyLock;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
#[cfg(feature = "unstable_encodings")]
use vortex_array::ArrayEq;
#[cfg(feature = "unstable_encodings")]
use vortex_array::EqMode;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::validity::Validity;
#[cfg(feature = "unstable_encodings")]
use vortex_block_residual::BlockResidual;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::BtrBlocksCompressor;
#[cfg(feature = "unstable_encodings")]
use crate::BtrBlocksCompressorBuilder;
#[cfg(feature = "unstable_encodings")]
use crate::SchemeExt;
#[cfg(feature = "unstable_encodings")]
use crate::schemes::integer::DeltaScheme;
static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

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

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_compressed() -> VortexResult<()> {
    let values = (0..8_192)
        .map(|index| {
            let block = index / 1_024;
            let residual = (index * 2_654_435_761_usize) % 1_024;
            (block as i64 - 4) * 1_000_000_000_000 + residual as i64
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values);
    #[cfg(not(feature = "unstable_encodings"))]
    let compressor = BtrBlocksCompressor::default();
    #[cfg(feature = "unstable_encodings")]
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DeltaScheme::default().id()])
        .build();
    let compressed =
        compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

    assert!(
        compressed.is::<BlockResidual>(),
        "expected BlockResidual, got tree:\n{}",
        compressed.display_tree()
    );
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_ignores_null_payloads() -> VortexResult<()> {
    let values = (0usize..8_192)
        .map(|index| {
            let block = index / 1_024;
            let residual = index.wrapping_mul(2_654_435_761) % 1_024;
            (block as i64 - 4) * 1_000_000_000_000 + residual as i64
        })
        .collect::<Vec<_>>();
    let validity = Validity::from_iter((0..values.len()).map(|index| index % 17 != 0));
    let mut alternate = values.clone();
    for index in (0..alternate.len()).step_by(17) {
        alternate[index] = i64::MAX - index as i64;
    }
    let first = PrimitiveArray::new(Buffer::copy_from(&values), validity.clone()).into_array();
    let second = PrimitiveArray::new(Buffer::copy_from(&alternate), validity).into_array();
    let compressor = BtrBlocksCompressor::default();
    let first = compressor.compress(&first, &mut SESSION.create_execution_ctx())?;
    let second = compressor.compress(&second, &mut SESSION.create_execution_ctx())?;

    assert!(first.is::<BlockResidual>());
    assert!(second.is::<BlockResidual>());
    assert!(first.array_eq(&second, EqMode::Value));
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_compresses_16_bit_integers() -> VortexResult<()> {
    let signed_values = (0..8_192)
        .map(|index| {
            let block = index / 1_024;
            let residual = (index * 2_654_435_761_usize) % 32;
            Ok(i16::try_from(block * 1_000 + residual)?)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let unsigned_values = signed_values
        .iter()
        .copied()
        .map(u16::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "unstable_encodings"))]
    let compressor = BtrBlocksCompressor::default();
    #[cfg(feature = "unstable_encodings")]
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DeltaScheme::default().id()])
        .build();

    for array in [
        PrimitiveArray::from_iter(signed_values),
        PrimitiveArray::from_iter(unsigned_values),
    ] {
        let compressed =
            compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(
            contains_block_residual(&compressed),
            "BlockResidual must encode this 16-bit input:\n{}",
            compressed.display_tree()
        );
    }
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_compresses_8_bit_integers() -> VortexResult<()> {
    let unsigned_values = (0..16_384)
        .map(|index| {
            let block = (index / 1_024) % 32;
            let residual = (index * 2_654_435_761_usize) % 8;
            u8::try_from(block * 8 + residual)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signed_values = unsigned_values
        .iter()
        .copied()
        .map(|value| i8::from_le_bytes([value]))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "unstable_encodings"))]
    let compressor = BtrBlocksCompressor::default();
    #[cfg(feature = "unstable_encodings")]
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DeltaScheme::default().id()])
        .build();

    for array in [
        PrimitiveArray::from_iter(signed_values),
        PrimitiveArray::from_iter(unsigned_values),
    ] {
        let compressed =
            compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(
            contains_block_residual(&compressed),
            "BlockResidual must encode this 8-bit input:\n{}",
            compressed.display_tree()
        );
    }
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_rejects_weak_8_bit_gain() -> VortexResult<()> {
    let unsigned_values = (0..16_384)
        .map(|index| {
            let block = (index / 1_024) % 2;
            let residual = (index * 2_654_435_761_usize) % 128;
            u8::try_from(block * 128 + residual)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signed_values = unsigned_values
        .iter()
        .copied()
        .map(|value| i8::from_le_bytes([value]))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "unstable_encodings"))]
    let compressor = BtrBlocksCompressor::default();
    #[cfg(feature = "unstable_encodings")]
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DeltaScheme::default().id()])
        .build();

    for array in [
        PrimitiveArray::from_iter(signed_values),
        PrimitiveArray::from_iter(unsigned_values),
    ] {
        let compressed =
            compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(
            !contains_block_residual(&compressed),
            "BlockResidual must reject this weak 8-bit gain:\n{}",
            compressed.display_tree()
        );
    }
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_rejects_uniform_8_bit_integers() -> VortexResult<()> {
    let unsigned_values = (0..16_384)
        .map(|index| u8::try_from((index * 2_654_435_761_usize) % 256))
        .collect::<Result<Vec<_>, _>>()?;
    let signed_values = unsigned_values
        .iter()
        .copied()
        .map(|value| i8::from_le_bytes([value]))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "unstable_encodings"))]
    let compressor = BtrBlocksCompressor::default();
    #[cfg(feature = "unstable_encodings")]
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DeltaScheme::default().id()])
        .build();

    for array in [
        PrimitiveArray::from_iter(signed_values),
        PrimitiveArray::from_iter(unsigned_values),
    ] {
        let compressed =
            compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(
            !contains_block_residual(&compressed),
            "BlockResidual must reject this uniform 8-bit input:\n{}",
            compressed.display_tree()
        );
    }
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_rejects_dense_patches() -> VortexResult<()> {
    let values = (0..8_192_u32).map(|index| if index % 4 == 0 { u32::MAX - index } else { 42 });
    let array = PrimitiveArray::from_iter(values);
    let compressed = BtrBlocksCompressor::default()
        .compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

    assert!(
        !contains_block_residual(&compressed),
        "dense patches must not select BlockResidual:\n{}",
        compressed.display_tree()
    );
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_composes_with_sparse() -> VortexResult<()> {
    let values = (0..65_536_usize).map(|index| {
        if index % 16 == 0 {
            let value_index = index / 16;
            let block = value_index / 1_024;
            let residual = value_index.wrapping_mul(2_654_435_761) % 1_024;
            block as u64 * 1_000_000_000_000 + residual as u64
        } else {
            42
        }
    });
    let array = PrimitiveArray::from_iter(values);
    let compressed = BtrBlocksCompressor::default()
        .compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

    assert!(compressed.is::<Sparse>());
    assert!(
        contains_block_residual(&compressed),
        "expected a BlockResidual child:\n{}",
        compressed.display_tree()
    );
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_composes_with_runend() -> VortexResult<()> {
    let values = (0..65_536_usize).map(|index| {
        let value_index = index / 16;
        let block = value_index / 1_024;
        let residual = value_index.wrapping_mul(2_654_435_761) % 1_024;
        block as u64 * 1_000_000_000_000 + residual as u64
    });
    let array = PrimitiveArray::from_iter(values);
    let compressed = BtrBlocksCompressor::default()
        .compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

    assert!(compressed.is::<RunEnd>());
    assert!(
        contains_block_residual(&compressed),
        "expected a BlockResidual child:\n{}",
        compressed.display_tree()
    );
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
fn contains_block_residual(array: &vortex_array::ArrayRef) -> bool {
    array.is::<BlockResidual>() || array.children().iter().any(contains_block_residual)
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

/// Same as [`test_delta_compressed`], but with a length that is not a multiple of 1024.
/// Zero-padding the trailing chunk used to inflate the delta span and cause DeltaScheme to skip.
#[cfg(feature = "unstable_encodings")]
#[test]
fn test_delta_compressed_unaligned_length() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    use vortex_array::assert_arrays_eq;
    use vortex_fastlanes::Delta;

    let mut rng = StdRng::seed_from_u64(7u64);
    let mut value = 500_000i32;
    let values: Vec<i32> = (0..1025)
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
        "expected Delta for unaligned near-monotone column, got tree:\n{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);
    Ok(())
}

/// Nullable unaligned monotone must round-trip through Delta (and a cascaded residual).
///
/// Mirrors `duckdb/aggregate_pushdown.slt`: `NULL` then `1..=100000` (length 100001).
#[cfg(feature = "unstable_encodings")]
#[test]
fn test_delta_nullable_unaligned_sum() -> VortexResult<()> {
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::assert_arrays_eq;
    use vortex_fastlanes::Delta;

    let mut ctx = SESSION.create_execution_ctx();
    let array =
        PrimitiveArray::from_option_iter(iter::once(None).chain((1i32..=100_000).map(Some)));

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.clone().into_array(), &mut ctx)?;
    assert!(
        compressed.is::<Delta>(),
        "expected Delta, got tree:\n{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);

    let expected_sum: i64 = (1i64..=100_000).sum();
    assert_eq!(
        sum(&compressed, &mut ctx)?.as_primitive().as_::<i64>(),
        Some(expected_sum),
    );
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
