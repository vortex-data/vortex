// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use tracing_test::traced_test;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::execute_tq_decode;
use super::execute_tq_encode;
use super::f32_vector_array;
use super::test_session;
use super::turboquant_storage;
use super::vector_array;
use super::vector_values_f32;
use crate::TurboQuantConfig;
use crate::vtable::tq_metadata;

#[rstest]
#[case::dim_64_default(64, None, vec![64])]
#[case::dim_128_default(128, None, vec![128])]
#[case::dim_768_default(768, None, vec![1024])]
#[case::dim_768_explicit(768, Some(vec![512, 256]), vec![512, 256])]
#[case::dim_384(384, Some(vec![256, 128]), vec![256, 128])]
#[case::dim_1536(1536, Some(vec![1024, 512]), vec![1024, 512])]
#[case::dim_837_with_overspill(837, Some(vec![512, 256, 64, 64]), vec![512, 256, 64, 64])]
fn encode_decode_roundtrip(
    #[case] dim: u32,
    #[case] config_block_sizes: Option<Vec<u32>>,
    #[case] expected_block_sizes: Vec<u32>,
) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(dim, 4, 0.125, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 17, 2, config_block_sizes)?;

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let metadata = tq_metadata(encoded.dtype())?;
    assert_eq!(metadata.block_sizes, expected_block_sizes);
    assert_eq!(metadata.dimensions, dim);

    let decoded = execute_tq_decode(encoded, &mut ctx)?;
    let decoded_values = vector_values_f32(decoded, &mut ctx)?;
    assert_eq!(decoded_values.len(), 4 * dim as usize);
    Ok(())
}

#[test]
fn encode_rejects_block_sizes_with_sum_less_than_dim() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 1, 1.0, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(2, 42, 3, Some(vec![64]))?;
    assert!(execute_tq_encode(input, &config, &mut ctx).is_err());
    Ok(())
}

#[test]
#[traced_test]
fn encode_warns_on_overspilling_final_block() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    // `dim = 65` with `block_sizes = [64, 64, 64]`. The third block (positions 128..192) starts
    // entirely past `dim = 65`, so `resolve_block_sizes` should fire its
    // "lies entirely past dimensions" warning for `block_index = 2`.
    let input = f32_vector_array(65, 1, 1.0, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(2, 42, 3, Some(vec![64, 64, 64]))?;
    let _encoded = execute_tq_encode(input, &config, &mut ctx)?;
    assert!(logs_contain("lies entirely past dimensions"));
    Ok(())
}

#[test]
#[traced_test]
fn encode_warns_on_sum_more_than_double_dimensions() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    // `dim = 128`, `block_sizes = [256, 256]` sums to `512 > 2 * 128`.
    let input = f32_vector_array(128, 1, 1.0, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(2, 42, 3, Some(vec![256, 256]))?;
    let _encoded = execute_tq_encode(input, &config, &mut ctx)?;
    assert!(logs_contain("exceeds 2 * dimensions"));
    Ok(())
}

/// Encode and decode a synthetic vector array and confirm that per-block normalized MSE stays
/// below an empirical bound that shrinks as `1 / 2^(2 * bit_width)`. The bound is loose enough
/// to be flake-free but tight enough to catch regressions in the centroid table, the SORF
/// rotation, or the per-block norm round-trip.
///
/// For a `b`-bit Lloyd-Max scalar quantizer applied to coordinates of a randomly rotated
/// unit-norm vector in dimension `d`, the per-coordinate marginal has variance roughly `1/d` and
/// the per-coordinate MSE is roughly `c / (d * 2^(2b))` for some distribution-dependent
/// constant `c`. Summed over the `d` coordinates of a block and normalized by the block's L2
/// norm squared, the expected normalized MSE is on the order of `1 / 2^(2b)`. The empirical
/// bound below is around `8 / 2^(2b)`, well above the theoretical floor but well below the
/// `~1.0` you would see if the centroid lookup or inverse SORF were silently broken.
#[rstest]
#[case::two_bit(2u8)]
#[case::four_bit(4u8)]
#[case::six_bit(6u8)]
fn encode_decode_per_block_mse_within_bound(#[case] bit_width: u8) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let block_sizes: Vec<u32> = vec![512, 256];
    let dim: u32 = block_sizes.iter().sum();
    let rows: usize = 16;

    // Generate deterministic pseudo-random f32 inputs without pulling a PRNG dep. The linear
    // congruential recurrence is good enough to produce coordinates whose per-block L2 norms
    // are non-trivially spread out across rows.
    let total = rows * dim as usize;
    let mut values = vec![0.0f32; total];
    let mut state: u32 = 0x1234_5678;
    for v in values.iter_mut() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        #[expect(
            clippy::cast_precision_loss,
            reason = "f32 precision is sufficient for the synthetic input distribution"
        )]
        let x = ((state as f32) / (u32::MAX as f32 / 2.0)) - 1.0;
        *v = x;
    }

    let input = vector_array(dim, &values, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(bit_width, 7, 3, Some(block_sizes.clone()))?;
    let encoded = execute_tq_encode(input.clone(), &config, &mut ctx)?;
    let decoded = execute_tq_decode(encoded, &mut ctx)?;

    let original = vector_values_f32(input, &mut ctx)?;
    let recovered = vector_values_f32(decoded, &mut ctx)?;

    #[expect(
        clippy::cast_precision_loss,
        reason = "`1 << (2 * bit_width)` fits a u32 for `bit_width <= 8`"
    )]
    let quant_levels_sq = (1u32 << (2 * bit_width)) as f32;
    let max_normalized_mse = 8.0_f32 / quant_levels_sq;
    let dim = dim as usize;
    for row in 0..rows {
        let mut offset = 0usize;
        for (block_index, &block) in block_sizes.iter().enumerate() {
            let block = block as usize;
            let orig = &original[row * dim + offset..][..block];
            let rec = &recovered[row * dim + offset..][..block];
            let norm_sq: f32 = orig.iter().map(|&x| x * x).sum();
            let err_sq: f32 = orig
                .iter()
                .zip(rec.iter())
                .map(|(&o, &r)| (o - r).powi(2))
                .sum();
            // Guard against the degenerate zero-norm row (the LCG above will not produce one in
            // practice, but the guard makes the invariant explicit).
            let normalized_mse = err_sq / norm_sq.max(1e-10);
            assert!(
                normalized_mse < max_normalized_mse,
                "row {row} block {block_index} normalized MSE {normalized_mse} exceeds bound \
                 {max_normalized_mse} for bit_width {bit_width}",
            );
            offset += block;
        }
    }
    Ok(())
}

#[test]
fn same_size_blocks_use_different_seeds() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    // Two identical 128-wide blocks on a 256-dim input. The first 128 coordinates of every row
    // match the second 128 coordinates exactly, so if the blocks shared seeds their `codes`
    // would also match. They should not.
    let mut values = vec![0.0f32; 4 * 256];
    for row in 0..4 {
        for j in 0..128 {
            let v = ((row * 128 + j) as f32) * 0.001;
            values[row * 256 + j] = v;
            values[row * 256 + 128 + j] = v;
        }
    }
    let input = vector_array(256, &values, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 11, 2, Some(vec![128, 128]))?;
    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let outer = turboquant_storage(encoded, &mut ctx)?;
    let block_0_codes = block_codes(&outer, 0, &mut ctx)?;
    let block_1_codes = block_codes(&outer, 1, &mut ctx)?;
    assert_ne!(block_0_codes, block_1_codes);
    Ok(())
}

/// Multi-block null-row coverage: a null outer row must produce zero placeholders on every
/// block's `norms` and `codes` children while valid rows still roundtrip per-block.
#[test]
fn encode_decode_multi_block_null_rows() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let dim = 768u32;
    let rows = 3usize;
    let mut values = vec![0.0f32; rows * dim as usize];
    for (i, v) in values.iter_mut().enumerate() {
        *v = ((i % 13) as f32) * 0.1 + 0.05;
    }
    let validity = Validity::from_iter([true, false, true]);
    let input = vector_array(dim, &values, validity)?;
    let block_sizes = [512usize, 256];
    let config = TurboQuantConfig::try_new(4, 23, 3, Some(vec![512, 256]))?;
    let encoded = execute_tq_encode(input, &config, &mut ctx)?;

    // The null row (row 1) must store zero placeholders in every block's `norms` and `codes`, not
    // merely reconstruct to zero on decode.
    let outer = turboquant_storage(encoded.clone(), &mut ctx)?;
    for (block_index, &block) in block_sizes.iter().enumerate() {
        let norms = block_norms(&outer, block_index, &mut ctx)?;
        let codes = block_codes(&outer, block_index, &mut ctx)?;
        assert_eq!(
            norms[1], 0.0,
            "block {block_index} null-row norm must be zero"
        );
        assert!(
            codes[block..2 * block].iter().all(|&c| c == 0),
            "block {block_index} null-row codes must be zero"
        );
    }

    let decoded = execute_tq_decode(encoded, &mut ctx)?;
    let validity =
        super::vector_validity(decoded.clone(), &mut ctx)?.execute_mask(rows, &mut ctx)?;
    assert!(validity.value(0));
    assert!(!validity.value(1));
    assert!(validity.value(2));
    // The null row's reconstructed coordinates should be zero placeholders.
    let values = vector_values_f32(decoded, &mut ctx)?;
    let null_row = &values[dim as usize..2 * dim as usize];
    assert!(null_row.iter().all(|&v| v == 0.0));
    Ok(())
}

/// Multi-block per-block zero-norm coverage: a row whose first block's slice is entirely zero
/// but whose second block carries energy must reconstruct the second block correctly while
/// leaving the first block at zero (the per-block `norm = 0` placeholder path).
#[test]
fn encode_decode_multi_block_zero_norm_block() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let dim = 768u32;
    let rows = 2usize;
    let mut values = vec![0.0f32; rows * dim as usize];
    // Row 0 valid everywhere; row 1 zero in `block_0` (positions 0..512), nonzero in `block_1`.
    for (i, v) in values[..dim as usize].iter_mut().enumerate() {
        *v = ((i % 13) as f32) * 0.1 + 0.05;
    }
    for v in values[dim as usize + 512..2 * dim as usize].iter_mut() {
        *v = 0.5;
    }
    let input = vector_array(dim, &values, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(4, 29, 3, Some(vec![512, 256]))?;
    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let decoded = execute_tq_decode(encoded, &mut ctx)?;
    let recovered = vector_values_f32(decoded, &mut ctx)?;
    let row1_block0 = &recovered[dim as usize..dim as usize + 512];
    let row1_block1 = &recovered[dim as usize + 512..2 * dim as usize];
    assert!(
        row1_block0.iter().all(|&v| v == 0.0),
        "zero-norm block expected to reconstruct as zeros"
    );
    let block1_energy: f32 = row1_block1.iter().map(|&v| v * v).sum();
    assert!(
        block1_energy > 0.0,
        "nonzero block expected to recover energy"
    );
    Ok(())
}

/// A dimension whose next power of two overflows `u32` must produce a clean error from the default
/// (`block_sizes = None`) path rather than panicking. Regression for the overflow guard that the
/// block-decomposition refactor dropped from the old `tq_padded_dim`.
#[test]
fn encode_rejects_dimension_overflow() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = vector_array::<f32>(2_147_483_649, &[], Validity::NonNullable)?;
    assert!(execute_tq_encode(input, &TurboQuantConfig::default(), &mut ctx).is_err());
    Ok(())
}

/// A finite f64 value far above `f32::MAX` casts to inf in the f32 quantization pipeline, making
/// the block norm non-finite. Encode must reject it cleanly rather than emit corrupt codes.
#[test]
fn encode_rejects_non_finite_f64_norm() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let mut values = vec![0.0f64; 64];
    values[0] = 1e300;
    let input = vector_array::<f64>(64, &values, Validity::NonNullable)?;
    assert!(execute_tq_encode(input, &TurboQuantConfig::default(), &mut ctx).is_err());
    Ok(())
}

/// A well-typed input that already contains a NaN or infinite coordinate makes the block norm
/// non-finite; encode must reject it cleanly (the guard is otherwise only reached via f64
/// overflow, so this pins the direct non-finite-input path).
#[rstest]
#[case::nan(f32::NAN)]
#[case::pos_inf(f32::INFINITY)]
#[case::neg_inf(f32::NEG_INFINITY)]
fn encode_rejects_non_finite_coordinate(#[case] bad: f32) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let mut values = vec![0.5f32; 64];
    values[0] = bad;
    let input = vector_array(64, &values, Validity::NonNullable)?;
    assert!(execute_tq_encode(input, &TurboQuantConfig::default(), &mut ctx).is_err());
    Ok(())
}

/// Whole-vector reconstruction fidelity over the real `dim` coordinates (overspill padding
/// dropped). Complements `encode_decode_per_block_mse_within_bound` by covering the default
/// single-block path (which the per-block test never exercises) and an overspilling block shape
/// whose final block mixes real coordinates with zero padding. The bound is a loose whole-vector
/// normalized MSE that still catches a broken centroid lookup, inverse SORF, or norm round-trip
/// (those drive normalized MSE toward ~1.0).
#[rstest]
#[case::default_single_block(768, None)]
#[case::overspill(837, Some(vec![512, 256, 64, 64]))]
#[case::two_block_384(384, Some(vec![256, 128]))]
#[case::two_block_1536(1536, Some(vec![1024, 512]))]
#[case::single_64(64, None)]
#[case::single_128(128, None)]
fn encode_decode_real_dim_fidelity(
    #[case] dim: u32,
    #[case] config_block_sizes: Option<Vec<u32>>,
) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let rows: usize = 8;
    let bit_width: u8 = 6;

    let total = rows * dim as usize;
    let mut values = vec![0.0f32; total];
    let mut state: u32 = 0x9E37_79B9;
    for v in values.iter_mut() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        #[expect(
            clippy::cast_precision_loss,
            reason = "f32 precision is sufficient for the synthetic input distribution"
        )]
        let x = ((state as f32) / (u32::MAX as f32 / 2.0)) - 1.0;
        *v = x;
    }

    let input = vector_array(dim, &values, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(bit_width, 7, 3, config_block_sizes)?;
    let encoded = execute_tq_encode(input.clone(), &config, &mut ctx)?;
    let decoded = execute_tq_decode(encoded, &mut ctx)?;

    let original = vector_values_f32(input, &mut ctx)?;
    let recovered = vector_values_f32(decoded, &mut ctx)?;
    assert_eq!(recovered.len(), rows * dim as usize);

    #[expect(
        clippy::cast_precision_loss,
        reason = "`1 << (2 * bit_width)` fits a u32 for `bit_width <= 8`"
    )]
    let quant_levels_sq = (1u32 << (2 * bit_width)) as f32;
    let max_normalized_mse = 16.0_f32 / quant_levels_sq;
    let dim = dim as usize;
    for row in 0..rows {
        let orig = &original[row * dim..][..dim];
        let rec = &recovered[row * dim..][..dim];
        let norm_sq: f32 = orig.iter().map(|&x| x * x).sum();
        let err_sq: f32 = orig
            .iter()
            .zip(rec.iter())
            .map(|(&o, &r)| (o - r).powi(2))
            .sum();
        let normalized_mse = err_sq / norm_sq.max(1e-10);
        assert!(
            normalized_mse < max_normalized_mse,
            "row {row} whole-vector normalized MSE {normalized_mse} exceeds bound \
             {max_normalized_mse} (dim {dim})",
        );
    }
    Ok(())
}

fn block_codes(
    outer: &StructArray,
    block_index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<u8>> {
    let inner: StructArray = outer
        .unmasked_field_by_name(format!("block_{block_index}"))?
        .clone()
        .execute(ctx)?;
    let codes: FixedSizeListArray = inner
        .unmasked_field_by_name("codes")?
        .clone()
        .execute(ctx)?;
    let elements: PrimitiveArray = codes.elements().clone().execute(ctx)?;
    Ok(elements.as_slice::<u8>().to_vec())
}

fn block_norms(
    outer: &StructArray,
    block_index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<f32>> {
    let inner: StructArray = outer
        .unmasked_field_by_name(format!("block_{block_index}"))?
        .clone()
        .execute(ctx)?;
    let norms: PrimitiveArray = inner
        .unmasked_field_by_name("norms")?
        .clone()
        .execute(ctx)?;
    Ok(norms.as_slice::<f32>().to_vec())
}
