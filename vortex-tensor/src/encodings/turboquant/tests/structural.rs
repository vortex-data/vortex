// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests that verify the internal structure of the encoded tree.

use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_error::VortexResult;

use super::*;
use crate::encodings::turboquant::centroids::compute_or_get_centroids;

/// Verify that the centroids stored in the DictArray match what `compute_or_get_centroids()` computes.
#[test]
fn stored_centroids_match_computed() -> VortexResult<()> {
    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: 123,
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

    let (_codes, centroids, _norms) = unwrap_codes_centroids_norms(&encoded, &mut ctx)?;
    let stored = centroids.as_slice::<f32>();

    // padded_dim for dim=128 is 128.
    let computed = compute_or_get_centroids(128, 3)?;

    assert_eq!(stored.len(), computed.len());
    for i in 0..stored.len() {
        assert_eq!(stored[i], computed[i], "Centroid mismatch at {i}");
    }
    Ok(())
}

/// Verify that the rotation is deterministic from seed by checking decode output.
#[test]
fn seed_deterministic_rotation_produces_correct_decode() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: 123,
        num_rounds: 4,
    };

    // Encode twice with the same seed → should produce identical results.
    let mut ctx = SESSION.create_execution_ctx();
    let encoded1 = turboquant_encode(ext.clone(), &config, &mut ctx)?;
    let decoded1 = encoded1.execute::<ExtensionArray>(&mut ctx)?;
    let fsl1 = decoded1
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let elems1 = fsl1
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;

    let mut ctx = SESSION.create_execution_ctx();
    let encoded2 = turboquant_encode(ext, &config, &mut ctx)?;
    let decoded2 = encoded2.execute::<ExtensionArray>(&mut ctx)?;
    let fsl2 = decoded2
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let elems2 = fsl2
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;

    assert_eq!(
        elems1.as_slice::<f32>(),
        elems2.as_slice::<f32>(),
        "Two encodes with same seed should produce identical decode output"
    );
    Ok(())
}

/// Verify that the encoded array's dtype is a Vector extension type.
#[test]
fn encoded_dtype_is_vector_extension() -> VortexResult<()> {
    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: 123,
        num_rounds: 2,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

    assert!(
        encoded.dtype().is_extension(),
        "TurboQuant dtype should be an extension type, got {}",
        encoded.dtype()
    );
    assert!(
        encoded.dtype().as_extension().is::<Vector>(),
        "TurboQuant dtype should be a Vector extension type"
    );
    Ok(())
}

/// Verify approximate cosine similarity in the quantized domain.
#[test]
fn cosine_similarity_quantized_accuracy() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 4,
        seed: 123,
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

    let input_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let input_f32 = input_prim.as_slice::<f32>();

    // Navigate tree to get codes, centroids, norms.
    let (codes_prim, centroids_prim, norms_prim) =
        unwrap_codes_centroids_norms(&encoded, &mut ctx)?;
    let all_codes = codes_prim.as_slice::<u8>();
    let centroid_vals = centroids_prim.as_slice::<f32>();
    let norms = norms_prim.as_slice::<f32>();

    // padded_dim for dim=128.
    let pd = 128usize;

    for (row_a, row_b) in [(0, 1), (5, 10), (0, 19)] {
        let vec_a = &input_f32[row_a * 128..(row_a + 1) * 128];
        let vec_b = &input_f32[row_b * 128..(row_b + 1) * 128];

        let dot: f32 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = vec_a.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = vec_b.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let exact_cos = dot / (norm_a * norm_b);

        let approx_cos = if norms[row_a] == 0.0 || norms[row_b] == 0.0 {
            0.0
        } else {
            let codes_a = &all_codes[row_a * pd..(row_a + 1) * pd];
            let codes_b = &all_codes[row_b * pd..(row_b + 1) * pd];
            codes_a
                .iter()
                .zip(codes_b.iter())
                .map(|(&ca, &cb)| centroid_vals[ca as usize] * centroid_vals[cb as usize])
                .sum::<f32>()
        };

        let error = (exact_cos - approx_cos).abs();
        assert!(
            error < 0.15,
            "cosine similarity error too large for ({row_a}, {row_b}): \
                 exact={exact_cos:.4}, approx={approx_cos:.4}, error={error:.4}"
        );
    }
    Ok(())
}

/// Verify approximate dot product in the quantized domain.
#[test]
fn dot_product_quantized_accuracy() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 8,
        seed: 123,
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

    let input_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let input_f32 = input_prim.as_slice::<f32>();

    let (codes_prim, centroids_prim, norms_prim) =
        unwrap_codes_centroids_norms(&encoded, &mut ctx)?;
    let all_codes = codes_prim.as_slice::<u8>();
    let centroid_vals = centroids_prim.as_slice::<f32>();
    let norms = norms_prim.as_slice::<f32>();

    let pd = 128usize;

    for (row_a, row_b) in [(0, 1), (5, 10), (0, 19)] {
        let vec_a = &input_f32[row_a * 128..(row_a + 1) * 128];
        let vec_b = &input_f32[row_b * 128..(row_b + 1) * 128];

        let exact_dot: f32 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| x * y).sum();

        let codes_a = &all_codes[row_a * pd..(row_a + 1) * pd];
        let codes_b = &all_codes[row_b * pd..(row_b + 1) * pd];
        let unit_dot: f32 = codes_a
            .iter()
            .zip(codes_b.iter())
            .map(|(&ca, &cb)| centroid_vals[ca as usize] * centroid_vals[cb as usize])
            .sum();
        let approx_dot = norms[row_a] * norms[row_b] * unit_dot;

        let scale = exact_dot.abs().max(1.0);
        let rel_error = (exact_dot - approx_dot).abs() / scale;
        assert!(
            rel_error < 0.15,
            "dot product error too large for ({row_a}, {row_b}): \
                 exact={exact_dot:.4}, approx={approx_dot:.4}, rel_error={rel_error:.4}"
        );
    }
    Ok(())
}

/// Verify SorfTransform in isolation: manually forward-rotate known data, wrap in
/// FSL(Dict), execute SorfTransform, and check inverse rotation recovers the original.
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "test uses known small dimensions"
)]
fn sorf_transform_roundtrip_isolation() -> VortexResult<()> {
    use vortex_array::IntoArray;
    use vortex_array::arrays::dict::DictArray;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;

    use crate::encodings::turboquant::centroids::compute_centroid_boundaries;
    use crate::encodings::turboquant::centroids::compute_or_get_centroids;
    use crate::encodings::turboquant::centroids::find_nearest_centroid;
    use crate::scalar_fns::sorf_transform::SorfMatrix;
    use crate::scalar_fns::sorf_transform::SorfOptions;
    use crate::scalar_fns::sorf_transform::SorfTransform;
    use crate::types::vector::Vector;

    let dim = 128usize;
    let seed = 99u64;
    let num_rounds = 3u8;
    let num_rows = 5;

    // Build a known input: simple increasing values, then normalize each row to unit norm.
    let mut input_f32 = vec![0.0f32; num_rows * dim];
    for row in 0..num_rows {
        let mut norm_sq = 0.0f32;
        for i in 0..dim {
            let val = ((row * dim + i) as f32 + 1.0) * 0.01;
            input_f32[row * dim + i] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        for i in 0..dim {
            input_f32[row * dim + i] /= norm;
        }
    }

    // Forward transform + quantize (mimicking what turboquant_quantize_core does).
    let padded_dim = dim.next_power_of_two();
    let rotation = SorfMatrix::try_new_padded(padded_dim, num_rounds as usize, seed)?;
    let centroids = compute_or_get_centroids(padded_dim as u32, 8)?;
    let boundaries = compute_centroid_boundaries(&centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        padded[..dim].copy_from_slice(&input_f32[row * dim..(row + 1) * dim]);
        padded[dim..].fill(0.0);
        rotation.rotate(&padded, &mut rotated);
        for j in 0..padded_dim {
            all_indices.push(find_nearest_centroid(rotated[j], &boundaries));
        }
    }

    // Build FSL(Dict(codes, centroids)).
    let codes = PrimitiveArray::new::<u8>(all_indices.freeze(), Validity::NonNullable);
    let mut centroids_buf = BufferMut::<f32>::with_capacity(centroids.len());
    centroids_buf.extend_from_slice(&centroids);
    let centroids_arr = PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable);
    let dict = DictArray::try_new(codes.into_array(), centroids_arr.into_array())?;
    let fsl = FixedSizeListArray::try_new(
        dict.into_array(),
        padded_dim as u32,
        Validity::NonNullable,
        num_rows,
    )?;

    // Wrap the padded FSL in a Vector extension so it can be the SorfTransform child.
    let padded_vector_dtype =
        ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    let padded_vector = ExtensionArray::new(padded_vector_dtype, fsl.into_array());

    // Wrap in SorfTransform and execute.
    let sorf_options = SorfOptions {
        seed,
        num_rounds,
        dimensions: dim as u32,
        element_ptype: vortex_array::dtype::PType::F32,
    };
    let sorf_array =
        SorfTransform::try_new_array(&sorf_options, padded_vector.into_array(), num_rows)?;

    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf_array.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    let result_prim: PrimitiveArray = result_fsl.elements().clone().execute(&mut ctx)?;
    let result_f32 = result_prim.as_slice::<f32>();

    assert_eq!(result_f32.len(), num_rows * dim);

    // At 8-bit quantization, reconstruction should be very close to input.
    for row in 0..num_rows {
        let orig = &input_f32[row * dim..(row + 1) * dim];
        let recon = &result_f32[row * dim..(row + 1) * dim];
        let err_sq: f32 = orig
            .iter()
            .zip(recon)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
        assert!(
            err_sq / norm_sq < 1e-3,
            "SorfTransform isolation: row {row} MSE too high: {:.6}",
            err_sq / norm_sq
        );
    }
    Ok(())
}
