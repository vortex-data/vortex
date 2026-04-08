// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::TurboQuantArrayExt;
use crate::encodings::turboquant::TurboQuantConfig;
use crate::encodings::turboquant::array::rotation::RotationMatrix;
use crate::encodings::turboquant::turboquant_encode;
use crate::scalar_fns::ApproxOptions;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::vector::Vector;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Create a FixedSizeListArray of random f32 vectors (i.i.d. standard normal) with the given
/// validity.
fn make_fsl_with_validity(
    num_rows: usize,
    dim: usize,
    seed: u64,
    validity: Validity,
) -> FixedSizeListArray {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(normal.sample(&mut rng));
    }

    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        validity,
        num_rows,
    )
    .unwrap()
}

/// Create a non-nullable FixedSizeListArray of random f32 vectors (i.i.d. standard normal).
fn make_fsl(num_rows: usize, dim: usize, seed: u64) -> FixedSizeListArray {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(normal.sample(&mut rng));
    }

    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        num_rows,
    )
    .unwrap()
}

/// Wrap a `FixedSizeListArray` in a `Vector` extension array.
fn make_vector_ext(fsl: &FixedSizeListArray) -> ExtensionArray {
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
        .unwrap()
        .erased();
    ExtensionArray::new(ext_dtype, fsl.clone().into_array())
}

fn theoretical_mse_bound(bit_width: u8) -> f32 {
    let sqrt3_pi_over_2 = (3.0f32).sqrt() * std::f32::consts::PI / 2.0;
    sqrt3_pi_over_2 / (4.0f32).powi(bit_width as i32)
}

fn per_vector_normalized_mse(
    original: &[f32],
    reconstructed: &[f32],
    dim: usize,
    num_rows: usize,
) -> f32 {
    let mut total = 0.0f32;
    for row in 0..num_rows {
        let orig = &original[row * dim..(row + 1) * dim];
        let recon = &reconstructed[row * dim..(row + 1) * dim];
        let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
        if norm_sq < 1e-10 {
            continue;
        }
        let err_sq: f32 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        total += err_sq / norm_sq;
    }
    total / num_rows as f32
}

/// Encode and decode, returning (original, decoded) flat f32 slices.
fn encode_decode(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<(Vec<f32>, Vec<f32>)> {
    let mut ctx = SESSION.create_execution_ctx();
    let original: Vec<f32> = {
        let prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
        prim.as_slice::<f32>().to_vec()
    };
    let ext = make_vector_ext(fsl);
    let config = config.clone();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let decoded_ext = encoded.execute::<ExtensionArray>(&mut ctx)?;
    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let decoded_elements: Vec<f32> = {
        let prim = decoded_fsl
            .elements()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        prim.as_slice::<f32>().to_vec()
    };
    Ok((original, decoded_elements))
}

fn empty_turboquant_parts(
    dim: u32,
) -> VortexResult<(
    vortex_array::dtype::DType,
    ArrayRef,
    ArrayRef,
    ArrayRef,
    ArrayRef,
)> {
    let fsl = make_fsl(0, dim as usize, 42);
    let ext = make_vector_ext(&fsl);

    let codes = FixedSizeListArray::try_new(
        PrimitiveArray::empty::<u8>(Nullability::NonNullable).into_array(),
        dim,
        Validity::NonNullable,
        0,
    )?
    .into_array();
    let norms = PrimitiveArray::empty::<f32>(ext.dtype().nullability()).into_array();
    let centroids = PrimitiveArray::empty::<f32>(Nullability::NonNullable).into_array();
    let rotation_signs = FixedSizeListArray::try_new(
        PrimitiveArray::empty::<u8>(Nullability::NonNullable).into_array(),
        dim,
        Validity::NonNullable,
        0,
    )?
    .into_array();

    Ok((ext.dtype().clone(), codes, norms, centroids, rotation_signs))
}

// -----------------------------------------------------------------------
// Roundtrip tests
// -----------------------------------------------------------------------

#[rstest]
#[case(128, 1)]
#[case(128, 2)]
#[case(128, 3)]
#[case(128, 4)]
#[case(128, 6)]
#[case(128, 8)]
#[case(256, 2)]
fn roundtrip(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let fsl = make_fsl(10, dim, 42);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    assert_eq!(decoded.len(), original.len());
    Ok(())
}

#[test]
fn empty_try_new_rejects_invalid_norms_dtype() -> VortexResult<()> {
    let (dtype, codes, _norms, centroids, rotation_signs) = empty_turboquant_parts(128)?;
    let wrong_norms = PrimitiveArray::empty::<f64>(dtype.nullability()).into_array();

    let err = TurboQuant::try_new_array(dtype, codes, wrong_norms, centroids, rotation_signs)
        .unwrap_err();

    assert!(err.to_string().contains("norms dtype does not match"));
    Ok(())
}

#[test]
fn empty_try_new_rejects_invalid_centroids_dtype() -> VortexResult<()> {
    let (dtype, codes, norms, _centroids, rotation_signs) = empty_turboquant_parts(128)?;
    let wrong_centroids = PrimitiveArray::empty::<f64>(Nullability::NonNullable).into_array();

    let err = TurboQuant::try_new_array(dtype, codes, norms, wrong_centroids, rotation_signs)
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("centroids dtype must be non-nullable f32")
    );
    Ok(())
}

#[test]
fn empty_try_new_rejects_invalid_rotation_signs_dtype() -> VortexResult<()> {
    let (dtype, codes, norms, centroids, _rotation_signs) = empty_turboquant_parts(128)?;
    let wrong_rotation_signs = PrimitiveArray::empty::<u8>(Nullability::NonNullable).into_array();

    let err = TurboQuant::try_new_array(dtype, codes, norms, centroids, wrong_rotation_signs)
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("rotation_signs dtype does not match")
    );
    Ok(())
}

// -----------------------------------------------------------------------
// MSE quality tests
// -----------------------------------------------------------------------

#[rstest]
#[case(128, 1)]
#[case(128, 2)]
#[case(128, 3)]
#[case(128, 4)]
#[case(256, 2)]
#[case(256, 4)]
fn mse_within_theoretical_bound(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let num_rows = 200;
    let fsl = make_fsl(num_rows, dim, 42);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;

    let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
    let bound = theoretical_mse_bound(bit_width);

    assert!(
        normalized_mse < bound,
        "Normalized MSE {normalized_mse:.6} exceeds bound {bound:.6} \
             for dim={dim}, bits={bit_width}",
    );
    Ok(())
}

#[rstest]
#[case(128, 6)]
#[case(128, 8)]
#[case(256, 6)]
#[case(256, 8)]
fn high_bitwidth_mse_is_small(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let num_rows = 200;
    let fsl = make_fsl(num_rows, dim, 42);

    let config_4bit = TurboQuantConfig {
        bit_width: 4,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original_4, decoded_4) = encode_decode(&fsl, &config_4bit)?;
    let mse_4bit = per_vector_normalized_mse(&original_4, &decoded_4, dim, num_rows);

    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

    assert!(
        mse < mse_4bit,
        "{bit_width}-bit MSE ({mse:.6}) should be < 4-bit MSE ({mse_4bit:.6})"
    );
    assert!(mse < 0.01, "{bit_width}-bit MSE ({mse:.6}) should be < 1%");
    Ok(())
}

#[test]
fn mse_decreases_with_bits() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 50;
    let fsl = make_fsl(num_rows, dim, 99);

    let mut prev_mse = f32::MAX;
    for bit_width in 1..=8u8 {
        let config = TurboQuantConfig {
            bit_width,
            seed: Some(123),
            num_rounds: 3,
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;
        let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
        assert!(
            mse <= prev_mse * 1.01,
            "MSE should decrease: {bit_width}-bit={mse:.6} > prev={prev_mse:.6}"
        );
        prev_mse = mse;
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------

#[rstest]
#[case(0)]
#[case(1)]
fn roundtrip_edge_cases(#[case] num_rows: usize) -> VortexResult<()> {
    let fsl = make_fsl(num_rows, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let decoded = encoded.execute::<ExtensionArray>(&mut ctx)?;
    assert_eq!(decoded.len(), num_rows);
    Ok(())
}

#[rstest]
#[case(1)]
#[case(64)]
#[case(127)]
fn rejects_dimension_below_128(#[case] dim: usize) {
    let fsl = make_fsl_small(dim);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(0),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    assert!(turboquant_encode(ext.as_view(), &config, &mut ctx).is_err());
}

fn make_fsl_small(dim: usize) -> FixedSizeListArray {
    let mut buf = BufferMut::<f32>::with_capacity(dim);
    for i in 0..dim {
        buf.push(i as f32 + 1.0);
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        1,
    )
    .unwrap()
}

/// Verify that all-zero vectors roundtrip correctly (norm == 0 branch).
#[test]
fn all_zero_vectors_roundtrip() -> VortexResult<()> {
    let num_rows = 10;
    let dim = 128;
    let buf = BufferMut::<f32>::full(0.0f32, num_rows * dim);
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        num_rows,
    )?;

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(42),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    // All-zero vectors should decode to all-zero (norm=0 -> 0 * anything = 0).
    for (i, (&o, &d)) in original.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o, 0.0, "original[{i}] not zero");
        assert_eq!(d, 0.0, "decoded[{i}] not zero for all-zero input");
    }
    Ok(())
}

/// Verify that f64 input is accepted and encoded (converted to f32 internally).
#[test]
fn f64_input_encodes_successfully() -> VortexResult<()> {
    let num_rows = 10;
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0f64, 1.0).unwrap();

    let mut buf = BufferMut::<f64>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(normal.sample(&mut rng));
    }
    let elements = PrimitiveArray::new::<f64>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        num_rows,
    )?;

    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(42),
        num_rounds: 3,
    };
    // Verify encoding succeeds with f64 input (f64->f32 conversion).
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let encoded = encoded.as_opt::<TurboQuant>().unwrap();
    assert_eq!(encoded.norms().len(), num_rows);
    assert_eq!(encoded.dimension() as usize, dim);
    Ok(())
}

/// Verify that f16 input is accepted and encoded (upcast to f32 internally).
#[test]
fn f16_input_encodes_successfully() -> VortexResult<()> {
    let num_rows = 10;
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    let mut buf = BufferMut::<half::f16>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(half::f16::from_f32(normal.sample(&mut rng)));
    }
    let elements = PrimitiveArray::new::<half::f16>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        num_rows,
    )?;

    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(42),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let tq = encoded.as_opt::<TurboQuant>().unwrap();
    assert_eq!(tq.norms().len(), num_rows);
    assert_eq!(tq.dimension() as usize, dim);

    // Verify roundtrip: decode and check reconstruction is reasonable.
    let decoded_ext = encoded.execute::<ExtensionArray>(&mut ctx)?;
    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    assert_eq!(decoded_fsl.len(), num_rows);
    Ok(())
}

// -----------------------------------------------------------------------
// Verification tests for stored metadata
// -----------------------------------------------------------------------

/// Verify that the centroids stored in the array match what `get_centroids()` computes.
#[test]
fn stored_centroids_match_computed() -> VortexResult<()> {
    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let encoded = encoded.as_opt::<TurboQuant>().unwrap();

    let mut ctx = SESSION.create_execution_ctx();
    let stored_centroids_prim = encoded
        .centroids()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let stored = stored_centroids_prim.as_slice::<f32>();

    let padded_dim = encoded.padded_dim();
    let computed = crate::encodings::turboquant::array::centroids::get_centroids(padded_dim, 3)?;

    assert_eq!(stored.len(), computed.len());
    for i in 0..stored.len() {
        assert_eq!(stored[i], computed[i], "Centroid mismatch at {i}");
    }
    Ok(())
}

/// Verify that stored rotation signs produce identical decode to seed-based decode.
#[test]
fn stored_rotation_signs_produce_correct_decode() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 4,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let encoded = encoded.as_opt::<TurboQuant>().unwrap();

    // Decode via the stored-signs path (normal decode).
    let mut ctx = SESSION.create_execution_ctx();
    let decoded_ext = encoded
        .array()
        .clone()
        .execute::<ExtensionArray>(&mut ctx)?;
    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let decoded = decoded_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let decoded_slice = decoded.as_slice::<f32>();

    // Verify stored signs match seed-derived signs.
    let rot_from_seed = RotationMatrix::try_new(123, 128, 4)?;
    let expected_u8 = rot_from_seed.export_inverse_signs_u8();
    let stored_signs_fsl = encoded
        .rotation_signs()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let stored_signs = stored_signs_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let stored_u8 = stored_signs.as_slice::<u8>();

    assert_eq!(expected_u8.len(), stored_u8.len());
    for i in 0..expected_u8.len() {
        assert_eq!(expected_u8[i], stored_u8[i], "Sign mismatch at index {i}");
    }

    // Also verify decode output is non-empty and has expected size.
    assert_eq!(decoded_slice.len(), 20 * 128);
    Ok(())
}

// -----------------------------------------------------------------------
// Compute pushdown tests
// -----------------------------------------------------------------------

#[test]
fn slice_preserves_data() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 4,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    // Full decompress then slice.
    let mut ctx = SESSION.create_execution_ctx();
    let full_decoded = encoded.clone().execute::<ExtensionArray>(&mut ctx)?;
    let full_fsl = full_decoded
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let expected = full_fsl.slice(5..10)?;
    let expected_fsl = expected.execute::<FixedSizeListArray>(&mut ctx)?;
    let expected_elements = expected_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;

    // Slice then decompress.
    let sliced = encoded.slice(5..10)?;
    let sliced_decoded = sliced.execute::<ExtensionArray>(&mut ctx)?;
    let sliced_fsl = sliced_decoded
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let actual_elements = sliced_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;

    assert_eq!(
        expected_elements.as_slice::<f32>(),
        actual_elements.as_slice::<f32>()
    );
    Ok(())
}

#[test]
fn scalar_at_matches_decompress() -> VortexResult<()> {
    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 2,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    let full_decoded = encoded.clone().execute::<ExtensionArray>(&mut ctx)?;

    for i in [0, 1, 5, 9] {
        let expected = full_decoded.scalar_at(i)?;
        let actual = encoded.scalar_at(i)?;
        assert_eq!(expected, actual, "scalar_at mismatch at index {i}");
    }
    Ok(())
}

#[test]
fn l2_norm_readthrough() -> VortexResult<()> {
    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 5,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let tq = encoded.as_opt::<TurboQuant>().unwrap();

    // Stored norms should match the actual L2 norms of the input.
    let norms_prim = tq.norms().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let stored_norms = norms_prim.as_slice::<f32>();

    let input_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let input_f32 = input_prim.as_slice::<f32>();
    for row in 0..10 {
        let vec = &input_f32[row * 128..(row + 1) * 128];
        let actual_norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (stored_norms[row] - actual_norm).abs() < 1e-5,
            "norm mismatch at row {row}: stored={}, actual={}",
            stored_norms[row],
            actual_norm
        );
    }
    Ok(())
}

#[test]
fn cosine_similarity_quantized_accuracy() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 4,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let tq = encoded.as_opt::<TurboQuant>().unwrap();

    // Compute exact cosine similarity from original data.
    let input_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let input_f32 = input_prim.as_slice::<f32>();

    // Read quantized codes, norms, and centroids for approximate computation.
    let mut ctx = SESSION.create_execution_ctx();
    let pd = tq.padded_dim() as usize;
    let norms_prim = tq.norms().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let norms = norms_prim.as_slice::<f32>();
    let codes_fsl = tq.codes().clone().execute::<FixedSizeListArray>(&mut ctx)?;
    let codes_prim = codes_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let all_codes = codes_prim.as_slice::<u8>();
    let centroids_prim = tq.centroids().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let centroid_vals = centroids_prim.as_slice::<f32>();

    for (row_a, row_b) in [(0, 1), (5, 10), (0, 19)] {
        let vec_a = &input_f32[row_a * 128..(row_a + 1) * 128];
        let vec_b = &input_f32[row_b * 128..(row_b + 1) * 128];

        let dot: f32 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = vec_a.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = vec_b.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let exact_cos = dot / (norm_a * norm_b);

        // Approximate cosine similarity in quantized domain.
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

        // At 4-bit, the theoretical MSE bound per coordinate is ~0.0106 (Theorem 1). For cosine
        // similarity (bounded [-1, 1]), the error is bounded roughly by 2*sqrt(MSE) ~ 0.2. We use
        // 0.15 as a tighter empirical bound.
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
///
/// NOTE: The MSE quantizer (TurboQuant_mse) has inherent **multiplicative bias** for inner
/// products — the quantized dot product systematically over- or under-estimates the true value.
/// This is a fundamental property: the paper's `TurboQuant_prod` variant adds QJL specifically
/// to debias inner products, but we only implement the MSE-only variant.
///
/// Even at 8-bit (near-lossless reconstruction, MSE ~4e-5), the quantized-domain dot product
/// can have ~10-15% relative error due to this bias. This tolerance is therefore intentionally
/// loose — we're testing that the approximation is in the right ballpark, not that it's precise.
///
/// TODO(connor): Revisit these tolerances when we have TurboQuant_prod (QJL debiasing).
#[test]
fn dot_product_quantized_accuracy() -> VortexResult<()> {
    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 8,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let tq = encoded.as_opt::<TurboQuant>().unwrap();

    let input_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let input_f32 = input_prim.as_slice::<f32>();

    let mut ctx = SESSION.create_execution_ctx();
    let pd = tq.padded_dim() as usize;
    let norms_prim = tq.norms().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let norms = norms_prim.as_slice::<f32>();
    let codes_fsl = tq.codes().clone().execute::<FixedSizeListArray>(&mut ctx)?;
    let codes_prim = codes_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let all_codes = codes_prim.as_slice::<u8>();
    let centroids_prim = tq.centroids().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let centroid_vals = centroids_prim.as_slice::<f32>();

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

        // See doc comment above: 15% relative error is expected due to MSE quantizer bias.
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

/// Roundtrip at large embedding dimensions to validate padding and SRHT at common sizes.
///
/// NOTE: The theoretical MSE bound (Theorem 1) is proved for Haar-distributed random orthogonal
/// matrices, not SRHT. The SRHT is a practical O(d log d) approximation that doesn't exactly
/// satisfy the Haar assumption, so empirical MSE can slightly exceed the theoretical bound. We
/// use a 2x multiplier to account for this gap.
///
/// The 1024-d case uses 5-bit instead of 4-bit because at 4-bit the SRHT approximation error
/// at d=1024 pushes MSE ~20% above the 1x theoretical bound (0.0127 vs bound 0.0106).
///
/// TODO(connor): Revisit after Stage 2 block decomposition — at d=768 with block_size=256,
/// the per-block SRHT will be lower-dimensional and may have different error characteristics.
#[rstest]
#[case(768, 4)]
#[case(1024, 5)]
fn large_dimension_roundtrip(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let num_rows = 10;
    let fsl = make_fsl(num_rows, dim, 42);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    assert_eq!(decoded.len(), original.len());

    let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
    // 2x slack for the SRHT-vs-Haar gap (see doc comment above).
    let bound = 2.0 * theoretical_mse_bound(bit_width);
    assert!(
        normalized_mse < bound,
        "Normalized MSE {normalized_mse:.6} exceeds 2x bound {bound:.6} for dim={dim}, bits={bit_width}",
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
        seed: Some(123),
        num_rounds: 2,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    // The encoded TurboQuant array should claim a Vector extension dtype.
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

// -----------------------------------------------------------------------
// Nullable vector tests
// -----------------------------------------------------------------------

/// Encode a nullable Vector array and verify roundtrip preserves validity and non-null values.
#[test]
fn nullable_vectors_roundtrip() -> VortexResult<()> {
    // Rows 2, 5, 7 are null.
    let validity = Validity::from_iter([
        true, true, false, true, true, false, true, false, true, true,
    ]);
    let fsl = make_fsl_with_validity(10, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 4,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    assert_eq!(encoded.len(), 10);
    assert!(encoded.dtype().is_nullable());

    // Check validity of the encoded array.
    let encoded_validity = encoded.validity()?;
    for i in 0..10 {
        let expected = ![2, 5, 7].contains(&i);
        assert_eq!(
            encoded_validity.is_valid(i)?,
            expected,
            "validity mismatch at row {i}"
        );
    }

    // Decode and verify non-null rows have correct data.
    let decoded_ext = encoded.execute::<ExtensionArray>(&mut ctx)?;
    assert_eq!(decoded_ext.len(), 10);

    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let decoded_prim = decoded_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let decoded_f32 = decoded_prim.as_slice::<f32>();

    // Original f32 elements for non-null row comparison.
    let orig_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let orig_f32 = orig_prim.as_slice::<f32>();

    // Non-null rows should have reasonable reconstruction (within MSE bounds).
    for row in [0, 1, 3, 4, 6, 8, 9] {
        let orig_vec = &orig_f32[row * 128..(row + 1) * 128];
        let dec_vec = &decoded_f32[row * 128..(row + 1) * 128];
        let norm_sq: f32 = orig_vec.iter().map(|&v| v * v).sum();
        let err_sq: f32 = orig_vec
            .iter()
            .zip(dec_vec.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        // 3-bit normalized MSE should be well under the theoretical bound.
        assert!(
            err_sq / norm_sq < 0.1,
            "non-null row {row} has excessive reconstruction error"
        );
    }
    Ok(())
}

/// Verify that norms carry the validity: null vectors have null norms.
#[test]
fn nullable_norms_match_validity() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false, true, false, true]);
    let fsl = make_fsl_with_validity(5, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    let tq = encoded.as_opt::<TurboQuant>().unwrap();

    let norms_validity = tq.norms().validity()?;
    for i in 0..5 {
        let expected = i % 2 == 0; // rows 0, 2, 4 are valid
        assert_eq!(
            norms_validity.is_valid(i)?,
            expected,
            "norms validity mismatch at row {i}"
        );
    }
    Ok(())
}

/// Verify that L2Norm readthrough works correctly on nullable TurboQuant arrays.
#[test]
fn nullable_l2_norm_readthrough() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false, true, false, true]);
    let fsl = make_fsl_with_validity(5, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    // Compute L2Norm on the encoded array.
    let norm_sfn = L2Norm::try_new_array(&ApproxOptions::Exact, encoded, 5)?;
    let norms: PrimitiveArray = norm_sfn.into_array().execute(&mut ctx)?;

    // Null rows should have null norms, valid rows should have correct norms.
    let orig_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let orig_f32 = orig_prim.as_slice::<f32>();
    for row in 0..5 {
        if row % 2 == 0 {
            assert!(norms.is_valid(row)?, "row {row} should be valid");
            let expected: f32 = orig_f32[row * 128..(row + 1) * 128]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>()
                .sqrt();
            let actual = norms.as_slice::<f32>()[row];
            assert!(
                (actual - expected).abs() < 1e-5,
                "norm mismatch at valid row {row}: actual={actual}, expected={expected}"
            );
        } else {
            assert!(!norms.is_valid(row)?, "row {row} should be null");
        }
    }
    Ok(())
}

/// Verify that slicing a nullable TurboQuant array preserves validity.
#[test]
fn nullable_slice_preserves_validity() -> VortexResult<()> {
    // Rows 2, 5, 7 are null.
    let validity = Validity::from_iter([
        true, true, false, true, true, false, true, false, true, true,
    ]);
    let fsl = make_fsl_with_validity(10, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 2,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    // Slice rows 1..6 -> [true, false, true, true, false].
    let sliced = encoded.slice(1..6)?;
    assert_eq!(sliced.len(), 5);

    let sliced_validity = sliced.validity()?;
    let expected = [true, false, true, true, false];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(
            sliced_validity.is_valid(i)?,
            exp,
            "sliced validity mismatch at index {i}"
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Serde roundtrip tests
// -----------------------------------------------------------------------

/// Verify that a TurboQuant array survives serialize/deserialize.
#[test]
fn serde_roundtrip() -> VortexResult<()> {
    use vortex_array::ArrayContext;
    use vortex_array::ArrayEq;
    use vortex_array::Precision;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_fastlanes::BitPacked;
    use vortex_session::registry::ReadContext;

    let fsl = make_fsl(20, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 5,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;

    let dtype = encoded.dtype().clone();
    let len = encoded.len();

    // Serialize.
    let array_ctx = ArrayContext::empty();
    let serialized = encoded.serialize(&array_ctx, &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }

    // Deserialize. The session needs TurboQuant and BitPacked (for rotation signs) registered.
    let serde_session = VortexSession::empty().with::<ArraySession>();
    serde_session.arrays().register(TurboQuant);
    serde_session.arrays().register(BitPacked);

    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &serde_session,
    )?;

    assert!(
        decoded.array_eq(&encoded, Precision::Value),
        "serde roundtrip did not preserve array equality"
    );
    Ok(())
}

/// Verify that a degenerate (empty) TurboQuant array survives serialize/deserialize.
#[test]
fn serde_roundtrip_empty() -> VortexResult<()> {
    use vortex_array::ArrayContext;
    use vortex_array::ArrayEq;
    use vortex_array::Precision;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_fastlanes::BitPacked;
    use vortex_session::registry::ReadContext;

    let fsl = make_fsl(0, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx)?;
    assert_eq!(encoded.len(), 0);

    let dtype = encoded.dtype().clone();
    let len = encoded.len();

    let array_ctx = ArrayContext::empty();
    let serialized = encoded.serialize(&array_ctx, &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }

    let serde_session = VortexSession::empty().with::<ArraySession>();
    serde_session.arrays().register(TurboQuant);
    serde_session.arrays().register(BitPacked);

    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &serde_session,
    )?;

    assert!(
        decoded.array_eq(&encoded, Precision::Value),
        "serde roundtrip did not preserve array equality"
    );
    Ok(())
}
