// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_error::VortexResult;

use super::*;
use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::l2_norm::L2Norm;

fn execute_l2_norm(
    input: ArrayRef,
    len: usize,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    L2Norm::try_new_array(input, len)?.into_array().execute(ctx)
}

fn execute_cosine_similarity(
    lhs: ArrayRef,
    rhs: ArrayRef,
    len: usize,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    CosineSimilarity::try_new_array(lhs, rhs, len)?
        .into_array()
        .execute(ctx)
}

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
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

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
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

    let full_decoded = encoded.clone().execute::<ExtensionArray>(&mut ctx)?;

    for i in [0, 1, 5, 9] {
        let expected =
            full_decoded.execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())?;
        let actual = encoded.execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())?;
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
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;
    let (_sorf_child, norms_child) = unwrap_l2denorm(&encoded);

    // Stored norms should match the actual L2 norms of the input.
    let norms_prim = norms_child.execute::<PrimitiveArray>(&mut ctx)?;
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

    // Also verify L2Norm readthrough shortcut works.
    let norms = execute_l2_norm(encoded, 10, &mut ctx)?;
    assert_eq!(norms.as_slice::<f32>(), stored_norms);
    assert_eq!(norms.len(), 10);
    Ok(())
}

#[test]
fn l2_norm_readthrough_is_authoritative_for_lossy_storage() -> VortexResult<()> {
    let num_rows = 12;
    let fsl = make_fsl(num_rows, 128, 7);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 1,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;
    let (_sorf_child, norms_child) = unwrap_l2denorm(&encoded);

    let stored_norms: PrimitiveArray = norms_child.execute(&mut ctx)?;
    let encoded_norms = execute_l2_norm(encoded.clone(), num_rows, &mut ctx)?;
    assert_eq!(
        encoded_norms.as_slice::<f32>(),
        stored_norms.as_slice::<f32>()
    );

    let decoded = encoded.execute::<ExtensionArray>(&mut ctx)?.into_array();
    let decoded_norms = execute_l2_norm(decoded, num_rows, &mut ctx)?;
    let max_gap = stored_norms
        .as_slice::<f32>()
        .iter()
        .zip(decoded_norms.as_slice::<f32>().iter())
        .map(|(&stored, &decoded)| (stored - decoded).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_gap > 1e-3,
        "expected at least one decoded norm to drift from the authoritative stored norms, got max gap {max_gap:.6}",
    );
    Ok(())
}

#[test]
fn cosine_similarity_readthrough_is_authoritative_for_lossy_storage() -> VortexResult<()> {
    let num_rows = 12;
    let fsl = make_fsl(num_rows, 128, 11);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 1,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;

    let encoded_cos =
        execute_cosine_similarity(encoded.clone(), encoded.clone(), num_rows, &mut ctx)?;
    let decoded = encoded.execute::<ExtensionArray>(&mut ctx)?.into_array();
    let decoded_cos = execute_cosine_similarity(decoded.clone(), decoded, num_rows, &mut ctx)?;

    let decoded_values = decoded_cos.as_slice::<f32>();
    assert!(
        decoded_values
            .iter()
            .all(|&value| (value - 1.0).abs() < 1e-5),
        "decoded cosine(x, x) should stay at 1.0",
    );

    let max_gap = encoded_cos
        .as_slice::<f32>()
        .iter()
        .zip(decoded_values.iter())
        .map(|(&encoded, &decoded)| (encoded - decoded).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap > 1e-3,
        "expected encoded cosine readthrough to differ from decoded recomputation, got max gap {max_gap:.6}",
    );
    Ok(())
}
