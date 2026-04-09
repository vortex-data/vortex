// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_error::VortexResult;

use super::*;
use crate::scalar_fns::ApproxOptions;

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
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;

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
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;

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
    use crate::scalar_fns::l2_norm::L2Norm;

    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 5,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;
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
    let norm_sfn = L2Norm::try_new_array(&ApproxOptions::Exact, encoded, 10)?;
    let norms: PrimitiveArray = norm_sfn.into_array().execute(&mut ctx)?;
    assert_eq!(norms.len(), 10);
    Ok(())
}
