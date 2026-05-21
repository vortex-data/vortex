// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::VortexSessionExecute;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_tensor::encodings::turboquant::TurboQuantConfig as OldTurboQuantConfig;
use vortex_tensor::encodings::turboquant::turboquant_encode;

use super::execute_tq_decode;
use super::execute_tq_encode;
use super::f32_vector_array;
use super::test_session;
use super::vector_values_f32;
use crate::TurboQuantConfig;
use crate::vector::storage::parse_storage;

/// Pins the exact relationship between new and legacy TurboQuant decode: for each row,
/// `new_value[i] == old_value[i] * (stored_norm / old_norm)`. The centroid table and SORF
/// transform are identical between the two encoders, so the inverse-transformed direction is
/// the same; the only mathematical difference is the per-row scalar correction the new decode
/// applies in flight to rescale the lossy quantized direction to unit norm before re-applying
/// the stored row norm.
#[test]
fn new_decode_rescales_old_decode_to_stored_norm() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(129, 2, 0.125, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let new_encoded = execute_tq_encode(input.clone(), &config, &mut ctx)?;
    let stored_norms = parse_storage(new_encoded.clone(), &mut ctx)?
        .norms
        .as_slice::<f32>()
        .to_vec();

    let new_decoded = execute_tq_decode(new_encoded, &mut ctx)?;
    let old_config = OldTurboQuantConfig {
        bit_width: config.bit_width(),
        seed: config.seed(),
        num_rounds: config.num_rounds(),
    };
    let old_decoded = turboquant_encode(input, &old_config, &mut ctx)?.execute(&mut ctx)?;

    let new_values = vector_values_f32(new_decoded, &mut ctx)?;
    let old_values = vector_values_f32(old_decoded, &mut ctx)?;
    let dim = new_values.len() / stored_norms.len();

    let mut any_correction_nontrivial = false;
    for (row, &stored_norm) in stored_norms.iter().enumerate() {
        let old_row = &old_values[row * dim..][..dim];
        let old_norm = old_row.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Skip rows where the legacy decode produced exact zero norm; division below would
        // be undefined and there is no meaningful relationship to pin.
        if old_norm == 0.0 {
            continue;
        }
        let correction = stored_norm / old_norm;
        if (correction - 1.0).abs() > 1e-4 {
            any_correction_nontrivial = true;
        }

        for col in 0..dim {
            let idx = row * dim + col;
            let new_v = new_values[idx];
            let expected = old_values[idx] * correction;
            let scale = new_v.abs().max(expected.abs()).max(1.0);
            assert!(
                (new_v - expected).abs() <= 1e-4 * scale,
                "row {row} col {col}: new {new_v} != old {} * correction {correction} \
                 (= {expected})",
                old_values[idx]
            );
        }
    }
    // Sanity: the correction is meaningfully non-trivial for at least one row. If the new
    // decode were a no-op rescaling, this whole test would silently pass.
    assert!(
        any_correction_nontrivial,
        "the in-flight correction should differ from 1.0 for at least one row"
    );
    Ok(())
}
