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

/// Pins down the exact relationship between new and legacy TurboQuant decode: for each row,
/// `new_value[i] == old_value[i] * inv_direction_norm[row]`. The centroid table and SORF
/// transform are identical between the two encoders, so the inverse-transformed direction is
/// the same; the only mathematical difference is the per-row scalar correction.
#[test]
fn encode_decode_applies_direction_norm_correction_after_old_turboquant_decode() -> VortexResult<()>
{
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(129, 2, 0.125, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let new_encoded = execute_tq_encode(input.clone(), &config, &mut ctx)?;
    let parsed = parse_storage(new_encoded.clone(), &mut ctx)?;
    let inv_direction_norms = parsed.inv_direction_norms.as_slice::<f32>().to_vec();

    let new_decoded = execute_tq_decode(new_encoded, &mut ctx)?;
    let old_config = OldTurboQuantConfig {
        bit_width: config.bit_width(),
        seed: config.seed(),
        num_rounds: config.num_rounds(),
    };
    let old_decoded = turboquant_encode(input, &old_config, &mut ctx)?.execute(&mut ctx)?;

    let new_values = vector_values_f32(new_decoded, &mut ctx)?;
    let old_values = vector_values_f32(old_decoded, &mut ctx)?;
    let dim = new_values.len() / inv_direction_norms.len();

    // Every coordinate of the new decode should equal the corresponding coordinate of the old
    // decode, multiplied by the row's stored `inv_direction_norm`. Tolerance is per-element,
    // scaled by the larger of the two values to handle exact zeros gracefully.
    for (row, &correction) in inv_direction_norms.iter().enumerate() {
        for col in 0..dim {
            let idx = row * dim + col;
            let new_v = new_values[idx];
            let old_v = old_values[idx];
            let expected = old_v * correction;
            let scale = new_v.abs().max(expected.abs()).max(1.0);
            assert!(
                (new_v - expected).abs() <= 1e-4 * scale,
                "row {row} col {col}: new {new_v} != old {old_v} * inv_direction_norm \
                 {correction} (= {expected})"
            );
        }
    }
    // Sanity: the correction is meaningfully non-trivial for at least one row (verifies the
    // direction-norm field is actually doing work, not a no-op).
    assert!(
        inv_direction_norms.iter().any(|&c| (c - 1.0).abs() > 1e-4),
        "inv_direction_norms should differ from 1.0 for at least one row"
    );
    Ok(())
}
