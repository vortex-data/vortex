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

#[test]
fn encode_decode_matches_old_turboquant_decode() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.125, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let new_encoded = execute_tq_encode(input.clone(), &config, &mut ctx)?;
    let new_decoded = execute_tq_decode(new_encoded, &mut ctx)?;
    let old_config = OldTurboQuantConfig {
        bit_width: config.bit_width(),
        seed: config.seed(),
        num_rounds: config.num_rounds(),
    };
    let old_decoded = turboquant_encode(input, &old_config, &mut ctx)?.execute(&mut ctx)?;

    let new_values = vector_values_f32(new_decoded, &mut ctx)?;
    let old_values = vector_values_f32(old_decoded, &mut ctx)?;

    assert_eq!(new_values, old_values);
    Ok(())
}
