// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayPlugin;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::execute_tq_encode;
use super::f32_vector_array;
use super::test_session;
use super::vector_validity;
use crate::TQDecode;
use crate::TQEncode;
use crate::TurboQuant;
use crate::TurboQuantConfig;
use crate::vtable::tq_metadata;

#[test]
fn scalar_fn_ids_and_config_options_roundtrip() -> VortexResult<()> {
    let session = test_session();
    let config = TurboQuantConfig::try_new(4, 7, 2)?;

    assert_eq!(TQEncode.id().as_ref(), "vortex.turboquant.encode");
    assert_eq!(TQDecode.id().as_ref(), "vortex.turboquant.decode");

    let encode_metadata = TQEncode.serialize(&config)?.unwrap();
    let decode_metadata = TQDecode.serialize(&config)?.unwrap();

    assert_eq!(TQEncode.deserialize(&encode_metadata, &session)?, config);
    assert_eq!(TQDecode.deserialize(&decode_metadata, &session)?, config);
    Ok(())
}

#[test]
fn scalar_fn_arrays_encode_and_decode_vectors() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.25, Validity::from_iter([true, false]))?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let encoded_lazy = TQEncode::try_new_array(input, &config, 2)?;
    let encoded_metadata = tq_metadata(encoded_lazy.dtype())?;
    assert_eq!(encoded_metadata.dimensions, 128);
    assert_eq!(encoded_metadata.bit_width, config.bit_width());
    assert!(encoded_lazy.dtype().as_extension().is::<TurboQuant>());

    let encoded = encoded_lazy.into_array().execute(&mut ctx)?;
    let decoded_lazy = TQDecode::try_new_array(encoded, &config, 2)?;
    let decoded = decoded_lazy.into_array().execute(&mut ctx)?;
    let validity = vector_validity(decoded, &mut ctx)?.execute_mask(2, &mut ctx)?;

    assert!(validity.value(0));
    assert!(!validity.value(1));
    Ok(())
}

#[test]
fn scalar_fn_array_metadata_stores_only_config() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.25, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;
    let config_metadata = TQDecode.serialize(&config)?.unwrap();

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let decoded_lazy = TQDecode::try_new_array(encoded, &config, 2)?.into_array();
    let decode_plugin = ScalarFnArrayPlugin::new(TQDecode);
    assert_eq!(
        decode_plugin.serialize(&decoded_lazy, &session)?.unwrap(),
        config_metadata
    );
    Ok(())
}

#[test]
fn decode_rejects_config_that_disagrees_with_turboquant_child() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 1, 0.25, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;
    let different_config = TurboQuantConfig::try_new(4, 42, 3)?;

    let encoded = TQEncode::try_new_array(input, &config, 1)?
        .into_array()
        .execute(&mut ctx)?;

    assert!(TQDecode::try_new_array(encoded, &different_config, 1).is_err());
    Ok(())
}
