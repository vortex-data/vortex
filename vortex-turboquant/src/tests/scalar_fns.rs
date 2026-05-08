// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayPlugin;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::execute_tq_pack;
use super::f32_vector_array;
use super::test_session;
use super::vector_validity;
use crate::TQPack;
use crate::TQUnpack;
use crate::TurboQuant;
use crate::TurboQuantConfig;
use crate::vtable::tq_metadata;

#[test]
fn scalar_fn_ids_and_config_options_roundtrip() -> VortexResult<()> {
    let session = test_session();
    let config = TurboQuantConfig::try_new(4, 7, 2)?;

    assert_eq!(TQPack.id().as_ref(), "vortex.turboquant.pack");
    assert_eq!(TQUnpack.id().as_ref(), "vortex.turboquant.unpack");

    let pack_metadata = TQPack.serialize(&config)?.unwrap();
    let unpack_metadata = TQUnpack.serialize(&config)?.unwrap();

    assert_eq!(TQPack.deserialize(&pack_metadata, &session)?, config);
    assert_eq!(TQUnpack.deserialize(&unpack_metadata, &session)?, config);
    Ok(())
}

#[test]
fn scalar_fn_arrays_pack_and_unpack_vectors() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.25, Validity::from_iter([true, false]))?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let packed_lazy = TQPack::try_new_array(input, &config, 2)?;
    let packed_metadata = tq_metadata(packed_lazy.dtype())?;
    assert_eq!(packed_metadata.dimensions, 128);
    assert_eq!(packed_metadata.bit_width, config.bit_width());
    assert!(packed_lazy.dtype().as_extension().is::<TurboQuant>());

    let packed = packed_lazy.into_array().execute(&mut ctx)?;
    let unpacked_lazy = TQUnpack::try_new_array(packed, &config, 2)?;
    let unpacked = unpacked_lazy.into_array().execute(&mut ctx)?;
    let validity = vector_validity(unpacked, &mut ctx)?.execute_mask(2, &mut ctx)?;

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
    let config_metadata = TQUnpack.serialize(&config)?.unwrap();

    let packed = execute_tq_pack(input, &config, &mut ctx)?;
    let unpacked_lazy = TQUnpack::try_new_array(packed, &config, 2)?.into_array();
    let unpack_plugin = ScalarFnArrayPlugin::new(TQUnpack);
    assert_eq!(
        unpack_plugin.serialize(&unpacked_lazy, &session)?.unwrap(),
        config_metadata
    );
    Ok(())
}

#[test]
fn unpack_rejects_config_that_disagrees_with_turboquant_child() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 1, 0.25, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;
    let different_config = TurboQuantConfig::try_new(4, 42, 3)?;

    let packed = TQPack::try_new_array(input, &config, 1)?
        .into_array()
        .execute(&mut ctx)?;

    assert!(TQUnpack::try_new_array(packed, &different_config, 1).is_err());
    Ok(())
}
