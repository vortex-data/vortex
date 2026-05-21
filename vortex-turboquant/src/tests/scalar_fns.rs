// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::f32_vector_array;
use super::test_session;
use super::vector_validity;
use crate::TQDecode;
use crate::TQEncode;
use crate::TurboQuant;
use crate::TurboQuantConfig;
use crate::vtable::tq_metadata;

#[test]
fn scalar_fn_ids_and_options_roundtrip() -> VortexResult<()> {
    let session = test_session();
    let config = TurboQuantConfig::try_new(4, 7, 2)?;

    assert_eq!(TQEncode.id().as_ref(), "vortex.turboquant.encode");
    assert_eq!(TQDecode.id().as_ref(), "vortex.turboquant.decode");

    let encode_metadata = TQEncode.serialize(&config)?.unwrap();
    let decode_metadata = TQDecode.serialize(&EmptyMetadata)?.unwrap();

    assert_eq!(TQEncode.deserialize(&encode_metadata, &session)?, config);
    assert!(decode_metadata.is_empty());
    assert_eq!(
        TQDecode.deserialize(&decode_metadata, &session)?,
        EmptyMetadata
    );
    Ok(())
}

#[test]
fn scalar_fn_arrays_encode_and_decode_vectors() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.25, Validity::from_iter([true, false]))?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let encoded_lazy = TQEncode::try_new_array(input, &config)?;
    let encoded_metadata = tq_metadata(encoded_lazy.dtype())?;
    assert_eq!(encoded_metadata.dimensions, 128);
    assert_eq!(encoded_metadata.bit_width, config.bit_width());
    assert!(encoded_lazy.dtype().as_extension().is::<TurboQuant>());

    let encoded = encoded_lazy.into_array().execute(&mut ctx)?;
    let decoded_lazy = TQDecode::try_new_array(encoded)?;
    let decoded = decoded_lazy.into_array().execute(&mut ctx)?;
    let validity = vector_validity(decoded, &mut ctx)?.execute_mask(2, &mut ctx)?;

    assert!(validity.value(0));
    assert!(!validity.value(1));
    Ok(())
}
