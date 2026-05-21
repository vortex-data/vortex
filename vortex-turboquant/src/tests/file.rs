// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexWriteOptions;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;
use vortex_tensor::vector::Vector;

use super::execute_tq_decode_from_metadata;
use super::execute_tq_encode;
use super::f32_vector_array;
use super::file_session;
use super::vector_validity;
use crate::TQDecode;
use crate::TurboQuantConfig;
use crate::vector::storage::parse_storage;
use crate::vtable::tq_metadata;

#[test]
fn file_roundtrip_with_initialize_session() -> VortexResult<()> {
    let runtime = SingleThreadRuntime::default();
    let session = file_session(&runtime);
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.25, Validity::from_iter([true, false]))?;
    let encoded = execute_tq_encode(input, &TurboQuantConfig::default(), &mut ctx)?;

    let mut file_bytes = Vec::new();
    VortexWriteOptions::new(session.clone())
        .blocking(&runtime)
        .write(&mut file_bytes, encoded.to_array_iterator())?;

    let file = session.open_options().open_buffer(file_bytes)?;
    let read = runtime.block_on(async { file.scan()?.into_array_stream()?.read_all().await })?;

    let metadata = tq_metadata(read.dtype())?;
    assert_eq!(metadata.dimensions, 128);
    let decoded = execute_tq_decode_from_metadata(read, &mut ctx)?;
    let validity = vector_validity(decoded, &mut ctx)?.execute_mask(2, &mut ctx)?;
    assert!(validity.value(0));
    assert!(!validity.value(1));
    Ok(())
}

/// File-roundtrip preserves the `L2Norm(TQDecode(_))` fast-path invariant. A regression that
/// silently broke the in-flight decode correction or the kernel would only show up downstream
/// as norm divergence; this test surfaces it at the IO layer.
#[test]
fn file_roundtrip_preserves_l2_norm_invariant() -> VortexResult<()> {
    let runtime = SingleThreadRuntime::default();
    let session = file_session(&runtime);
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 4, 0.25, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;
    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let original_norms: PrimitiveArray = parse_storage(encoded.clone(), &mut ctx)?.norms;

    let mut file_bytes = Vec::new();
    VortexWriteOptions::new(session.clone())
        .blocking(&runtime)
        .write(&mut file_bytes, encoded.to_array_iterator())?;

    let file = session.open_options().open_buffer(file_bytes)?;
    let read = runtime.block_on(async { file.scan()?.into_array_stream()?.read_all().await })?;

    // Fast-path `L2Norm(TQDecode(_))` must still return the originally stored row norms after
    // the file roundtrip. If the kernel or the in-flight decode correction had silently
    // broken at serialization, this is where it would surface.
    let decoded = TQDecode::try_new_array(read)?.into_array();
    let kernel_norms: PrimitiveArray = L2Norm::try_new_array(decoded, 4)?
        .into_array()
        .execute(&mut ctx)?;
    assert_eq!(
        kernel_norms.as_slice::<f32>(),
        original_norms.as_slice::<f32>(),
        "L2Norm(TQDecode(read_back)) must equal the originally stored row norms"
    );
    Ok(())
}

#[test]
fn file_roundtrip_lazy_decode_scalar_fn_with_initialize_session() -> VortexResult<()> {
    let runtime = SingleThreadRuntime::default();
    let session = file_session(&runtime);
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(128, 2, 0.25, Validity::from_iter([true, false]))?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;
    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let decoded = TQDecode::try_new_array(encoded)?.into_array();

    let mut file_bytes = Vec::new();
    VortexWriteOptions::new(session.clone())
        .blocking(&runtime)
        .write(&mut file_bytes, decoded.to_array_iterator())?;

    let file = session.open_options().open_buffer(file_bytes)?;
    let read = runtime.block_on(async { file.scan()?.into_array_stream()?.read_all().await })?;

    assert!(read.dtype().as_extension().is::<Vector>());

    let validity = vector_validity(read, &mut ctx)?.execute_mask(2, &mut ctx)?;
    assert!(validity.value(0));
    assert!(!validity.value(1));
    Ok(())
}
