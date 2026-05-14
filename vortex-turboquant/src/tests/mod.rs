// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::unwrap_in_result)
)]

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::memory::MemorySession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_tensor::vector::Vector;

use crate::TQDecode;
use crate::TQEncode;
use crate::TurboQuantConfig;
use crate::initialize;

mod encode_decode;
mod file;
mod malformed;
mod metadata;
mod parity;
mod scalar_fns;

fn test_session() -> VortexSession {
    let session = VortexSession::empty().with::<ArraySession>();
    initialize(&session);
    session
}

fn file_session(runtime: &SingleThreadRuntime) -> VortexSession {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<MemorySession>()
        .with::<RuntimeSession>()
        .with_handle(runtime.handle());
    vortex_file::register_default_encodings(&session);
    vortex_tensor::initialize(&session);
    initialize(&session);
    session
}

fn vector_array<T: NativePType>(
    dimensions: u32,
    values: &[T],
    validity: Validity,
) -> VortexResult<ArrayRef> {
    assert!(dimensions > 0, "dimensions must be > 0");
    let row_count = values.len() / dimensions as usize;

    let elements = PrimitiveArray::new::<T>(
        values.iter().copied().collect::<Buffer<T>>(),
        Validity::NonNullable,
    );
    let fsl = FixedSizeListArray::try_new(elements.into_array(), dimensions, validity, row_count)?;

    Ok(ExtensionArray::try_new_from_vtable(Vector, EmptyMetadata, fsl.into_array())?.into_array())
}

fn f32_vector_array(
    dimensions: u32,
    rows: usize,
    scale: f32,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let values = (0..rows * dimensions as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * scale)
        .collect::<Vec<_>>();
    vector_array(dimensions, &values, validity)
}

fn vector_values_f32(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<f32>> {
    let ext: ExtensionArray = array.execute(ctx)?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    Ok(elements.as_slice::<f32>().to_vec())
}

fn vector_validity(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Validity> {
    let ext: ExtensionArray = array.execute(ctx)?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    fsl.validity()
}

fn vector_element_ptype(array: &ExtensionArray) -> VortexResult<PType> {
    Ok(array
        .storage_array()
        .dtype()
        .as_fixed_size_list_element_opt()
        .ok_or_else(|| vortex_err!("expected FixedSizeList vector storage"))?
        .as_ptype())
}

fn turboquant_storage(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<StructArray> {
    let ext: ExtensionArray = array.execute(ctx)?;
    ext.storage_array().clone().execute(ctx)
}

fn execute_tq_encode(
    input: ArrayRef,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    TQEncode::try_new_array(input, config)?
        .into_array()
        .execute(ctx)
}

fn execute_tq_decode(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    TQDecode::try_new_array(input)?.into_array().execute(ctx)
}

fn execute_tq_decode_from_metadata(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    execute_tq_decode(input, ctx)
}
