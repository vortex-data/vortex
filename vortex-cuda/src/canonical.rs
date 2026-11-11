// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::future::try_join_all;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::bool::BoolDataParts;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::listview::ListViewDataParts;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::struct_::StructDataParts;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::arrays::varbinview::VarBinViewDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

/// Copy a child array to the host, executing it there if CUDA execution left it encoded.
async fn child_into_host(child: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let child = if child.is_canonical() {
        child
    } else {
        buffers_into_host(child).await?
    };
    Ok(child
        .execute::<Canonical>(ctx)?
        .into_host(ctx)
        .await?
        .into_array())
}

/// Copies every device buffer of `array` and of its children to the host, keeping the encoding.
///
/// A child that CUDA execution left encoded, such as a validity mask that is still a lazy `take`
/// through device-resident dictionary codes or a run-end encoded mask, can only be executed on
/// the host once all of its buffers are there.
fn buffers_into_host(array: ArrayRef) -> BoxFuture<'static, VortexResult<ArrayRef>> {
    Box::pin(async move {
        if array.is_host() {
            return Ok(array);
        }
        let mut array = array;
        if array.children_iter().any(|child| !child.is_host()) {
            let mut slots = Vec::with_capacity(array.slots().len());
            for slot in array.slots().iter() {
                slots.push(match slot {
                    Some(child) => Some(buffers_into_host(child.clone()).await?),
                    None => None,
                });
            }
            // SAFETY: the slots are byte-for-byte copies; only their placement changes.
            array = unsafe { array.with_slots(slots.into())? };
        }
        let buffers = array.buffer_handles();
        // Some encodings, such as Sparse, reject buffer replacement, so only replace when a
        // buffer actually moves.
        if buffers.iter().all(BufferHandle::is_on_host) {
            return Ok(array);
        }
        let buffers = try_join_all(buffers.into_iter().map(|buffer| async move {
            if buffer.is_on_host() {
                return Ok::<_, VortexError>(buffer);
            }
            Ok(BufferHandle::new_host(buffer.try_into_host()?.await?))
        }))
        .await?;
        // SAFETY: the buffers are byte-for-byte copies; only their placement changes.
        unsafe { array.with_buffers(buffers) }
    })
}

/// Move all canonical data from to_host from device.
#[async_trait]
pub trait CanonicalCudaExt {
    /// Copies the canonical data to the host, using `ctx` to canonicalize any child arrays.
    async fn into_host(self, ctx: &mut ExecutionCtx) -> VortexResult<Self>
    where
        Self: Sized;
}

/// Copies an array-backed validity mask back to the host.
///
/// Only [`Validity::Array`] owns a buffer; the other variants are metadata and pass through.
/// Migrating the values of a nullable array without its validity leaves the mask on the
/// device, and the first host read of it — canonicalising to Arrow, say — panics in
/// `BufferHandle::unwrap_host`.
async fn validity_into_host(validity: Validity, ctx: &mut ExecutionCtx) -> VortexResult<Validity> {
    let Validity::Array(array) = validity else {
        return Ok(validity);
    };
    Ok(Validity::Array(child_into_host(array, ctx).await?))
}

#[async_trait]
impl CanonicalCudaExt for Canonical {
    async fn into_host(self, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match self {
            Canonical::Struct(struct_array) => {
                // Children should all be canonical now
                let len = struct_array.len();
                let StructDataParts {
                    fields,
                    struct_fields,
                    validity,
                    ..
                } = struct_array.into_data_parts();

                let mut host_fields = vec![];
                for field in fields.iter() {
                    host_fields.push(child_into_host(field.clone(), ctx).await?);
                }

                Ok(Canonical::Struct(StructArray::new(
                    struct_fields.names().clone(),
                    host_fields,
                    len,
                    validity_into_host(validity, ctx).await?,
                )))
            }
            n @ Canonical::Null(_) => Ok(n),
            Canonical::Bool(bool) => {
                let len = bool.len();
                let validity = validity_into_host(bool.validity()?, ctx).await?;
                let BoolDataParts { bits, meta } = bool.into_data().into_parts(len);

                let bits = BitBuffer::new_with_offset(
                    bits.try_into_host()?.await?,
                    meta.len(),
                    meta.offset(),
                );
                Ok(Canonical::Bool(BoolArray::new(bits, validity)))
            }
            Canonical::Primitive(prim) => {
                let PrimitiveDataParts {
                    ptype,
                    buffer,
                    validity,
                    ..
                } = prim.into_data_parts();
                Ok(Canonical::Primitive(PrimitiveArray::from_byte_buffer(
                    buffer.try_into_host()?.await?,
                    ptype,
                    validity_into_host(validity, ctx).await?,
                )))
            }
            Canonical::Decimal(decimal) => {
                let DecimalDataParts {
                    decimal_dtype,
                    values,
                    values_type,
                    validity,
                    ..
                } = decimal.into_data_parts();
                let validity = validity_into_host(validity, ctx).await?;
                Ok(Canonical::Decimal(unsafe {
                    DecimalArray::new_unchecked_handle(
                        BufferHandle::new_host(values.try_into_host()?.await?),
                        values_type,
                        decimal_dtype,
                        validity,
                    )
                }))
            }
            Canonical::VarBinView(varbinview) => {
                let VarBinViewDataParts {
                    views,
                    buffers,
                    validity,
                    dtype,
                } = varbinview.into_data_parts();
                let validity = validity_into_host(validity, ctx).await?;

                // Copy all device views to host
                let host_views = views.try_into_host()?.await?;
                let host_views = Buffer::<BinaryView>::from_byte_buffer(host_views);

                // Copy any string data buffers back over to the host
                let host_buffers = buffers
                    .iter()
                    .cloned()
                    .map(|b| b.try_into_host())
                    .collect::<VortexResult<Vec<_>>>()?;
                let host_buffers = try_join_all(host_buffers).await?;
                let host_buffers: Arc<[ByteBuffer]> = Arc::from(host_buffers);

                Ok(Canonical::VarBinView(unsafe {
                    VarBinViewArray::new_unchecked(host_views, host_buffers, dtype, validity)
                }))
            }
            Canonical::List(list) => {
                let ListViewDataParts {
                    elements,
                    offsets,
                    sizes,
                    validity,
                    ..
                } = list.into_data_parts();
                let validity = validity_into_host(validity, ctx).await?;

                let elements = child_into_host(elements, ctx).await?;
                let offsets = child_into_host(offsets, ctx).await?;
                let sizes = child_into_host(sizes, ctx).await?;
                // Device-resident offsets and sizes skip numeric validation when the list is
                // built, so check them now that they are on the host.
                Ok(Canonical::List(ListViewArray::try_new(
                    elements, offsets, sizes, validity, ctx,
                )?))
            }
            Canonical::Extension(ext) => {
                // Copy the storage array to host and rewrap in ExtensionArray.
                let host_storage = child_into_host(ext.storage_array().clone(), ctx).await?;
                Ok(Canonical::Extension(ExtensionArray::new(
                    ext.ext_dtype().clone(),
                    host_storage,
                )))
            }
            c => vortex_bail!("into_host is not implemented for {}", c.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::DictArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::PType;
    use vortex::encodings::runend::RunEnd;
    use vortex::encodings::sparse::Sparse;
    use vortex::scalar::Scalar;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CudaSession;
    use crate::executor::CudaExecute;
    use crate::executor::CudaExecutionCtx;
    use crate::kernel::DictExecutor;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[crate::test]
    async fn test_into_host_rejects_out_of_bounds_device_list_view() -> VortexResult<()> {
        let cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let offsets = cuda_ctx.stream().copy_to_device(vec![0u32, 2])?.await?;
        let sizes = cuda_ctx.stream().copy_to_device(vec![2u32, 5])?.await?;
        let mut ctx = SESSION.create_execution_ctx();
        // The second list reads elements 2..7 of 3. The range check is skipped while the offsets
        // and sizes are on the device, so it must run once they are copied to the host.
        let list = ListViewArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            PrimitiveArray::from_buffer_handle(offsets, PType::U32, Validity::NonNullable)
                .into_array(),
            PrimitiveArray::from_buffer_handle(sizes, PType::U32, Validity::NonNullable)
                .into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?;

        assert!(Canonical::List(list).into_host(&mut ctx).await.is_err());
        Ok(())
    }

    /// Moves every buffer of `array` and of its children to the device without decoding it.
    fn encoded_on_device(array: ArrayRef, cuda_ctx: &CudaExecutionCtx) -> VortexResult<ArrayRef> {
        let mut slots = Vec::new();
        for slot in array.slots().iter() {
            slots.push(match slot {
                Some(child) => Some(encoded_on_device(child.clone(), cuda_ctx)?),
                None => None,
            });
        }
        // SAFETY: the slots are byte-for-byte copies; only their placement changes.
        let array = unsafe { array.with_slots(slots.into())? };
        // Sparse keeps its fill value in a host buffer that it doesn't allow to be replaced.
        if array.as_opt::<Sparse>().is_some() {
            return Ok(array);
        }
        let mut buffers = Vec::new();
        for buffer in array.buffer_handles() {
            buffers.push(cuda_ctx.ensure_on_device_sync(buffer)?);
        }
        // SAFETY: the buffers are byte-for-byte copies; only their placement changes.
        unsafe { array.with_buffers(buffers) }
    }

    /// A GPU decode can leave a validity mask encoded on the device, here run-end encoded.
    /// Copying the values to the host used to execute that mask on the host while its buffers
    /// were still on the device, which panicked with "expected host buffer".
    #[crate::test]
    async fn test_into_host_copies_encoded_device_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let mut ctx = SESSION.create_execution_ctx();
        let mask = RunEnd::new(
            PrimitiveArray::new(Buffer::from(vec![2u32, 5, 6]), Validity::NonNullable).into_array(),
            BoolArray::from_iter([true, false, true]).into_array(),
            &mut ctx,
        )
        .into_array();
        let mask = encoded_on_device(mask, &cuda_ctx)?;
        let values = cuda_ctx
            .stream()
            .copy_to_device(vec![1i32, 2, 3, 4, 5, 6])?
            .await?;
        let array = PrimitiveArray::from_buffer_handle(values, PType::I32, Validity::Array(mask));

        let host = Canonical::Primitive(array)
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_array();
        let expected =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, None, None, Some(6)]);
        assert_arrays_eq!(host, expected, &mut ctx);
        Ok(())
    }

    /// A sparse validity mask keeps its fill value in a host buffer, which Sparse doesn't allow to
    /// be replaced, while its patch indices and values are on the device. Only the children move
    /// to the host.
    #[crate::test]
    async fn test_into_host_copies_sparse_device_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let mut ctx = SESSION.create_execution_ctx();
        let mask = Sparse::try_new(
            PrimitiveArray::new(Buffer::from(vec![1u32, 4]), Validity::NonNullable).into_array(),
            BoolArray::from_iter([false, false]).into_array(),
            6,
            Scalar::from(true),
            &mut ctx,
        )?
        .into_array();
        let mask = encoded_on_device(mask, &cuda_ctx)?;
        let values = cuda_ctx
            .stream()
            .copy_to_device(vec![1i32, 2, 3, 4, 5, 6])?
            .await?;
        let array = PrimitiveArray::from_buffer_handle(values, PType::I32, Validity::Array(mask));

        let host = Canonical::Primitive(array)
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_array();
        let expected =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None, Some(6)]);
        assert_arrays_eq!(host, expected, &mut ctx);
        Ok(())
    }

    /// A dictionary with nullable values returns its validity as a lazy gather through the codes.
    /// With the codes on the device, as a file scan leaves them, copying the decoded dictionary to
    /// the host used to panic on a host-buffer read.
    async fn assert_device_dict_into_host(
        codes: PrimitiveArray,
        values: ArrayRef,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let mut ctx = SESSION.create_execution_ctx();
        let expected = DictArray::try_new(codes.clone().into_array(), values.clone())?;
        let device_codes = encoded_on_device(codes.into_array(), &cuda_ctx)?;
        let dict = DictArray::try_new(device_codes, values)?;
        let host = DictExecutor
            .execute(dict.into_array(), &mut cuda_ctx)
            .await?
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_array();
        assert_arrays_eq!(host, expected, &mut ctx);
        Ok(())
    }

    #[crate::test]
    async fn test_into_host_string_dict_with_nullable_values() -> VortexResult<()> {
        let values = VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            None,
            Some("a string longer than twelve bytes"),
        ]);
        let codes = PrimitiveArray::new(
            Buffer::from(vec![0u8, 1, 2, 1, 0, 2]),
            Validity::NonNullable,
        );
        assert_device_dict_into_host(codes, values.into_array()).await
    }

    #[crate::test]
    async fn test_into_host_primitive_dict_with_nullable_values() -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter([Some(7i32), None, Some(-3)]);
        let codes =
            PrimitiveArray::new(Buffer::from(vec![2u32, 0, 1, 1, 0]), Validity::NonNullable);
        assert_device_dict_into_host(codes, values.into_array()).await
    }
}
