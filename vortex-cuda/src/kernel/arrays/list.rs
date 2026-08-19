// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::List;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::list::ListArrayExt;
use vortex::array::arrays::list::ListArraySlotsExt;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::executor::execute_validity_cuda;

/// CUDA executor for `ListArray`.
///
/// `List` stores `len + 1` Arrow-style offsets; its canonical form, `ListView`, stores one
/// offset and one size per list. Decode the elements on the GPU and derive the view pair from
/// the offsets with a single kernel.
#[derive(Debug)]
pub(crate) struct ListExecutor;

#[async_trait]
impl CudaExecute for ListExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let list = array
            .try_downcast::<List>()
            .map_err(|_| vortex_err!("Expected ListArray"))?;

        let list_len = list.len();
        let validity = execute_validity_cuda(list.list_validity(), list_len, ctx).await?;
        let elements = list
            .elements()
            .clone()
            .execute_cuda(ctx)
            .await?
            .into_array();

        if list_len == 0 {
            let empty = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
            return Ok(Canonical::List(ListViewArray::try_new(
                elements,
                empty.clone().into_array(),
                empty.into_array(),
                validity,
            )?));
        }

        let offsets = list
            .offsets()
            .clone()
            .execute_cuda(ctx)
            .await?
            .into_primitive();
        vortex_ensure!(
            offsets.len() == list_len + 1,
            "ListArray must have {} offsets, got {}",
            list_len + 1,
            offsets.len()
        );

        let offsets_ptype = offsets.ptype();
        match_each_integer_ptype!(offsets_ptype, |O| {
            list_views_typed::<O>(offsets, elements, validity, list_len, ctx).await
        })
    }
}

async fn list_views_typed<O: DeviceRepr + NativePType>(
    offsets: PrimitiveArray,
    elements: ArrayRef,
    validity: Validity,
    list_len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let PrimitiveDataParts {
        buffer: offsets_buffer,
        ..
    } = offsets.into_data_parts();

    let offsets_device = ctx.ensure_on_device(offsets_buffer).await?;
    let offsets_view = offsets_device.cuda_view::<O>()?;

    let mut view_offsets = ctx.device_alloc::<O>(list_len)?;
    let mut view_sizes = ctx.device_alloc::<O>(list_len)?;
    let list_len_u64 = list_len as u64;

    let offsets_ptype = O::PTYPE.to_string();
    let cuda_function = ctx.load_function_with_suffixes("list", &["views", &offsets_ptype])?;
    ctx.launch_kernel(&cuda_function, list_len, |args| {
        args.arg(&offsets_view)
            .arg(&mut view_offsets)
            .arg(&mut view_sizes)
            .arg(&list_len_u64);
    })?;

    let view_offsets = PrimitiveArray::from_buffer_handle(
        BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(view_offsets))),
        O::PTYPE,
        Validity::NonNullable,
    );
    let view_sizes = PrimitiveArray::from_buffer_handle(
        BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(view_sizes))),
        O::PTYPE,
        Validity::NonNullable,
    );

    Ok(Canonical::List(ListViewArray::try_new(
        elements,
        view_offsets.into_array(),
        view_sizes.into_array(),
        validity,
    )?))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::ListArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[rstest]
    #[case::single_run(vec![0i32, 2, 5, 9])]
    #[case::empty_lists(vec![0i32, 0, 3, 3])]
    #[crate::test]
    async fn test_cuda_list_decompression(#[case] offsets: Vec<i32>) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let element_count = *offsets.last().vortex_expect("offsets are non-empty");
        let elements = PrimitiveArray::new(
            (0..element_count).collect::<Buffer<i32>>(),
            Validity::NonNullable,
        )
        .into_array();
        let offsets_array =
            PrimitiveArray::new(Buffer::from(offsets), Validity::NonNullable).into_array();
        let list = ListArray::try_new(elements, offsets_array, Validity::NonNullable)?;

        let gpu_result = ListExecutor
            .execute(list.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(list, gpu_result, &mut ctx);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_list_with_nulls() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let elements =
            PrimitiveArray::new(buffer![10i32, 20, 30, 40], Validity::NonNullable).into_array();
        let offsets =
            PrimitiveArray::new(buffer![0i32, 2, 2, 4], Validity::NonNullable).into_array();
        let validity =
            Validity::Array(BoolArray::from_iter([true, false, true].into_iter()).into_array());
        let list = ListArray::try_new(elements, offsets, validity)?;

        let gpu_result = ListExecutor
            .execute(list.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(list, gpu_result, &mut ctx);

        Ok(())
    }
}
