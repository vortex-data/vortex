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
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_signed_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::datetime_parts::DateTimeParts;
use vortex::encodings::datetime_parts::DateTimePartsArraySlotsExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::scalar::Scalar;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for DateTimeParts arrays.
///
/// Combines the days, seconds, and subseconds components into a single i64 timestamp array.
#[derive(Debug)]
pub(crate) struct DateTimePartsExecutor;

#[async_trait]
impl CudaExecute for DateTimePartsExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let output_len = array.len();
        let array = array
            .try_downcast::<DateTimeParts>()
            .map_err(|_| vortex_err!("Expected DateTimePartsArray"))?;

        // Extract the temporal metadata from the dtype
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_bail!("DateTimePartsArray dtype must be an Extension type")
        };

        let Some(options) = ext.metadata_opt::<Timestamp>() else {
            vortex_bail!("DateTimePartsArray must have Timestamp metadata")
        };

        let time_unit = options.unit;
        let time_zone = options.tz.clone();
        let validity = array.validity()?;

        if output_len == 0 {
            return Ok(Canonical::empty(array.dtype()));
        }

        if validity.definitely_all_null() {
            let storage_ptype = ext.storage_dtype().as_ptype();
            return Ok(Canonical::Extension(
                TemporalArray::new_timestamp(
                    ConstantArray::new(
                        Scalar::null(DType::Primitive(storage_ptype, Nullability::Nullable)),
                        output_len,
                    )
                    .into_array(),
                    time_unit,
                    time_zone,
                )
                .into(),
            ));
        }

        let divisor: i64 = match options.unit {
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Seconds => 1,
            TimeUnit::Days => vortex_bail!("Cannot decode DateTimeParts with TimeUnit::Days"),
        };

        let days_canonical = array.days().clone().execute_cuda(ctx).await?;
        let seconds_canonical = array.seconds().clone().execute_cuda(ctx).await?;
        let subseconds_canonical = array.subseconds().clone().execute_cuda(ctx).await?;

        let days_prim = days_canonical.into_primitive();

        // TODO(0ax1): Figure out how to handle constant arrays in CUDA kernels.
        let seconds_prim = seconds_canonical.into_primitive();
        let subseconds_prim = subseconds_canonical.into_primitive();

        let days_ptype = days_prim.ptype();
        let seconds_ptype = seconds_prim.ptype();
        let subseconds_ptype = subseconds_prim.ptype();

        match_each_signed_integer_ptype!(days_ptype, |DaysT| {
            match_each_signed_integer_ptype!(seconds_ptype, |SecondsT| {
                match_each_signed_integer_ptype!(subseconds_ptype, |SubsecondsT| {
                    decode_datetimeparts_typed::<DaysT, SecondsT, SubsecondsT>(
                        days_prim,
                        seconds_prim,
                        subseconds_prim,
                        divisor,
                        time_unit,
                        time_zone,
                        validity,
                        ctx,
                    )
                    .await
                })
            })
        })
    }
}

#[expect(clippy::too_many_arguments)]
async fn decode_datetimeparts_typed<DaysT, SecondsT, SubsecondsT>(
    days: PrimitiveArray,
    seconds: PrimitiveArray,
    subseconds: PrimitiveArray,
    divisor: i64,
    time_unit: TimeUnit,
    time_zone: Option<Arc<str>>,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    DaysT: NativePType + DeviceRepr,
    SecondsT: NativePType + DeviceRepr,
    SubsecondsT: NativePType + DeviceRepr,
{
    let output_len = days.len();

    let PrimitiveDataParts {
        buffer: days_buffer,
        ..
    } = days.into_data_parts();
    let PrimitiveDataParts {
        buffer: seconds_buffer,
        ..
    } = seconds.into_data_parts();
    let PrimitiveDataParts {
        buffer: subseconds_buffer,
        ..
    } = subseconds.into_data_parts();

    // Move buffers to device if not already there
    let days_device = ctx.ensure_on_device(days_buffer).await?;
    let seconds_device = ctx.ensure_on_device(seconds_buffer).await?;
    let subseconds_device = ctx.ensure_on_device(subseconds_buffer).await?;

    // Allocate output buffer
    let output_slice = ctx.device_alloc::<i64>(output_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    let days_view = days_device.cuda_view::<DaysT>()?;
    let seconds_view = seconds_device.cuda_view::<SecondsT>()?;
    let subseconds_view = subseconds_device.cuda_view::<SubsecondsT>()?;
    let output_view = output_device.as_view::<i64>();

    let cuda_function = ctx.load_function(
        "date_time_parts",
        &[DaysT::PTYPE, SecondsT::PTYPE, SubsecondsT::PTYPE],
    )?;

    let array_len_u64 = output_len as u64;

    ctx.launch_kernel(&cuda_function, output_len, |args| {
        args.arg(&days_view)
            .arg(&seconds_view)
            .arg(&subseconds_view)
            .arg(&divisor)
            .arg(&output_view)
            .arg(&array_len_u64);
    })?;

    let output_buffer = BufferHandle::new_device(Arc::new(output_device));
    let output_primitive = PrimitiveArray::from_buffer_handle(output_buffer, PType::I64, validity);

    Ok(Canonical::Extension(
        TemporalArray::new_timestamp(output_primitive.into_array(), time_unit, time_zone).into(),
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::encodings::datetime_parts::DateTimeParts;
    use vortex::encodings::datetime_parts::DateTimePartsArray;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::extension::datetime::TimeUnit;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    fn make_datetimeparts_array(
        days: Vec<i32>,
        seconds: Vec<i32>,
        subseconds: Vec<i64>,
        time_unit: TimeUnit,
    ) -> DateTimePartsArray {
        let len = days.len();
        let days_arr = PrimitiveArray::new(Buffer::from(days), Validity::NonNullable).into_array();
        let seconds_arr =
            PrimitiveArray::new(Buffer::from(seconds), Validity::NonNullable).into_array();
        let subseconds_arr =
            PrimitiveArray::new(Buffer::from(subseconds), Validity::NonNullable).into_array();

        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![0i64; len], Validity::NonNullable).into_array(),
            time_unit,
            None,
        );

        DateTimeParts::try_new(
            temporal.dtype().clone(),
            days_arr,
            seconds_arr,
            subseconds_arr,
        )
        .vortex_expect("Failed to create DateTimePartsArray")
    }

    #[rstest]
    #[case::seconds(
        vec![1i32, 2, -1],
        vec![3600i32, 0, 3600],
        vec![0i64, 0, 0],
        TimeUnit::Seconds
    )]
    #[case::milliseconds(
        vec![1i32, 0, -1],
        vec![0i32, 1, 0],
        vec![500i64, 0, 999],
        TimeUnit::Milliseconds
    )]
    #[case::microseconds(
        vec![0i32, 1, 2],
        vec![0i32, 0, 0],
        vec![1i64, 1000, 1000000],
        TimeUnit::Microseconds
    )]
    #[case::nanoseconds(
        vec![0i32, 0, 1],
        vec![1i32, 60, 0],
        vec![123456789i64, 0, 0],
        TimeUnit::Nanoseconds
    )]
    #[crate::test]
    async fn test_cuda_datetimeparts_decompression(
        #[case] days: Vec<i32>,
        #[case] seconds: Vec<i32>,
        #[case] subseconds: Vec<i64>,
        #[case] time_unit: TimeUnit,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let dtp_array = make_datetimeparts_array(days, seconds, subseconds, time_unit);
        let cpu_result = crate::canonicalize_cpu(dtp_array.clone())?;

        let gpu_result = DateTimePartsExecutor
            .execute(dtp_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_datetimeparts_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let len = 2050;
        let days: Vec<i32> = (0..len).collect();
        let seconds: Vec<i32> = (0..len).map(|i| i % 86400).collect();
        let subseconds: Vec<i64> = (0..len).map(|i| (i % 1000) as i64).collect();

        let dtp_array = make_datetimeparts_array(days, seconds, subseconds, TimeUnit::Milliseconds);
        let cpu_result = crate::canonicalize_cpu(dtp_array.clone())?;

        let gpu_result = DateTimePartsExecutor
            .execute(dtp_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_datetimeparts_with_nulls() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let days_arr = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5],
            Validity::from_iter([true, false, true, false, true]),
        )
        .into_array();
        let seconds_arr =
            PrimitiveArray::new(buffer![0i32, 0, 0, 0, 0], Validity::NonNullable).into_array();
        let subseconds_arr =
            PrimitiveArray::new(buffer![0i64, 0, 0, 0, 0], Validity::NonNullable).into_array();

        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::new(
                buffer![0i64; 5],
                Validity::from_iter([true, false, true, false, true]),
            )
            .into_array(),
            TimeUnit::Seconds,
            None,
        );

        let dtp_array = DateTimeParts::try_new(
            temporal.dtype().clone(),
            days_arr,
            seconds_arr,
            subseconds_arr,
        )
        .vortex_expect("Failed to create DateTimePartsArray");

        let cpu_result = crate::canonicalize_cpu(dtp_array.clone())?;

        let gpu_result = DateTimePartsExecutor
            .execute(dtp_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
