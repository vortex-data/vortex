// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Trait and utilities for scalar GPU decoders.
//!
//! Scalar GPU decoders are element-wise operations that decode encoded arrays
//! back to their canonical primitive form. This module provides a trait
//! [`ScalarGpuDecoder`] that captures the common pattern and a generic
//! [`execute_scalar_decoder`] function that handles the common execution flow.

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::DeviceRepr;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_error::VortexResult;

use crate::CudaDeviceBuffer;
use crate::CudaKernelEvents;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecutionCtx;

/// Trait for scalar GPU decoders that operate element-wise on encoded arrays.
///
/// Implement this trait for each (Encoding, PrimitiveType) combination to enable
/// GPU-accelerated decoding. The trait captures the encoding-specific parts
/// while the common execution flow is handled by [`execute_scalar_decoder`].
///
/// # Type Parameters
///
/// The associated types define the input/output primitive types, which must be
/// known at compile time for type-safe kernel argument passing.
pub trait ScalarGpuDecoder: Debug + Send + Sync + 'static {
    /// The specific array type (e.g., `FoRArray`, `ZigZagArray`, `ALPArray`).
    type Array: Clone + Send + Sync;

    /// Encoding-specific metadata extracted from the array (e.g., reference scalar, exponents).
    type Metadata: Send;

    /// The primitive type of the encoded (input) buffer.
    type InputPType: NativePType + DeviceRepr + Send + Sync + 'static;

    /// The primitive type of the decoded (output) buffer.
    type OutputPType: NativePType + DeviceRepr + Send + Sync + 'static;

    /// Kernel module name (e.g., "for", "zigzag", "alp").
    const KERNEL_MODULE: &'static str;

    /// Whether decoding happens in-place (true) or needs a separate output buffer (false).
    ///
    /// In-place decoding reuses the input buffer for output, which is valid when:
    /// - Input and output types have the same size (e.g., `u32` -> `u32` for FoR)
    /// - The bits can be reinterpreted (e.g., `u32` -> `i32` for ZigZag)
    ///
    /// Out-of-place decoding allocates a new buffer, required when:
    /// - Types have different sizes or representations (e.g., `i32` -> `f32` for ALP)
    const IN_PLACE: bool;

    /// Get the encoded child array from the specific array type.
    fn encoded(array: &Self::Array) -> &ArrayRef;

    /// Extract encoding-specific metadata from the array.
    fn extract_metadata(array: &Self::Array) -> VortexResult<Self::Metadata>;

    /// Get the primitive types for kernel function lookup.
    ///
    /// The kernel name is constructed as `{module}_{ptype1}_{ptype2}_...`.
    fn kernel_ptypes() -> &'static [PType];

    /// Launch the kernel with encoding-specific arguments.
    ///
    /// This method is responsible for building kernel arguments and launching the kernel.
    /// The input buffer handle is provided, along with optional output handle for out-of-place
    /// operations, the encoding metadata, and the execution context.
    ///
    /// For in-place operations, `output_handle` will be `None` and the kernel should
    /// modify `input_handle` in place.
    fn launch_kernel(
        ctx: &CudaExecutionCtx,
        input_handle: &BufferHandle,
        output_handle: Option<&BufferHandle>,
        metadata: &Self::Metadata,
        len: usize,
    ) -> VortexResult<CudaKernelEvents>;

    /// Get the output primitive type.
    ///
    /// By default returns `Self::OutputPType::PTYPE`, but can be overridden for
    /// encodings that need dynamic output type determination.
    fn output_ptype() -> PType {
        Self::OutputPType::PTYPE
    }
}

/// Execute a scalar GPU decoder using the common pattern.
///
/// This function handles the common execution flow for all scalar decoders:
/// 1. Extract encoding metadata
/// 2. Recursively execute child array on GPU
/// 3. Ensure buffer is on device
/// 4. Allocate output buffer (if not in-place)
/// 5. Launch kernel via the decoder's `launch_kernel` method
/// 6. Return result as `PrimitiveArray`
pub async fn execute_scalar_decoder<D: ScalarGpuDecoder>(
    array: D::Array,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let array_len = D::encoded(&array).len();
    assert!(array_len > 0, "Cannot decode empty array");

    // Step 1: Extract encoding metadata
    let metadata = D::extract_metadata(&array)?;

    // Step 2: Recursively execute child array on GPU
    let canonical = D::encoded(&array).clone().execute_cuda(ctx).await?;
    let (_, buffer, validity, ..) = canonical.into_primitive().into_parts();

    // Step 3: Ensure buffer is on device
    let device_input: BufferHandle = if buffer.is_on_device() {
        buffer
    } else {
        ctx.copy_buffer_to_device_async::<D::InputPType>(buffer)?
            .await?
    };

    // Step 4: Handle in-place vs out-of-place allocation
    let (output_handle, output_for_kernel) = if D::IN_PLACE {
        (device_input.clone(), None)
    } else {
        let output_slice = ctx.device_alloc::<D::OutputPType>(array_len)?;
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let handle = BufferHandle::new_device(Arc::new(output_buf));
        (handle.clone(), Some(handle))
    };

    // Step 5: Launch kernel
    let _events = D::launch_kernel(
        ctx,
        &device_input,
        output_for_kernel.as_ref(),
        &metadata,
        array_len,
    )?;

    // Step 6: Return result
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        D::output_ptype(),
        validity,
    )))
}

/// Macro to generate ScalarGpuDecoder implementations for FoR encoding.
///
/// FoR (Frame of Reference) decodes by adding a reference value to each element.
/// This is an in-place operation since input and output types are the same.
#[macro_export]
macro_rules! impl_for_scalar_decoder {
    ($($ptype:ty => $decoder:ident),* $(,)?) => {
        $(
            #[doc = concat!("FoR decoder for `", stringify!($ptype), "` values.")]
            #[derive(Debug)]
            pub struct $decoder;

            impl $crate::kernel::scalar::ScalarGpuDecoder for $decoder {
                type Array = vortex_fastlanes::FoRArray;
                type Metadata = $ptype;
                type InputPType = $ptype;
                type OutputPType = $ptype;

                const KERNEL_MODULE: &'static str = "for";
                const IN_PLACE: bool = true;

                fn encoded(array: &Self::Array) -> &vortex_array::ArrayRef {
                    array.encoded()
                }

                fn extract_metadata(array: &Self::Array) -> vortex_error::VortexResult<$ptype> {
                    use vortex_error::VortexExpect;
                    Ok(array
                        .reference_scalar()
                        .as_primitive()
                        .as_::<$ptype>()
                        .vortex_expect("reference cannot be null"))
                }

                fn kernel_ptypes() -> &'static [vortex_dtype::PType] {
                    &[<$ptype as vortex_dtype::NativePType>::PTYPE]
                }

                fn launch_kernel(
                    ctx: &$crate::CudaExecutionCtx,
                    input_handle: &vortex_array::buffer::BufferHandle,
                    _output_handle: Option<&vortex_array::buffer::BufferHandle>,
                    metadata: &$ptype,
                    len: usize,
                ) -> vortex_error::VortexResult<$crate::CudaKernelEvents> {
                    use cudarc::driver::PushKernelArg;
                    use $crate::CudaBufferExt;
                    let input_view = input_handle.cuda_view::<$ptype>()?;
                    let len_u64 = len as u64;
                    let reference = *metadata;
                    let events = $crate::launch_cuda_kernel!(
                        execution_ctx: ctx,
                        module: "for",
                        ptypes: Self::kernel_ptypes(),
                        launch_args: [input_view, reference, len_u64],
                        event_recording: cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                        array_len: len
                    );
                    Ok(events)
                }
            }
        )*
    };
}

/// Macro to generate ScalarGpuDecoder implementations for ZigZag encoding.
///
/// ZigZag decodes unsigned integers back to signed integers. This is an in-place
/// operation since the bit width is the same, just reinterpreted.
#[macro_export]
macro_rules! impl_zigzag_scalar_decoder {
    ($($unsigned:ty, $signed:ty => $decoder:ident),* $(,)?) => {
        $(
            #[doc = concat!("ZigZag decoder for `", stringify!($unsigned), "` -> `", stringify!($signed), "`.")]
            #[derive(Debug)]
            pub struct $decoder;

            impl $crate::kernel::scalar::ScalarGpuDecoder for $decoder {
                type Array = vortex_zigzag::ZigZagArray;
                type Metadata = ();
                type InputPType = $unsigned;
                type OutputPType = $signed;

                const KERNEL_MODULE: &'static str = "zigzag";
                const IN_PLACE: bool = true;

                fn encoded(array: &Self::Array) -> &vortex_array::ArrayRef {
                    array.encoded()
                }

                fn extract_metadata(_array: &Self::Array) -> vortex_error::VortexResult<()> {
                    Ok(())
                }

                fn kernel_ptypes() -> &'static [vortex_dtype::PType] {
                    &[<$unsigned as vortex_dtype::NativePType>::PTYPE]
                }

                fn launch_kernel(
                    ctx: &$crate::CudaExecutionCtx,
                    input_handle: &vortex_array::buffer::BufferHandle,
                    _output_handle: Option<&vortex_array::buffer::BufferHandle>,
                    _metadata: &(),
                    len: usize,
                ) -> vortex_error::VortexResult<$crate::CudaKernelEvents> {
                    use cudarc::driver::PushKernelArg;
                    use $crate::CudaBufferExt;
                    let input_view = input_handle.cuda_view::<$unsigned>()?;
                    let len_u64 = len as u64;
                    let events = $crate::launch_cuda_kernel!(
                        execution_ctx: ctx,
                        module: "zigzag",
                        ptypes: Self::kernel_ptypes(),
                        launch_args: [input_view, len_u64],
                        event_recording: cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                        array_len: len
                    );
                    Ok(events)
                }

                fn output_ptype() -> vortex_dtype::PType {
                    <$signed as vortex_dtype::NativePType>::PTYPE
                }
            }
        )*
    };
}

/// Macro to generate ScalarGpuDecoder implementations for ALP encoding.
///
/// ALP (Adaptive Lossless floating-Point) decodes integers back to floats using
/// exponent factors. This is an out-of-place operation since types differ.
#[macro_export]
macro_rules! impl_alp_scalar_decoder {
    ($($int:ty, $float:ty => $decoder:ident, $metadata:ident),* $(,)?) => {
        $(
            #[doc = concat!("ALP metadata for `", stringify!($float), "` values.")]
            #[derive(Debug)]
            pub struct $metadata {
                /// Multiply factor from F10 lookup table
                pub f: $float,
                /// Inverse factor from IF10 lookup table
                pub e: $float,
            }

            #[doc = concat!("ALP decoder for `", stringify!($int), "` -> `", stringify!($float), "`.")]
            #[derive(Debug)]
            pub struct $decoder;

            impl $crate::kernel::scalar::ScalarGpuDecoder for $decoder {
                type Array = vortex_alp::ALPArray;
                type Metadata = $metadata;
                type InputPType = $int;
                type OutputPType = $float;

                const KERNEL_MODULE: &'static str = "alp";
                const IN_PLACE: bool = false;

                fn encoded(array: &Self::Array) -> &vortex_array::ArrayRef {
                    array.encoded()
                }

                fn extract_metadata(array: &Self::Array) -> vortex_error::VortexResult<$metadata> {
                    use vortex_alp::ALPFloat;
                    let exp = array.exponents();
                    Ok($metadata {
                        f: <$float>::F10[exp.f as usize],
                        e: <$float>::IF10[exp.e as usize],
                    })
                }

                fn kernel_ptypes() -> &'static [vortex_dtype::PType] {
                    &[
                        <$int as vortex_dtype::NativePType>::PTYPE,
                        <$float as vortex_dtype::NativePType>::PTYPE,
                    ]
                }

                fn launch_kernel(
                    ctx: &$crate::CudaExecutionCtx,
                    input_handle: &vortex_array::buffer::BufferHandle,
                    output_handle: Option<&vortex_array::buffer::BufferHandle>,
                    metadata: &$metadata,
                    len: usize,
                ) -> vortex_error::VortexResult<$crate::CudaKernelEvents> {
                    use cudarc::driver::PushKernelArg;
                    use vortex_error::vortex_err;
                    use $crate::CudaBufferExt;
                    let input_view = input_handle.cuda_view::<$int>()?;
                    let output = output_handle.ok_or_else(|| vortex_err!("ALP requires output buffer"))?;
                    let output_view = output.cuda_view::<$float>()?;
                    let len_u64 = len as u64;
                    let f = metadata.f;
                    let e = metadata.e;
                    let events = $crate::launch_cuda_kernel!(
                        execution_ctx: ctx,
                        module: "alp",
                        ptypes: Self::kernel_ptypes(),
                        launch_args: [input_view, output_view, f, e, len_u64],
                        event_recording: cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                        array_len: len
                    );
                    Ok(events)
                }
            }
        )*
    };
}
