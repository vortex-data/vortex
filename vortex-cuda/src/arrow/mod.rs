// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module implements the Arrow C Device data interface extension for sharing GPU-resident
//! data.
//!
//! This is an extension to the Arrow C Data Interface.
//!
//! More documentation at <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>

mod canonical;
mod varbinview;

use std::ffi::c_void;
use std::fmt::Debug;
use std::ptr;
use std::sync::Arc;

use async_trait::async_trait;
pub(crate) use canonical::CanonicalDeviceArrayExport;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::runtime::sys::cudaEvent_t;
use vortex::array::ArrayRef;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
mod arrow_c_abi {
    include!(concat!(env!("OUT_DIR"), "/arrow_c_abi.rs"));
}

pub use arrow_c_abi::ArrowArray;
pub use arrow_c_abi::ArrowDeviceArray;
pub use arrow_c_abi::ArrowDeviceType;

/// CUDA device memory.
pub const ARROW_DEVICE_CUDA: ArrowDeviceType = arrow_c_abi::ARROW_DEVICE_CUDA as ArrowDeviceType;

/// A pointer to a device-specific synchronization event, or null if synchronization is not needed.
pub type SyncEvent = *mut c_void;

impl ArrowArray {
    pub fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

unsafe impl Send for ArrowArray {}
unsafe impl Sync for ArrowArray {}
unsafe impl Send for ArrowDeviceArray {}

#[expect(
    unused,
    reason = "cuda_stream and cuda_buffers need to have deferred drop"
)]
pub(crate) struct PrivateData {
    /// Hold a reference to the CudaStream so that it stays alive even after CudaExecutionCtx
    /// has been dropped.
    pub(crate) cuda_stream: Arc<CudaStream>,
    /// The single boxed slice which owns all buffers that the Rust code allocated on the device.
    pub(crate) buffers: Box<[Option<BufferHandle>]>,
    /// Boxed slice of buffer pointers. We return a pointer to the start of this allocation over
    /// the interface, so we hold it here so the Box contents are not freed.
    pub(crate) buffer_ptrs: Box<[*const c_void]>,
    pub(crate) cuda_event: CudaEvent,
    pub(crate) cuda_event_ptr: cudaEvent_t,
    pub(crate) children: Box<[*mut ArrowArray]>,
}

impl PrivateData {
    pub(crate) fn new(
        buffers: Vec<Option<BufferHandle>>,
        children: Vec<ArrowArray>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Box<Self>> {
        let buffers = buffers.into_boxed_slice();
        let buffer_ptrs: Box<[*const c_void]> = buffers
            .iter()
            .map(|buf| {
                match buf {
                    None => {
                        // null pointer
                        Ok(ptr::null())
                    }
                    Some(handle) => Ok(handle.cuda_device_ptr()? as usize as *const c_void),
                }
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();

        let children = children
            .into_iter()
            .map(|array| Box::into_raw(Box::new(array)))
            .collect::<Box<[_]>>();

        // generate the synchronization event
        let cuda_event = ctx
            .stream()
            .record_event(None)
            .map_err(|_| vortex_err!("failed to create cudaEvent_t"))?;

        Ok(Box::new(Self {
            buffers,
            buffer_ptrs,
            cuda_stream: Arc::clone(ctx.stream()),
            children,
            cuda_event_ptr: cuda_event.cu_event().cast(),
            cuda_event,
        }))
    }

    pub(crate) fn sync_event(&mut self) -> SyncEvent {
        (&raw mut self.cuda_event_ptr).cast()
    }
}

#[async_trait]
pub trait DeviceArrayExt {
    async fn export_device_array(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray>;
}

#[async_trait]
impl DeviceArrayExt for ArrayRef {
    async fn export_device_array(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray> {
        let exporter = Arc::clone(ctx.exporter());
        exporter.export_device_array(self, ctx).await
    }
}

/// A type that can convert a Vortex array into an [`ArrowDeviceArray`].
#[async_trait]
pub trait ExportDeviceArray: Debug + Send + Sync + 'static {
    /// Export a Vortex array as an [`ArrowDeviceArray`].
    ///
    /// The Arrow Device Array is part of the Arrow C Device data interface extension to the Arrow
    /// specification. It enables passing Vortex arrays to other processes that consume Arrow
    /// arrays, such as cudf.
    ///
    /// See <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>.
    async fn export_device_array(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray>;
}

/// Check that the validity buffer is empty and does not need to be copied over the device boundary.
pub(crate) fn check_validity_empty(validity: &Validity) -> VortexResult<()> {
    if let Validity::AllInvalid | Validity::Array(_) = validity {
        vortex_bail!("Exporting array with non-trivial validity not supported yet")
    }

    Ok(())
}
