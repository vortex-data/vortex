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
use std::ptr::NonNull;
use std::sync::Arc;

use async_trait::async_trait;
pub(crate) use canonical::CanonicalDeviceArrayExport;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::sys;
use cudarc::runtime::sys::cudaEvent_t;
use vortex::array::ArrayRef;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

#[derive(Debug, Copy, Clone)]
#[repr(i32)]
pub enum DeviceType {
    /// Host-resident data buffer
    Cpu = 1,
    Cuda = 2,
    CudaHost = 3,
    // OpenCL = 4,
    // Vulkan = 7,
    // Metal = 8,
    // Vpi = 9,
    // Rocm = 10,
    // RocmHost = 11,
    CudaManaged = 13,
    // OneApi = 14,
    // WebGPU = 15,
    // Hexagon = 16,
}

/// A (potentially null) pointer to a `cudaEvent_t`.
pub type SyncEvent = Option<NonNull<cudaEvent_t>>;

/// The C Device data interface representation of an Arrow array.
///
/// This array contains on-device pointers to Arrow array data, along with a synchronization
/// event that the client must wait on.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowDeviceArray {
    array: ArrowArray,
    device_id: i64,
    device_type: DeviceType,
    sync_event: SyncEvent,

    // unused space reserved for future fields
    _reserved: [i64; 3],
}

unsafe impl Send for ArrowDeviceArray {}

/// An FFI-compatible version of the ArrowArray that holds pointers to device buffers.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut sys::CUdeviceptr,
    children: *mut *mut ArrowArray,
    // NOTE: we don't support exporting dictionary arrays, so we leave this as an opaque pointer.
    dictionary: *mut (),
    release: Option<unsafe extern "C" fn(arg1: *mut ArrowArray)>,
    // When exported, this MUST contain everything that is owned by this array.
    // for example, any buffer pointed to in `buffers` must be here, as well
    // as the `buffers` pointer itself.
    // In other words, everything in ArrowArray must be owned by
    // `private_data` and can assume that they do not outlive `private_data`.
    private_data: *mut c_void,
}

impl ArrowArray {
    #[allow(unused)]
    pub fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: std::ptr::null_mut(),
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

unsafe impl Send for ArrowArray {}
unsafe impl Sync for ArrowArray {}

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
    pub(crate) buffer_ptrs: Box<[sys::CUdeviceptr]>,
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
        let buffer_ptrs: Box<[sys::CUdeviceptr]> = buffers
            .iter()
            .map(|buf| {
                match buf {
                    None => {
                        // null pointer
                        Ok(sys::CUdeviceptr::default())
                    }
                    Some(handle) => handle.cuda_device_ptr(),
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
        NonNull::new(&raw mut self.cuda_event_ptr)
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
