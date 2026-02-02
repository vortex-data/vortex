// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module implements the Arrow C Data Device Interface extension for sharing GPU-resident
//! data.
//!
//! This is an extension to the Arrow C Data Interface.
//!
//! More documentation at <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>

mod canonical;

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaStream;
use cudarc::driver::sys;
use cudarc::runtime::sys::cudaEvent_t;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;

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

pub type SyncEvent = Option<NonNull<cudaEvent_t>>;

/// The C Data Device Interface representation of an Arrow array.
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
    dictionary: *mut ArrowArray,
    release: Option<unsafe extern "C" fn(arg1: *mut ArrowArray)>,
    // When exported, this MUST contain everything that is owned by this array.
    // for example, any buffer pointed to in `buffers` must be here, as well
    // as the `buffers` pointer itself.
    // In other words, everything in [FFI_ArrowArray] must be owned by
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

#[expect(unused, reason = "cuda_stream and cuda_buffers need to have deferred drop")]
pub(crate) struct CudaPrivateData {
    /// Hold a reference to the CudaStream so that it stays alive even after CudaExecutionCtx
    /// has been dropped.
    pub(crate) cuda_stream: Arc<CudaStream>,
    /// The single boxed slice which owns all buffers that the Rust code allocated on the device.
    pub(crate) buffers: Box<[Option<BufferHandle>]>,
    /// Boxed slice of buffer pointers. We return a pointer to the start of this allocation over
    /// the interface, so we hold it here so the Box contents are not freed.
    pub(crate) buffer_ptrs: Box<[sys::CUdeviceptr]>,
}

/// Trait implemented for types that can be exported to [`ArrowDeviceArray`].
#[async_trait]
pub trait CudaDeviceArrayExecute {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray>;
}
