// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Raw CUDA driver API: the types, constants and functions Vortex uses.
//!
//! The driver library is opened lazily on first use, so binaries build and start on hosts
//! without CUDA; [`is_culib_present`] tests for it without failing. Each function forwards to
//! the driver symbol of the same name and returns the raw [`CUresult`].

#![allow(
    non_camel_case_types,
    non_snake_case,
    missing_docs,
    clippy::too_many_arguments,
    reason = "mirrors the driver header's names and signatures"
)]

use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::sync::OnceLock;

use libloading::Library;

#[repr(C)]
pub struct CUctx_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUstream_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUevent_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUmod_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUfunc_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUmemPoolHandle_st {
    _private: [u8; 0],
}

pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut CUctx_st;
pub type CUstream = *mut CUstream_st;
pub type CUevent = *mut CUevent_st;
pub type CUmodule = *mut CUmod_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUmemoryPool = *mut CUmemPoolHandle_st;
pub type CUhostFn = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

/// A CUDA driver API status code.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUresult(pub c_uint);

impl CUresult {
    pub const CUDA_SUCCESS: Self = Self(0);
    pub const CUDA_ERROR_INVALID_VALUE: Self = Self(1);
    pub const CUDA_ERROR_OUT_OF_MEMORY: Self = Self(2);
    pub const CUDA_ERROR_NOT_INITIALIZED: Self = Self(3);
    pub const CUDA_ERROR_NO_DEVICE: Self = Self(100);
    pub const CUDA_ERROR_INVALID_CONTEXT: Self = Self(201);
    pub const CUDA_ERROR_INVALID_HANDLE: Self = Self(400);
    pub const CUDA_ERROR_NOT_READY: Self = Self(600);
}

/// The status type under the name of the driver header's enum.
pub type cudaError_enum = CUresult;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUevent_flags(pub c_uint);

impl CUevent_flags {
    pub const CU_EVENT_DEFAULT: Self = Self(0x0);
    pub const CU_EVENT_BLOCKING_SYNC: Self = Self(0x1);
    pub const CU_EVENT_DISABLE_TIMING: Self = Self(0x2);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUevent_wait_flags(pub c_uint);

impl CUevent_wait_flags {
    pub const CU_EVENT_WAIT_DEFAULT: Self = Self(0x0);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUdevice_attribute(pub c_int);

impl CUdevice_attribute {
    pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Self = Self(16);
    pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Self = Self(75);
    pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Self = Self(76);
    pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: Self = Self(100);
    pub const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: Self = Self(115);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUmemPool_attribute(pub c_int);

impl CUmemPool_attribute {
    pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: Self = Self(4);
}

/// The memory pool attribute type under the name of the driver header's enum.
pub type CUmemPool_attribute_enum = CUmemPool_attribute;

pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x1;
pub const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 0x2;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: c_uint = 0x4;

/// Shared-library names tried, in order, when loading the driver.
const DRIVER_LIBRARIES: &[&str] = &["libcuda.so.1", "libcuda.so", "nvcuda.dll"];

/// Opens the driver library, if any of its names resolve.
fn open_library() -> Option<Library> {
    DRIVER_LIBRARIES.iter().find_map(|name| {
        // SAFETY: Loading the CUDA driver runs only its own initializers, which have no
        // preconditions on the calling process.
        unsafe { Library::new(name) }.ok()
    })
}

/// Whether the CUDA driver library can be loaded on this host.
///
/// # Safety
///
/// Loads a shared library, running its initializers.
pub unsafe fn is_culib_present() -> bool {
    driver().is_ok()
}

macro_rules! driver_api {
    ($( fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ); )*) => {
        /// Resolved driver entry points, kept alive with their library.
        struct Driver {
            $( $name: unsafe extern "C" fn( $( $ty ),* ) -> CUresult, )*
            _library: Library,
        }

        impl Driver {
            fn load(library: Library) -> Result<Self, String> {
                // SAFETY: Each symbol is declared with the signature of the driver header.
                unsafe {
                    Ok(Self {
                        $( $name: *library
                            .get::<unsafe extern "C" fn( $( $ty ),* ) -> CUresult>(
                                concat!(stringify!($name), "\0").as_bytes(),
                            )
                            .map_err(|error| format!("{}: {error}", stringify!($name)))?, )*
                        _library: library,
                    })
                }
            }
        }

        $(
            /// Calls the driver function of the same name.
            ///
            /// # Safety
            ///
            /// The arguments must satisfy the driver API's requirements for this function.
            ///
            /// # Panics
            ///
            /// Panics if the CUDA driver library cannot be loaded.
            #[inline]
            pub unsafe fn $name( $( $arg: $ty ),* ) -> CUresult {
                let driver = match driver() {
                    Ok(driver) => driver,
                    Err(error) => panic!("CUDA driver unavailable: {error}"),
                };
                // SAFETY: Forwarded under the caller's guarantees.
                unsafe { (driver.$name)( $( $arg ),* ) }
            }
        )*
    };
}

driver_api! {
    fn cuInit(flags: c_uint);
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int);
    fn cuDeviceGetCount(count: *mut c_int);
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: CUdevice_attribute, device: CUdevice);
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: CUdevice);
    fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, device: CUdevice);
    fn cuDevicePrimaryCtxRetain(context: *mut CUcontext, device: CUdevice);
    fn cuDevicePrimaryCtxRelease_v2(device: CUdevice);
    fn cuCtxCreate_v2(context: *mut CUcontext, flags: c_uint, device: CUdevice);
    fn cuCtxDestroy_v2(context: CUcontext);
    fn cuCtxGetCurrent(context: *mut CUcontext);
    fn cuCtxSetCurrent(context: CUcontext);
    fn cuCtxSynchronize();
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize);
    fn cuStreamCreate(stream: *mut CUstream, flags: c_uint);
    fn cuStreamDestroy_v2(stream: CUstream);
    fn cuStreamSynchronize(stream: CUstream);
    fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint);
    fn cuLaunchHostFunc(stream: CUstream, function: CUhostFn, user_data: *mut c_void);
    fn cuEventCreate(event: *mut CUevent, flags: c_uint);
    fn cuEventDestroy_v2(event: CUevent);
    fn cuEventRecord(event: CUevent, stream: CUstream);
    fn cuEventSynchronize(event: CUevent);
    fn cuEventQuery(event: CUevent);
    fn cuEventElapsedTime(milliseconds: *mut f32, start: CUevent, end: CUevent);
    fn cuMemAllocAsync(pointer: *mut CUdeviceptr, bytes: usize, stream: CUstream);
    fn cuMemFreeAsync(pointer: CUdeviceptr, stream: CUstream);
    fn cuMemAlloc_v2(pointer: *mut CUdeviceptr, bytes: usize);
    fn cuMemFree_v2(pointer: CUdeviceptr);
    fn cuMemsetD8Async(target: CUdeviceptr, value: u8, count: usize, stream: CUstream);
    fn cuMemsetD32Async(target: CUdeviceptr, value: c_uint, count: usize, stream: CUstream);
    fn cuMemcpyHtoDAsync_v2(
        target: CUdeviceptr,
        source: *const c_void,
        bytes: usize,
        stream: CUstream,
    );
    fn cuMemcpyDtoHAsync_v2(
        target: *mut c_void,
        source: CUdeviceptr,
        bytes: usize,
        stream: CUstream,
    );
    fn cuMemcpyDtoDAsync_v2(
        target: CUdeviceptr,
        source: CUdeviceptr,
        bytes: usize,
        stream: CUstream,
    );
    fn cuMemcpyDtoH_v2(target: *mut c_void, source: CUdeviceptr, bytes: usize);
    fn cuMemHostAlloc(pointer: *mut *mut c_void, bytes: usize, flags: c_uint);
    fn cuMemFreeHost(pointer: *mut c_void);
    fn cuMemPoolSetAttribute(
        pool: CUmemoryPool,
        attribute: CUmemPool_attribute,
        value: *mut c_void,
    );
    fn cuMemPoolGetAttribute(
        pool: CUmemoryPool,
        attribute: CUmemPool_attribute,
        value: *mut c_void,
    );
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void);
    fn cuModuleUnload(module: CUmodule);
    fn cuModuleGetFunction(function: *mut CUfunction, module: CUmodule, name: *const c_char);
    fn cuLaunchKernel(
        function: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    );
    fn cuGetErrorName(error: CUresult, name: *mut *const c_char);
    fn cuGetErrorString(error: CUresult, description: *mut *const c_char);
}

/// The process-wide driver entry points, loaded on first use.
fn driver() -> Result<&'static Driver, &'static str> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            let library = open_library()
                .ok_or_else(|| format!("none of {DRIVER_LIBRARIES:?} could be loaded"))?;
            Driver::load(library)
        })
        .as_ref()
        .map_err(String::as_str)
}
