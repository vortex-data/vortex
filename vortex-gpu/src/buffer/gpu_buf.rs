// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream};
use vortex_dtype::{NativePType, PType};
use vortex_error::vortex_panic;

pub struct ErasedCudaSlice {
    ptr: CUdeviceptr,
    len: usize,
    ptype: PType,
}

impl ErasedCudaSlice {
    pub fn new<T: NativePType>(slice: impl Into<CudaSlice<T>>) -> Self {
        let slice = slice.into();
        let len = slice.len();
        Self {
            ptr: slice.leak(),
            len,
            ptype: T::PTYPE,
        }
    }

    pub fn ptype(&self) -> PType {
        self.ptype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice<T: NativePType>(&self, stream: &Arc<CudaStream>) -> CudaSlice<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get slice of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }

        unsafe { stream.upgrade_device_ptr::<T>(self.ptr, self.len) }
    }
}
