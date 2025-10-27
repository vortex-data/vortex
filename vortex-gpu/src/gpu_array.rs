// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use vortex_dtype::NativePType;

use crate::buffer::ErasedCudaSlice;

pub enum GpuArray {
    Primitive(GpuPrimitiveArray),
    Bool(GpuBoolArray),
    Struct(GpuStructArray),
    Chunked(GpuChunkedArray),
}

pub struct GpuPrimitiveArray {
    values: ErasedCudaSlice,
}

impl GpuPrimitiveArray {
    fn as_slice<T: NativePType>(&self) -> CudaSlice<T> {
        self.values.as_slice()
    }
}

pub struct GpuBoolArray {
    values: CudaSlice<bool>,
}

impl GpuBoolArray {
    fn values(&self) -> CudaSlice<bool> {
        self.values.clone()
    }
}

pub struct GpuChunkedArray {
    gpu_arrays: Arc<[GpuArray]>,
}

impl GpuChunkedArray {
    fn child(&self, idx: usize) -> &GpuArray {
        &self.gpu_arrays[idx]
    }
}

pub struct GpuStructArray {
    gpu_arrays: Arc<[GpuArray]>,
}

impl GpuStructArray {
    fn child(&self, idx: usize) -> &GpuArray {
        &self.gpu_arrays[idx]
    }
}
