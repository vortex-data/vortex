// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::buffer::ErasedCudaSlice;

pub type GpuArrayRef = Arc<dyn GpuArray>;

pub trait GpuArray {
    fn child(&self, idx: usize) -> GpuArrayRef;

    fn buffer(&self, idx: usize) -> ErasedCudaSlice;
}
