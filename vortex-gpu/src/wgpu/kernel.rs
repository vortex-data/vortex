// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub type WgpuKernelRef = Box<dyn WgpuKernel>;

/// A trait representing a kernel for executing computations over a WebGPU backend.
pub trait WgpuKernel {}
