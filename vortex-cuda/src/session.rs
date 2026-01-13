// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::vtable::ArrayId;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_utils::aliases::dash_map::DashMap;

use crate::executor::CudaExecute;

/// CUDA session for GPU accelerated execution.
///
/// Maintains a registry of CUDA kernel implementations for array encodings.
#[derive(Debug, Default)]
pub struct CudaSession {
    kernels: DashMap<ArrayId, &'static dyn CudaExecute>,
}

impl CudaSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers CUDA support for an array encoding.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to register support for
    /// * `executor` - A static reference to the CUDA support implementation
    pub fn register_kernel(&self, array_id: ArrayId, executor: &'static dyn CudaExecute) {
        self.kernels.insert(array_id, executor);
    }

    /// Retrieves the CUDA support implementation for an encoding, if registered.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to look up
    pub fn kernel(&self, array_id: &ArrayId) -> Option<&'static dyn CudaExecute> {
        self.kernels.get(array_id).map(|entry| *entry.value())
    }
}

/// Extension trait for accessing the CUDA session from a Vortex session.
pub trait CudaSessionExt: SessionExt {
    /// Returns the CUDA session.
    fn cuda(&self) -> Ref<'_, CudaSession> {
        self.get::<CudaSession>()
    }
}
impl<S: SessionExt> CudaSessionExt for S {}
