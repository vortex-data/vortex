// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::vtable::ArrayId;
use vortex_utils::aliases::dash_map::DashMap;

use crate::executor::CudaExecute;

/// CUDA session for GPU accelerated execution.
///
/// Maintains a registry of CUDA kernel implementations for array encodings.
#[derive(Debug, Default)]
pub struct CudaSession {
    executors: DashMap<ArrayId, &'static dyn CudaExecute>,
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
    pub fn register(&self, array_id: ArrayId, executor: &'static dyn CudaExecute) {
        self.executors.insert(array_id, executor);
    }

    /// Retrieves the CUDA support implementation for an encoding, if registered.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to look up
    pub fn get_executor(&self, array_id: &ArrayId) -> Option<&'static dyn CudaExecute> {
        self.executors.get(array_id).map(|entry| *entry.value())
    }

    /// Returns the number of registered executors.
    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }

    /// Checks whether CUDA support is registered for a specific encoding.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to check
    pub fn has_executor(&self, array_id: &ArrayId) -> bool {
        self.executors.contains_key(array_id)
    }
}
