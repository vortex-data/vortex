// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel configuration.
//!
//! This module provides types for configuring CUDA kernels that are used
//! within `CudaSupport` trait implementations.

use std::fmt::Debug;

use vortex_array::vtable::ArrayId;
use vortex_utils::aliases::dash_map::DashMap;

/// Represents configuration for a CUDA kernel.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// The name of the kernel function.
    pub name: String,
    /// PTX source code for the kernel (optional - may be loaded from file).
    pub ptx: Option<String>,
    /// Number of threads per block.
    pub block_size: u32,
    /// Number of blocks in the grid.
    pub grid_size: u32,
}

impl KernelConfig {
    /// Creates a new kernel configuration.
    pub fn new(name: String, ptx: Option<String>, block_size: u32, grid_size: u32) -> Self {
        Self {
            name,
            ptx,
            block_size,
            grid_size,
        }
    }
}

/// Registry for CUDA kernels, keyed by array ID.
#[derive(Debug, Default)]
pub struct KernelRegistry {
    /// Kernel configurations indexed by array_id.
    configs: DashMap<ArrayId, KernelConfig>,
}

impl KernelRegistry {
    /// Creates a new kernel registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a kernel for an array encoding.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to register the kernel for
    /// * `config` - The kernel configuration
    pub fn register(&self, array_id: ArrayId, config: KernelConfig) {
        self.configs.insert(array_id, config);
    }

    /// Retrieves a kernel configuration by array ID.
    ///
    /// Returns `None` if the kernel hasn't been registered.
    pub fn get_config(&self, array_id: ArrayId) -> Option<KernelConfig> {
        self.configs.get(&array_id).map(|entry| entry.clone())
    }

    /// Checks if a kernel is registered for an array encoding.
    pub fn has_kernel(&self, array_id: ArrayId) -> bool {
        self.configs.contains_key(&array_id)
    }

    /// Returns the number of registered kernels.
    pub fn kernel_count(&self) -> usize {
        self.configs.len()
    }

    /// Clears all registered kernels.
    pub fn clear(&self) {
        self.configs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_creation() {
        let config = KernelConfig::new("test_kernel".to_string(), None, 256, 1024);

        assert_eq!(config.name, "test_kernel");
        assert_eq!(config.block_size, 256);
        assert_eq!(config.grid_size, 1024);
    }
}
