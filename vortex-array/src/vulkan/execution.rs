// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::LazyCell;
use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use vortex_error::{
    vortex_bail, vortex_err, SharedVortexResult, VortexError, VortexExpect, VortexResult,
};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::{Version, VulkanLibrary};

pub struct VulkanExecutor {
    device: Arc<Device>,
    queues: Vec<Arc<Queue>>,
}

impl VulkanExecutor {
    pub fn builder() -> VortexResult<VulkanExecutorBuilder> {
        let instance = get_instance()?;
        Ok(VulkanExecutorBuilder {
            instance,
            ..Default::default()
        })
    }
}

pub struct VulkanExecutorBuilder {
    instance: Arc<Instance>,
    physical_device: Option<Arc<PhysicalDevice>>,
}

impl VulkanExecutorBuilder {
    pub fn list_devices(&self) -> VortexResult<impl Iterator<Item = Arc<PhysicalDevice>>> {
        self.instance
            .enumerate_physical_devices()
            .map_err(|e| vortex_err!("Failed to enumerate physical devices: {}", e))?
    }

    pub fn build(self) -> VortexResult<VulkanExecutor> {
        let physical_device = match self.physical_device {
            Some(device) => device,
            None => self
                .instance
                .enumerate_physical_devices()
                .map_err(|e| vortex_err!("Failed to enumerate physical devices: {}", e))?
                .next()
                .ok_or_else(|| vortex_err!("No physical devices available"))?,
        };

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE)
            })
            .ok_or_else(|| vortex_err!("No queue family found in physical device"))?;
        let queue_family_index =
            u32::try_from(queue_family_index).vortex_expect("queue family index out of range");

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| vortex_err!("Failed to create logical device: {}", e))?;
        let queues: Vec<_> = queues.collect();

        Ok(VulkanExecutor { device, queues })
    }
}

/// Returns the shared Vulkan instance.
fn get_instance() -> VortexResult<Arc<Instance>> {
    static INSTANCE: LazyLock<SharedVortexResult<Arc<Instance>>> = LazyLock::new(|| {
        let lib = VulkanLibrary::new()
            .map_err(|e| vortex_err!("Failed to load Vulkan library: {}", e))
            .map_err(Arc::from)?;

        const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");
        let version = VERSION
            .and_then(|v| Version::from_str(v).ok())
            .unwrap_or_default();

        Instance::new(
            lib,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                application_name: Some("vortex".into()),
                application_version: version,
                ..Default::default()
            },
        )
        .map_err(|e| vortex_err!("{:?}", e))
        .map_err(Arc::from)
    });

    INSTANCE.clone().map_err(VortexError::from)
}

#[cfg(test)]
mod test {
    use crate::vulkan::execution::VulkanExecutor;
    use vortex_error::VortexResult;

    #[test]
    fn test_vulkan() -> VortexResult<()> {
        let builder = VulkanExecutor::builder()?;
        let devices: Vec<_> = builder.list_devices()?.collect();
        println!("{:?}", devices);
        assert!(false);
        Ok(())
    }
}
