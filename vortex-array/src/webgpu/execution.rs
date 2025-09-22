// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::str::FromStr;
use std::sync::LazyLock;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use wgpu::hal::DynDevice;
use wgpu::{
    Backends, ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    PowerPreference, Queue, RequestAdapterOptions,
};

static INSTANCE: LazyLock<Instance> = LazyLock::new(|| {
    Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    })
});

pub struct WebGpuExecutor {
    device: Device,
    queue: Queue,
}

impl WebGpuExecutor {
    pub fn builder() -> WebGpuExecutorBuilder {
        WebGpuExecutorBuilder {
            instance,
            ..Default::default()
        }
    }

    pub fn do_something(&self) {
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &(),
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
    }
}

pub struct WebGpuExecutorBuilder {
    instance: Instance,
}

impl WebGpuExecutorBuilder {
    pub async fn build(self) -> VortexResult<WebGpuExecutor> {
        let adapter = INSTANCE
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None, // No surface needed for compute
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| vortex_err!("Failed to request adapter: {}", e))?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("vortex"),
                ..Default::default()
            })
            .await
            .map_err(|e| vortex_err!("Failed to request device: {}", e))?;

        Ok(WebGpuExecutor { device, queue })
    }
}

/// Returns the shared WebGpu instance.
fn get_instance() -> Instance {
    INSTANCE.clone()
}

#[cfg(test)]
mod test {
    use crate::webgpu::execution::WebGpuExecutor;
    use futures::executor::block_on;
    use vortex_error::VortexResult;

    #[test]
    fn test_webgpu() -> VortexResult<()> {
        let executor = block_on(WebGpuExecutor::builder().build())?;
        println!("{:?}", devices);
        assert!(false);
        Ok(())
    }
}
