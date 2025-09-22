// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;
use std::sync::LazyLock;

use vortex_error::{VortexExpect, VortexResult, vortex_err};
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, DeviceDescriptor, Instance,
    InstanceDescriptor, PowerPreference, Queue, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
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
        WebGpuExecutorBuilder {}
    }

    pub fn do_something(&self) {
        // Define the shader that subtracts a value from each element in an i32 buffer
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<i32>;
            @group(0) @binding(1) var<uniform> subtract_value: i32;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index < arrayLength(&data)) {
                    data[index] = data[index] - subtract_value;
                }
            }
        "#;

        // Create shader module
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("subtract_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("subtract_bind_group_layout"),
                entries: &[
                    // Storage buffer for i32 array
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Uniform buffer for subtract value
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("subtract_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("subtract_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Example usage: Create buffers and execute the kernel
        let data = vec![10i32, 20, 30, 40, 50];
        let data_size = (data.len() * size_of::<i32>()) as usize;

        // Create data buffer with initial values
        let data_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("data_buffer"),
            size: data_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Write initial data
        let data_bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data_size) };
        self.queue.write_buffer(&data_buffer, 0, data_bytes);

        // Create uniform buffer for subtract value
        let subtract_value = 5i32;
        let uniform_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buffer"),
            size: size_of::<i32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write subtract value
        let subtract_bytes =
            unsafe { std::slice::from_raw_parts(&raw const subtract_value as *const u8, 4) };
        self.queue.write_buffer(&uniform_buffer, 0, subtract_bytes);

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("subtract_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch compute work
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("subtract_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("subtract_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with ceiling division to handle all elements
            let workgroups = u32::try_from(data.len())
                .vortex_expect("data too large")
                .div_ceil(64); // 64 is the workgroup size
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Submit the work
        self.queue.submit(std::iter::once(encoder.finish()));

        // For demonstration: Read back the results
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("staging_buffer"),
            size: data_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging_buffer, 0, data_size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Note: In a real implementation, we'd read back the results from the staging buffer
        // For now, just ensure the GPU work is submitted

        log::info!(
            "WebGPU kernel executed: subtracted {} from each element on {:?}",
            subtract_value,
            self.device
        );
        log::info!("Original data: {:?}", data);
        log::info!(
            "Expected result: {:?}",
            data.iter().map(|x| x - subtract_value).collect::<Vec<_>>()
        );
    }
}

pub struct WebGpuExecutorBuilder {}

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
    use futures::executor::block_on;
    use vortex_error::VortexResult;

    use crate::webgpu::execution::WebGpuExecutor;

    #[test]
    // #[ignore] // Ignore by default since it requires GPU hardware
    fn test_webgpu() -> VortexResult<()> {
        let executor = block_on(WebGpuExecutor::builder().build())?;
        executor.do_something();
        Ok(())
    }
}
