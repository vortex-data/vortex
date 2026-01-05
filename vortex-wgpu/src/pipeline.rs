// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use vortex_error::VortexResult;
use wgpu::{ComputePipelineDescriptor, Device, ShaderModuleDescriptor, ShaderSource};

pub struct Pipeline {
    wgpu: wgpu::ComputePipeline,
}

pub struct PipelineBuilder<'a> {
    device: &'a Device,
    cache: Option<&'a wgpu::PipelineCache>,
    next_bind_group_index: u32,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self {
            device,
            next_bind_group_index: 0,
        }
    }

    pub fn build(self) -> VortexResult<Pipeline> {
        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed("")),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: self.cache,
            });

        Ok(Pipeline { wgpu: pipeline })
    }
}
