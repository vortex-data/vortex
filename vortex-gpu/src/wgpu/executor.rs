// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use crate::vector::GpuVector;
use crate::vector::PrimitiveGpuVector;
use crate::wgpu::INSTANCE;
use crate::wgpu::session::WgpuSession;
use vortex_array::Canonical;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::PType;
use vortex_error::vortex_err;
use vortex_error::{VortexExpect, VortexResult};
use vortex_fastlanes::FoRVTable;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::DeviceDescriptor;
use wgpu::{BindGroup, BindGroupEntry, BufferBindingType};
use wgpu::{BindGroupDescriptor, RequestAdapterOptions};
use wgpu::{BindGroupLayoutDescriptor, ShaderStages};
use wgpu::{BindGroupLayoutEntry, ShaderModuleDescriptor, ShaderSource};
use wgpu::{BindingType, ComputePipelineDescriptor};

pub struct WgpuExecutor {
    session: Arc<WgpuSession>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuExecutor {
    pub async fn try_new(session: Arc<WgpuSession>) -> VortexResult<Self> {
        let adapter = INSTANCE
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .map_err(|e| vortex_err!("Failed to request adapter: {}", e))?;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .map_err(|e| vortex_err!("Failed to request device: {}", e))?;
        Ok(Self {
            session,
            device,
            queue,
        })
    }
}

/// Shader support for WebGPU execution.
pub trait WgpuSupport {
    fn shader(&self, array: &ArrayRef, args: WgpuShaderArgs) -> VortexResult<WgpuShader>;
}

pub struct WgpuShaderArgs<'a> {
    /// The device to compile the shader for.
    pub device: &'a wgpu::Device,
    /// The input vectors to the shader.
    pub inputs: &'a [WgpuShaderInput],
    /// The bind group index for additional self inputs.
    pub self_bind_group: u32,
    /// The output bind group index.
    pub output_bind_group: u32,
}

pub struct WgpuShaderInput {
    pub bind_group: u32,
    pub vector: GpuVector<BindGroupLayoutEntry>,
}

pub struct WgpuShader {
    pub source: String,
    pub output: GpuVector<BindGroupLayoutEntry>,
    pub self_input: Option<BindGroup>,
}

struct FoRWgpuSupport;
impl WgpuSupport for FoRWgpuSupport {
    fn shader(&self, array: &ArrayRef, args: WgpuShaderArgs) -> VortexResult<WgpuShader> {
        let for_array = array.as_::<FoRVTable>();
        let input = &args.inputs[0];

        let source = format!(
            "
            @group({}) @binding(0) var<storage, read> input: array<vec4<u32>>;
            @group({}) @binding(0) var<uniform> ref_val: u32;
            @group({}) @binding(0) var<storage, read_write> output: array<vec4<u32>>;

            @compute @workgroup_size(256)
            fn decode_for(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let r = vec4<u32>(ref_val);
                output[global_id.x] = input[global_id.x] + r;
            }}
            ",
            input.bind_group, args.self_bind_group, args.output_bind_group,
        );

        let output = GpuVector::Primitive(unsafe {
            PrimitiveGpuVector::new_unchecked(
                for_array.ptype(),
                for_array.len(),
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
            )
        });

        let reference_bytes = for_array
            .reference_scalar()
            .as_primitive()
            .typed_value::<u32>()
            .vortex_expect("FoR reference is u32")
            .to_le_bytes();
        let reference_buffer = args.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("FoR Reference Buffer"),
            contents: &reference_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let self_input_layout = args
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("FoR Self Bind Group Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let self_input = args.device.create_bind_group(&BindGroupDescriptor {
            label: Some("FoR Self Bind Group"),
            layout: &self_input_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: reference_buffer.as_entire_binding(),
            }],
        });

        Ok(WgpuShader {
            source,
            output,
            self_input: Some(self_input),
        })
    }
}

impl WgpuExecutor {
    /// Execute the array performing minimal work to convert it to its canonical form.
    pub async fn execute_canonical(&self, array: ArrayRef) -> VortexResult<Canonical> {
        // For a given array, we first check if any of its children can "execute_parent", if so
        // we allow them to perform the execution.
        // TODO(ngates): check children for execute_parent optimizations

        // Otherwise, we need to execute this array ourselves.

        // In order of preference, an array should:
        // 1. Produce a WGSL shader (this has the best chance of fused compilation as part of a
        //    larger pipeline).
        // 2. Produce a WebGPU pipeline (data is passed between pipelines via GPU memory, so
        //    cannot benefit from passing data via registers).
        // 3. Fall back to CPU execution (data is copied back to CPU memory).

        let Some(support) = self.session.get_executor(&array.encoding_id()) else {
            todo!(
                "No WebGPU executor registered for array encoding {}",
                array.encoding_id()
            );
        };

        // The input vectors are the array's children.
        // TODO(ngates): implement this recursion.
        let input_layout_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let input = WgpuShaderInput {
            bind_group: 0,
            vector: GpuVector::Primitive(unsafe {
                PrimitiveGpuVector::new_unchecked(PType::U32, array.len(), input_layout_entry)
            }),
        };
        let input_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Input Bind Group Layout"),
                entries: &[input_layout_entry],
            });

        let args = WgpuShaderArgs {
            device: &self.device,
            inputs: &[input],
            self_bind_group: 1,
            output_bind_group: 2,
        };
        let shader = support.shader(&array, args)?;
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(&shader.source)),
        });

        let output_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Output Bind Group Layout"),
                entries: &[shader.output.bind_group_layout_entry(0)?],
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &input_layout,
                    shader.self_input.vortex_expect("missing"),
                    output_layout,
                ],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        todo!()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::vtable::VTable;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;
    use vortex_error::VortexResult;
    use vortex_fastlanes::{FoRArray, FoRVTable};
    use vortex_io::runtime::single::block_on;

    use crate::wgpu::executor::{FoRWgpuSupport, WgpuExecutor};
    use crate::wgpu::session::WgpuSession;

    #[test]
    fn test_for_wgpu() -> VortexResult<()> {
        let mut session = WgpuSession::default();
        session.register_executor(FoRVTable.id(), &FoRWgpuSupport);
        let session = Arc::new(session);

        let array = FoRArray::try_new(
            buffer![0_u32, 1, 2, 3, 4, 5, 6, 7].into_array(),
            10_u32.into(),
        )?
        .into_array();

        let canonical = block_on(|_| async move {
            let executor = WgpuExecutor::try_new(session).await?;
            let canonical = executor.execute_canonical(array.clone()).await?;
            Ok::<_, VortexError>(canonical.into_primitive())
        })?;

        assert_eq!(canonical.len(), 8);
        assert_eq!(
            canonical.buffer::<u32>().as_slice(),
            &[10, 11, 12, 13, 14, 15, 16, 17]
        );

        Ok(())
    }
}
