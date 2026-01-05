// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::INSTANCE;
use crate::session::WgpuSession;
use async_trait::async_trait;
use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_array::arrays::{PrimitiveArray, PrimitiveVTable};
use vortex_array::buffer::{BufferHandle, DeviceBuffer};
use vortex_array::{Array, ArrayRef};
use vortex_array::{Canonical, IntoArray};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_error::{vortex_bail, vortex_err};
use vortex_fastlanes::FoRVTable;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::{CommandEncoderDescriptor, DeviceDescriptor, PollType};
use wgpu::{BindGroupDescriptor, RequestAdapterOptions};
use wgpu::{BindGroupEntry, MapMode};
use wgpu::{BufferUsages, ComputePipelineDescriptor};
use wgpu::{ShaderModuleDescriptor, ShaderSource};

#[derive(Debug)]
pub struct WgpuBuffer {
    buffer: wgpu::Buffer,
    queue: wgpu::Queue,
    device: wgpu::Device,
}

impl DeviceBuffer for WgpuBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        usize::try_from(self.buffer.size()).vortex_expect("wgpu buffer size fits in usize")
    }

    fn to_host(self: Arc<Self>) -> VortexResult<ByteBuffer> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: self.buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("To Host Command Encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.buffer.size());
        let submission_index = self.queue.submit(Some(encoder.finish()));

        let (send, recv) = oneshot::channel();
        staging_buffer.map_async(MapMode::Read, .., move |result| drop(send.send(result)));

        self.device
            .poll(PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map_err(|e| vortex_err!("Device poll failed: {}", e))?;

        recv.recv()
            .map_err(|e| vortex_err!("Buffer mapping channel error: {}", e))?
            .map_err(|e| vortex_err!("Buffer mapping failed: {}", e))?;

        // FIXME(ngates): we have no idea on alignment requirements here.
        Ok(ByteBuffer::copy_from(
            staging_buffer.get_mapped_range(..).as_ref(),
        ))
    }
}

impl PartialEq for WgpuBuffer {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Hash for WgpuBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // FIXME(ngates): get HAL address?
        state.write_usize(self.len());
    }
}

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

pub struct WgpuCtx {
    pub session: Arc<WgpuSession>,
    pub device: wgpu::Device,
    pub cache: Option<wgpu::PipelineCache>,
    queue: wgpu::Queue,
}

/// Shader support for WebGPU execution.
#[async_trait]
pub trait WgpuSupport: 'static + Send + Sync + Debug {
    async fn execute(&self, array: &ArrayRef, ctx: &WgpuCtx) -> VortexResult<ArrayRef>;
}

#[async_trait]
pub trait WgpuArrayExt: Array {
    async fn execute_wgpu(&self, ctx: &WgpuCtx) -> VortexResult<ArrayRef>;
}

#[async_trait]
impl WgpuArrayExt for ArrayRef {
    async fn execute_wgpu(&self, ctx: &WgpuCtx) -> VortexResult<ArrayRef> {
        if self.is_canonical() {
            return Ok(self.clone());
        }

        let Some(support) = ctx.session.get_executor(&self.encoding_id()) else {
            todo!(
                "No WebGPU executor registered for array encoding {}",
                self.encoding_id()
            );
        };
        support.execute(self, ctx).await
    }
}

#[derive(Debug)]
struct FoRWgpuSupport;

#[async_trait]
impl WgpuSupport for FoRWgpuSupport {
    async fn execute(&self, array: &ArrayRef, ctx: &WgpuCtx) -> VortexResult<ArrayRef> {
        // First, we execute the child array to get the input values.
        let for_array = array.as_::<FoRVTable>();
        let input = for_array.encoded().execute_wgpu(ctx).await?;
        let input_primitive = input.as_::<PrimitiveVTable>();

        let source = "
            @group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;
            @group(0) @binding(1) var<uniform> ref_val: u32;
            @group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;

            @compute @workgroup_size(256)
            fn decode_for(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let r = vec4<u32>(ref_val);
                output[global_id.x] = input[global_id.x] + r;
            }
            ";

        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(source)),
        });

        let compute_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: ctx.cache.as_ref(),
            });

        let encoded_buffer = match input_primitive.buffer_handle() {
            BufferHandle::Host(buffer) => ctx.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("FoR Input Buffer"),
                contents: buffer.as_slice(),
                usage: BufferUsages::STORAGE,
            }),
            BufferHandle::Device(device) => match device.as_any().downcast_ref::<wgpu::Buffer>() {
                None => {
                    // We could go via host memory
                    vortex_bail!("FoR input buffer is not a wgpu::Buffer");
                }
                Some(buffer) => buffer.clone(),
            },
        };

        let ref_val = for_array
            .reference_scalar()
            .as_primitive()
            .typed_value::<u32>()
            .vortex_expect("FoR reference is u32");
        let ref_val_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("FoR Reference Buffer"),
            contents: &ref_val.to_le_bytes(),
            usage: BufferUsages::UNIFORM,
        });

        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FoR Output Buffer"),
            size: encoded_buffer.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("FoR Bind Group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: encoded_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: ref_val_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("FoR Command Encoder"),
            });

        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FoR Compute Pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute for iterations");

            let workgroup_count = u32::try_from(for_array.len().div_ceil(256))
                .map_err(|_| vortex_err!("FoR array workgroup count exceeds u32"))?;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue.submit(Some(enc.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| vortex_err!("Device poll failed: {}", e))?;

        let output_array = unsafe {
            PrimitiveArray::new_unchecked_raw(
                for_array.dtype().clone(),
                BufferHandle::Device(Arc::new(WgpuBuffer {
                    buffer: output_buffer,
                    queue: ctx.queue.clone(),
                    device: ctx.device.clone(),
                })),
                input_primitive.validity()?,
            )
        };

        Ok(output_array.into_array())
    }
}

impl WgpuExecutor {
    /// Execute the array performing minimal work to convert it to its canonical form.
    pub async fn execute_canonical(&self, array: ArrayRef) -> VortexResult<Canonical> {
        // For a given array, we first check if any of its children can "execute_parent", if so
        // we allow them to perform the execution.
        // TODO(ngates): check children for execute_parent optimizations
        let ctx = WgpuCtx {
            session: self.session.clone(),
            device: self.device.clone(),
            cache: None,
            queue: self.queue.clone(),
        };
        Ok(array.execute_wgpu(&ctx).await?.to_canonical())
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

    use crate::executor::{FoRWgpuSupport, WgpuExecutor};
    use crate::session::WgpuSession;

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
