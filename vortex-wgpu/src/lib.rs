use crate::session::WgpuSession;
use std::sync::LazyLock;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

pub mod executor;
mod pipeline;
pub mod session;
mod vector;
mod vectors;

pub(crate) static INSTANCE: LazyLock<wgpu::Instance> =
    LazyLock::new(|| wgpu::Instance::new(&wgpu::InstanceDescriptor::default()));

/// Initializes the WebGPU session within the given Vortex session.
pub fn initialize(session: &mut VortexSession) {
    let gpu = session.get_mut::<WgpuSession>();
}
