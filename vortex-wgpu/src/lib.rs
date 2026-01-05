use std::sync::LazyLock;
use vortex_session::VortexSession;

pub mod executor;
pub mod session;

pub(crate) static INSTANCE: LazyLock<wgpu::Instance> =
    LazyLock::new(|| wgpu::Instance::new(&wgpu::InstanceDescriptor::default()));

/// Initializes the WebGPU session within the given Vortex session.
pub fn initialize(_session: &mut VortexSession) {}
