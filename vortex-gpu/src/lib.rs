use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::hal::Hal;
use crate::session::GpuSession;

pub mod cuda;
pub mod hal;
pub mod session;
pub mod vector;
pub mod wgpu;

/// Initializes the GPU session within the given Vortex session for the specified HAL.
///
/// This function can be called multiple times for different HALs.
pub fn initialize<H: Hal>(session: &mut VortexSession) {
    let gpu = session.get_mut::<GpuSession>();
}
