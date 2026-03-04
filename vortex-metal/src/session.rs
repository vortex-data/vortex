// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandQueue;
use objc2_metal::MTLCreateSystemDefaultDevice;
use objc2_metal::MTLDevice;
use parking_lot::RwLock;
use vortex::array::VortexSessionExecute;
use vortex::array::vtable::ArrayId;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::Ref;
use vortex::session::SessionExt;

use crate::MetalExecute;
use crate::MetalExecutionCtx;
use crate::MetalLibraryLoader;
use crate::initialize_metal;

/// Metal session for GPU accelerated execution.
///
/// Maintains a registry of Metal kernel implementations for array encodings.
/// Holds the Metal device and command queue for all GPU operations.
#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct MetalSession {
    /// The Metal device
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for work submission
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Registry of kernel implementations
    kernels: Arc<RwLock<Vec<(ArrayId, &'static dyn MetalExecute)>>>,
    /// Library loader with caching
    library_loader: Arc<MetalLibraryLoader>,
}

impl Debug for MetalSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSession")
            .field("device", &self.device.name())
            .field("kernels_registered", &self.kernels.read().len())
            .finish()
    }
}

impl MetalSession {
    /// Creates a new Metal session with the system default device.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available.
    pub fn new() -> VortexResult<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| vortex_err!("No Metal device available"))?;

        let command_queue = device
            .newCommandQueue()
            .ok_or_else(|| vortex_err!("Failed to create Metal command queue"))?;

        let library_loader = MetalLibraryLoader::new(device.clone());

        Ok(Self {
            device,
            command_queue,
            kernels: Arc::new(RwLock::new(Vec::new())),
            library_loader: Arc::new(library_loader),
        })
    }

    /// Creates a new Metal execution context.
    pub fn create_execution_ctx(
        &self,
        vortex_session: &vortex::session::VortexSession,
    ) -> VortexResult<MetalExecutionCtx> {
        MetalExecutionCtx::new(self.clone(), vortex_session.create_execution_ctx())
    }

    /// Returns a reference to the Metal device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Returns a reference to the command queue.
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }

    /// Returns a reference to the library loader.
    pub fn library_loader(&self) -> &MetalLibraryLoader {
        &self.library_loader
    }

    /// Registers Metal support for an array encoding.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to register support for
    /// * `executor` - A static reference to the Metal support implementation
    pub fn register_kernel(&self, array_id: ArrayId, executor: &'static dyn MetalExecute) {
        let mut kernels = self.kernels.write();
        // Remove any existing registration for this array_id
        kernels.retain(|(id, _)| *id != array_id);
        kernels.push((array_id, executor));
    }

    /// Retrieves the Metal support implementation for an encoding, if registered.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to look up
    pub fn kernel(&self, array_id: &ArrayId) -> Option<&'static dyn MetalExecute> {
        let kernels = self.kernels.read();
        kernels
            .iter()
            .find(|(id, _)| id == array_id)
            .map(|(_, executor)| *executor)
    }
}

impl Default for MetalSession {
    /// Creates a default Metal session using the system default device,
    /// with all GPU array kernels preloaded.
    ///
    /// # Panics
    ///
    /// Panics if no Metal device is available.
    fn default() -> Self {
        #[expect(clippy::expect_used)]
        let session = Self::new().expect("Failed to initialize Metal session");
        initialize_metal(&session);
        session
    }
}

/// Extension trait for accessing the Metal session from a Vortex session.
pub trait MetalSessionExt: SessionExt {
    /// Returns the Metal session.
    fn metal_session(&self) -> Ref<'_, MetalSession> {
        self.get::<MetalSession>()
    }
}
impl<S: SessionExt> MetalSessionExt for S {}
