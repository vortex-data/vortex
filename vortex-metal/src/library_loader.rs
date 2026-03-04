// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metal library loading and caching.

use std::fmt::Debug;
use std::path::Path;
use std::path::PathBuf;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::MTLCompileOptions;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLDevice;
use objc2_metal::MTLLibrary;
use parking_lot::RwLock;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

/// Loader for Metal shader libraries with caching.
///
/// Handles compiling Metal shader source, caching compiled libraries,
/// and creating compute pipeline states.
#[allow(clippy::type_complexity)]
pub struct MetalLibraryLoader {
    /// The Metal device
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Cache of compiled Metal libraries, keyed by module name
    libraries: RwLock<Vec<(String, Retained<ProtocolObject<dyn MTLLibrary>>)>>,
    /// Cache of pipeline states, keyed by function name
    pipelines: RwLock<
        Vec<(
            String,
            Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        )>,
    >,
}

impl Debug for MetalLibraryLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalLibraryLoader")
            .field("libraries_cached", &self.libraries.read().len())
            .field("pipelines_cached", &self.pipelines.read().len())
            .finish()
    }
}

impl MetalLibraryLoader {
    /// Creates a new library loader.
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            libraries: RwLock::new(Vec::new()),
            pipelines: RwLock::new(Vec::new()),
        }
    }

    /// Loads a Metal library from source code.
    ///
    /// The library is cached for future use.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (used as cache key)
    /// * `source` - Metal shader source code
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation fails.
    pub fn load_library_from_source(
        &self,
        module_name: &str,
        source: &str,
    ) -> VortexResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
        // Check cache first
        {
            let libraries = self.libraries.read();
            if let Some((_, lib)) = libraries.iter().find(|(name, _)| name == module_name) {
                return Ok(lib.clone());
            }
        }

        // Compile the library
        let source_ns = NSString::from_str(source);
        let options = MTLCompileOptions::new();

        let library = self
            .device
            .newLibraryWithSource_options_error(&source_ns, Some(&options))
            .map_err(|e| vortex_err!("Failed to compile Metal shader '{}': {}", module_name, e))?;

        // Cache the library
        {
            let mut libraries = self.libraries.write();
            libraries.push((module_name.to_string(), library.clone()));
        }

        Ok(library)
    }

    /// Loads a Metal library from a file.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (used as cache key)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or compilation fails.
    pub fn load_library_from_file(
        &self,
        module_name: &str,
    ) -> VortexResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
        // Check cache first
        {
            let libraries = self.libraries.read();
            if let Some((_, lib)) = libraries.iter().find(|(name, _)| name == module_name) {
                return Ok(lib.clone());
            }
        }

        let shader_path = Self::shader_path_for_module(module_name);
        let source = std::fs::read_to_string(&shader_path).map_err(|e| {
            vortex_err!(
                "Failed to read Metal shader '{}' at {}: {}",
                module_name,
                shader_path.display(),
                e
            )
        })?;

        self.load_library_from_source(module_name, &source)
    }

    /// Creates a compute pipeline state for a function in a library.
    ///
    /// The pipeline state is cached for future use.
    ///
    /// # Arguments
    ///
    /// * `library` - The Metal library containing the function
    /// * `function_name` - Name of the kernel function
    ///
    /// # Errors
    ///
    /// Returns an error if the function is not found or pipeline creation fails.
    pub fn create_pipeline(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> VortexResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // Check cache first
        {
            let pipelines = self.pipelines.read();
            if let Some((_, pipeline)) = pipelines.iter().find(|(name, _)| name == function_name) {
                return Ok(pipeline.clone());
            }
        }

        // Get the function from the library
        let function_ns = NSString::from_str(function_name);
        let function = library.newFunctionWithName(&function_ns).ok_or_else(|| {
            vortex_err!("Function '{}' not found in Metal library", function_name)
        })?;

        // Create the pipeline state
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                vortex_err!(
                    "Failed to create compute pipeline for '{}': {}",
                    function_name,
                    e
                )
            })?;

        // Cache the pipeline
        {
            let mut pipelines = self.pipelines.write();
            pipelines.push((function_name.to_string(), pipeline.clone()));
        }

        Ok(pipeline)
    }

    /// Loads a function and creates a pipeline state in one step.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (shader file without extension)
    /// * `function_name` - Name of the kernel function
    ///
    /// # Errors
    ///
    /// Returns an error if loading or pipeline creation fails.
    pub fn load_pipeline(
        &self,
        module_name: &str,
        function_name: &str,
    ) -> VortexResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let library = self.load_library_from_file(module_name)?;
        self.create_pipeline(&library, function_name)
    }

    /// Returns the shader file path for a given module name.
    ///
    /// Checks for `VORTEX_METAL_SHADERS_DIR` environment variable at runtime first,
    /// falling back to a default path relative to the crate.
    fn shader_path_for_module(module_name: &str) -> PathBuf {
        let shaders_dir = std::env::var("VORTEX_METAL_SHADERS_DIR").unwrap_or_else(|_| {
            // Default to the shaders directory relative to the crate
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            Path::new(manifest_dir)
                .join("shaders")
                .to_string_lossy()
                .to_string()
        });
        Path::new(&shaders_dir).join(format!("{}.metal", module_name))
    }

    /// Returns a reference to the Metal device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}
