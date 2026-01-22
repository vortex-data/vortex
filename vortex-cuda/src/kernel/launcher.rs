// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-safe CUDA kernel launcher that replaces the `launch_cuda_kernel!` macro.
//!
//! This module provides a builder-based API for launching CUDA kernels without
//! requiring a macro. The key challenge is that CUDA's `cuLaunchKernel` requires
//! a `void**` (array of pointers to arguments), so we need to store argument values
//! and provide stable pointers to them.
//!
//! # Example
//!
//! ```ignore
//! use vortex_cuda::kernel::KernelLauncher;
//! use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
//!
//! // Instead of:
//! // launch_cuda_kernel!(
//! //     execution_ctx: ctx,
//! //     module: "for",
//! //     ptypes: &[array.ptype()],
//! //     launch_args: [cuda_view, reference, array_len],
//! //     event_recording: CU_EVENT_DISABLE_TIMING,
//! //     array_len: array.len()
//! // );
//!
//! // Use:
//! let events = KernelLauncher::new(ctx, "for", &[array.ptype()])?
//!     .arg_view(&cuda_view)
//!     .arg(&reference)
//!     .arg(&array_len)
//!     .event_flags(CU_EVENT_DISABLE_TIMING)
//!     .launch(array.len())?;
//! ```

use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaStream;
use cudarc::driver::CudaView;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CudaKernelEvents;
use crate::executor::CudaExecutionCtx;

/// A builder for launching CUDA kernels with type-safe argument handling.
///
/// This struct collects kernel arguments and configuration, then launches the kernel.
/// Arguments are stored internally to ensure their memory remains valid until launch.
///
/// # Memory Layout
///
/// Arguments are stored as `u64` values (8 bytes each), which is sufficient for:
/// - All primitive scalar types (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64)
/// - Device pointers (`CUdeviceptr` is a `u64`)
///
/// The arguments are added to cudarc's `LaunchArgs` builder at launch time,
/// after all arguments have been collected and storage is stable.
pub struct KernelLauncher<'a> {
    stream: &'a Arc<CudaStream>,
    function: CudaFunction,
    /// Storage for argument values. Each value occupies one u64 slot.
    storage: Vec<u64>,
    /// Event recording flags (None means no event recording).
    event_flags: Option<CUevent_flags>,
}

impl<'a> KernelLauncher<'a> {
    /// Creates a new kernel launcher for the specified module and ptypes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The CUDA execution context
    /// * `module_name` - Name of the PTX module (e.g., "for")
    /// * `ptypes` - Primitive types that determine the kernel variant (e.g., `&[PType::U32]`)
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel function cannot be loaded.
    pub fn new(
        ctx: &'a CudaExecutionCtx,
        module_name: &str,
        ptypes: &[PType],
    ) -> VortexResult<Self> {
        let function = ctx.load_function(module_name, ptypes)?;
        let stream = ctx.stream();
        Ok(Self {
            stream,
            function,
            storage: Vec::new(),
            event_flags: None,
        })
    }

    /// Adds a scalar argument to the kernel launch.
    ///
    /// Supports any type that implements `DeviceRepr` and `Copy` and fits in 8 bytes:
    /// - Integers: u8, u16, u32, u64, i8, i16, i32, i64
    /// - Floats: f32, f64
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<T>() > 8`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// launcher
    ///     .arg(&42u32)
    ///     .arg(&3.14f64)
    ///     .arg(&array_len);
    /// ```
    pub fn arg<T: DeviceRepr + Copy>(mut self, value: &T) -> Self {
        assert!(
            size_of::<T>() <= 8,
            "Scalar argument must fit in 8 bytes, got {} bytes for {}",
            size_of::<T>(),
            std::any::type_name::<T>()
        );

        // Store the value as a u64 (zeroed first to ensure padding is deterministic)
        let mut storage_value: u64 = 0;
        // SAFETY: We've asserted that T fits in 8 bytes, and storage_value is properly aligned
        // for u64 which is at least as aligned as any primitive type <= 8 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                value as *const T as *const u8,
                (&raw mut storage_value).cast::<u8>(),
                size_of::<T>(),
            );
        }
        self.storage.push(storage_value);
        self
    }

    /// Adds a CUDA device buffer view as an argument.
    ///
    /// This extracts the device pointer from the view and stores it for the kernel.
    /// The view's underlying memory must remain valid until the kernel completes execution.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cuda_view = buffer_handle.cuda_view::<f32>()?;
    /// launcher.arg_view(&cuda_view);
    /// ```
    pub fn arg_cuda_view<T: DeviceRepr>(mut self, view: &CudaView<'_, T>) -> Self {
        // Get the device pointer value (CUdeviceptr is a u64)
        // The _sync guard is dropped immediately, but that's fine since we're just
        // reading the pointer value, not scheduling any work yet.
        let (device_ptr, _sync) = view.device_ptr(self.stream);
        self.storage.push(device_ptr);
        self
    }

    /// Sets the event recording flags for kernel launch timing.
    ///
    /// Events are recorded before and after the kernel launch for synchronization
    /// and optional timing measurements.
    ///
    /// # Arguments
    ///
    /// * `flags` - Event flags. Use `CU_EVENT_DISABLE_TIMING` for minimal overhead,
    ///   or `CU_EVENT_DEFAULT` to enable timestamps for profiling.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
    ///
    /// launcher.event_flags(CU_EVENT_DISABLE_TIMING);
    /// ```
    pub fn event_flags(mut self, flags: CUevent_flags) -> Self {
        self.event_flags = Some(flags);
        self
    }

    /// Launches the kernel with the configured arguments.
    ///
    /// # Arguments
    ///
    /// * `array_len` - The total number of elements to process
    ///
    /// # Launch Configuration
    ///
    /// The kernel is launched with:
    /// - `grid_dim`: `(ceil(array_len / 2048), 1, 1)` blocks
    /// - `block_dim`: `(64, 1, 1)` threads per block (2 warps)
    /// - Each thread processes 32 elements
    /// - Each block processes 2048 elements
    /// - The last block/thread may process fewer elements
    ///
    /// # Returns
    ///
    /// Returns `CudaKernelEvents` with before/after launch events for synchronization.
    ///
    /// # Safety
    ///
    /// The kernel launch is inherently unsafe because:
    /// - We cannot verify that arguments match the kernel signature
    /// - We cannot verify argument order or types at compile time
    /// - The kernel may access memory outside bounds
    /// - Device buffers passed as arguments may be mutated by the kernel
    ///
    /// The caller is responsible for ensuring arguments are correct.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Event flags were not set (required for this API)
    /// - Event recording fails
    /// - The kernel launch fails
    pub fn launch(self, array_len: usize) -> VortexResult<CudaKernelEvents> {
        let num_chunks = u32::try_from(array_len.div_ceil(2048))?;

        let config = LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get the event flags - required for this API to match macro behavior
        let event_flags = self
            .event_flags
            .ok_or_else(|| vortex_err!("Event flags must be set before launch"))?;

        // Build LaunchArgs using cudarc's builder.
        // Storage is now stable (no more additions), so references remain valid.
        let mut launch_args = self.stream.launch_builder(&self.function);

        // Add all stored arguments to the launch builder
        for storage_val in self.storage.iter() {
            launch_args.arg(storage_val);
        }

        // Enable event recording
        launch_args.record_kernel_launch(event_flags);

        // Launch the kernel
        // SAFETY: This is unsafe because we cannot verify argument types match the kernel.
        // The caller is responsible for ensuring arguments are correct.
        unsafe {
            launch_args
                .launch(config)
                .map_err(|e| vortex_err!("Failed to launch kernel: {}", e))
                .and_then(|events| {
                    events
                        .ok_or_else(|| vortex_err!("CUDA events not recorded"))
                        .map(|(before_launch, after_launch)| CudaKernelEvents {
                            before_launch,
                            after_launch,
                        })
                })
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_arg_storage_size() {
        // Verify that all supported scalar types fit in 8 bytes
        assert!(size_of::<u8>() <= 8);
        assert!(size_of::<u16>() <= 8);
        assert!(size_of::<u32>() <= 8);
        assert!(size_of::<u64>() <= 8);
        assert!(size_of::<i8>() <= 8);
        assert!(size_of::<i16>() <= 8);
        assert!(size_of::<i32>() <= 8);
        assert!(size_of::<i64>() <= 8);
        assert!(size_of::<f32>() <= 8);
        assert!(size_of::<f64>() <= 8);
    }
}
