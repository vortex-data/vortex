// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernel launches with stream-ordered device memory arguments.

use std::ffi::c_void;

use crate::memory::AccessKind;
use crate::memory::ArgAccess;
use crate::memory::CudaSlice;
use crate::memory::CudaView;
use crate::memory::CudaViewMut;
use crate::module::CudaFunction;
use crate::repr::DeviceRepr;
use crate::result::DriverError;
use crate::stream::CudaEvent;
use crate::stream::CudaStream;
use crate::sys;

/// Grid, block and dynamic shared memory of a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Blocks per grid dimension.
    pub grid_dim: (u32, u32, u32),
    /// Threads per block dimension.
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory per block, in bytes.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// One-dimensional blocks of 1024 threads covering `n` elements.
    pub fn for_num_elems(n: u32) -> Self {
        Self {
            grid_dim: (n.div_ceil(1024), 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// A kernel launch being assembled on a stream; create with [`CudaStream::launch_builder`].
///
/// Device memory arguments are ordered after earlier conflicting accesses and recorded as
/// accesses of this launch.
pub struct LaunchArgs<'a> {
    stream: &'a CudaStream,
    function: &'a CudaFunction,
    // `Vec`, unlike `SmallVec`, lets argument borrows end before the builder drops.
    args: Vec<*mut c_void>,
    accesses: Vec<ArgAccess<'a>>,
    timing: Option<sys::CUevent_flags>,
}

impl std::fmt::Debug for LaunchArgs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaunchArgs")
            .field("args", &self.args.len())
            .finish()
    }
}

impl CudaStream {
    /// Starts a launch of `function` on this stream.
    pub fn launch_builder<'a>(&'a self, function: &'a CudaFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream: self,
            function,
            args: Vec::with_capacity(8),
            accesses: Vec::with_capacity(4),
            timing: None,
        }
    }
}

/// A value that can be passed as a kernel argument.
///
/// # Safety
///
/// Implementations must push a pointer to storage that lives as long as the [`LaunchArgs`] and
/// holds the argument in the kernel's expected representation.
pub unsafe trait PushKernelArg<T> {
    /// Appends the argument.
    fn arg(&mut self, arg: T) -> &mut Self;
}

// SAFETY: The reference outlives the launch and `T` has a device representation.
unsafe impl<'a, 'b: 'a, T: DeviceRepr> PushKernelArg<&'b T> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b T) -> &mut Self {
        self.args.push((arg as *const T).cast_mut().cast());
        self
    }
}

// SAFETY: The device pointer lives in the slice, which outlives the launch.
unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b CudaSlice<T>> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b CudaSlice<T>) -> &mut Self {
        let (access, pointer) = arg.arg_access(AccessKind::Read);
        self.accesses.push(access);
        self.args
            .push((pointer as *const sys::CUdeviceptr).cast_mut().cast());
        self
    }
}

// SAFETY: The device pointer lives in the slice, which outlives the launch.
unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut CudaSlice<T>> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b mut CudaSlice<T>) -> &mut Self {
        let (access, pointer) = arg.arg_access(AccessKind::Write);
        self.accesses.push(access);
        self.args
            .push((pointer as *const sys::CUdeviceptr).cast_mut().cast());
        self
    }
}

// SAFETY: The device pointer lives in the view, which outlives the launch.
unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b CudaView<'c, T>> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b CudaView<'c, T>) -> &mut Self {
        let (access, pointer) = arg.arg_access();
        self.accesses.push(access);
        self.args
            .push((pointer as *const sys::CUdeviceptr).cast_mut().cast());
        self
    }
}

// SAFETY: The device pointer lives in the view, which outlives the launch.
unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut CudaViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b mut CudaViewMut<'c, T>) -> &mut Self {
        let (access, pointer) = arg.arg_access();
        self.accesses.push(access);
        self.args
            .push((pointer as *const sys::CUdeviceptr).cast_mut().cast());
        self
    }
}

impl LaunchArgs<'_> {
    /// Makes [`launch`](Self::launch) return events recorded before and after the kernel.
    pub fn record_kernel_launch(&mut self, flags: sys::CUevent_flags) -> &mut Self {
        self.timing = Some(flags);
        self
    }

    /// Enqueues the kernel after the arguments' earlier conflicting accesses.
    ///
    /// # Safety
    ///
    /// The arguments must match the kernel's parameters, and the kernel must access device
    /// memory only within the arguments, writing only those passed mutably.
    pub unsafe fn launch(
        &mut self,
        config: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        self.stream.bind()?;
        for access in &self.accesses {
            access
                .access
                .before(access.owner, self.stream, access.kind)?;
        }
        let start = self
            .timing
            .map(|flags| self.stream.record_event(Some(flags)))
            .transpose()?;
        // SAFETY: Under the caller's guarantees; the argument pointers outlive the call.
        unsafe {
            sys::cuLaunchKernel(
                self.function.cu_function(),
                config.grid_dim.0,
                config.grid_dim.1,
                config.grid_dim.2,
                config.block_dim.0,
                config.block_dim.1,
                config.block_dim.2,
                config.shared_mem_bytes,
                self.stream.cu_stream(),
                self.args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
        .result()?;
        // The kernel is already queued: no failure may skip tracking any of its accesses.
        let end = self
            .timing
            .map(|flags| self.stream.record_event(Some(flags)))
            .transpose();
        let mut tracked = Ok(());
        for access in &self.accesses {
            let recorded = access
                .access
                .after_or_wait(access.owner, self.stream, access.kind);
            if tracked.is_ok() {
                tracked = recorded;
            }
        }
        tracked?;
        Ok(start.zip(end?))
    }
}
