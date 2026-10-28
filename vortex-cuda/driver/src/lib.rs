// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimal CUDA driver API bindings for Vortex.
//!
//! Covers the driver surface Vortex uses: the primary context, streams, events, modules,
//! kernel launches, and stream-ordered device memory. The driver library is loaded lazily, so
//! dependents build and run without CUDA until they first use it.
//!
//! Device memory orders its accesses across streams automatically, paying for driver calls
//! only when an access crosses streams; see [`CudaSlice`].

mod context;
mod launch;
mod memory;
mod module;
mod repr;
pub mod result;
mod stream;
pub mod sys;

pub use context::CudaContext;
pub use launch::LaunchArgs;
pub use launch::LaunchConfig;
pub use launch::PushKernelArg;
pub use memory::CudaSlice;
pub use memory::CudaView;
pub use memory::CudaViewMut;
pub use memory::DevicePtr;
pub use memory::DevicePtrMut;
pub use memory::DeviceSlice;
pub use memory::HostSlice;
pub use memory::SyncOnDrop;
pub use module::CudaFunction;
pub use module::CudaModule;
pub use repr::DeviceRepr;
pub use repr::ValidAsZeroBits;
pub use result::DriverError;
pub use stream::CudaEvent;
pub use stream::CudaStream;

#[cfg(test)]
mod tests;
