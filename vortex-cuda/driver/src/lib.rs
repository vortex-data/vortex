// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimal CUDA driver API bindings for Vortex.
//!
//! Covers the driver surface Vortex uses: the primary context, streams, events, modules,
//! kernel launches, and stream-ordered device memory. The driver library is loaded lazily, so
//! dependents build and run without CUDA until they first use it.
//!
//! Device memory orders its accesses across streams automatically; see [`CudaSlice`].
//!
//! Operations return errors directly. Finish access guards with [`SyncOnDrop::finish`] to
//! receive tracking errors; dropping a guard still performs cleanup but only logs failures.
//! Primary errors are returned without logging. Destructor failures and secondary errors that
//! cannot be returned are reported through `tracing`, not stored for a later caller. CUDA's
//! asynchronous device errors can still surface at a later driver call or synchronization.
//!
//! [`sys`] exposes the CUDA bindings. Resource-specific modules provide Rust wrappers;
//! the managed types own handles and enforce lifetimes and stream ordering.

mod context;
pub mod device;
mod error;
pub mod event;
mod launch;
pub mod mem_pool;
pub mod memory;
mod module;
mod repr;
pub mod stream;
pub mod sys;

pub use context::CudaContext;
pub use error::DriverError;
pub use event::CudaEvent;
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
pub use stream::CudaStream;

#[cfg(test)]
mod tests;
