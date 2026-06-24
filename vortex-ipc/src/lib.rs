// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex IPC messages and associated readers and writers.
//!
//! Vortex provides an IPC messaging format to exchange array data over a streaming
//! interface. The format emits message headers in FlatBuffer format, along with their
//! data buffers.
//!
//! This crate provides both in-memory message representations for holding IPC messages
//! before/after serialization, and streaming readers and writers that sit on top
//! of any type implementing `VortexRead` or `VortexWrite` respectively.

pub mod iterator;
pub mod messages;
pub mod stream;

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::aggregate_fn::session::AggregateFnSession;
    use vortex_array::dtype::session::DTypeSession;
    use vortex_array::optimizer::kernels::KernelSession;
    use vortex_array::session::ArraySession;
    use vortex_session::VortexSession;

    pub(crate) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::builder()
            .with::<DTypeSession>()
            .with::<ArraySession>()
            .with::<KernelSession>()
            .with::<AggregateFnSession>()
            .build()
    });
}
