// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use compress::*;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::FixedWidthUncompressedSizeInBytesKernel;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod compress;
mod compute;
mod kernel;
mod rules;
mod slice;

/// Initialize ZigZag encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(ZigZag);
    session.aggregate_fns().register_aggregate_kernel(
        ZigZag.id(),
        Some(UncompressedSizeInBytes.id()),
        &FixedWidthUncompressedSizeInBytesKernel,
    );
}
