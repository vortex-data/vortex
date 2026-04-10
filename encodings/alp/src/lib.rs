// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate contains an implementation of the floating point compression algorithm from the
//! paper ["ALP: Adaptive Lossless floating-Point Compression"][paper] by Afroozeh et al.
//!
//! The compressor has two variants, classic ALP which is well-suited for data that does not use
//! the full precision, and "real doubles", values that do.
//!
//! Classic ALP will return small integers, and it is meant to be cascaded with other integer
//! compression techniques such as bit-packing and frame-of-reference encoding. Combined this allows
//! for significant compression on the order of what you can get for integer values.
//!
//! ALP-RD is generally terminal, and in the ideal case it can represent an f64 is just 49 bits,
//! though generally it is closer to 54 bits per value or ~12.5% compression.
//!
//! [paper]: https://ir.cwi.nl/pub/33334/33334.pdf

pub use alp::*;
pub use alp_rd::*;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::patched::USE_EXPERIMENTAL_PATCHES;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod alp;
mod alp_rd;

/// Initialize ALP encoding in the given session.
pub fn initialize(session: &VortexSession) {
    // If we're using the experimental Patched encoding, register a shim
    // for ALP with interior patches to decode as Patched array.
    if *USE_EXPERIMENTAL_PATCHES {
        session.arrays().register(ALPPatchedPlugin);
    } else {
        session.arrays().register(ALP);
    }
    session.arrays().register(ALPRD);

    // Register the ALP-specific NaN count aggregate kernel.
    session.aggregate_fns().register_aggregate_kernel(
        ALP::ID,
        Some(NanCount.id()),
        &compute::nan_count::ALPNanCountKernel,
    );
}
