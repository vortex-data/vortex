// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Elias-Fano works in sign-extended 64-bit patterns and narrows back to the array's own width on
// the way out; see `EliasFanoData::reference_bits`. Both halves of that pair are exact.
#![expect(clippy::cast_possible_truncation)]

//! Elias-Fano encoding for monotonically non-decreasing integer sequences.
//!
//! Stores about `log2(u / n) + 2` bits per value for `n` values over a universe of `u`, while still
//! answering random access in constant time. Against bit-packing at `ceil(log2(u))` bits the saving
//! is `log2(n)`, so it widens with row count.
//!
//! Inputs must be non-decreasing and non-nullable; duplicates are fine. See [`elias_fano_encode`]
//! for the compression entry point and [`initialize`] to register the encoding in a session.
//!
//! [`ef`] holds the codec; everything else here binds it to Vortex's array model.
//!
//! The sampled select index follows Vigna's [broadword][] construction, with the two-table sampling
//! scheme and its parameters after [`rise-rs`][] (MIT); [`vers`][] was consulted as a further
//! reference. No code is taken from either.
//!
//! [broadword]: https://vigna.di.unimi.it/ftp/papers/Broadword.pdf
//! [`rise-rs`]: https://github.com/AngeloSav/rise-rs
//! [`vers`]: https://github.com/Cydhra/vers

mod access;
mod array;
mod compress;
mod compute;
pub mod ef;
mod lower;
mod rules;

pub use array::EliasFano;
pub use array::EliasFanoArray;
pub use array::EliasFanoArraySlotsExt;
pub use array::EliasFanoData;
pub use array::EliasFanoMetadata;
pub use array::EliasFanoSlots;
pub use compress::elias_fano_encode;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_error::VortexError;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// The codec's own errors, carried into Vortex's.
///
/// Functions rather than `From` impls: once [`ef`] is its own crate both types are foreign here and
/// the orphan rule refuses the impl.
pub(crate) fn unrepresentable(error: ef::Error) -> VortexError {
    vortex_err!("Elias-Fano {error}")
}

pub(crate) fn malformed(error: ef::Malformed) -> VortexError {
    vortex_err!("Elias-Fano {error}")
}

/// Initialize the Elias-Fano encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(EliasFano);

    // Answered from the layout rather than the data: the encoding only decodes correctly for a
    // non-decreasing sequence, so sortedness is a precondition rather than a measurement.
    session.aggregate_fns().register_aggregate_kernel(
        EliasFano.id(),
        Some(IsSorted.id()),
        &compute::is_sorted::EliasFanoIsSortedKernel,
    );
}

#[cfg(test)]
mod tests;
