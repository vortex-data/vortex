// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Elias-Fano works in sign-extended 64-bit patterns and narrows back to the array's own width on
// the way out; see `EliasFanoData::reference_bits`. Both halves of that pair are exact.
#![expect(clippy::cast_possible_truncation)]

//! Elias-Fano encoding for monotonically non-decreasing integer sequences.
//!
//! Stores about `log2(u / n) + 2` bits per value for `n` values over a universe of `u`, while still
//! answering random access, rank, and predecessor queries in constant time. Against bit-packing at
//! `ceil(log2(u))` bits per value the saving is `log2(n)` bits, so it widens with row count.
//!
//! Inputs must be non-decreasing and non-nullable. Duplicates are fine; anything else is refused
//! rather than silently mangled.
//!
//! See [`elias_fano_encode`] for the compression entry point, [`EliasFanoCursor`] for point
//! lookups, rank, and seeks, and [`initialize`] to register the encoding in a session. The
//! crate-private `params` module documents the bit layout; its sampled select index follows
//! Vigna's [broadword][] construction. `rise-rs` and `vers` were consulted as references; no code
//! is taken from either.
//!
//! [broadword]: https://vigna.di.unimi.it/ftp/papers/Broadword.pdf

mod array;
mod compress;
mod compute;
mod cursor;
mod kernel;
pub(crate) mod params;
mod rules;

pub use array::EliasFano;
pub use array::EliasFanoArray;
pub use array::EliasFanoArraySlotsExt;
pub use array::EliasFanoData;
pub use array::EliasFanoMetadata;
pub use array::EliasFanoSlots;
pub use compress::elias_fano_encode;
pub use cursor::EliasFanoCursor;
pub use params::encoded_bit_size;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize the Elias-Fano encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(EliasFano);
    kernel::initialize(session);

    // Both answer from the layout rather than the data.
    session.aggregate_fns().register_aggregate_kernel(
        EliasFano.id(),
        Some(MinMax.id()),
        &compute::min_max::EliasFanoMinMaxKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        EliasFano.id(),
        Some(IsSorted.id()),
        &compute::is_sorted::EliasFanoIsSortedKernel,
    );
}

#[cfg(test)]
mod tests;
