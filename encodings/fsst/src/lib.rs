// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An array that uses the [Fast Static Symbol Table][fsst] compression scheme
//! to compress string arrays.
//!
//! FSST arrays can generally compress string data up to 2x through the use of
//! string tables. The string table is static for an entire array, and occupies
//! up to 2048 bytes of buffer space. Thus, FSST is only worth reaching for when
//! dealing with larger arrays of potentially hundreds of kilobytes or more.
//!
//! [fsst]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

mod array;
mod canonical;
mod compress;
mod compute;
mod decoder;
mod dfa;
mod kernel;
mod ops;
mod rules;
mod slice;
#[cfg(feature = "_test-harness")]
pub mod test_utils;
#[cfg(test)]
mod tests;

pub use array::*;
pub use compress::*;

/// Re-export of the local FSST decoder for use from benchmarks.
///
/// Hidden behind `_test-harness` because the decoder is an implementation
/// detail of canonicalization; downstream crates should keep going through
/// `execute::<Canonical>`.
#[cfg(feature = "_test-harness")]
pub mod bench_decoder {
    pub use crate::decoder::Decoder;
}
