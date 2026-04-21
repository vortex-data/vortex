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
mod dfa;
mod kernel;
mod ops;
mod rules;
mod slice;

/// Test/benchmark-only handles to the internal DFA matchers.
///
/// Exposes the DFA constructors and their `matches` variants so that
/// bench harnesses can measure the different scan strategies (default
/// skip-assisted vs. zero-branch). Not part of the stable API.
#[cfg(feature = "_test-harness")]
pub mod dfa_bench_api {
    pub use crate::dfa::ContainsBench;
}

#[cfg(feature = "_test-harness")]
pub mod test_utils;
#[cfg(test)]
mod tests;

pub use array::*;
pub use compress::*;
