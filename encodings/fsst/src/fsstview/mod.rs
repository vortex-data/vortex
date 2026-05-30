// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A "ListView"-style variant of the [`FSST`][crate::FSST] encoding.
//!
//! Where [`FSST`][crate::FSST] addresses its compressed codes with a single monotonic
//! offsets array (`len + 1` offsets, exactly like `VarBin`/`List`), [`FSSTView`] addresses
//! them with a pair of `offsets` **and** `sizes` arrays (exactly like
//! [`ListView`][vortex_array::arrays::ListView]). Element `i`'s compressed bytecodes live in
//! `codes_bytes[offsets[i] .. offsets[i] + sizes[i]]`.
//!
//! Decoupling the start (`offset`) from the length (`size`) means the offsets are no longer
//! required to be monotonic or contiguous, so `filter`, `take`, and `slice` become
//! metadata-only operations: they rewrite only the (small) `offsets`/`sizes`/lengths/validity
//! arrays and **reuse the compressed byte heap untouched**. The plain [`FSST`][crate::FSST]
//! encoding has to rewrite the entire compressed heap for `filter`/`take` because it delegates
//! to `VarBin`. This is the same trade-off `ListView` makes over `List`.

mod array;
mod canonical;
mod compute;
mod from_fsst;
mod kernel;
mod ops;
mod rules;
mod slice;
#[cfg(test)]
mod tests;

pub use array::*;
pub use canonical::FsstViewByteStats;
pub use canonical::FsstViewCompaction;
pub use canonical::canonicalize_fsstview_to_varbin;
pub use canonical::canonicalize_fsstview_with;
pub use canonical::fsstview_byte_stats;
pub use from_fsst::fsst_filter_to_view;
pub use from_fsst::fsst_take_to_view;
