// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A "ListView"-style variant of the [`FSST`][crate::FSST] encoding.
//!
//! Where [`FSST`][crate::FSST] addresses its compressed codes with a single monotonic
//! offsets array (`len + 1` offsets, exactly like `VarBin`/`List`), [`FSSTView`] addresses
//! them with a pair of per-element `offsets` **and** `ends` arrays (the ListView idea, storing
//! the end offset rather than the size — see [`ListView`][vortex_array::arrays::ListView]).
//! Element `i`'s compressed bytecodes live in `codes_bytes[offsets[i] .. ends[i]]`, and its size
//! is the derived `ends[i] - offsets[i]`.
//!
//! Decoupling the start (`offset`) from the end means the offsets are no longer required to be
//! monotonic or contiguous, so `filter`, `take`, and `slice` become metadata-only operations:
//! they rewrite only the (small) `offsets`/`ends`/lengths/validity arrays and **reuse the
//! compressed byte heap untouched**. The plain [`FSST`][crate::FSST] encoding has to rewrite the
//! entire compressed heap for `filter`/`take` because it delegates to `VarBin`. This is the same
//! trade-off `ListView` makes over `List`.
//!
//! Storing the *end* offset (instead of the size) additionally makes the [`FSSTArray`] →
//! [`FSSTViewArray`] conversion allocation-free: a freshly converted heap is contiguous, so both
//! `offsets` and `ends` are zero-copy slices of the FSST's monotonic offsets buffer
//! (`offsets[0..len]` and `offsets[1..len + 1]`). A selective `filter`/`take` therefore never
//! pays to derive sizes for the rows it discards.

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
pub use canonical::FsstViewCompaction;
pub use canonical::canonicalize_fsstview_to_varbin;
pub use canonical::canonicalize_fsstview_with;
pub use from_fsst::fsst_filter_to_view;
pub use from_fsst::fsst_take_to_view;
