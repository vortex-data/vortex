// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Point functions — single-point queries against an array.
//!
//! Point functions are the third class of operations in Vortex, alongside
//! [`scalar_fn`](crate::scalar_fn) (element-wise) and
//! [`aggregate_fn`](crate::aggregate_fn) (reductions). They answer point queries
//! on an array — given a small input (a position, a value), produce a small output
//! (a scalar, an index) — without materializing the array.
//!
//! ## Layers
//!
//! - [`PointDispatch`] is the trait kernels call to recurse / cache.
//! - [`PointRuntime`] is the bare, one-shot dispatcher with no caching.
//! - [`PointSession`] is the caching dispatcher; hold it across many calls to
//!   amortize block decode and repeated scalar lookups.
//! - [`algorithms`] contains generic fallbacks like `generic_search_sorted`.
//!
//! See `docs/developer-guide/internals/point-fn.md` for the full design rationale.

pub mod algorithms;
mod dispatch;
mod dispatch_table;
mod runtime;
mod session;
#[cfg(test)]
mod tests;

pub use dispatch::AnyBlock;
pub use dispatch::BlockKey;
pub use dispatch::PointDispatch;
pub use dispatch::PointDispatchExt;
pub use runtime::PointRuntime;
pub use session::PointSession;

// Re-export the existing search result types so callers find them under the
// new module path too. These will move into this module in a later phase.
pub use crate::search_sorted::SearchResult;
pub use crate::search_sorted::SearchSortedSide;
