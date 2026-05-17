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
//! ## Public API
//!
//! - [`RepeatedAccess`] — the user-facing handle, obtained via
//!   [`ArrayRef::repeated_access`](crate::ArrayRef::repeated_access).
//!   Provides `scalar_at`, `search_sorted`, plus procedures (`rank`,
//!   `position_of`, `search_range`, `count_in_range`).
//! - [`PointDispatch`] / [`PointDispatchExt`] — the trait that encoding
//!   kernels see; `point_scalar_at` / `point_search_sorted` overrides receive
//!   `&mut dyn PointDispatch` and recurse via `d.scalar_at` /
//!   `d.search_sorted` / [`PointDispatchExt::cached_block`].
//! - [`algorithms`] — generic fallbacks like `generic_search_sorted`.
//!
//! ## Internals
//!
//! - `PointSession` and `PointRuntime` are the two concrete `PointDispatch`
//!   implementations (caching and one-shot respectively). They're
//!   `pub(crate)` — external users go through [`RepeatedAccess`].
//!
//! See `docs/developer-guide/internals/point-fn.md` for the full design rationale.

mod access;
pub mod algorithms;
mod dispatch;
#[cfg(test)]
mod runtime;
mod session;
#[cfg(test)]
mod tests;

pub use access::RepeatedAccess;
pub use dispatch::AnyBlock;
pub use dispatch::BlockKey;
pub use dispatch::PointDispatch;
pub use dispatch::PointDispatchExt;
#[cfg(test)]
pub(crate) use runtime::PointRuntime;
pub(crate) use session::PointSession;

// Re-export the existing search result types so callers find them under the
// new module path too. These will move into this module in a later phase.
pub use crate::search_sorted::SearchResult;
pub use crate::search_sorted::SearchSortedSide;
