// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The five operator state machines: FLAT, CHUNKED, STRUCT, CONJUNCT and FILTER.

mod chunked;
mod conjunct;
mod filter;
mod flat;
mod struct_;

pub use chunked::ChunkedExec;
pub use conjunct::ConjunctExec;
pub use conjunct::ConjunctMode;
pub use conjunct::ConjunctSlot;
pub use filter::FilterExec;
pub use flat::FlatExec;
pub use struct_::StructExec;

/// The mask density at or above which a predicate is evaluated over the whole range and
/// intersected afterwards, rather than over the selected rows only.
///
/// Mirrors `EXPR_EVAL_THRESHOLD` in the V1 flat reader so the two executors make the same
/// regime choice on the same data.
pub(crate) const EXPR_EVAL_THRESHOLD: f64 = 0.2;
