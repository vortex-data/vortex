// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain search: equality, prefix, and substring queries answered
//! directly over the code stream, without decoding rows back to bytes.
//!
//! These operations require a conformant dictionary — one that is sorted,
//! complete, and unique. A dictionary produced by the trainer or passed through
//! `validate` satisfies this precondition. [`crate::DictionaryView`]
//! only guarantees structurally safe access; it does not establish these semantic
//! properties:
//!
//! * **Sorted** — tokens are in bytewise-lexicographic order, so a needle can be
//!   tokenized ([`tokenize`](tokenize())) and prefix-ranged ([`prefix_range`]) by
//!   binary search.
//! * **Complete** — all 256 single-byte tokens are present, so any query string
//!   is encodable into codes.
//! * **Unique** — no two tokens are equal, so distinct code sequences denote
//!   distinct strings; equality then reduces to comparing code slices.
//!
//! # Shape
//! Each query is **prepared once** (tokenize the needle, or build a transition
//! table) into immutable data, then applied per row by a **free function over a
//! `&[`[`Token`](crate::Token)`]`**. The row to scan is
//! [`ColumnView::row_codes`](crate::ColumnView::row_codes), but the predicates
//! take any code slice, so a caller can scan a prefiltered subset of rows
//! rather than the whole column.
//!
//! # Operations
//! * [`tokenize`](tokenize()) — segment a needle into its canonical code sequence.
//! * [`equals`](equals()) — rows equal to a needle.
//! * [`starts_with`] — rows beginning with a needle, via a prepared
//!   [`PrefixQuery`].
//! * [`contains`](contains()) — rows containing a pattern, via a precomputed
//!   token-level KMP [`ContainsTable`].
//! * [`prefix_range`] — the sorted-dictionary primitive prefix search builds on.

mod contains;
mod equals;
mod lookup;
mod prefix;
mod tokenize;

pub use contains::{ContainsTable, contains};
pub use equals::equals;
pub use lookup::prefix_range;
pub use prefix::{PrefixQuery, starts_with};
pub use tokenize::tokenize;
