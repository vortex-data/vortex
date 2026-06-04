// SPDX-License-Identifier: Apache-2.0
//! # string-skip
//!
//! Block-level skip indexes for dictionary-coded string columns.
//! Supports range / equality / prefix / substring / wildcard / length /
//! null predicates with sound (no-false-negative) pruning.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use string_skip::{ChunkStats, Pred, prune::chunk_might_match};
//!
//! let stats = ChunkStats::from_rows(&rows);
//! let presence = DictPresence::build(&dict, &codes);
//! let bloom = HybridBloom::build(&dict, &codes, 16, &ubiq);
//!
//! let keep = chunk_might_match(
//!     &Pred::Contains("google".into()),
//!     &stats, &presence, &bloom, &ubiq, &dict);
//! ```
//!
//! ## Variants
//!
//! - [`DictPresence`] — bitmap over dict ids. Exact for equality and
//!   anchored prefix when sorted; weak for substring.
//! - [`HybridBloom`] — BitFunnel-style code-bigram bloom that skips
//!   ubiquitous bigrams. Tight for substring on URL-like data.
//! - [`TieredBloom`] — variable-k bloom: common bigrams get fewer hash
//!   bits, rare bigrams get more. Tightest on high-diversity columns.
//! - [`ChunkStats`] — min/max/length/null per chunk. Exact pruning for
//!   range and length predicates on sorted data.
//!
//! ## Soundness
//!
//! Every public predicate evaluator returns `true` for any chunk that
//! contains a matching row. This is a hard invariant enforced by
//! `proptest` and unit tests.

#![warn(missing_docs)]

pub mod bloom;
pub mod chunk_stats;
pub mod dict;
pub mod hash;
pub mod pred;
pub mod presence;
pub mod prune;
pub mod tiers;
pub mod ubiq;

pub use bloom::Bloom;
pub use chunk_stats::ChunkStats;
pub use dict::DictIndex;
pub use dict::TokenDict;
pub use dict::tokenize_needle;
pub use pred::Pred;
pub use presence::DictPresence;
pub use prune::HybridBloom;
pub use prune::PruneResult;
pub use prune::TieredBloom;
pub use prune::chunk_might_match;
pub use tiers::BigramTiers;
pub use ubiq::UbiquitousBigrams;

/// Error type for serialization and other recoverable failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Wrong magic / version in the serialized payload.
    #[error("invalid skip-index payload: {0}")]
    InvalidPayload(&'static str),
    /// `bincode` (de)serialization error.
    #[error("bincode: {0}")]
    Bincode(#[from] bincode::Error),
}

/// Result type for fallible operations.
pub type Result<T> = std::result::Result<T, Error>;
