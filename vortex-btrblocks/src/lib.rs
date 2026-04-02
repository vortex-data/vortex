// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Vortex's [BtrBlocks]-inspired adaptive compression framework.
//!
//! This crate provides a sophisticated multi-level compression system that adaptively selects
//! optimal compression schemes based on data characteristics. The compressor analyzes arrays
//! to determine the best encoding strategy, supporting cascaded compression with multiple
//! encoding layers for maximum efficiency.
//!
//! # Key Features
//!
//! - **Adaptive Compression**: Automatically selects the best compression scheme based on data
//!   patterns.
//! - **Unified Scheme Trait**: A single [`Scheme`] trait covers all data types (integers, floats,
//!   strings, etc.) with a [`SchemeId`] for identity.
//! - **Cascaded Encoding**: Multiple compression layers can be applied for optimal results.
//! - **Statistical Analysis**: Uses data sampling and statistics to predict compression ratios.
//! - **Recursive Structure Handling**: Compresses nested structures like structs and lists.
//!
//! # How It Works
//!
//! [`BtrBlocksCompressor::compress()`] takes an `&ArrayRef` and returns an `ArrayRef` that may
//! use a different encoding. It first canonicalizes the input, then dispatches by type.
//! Primitives and strings go through `choose_and_compress`, which evaluates every enabled
//! [`Scheme`] and picks the one with the best compression ratio. Compound types like structs
//! and lists recurse into their fields and elements.
//!
//! Each `Scheme` implementation declares whether it [`matches`](Scheme::matches) a given
//! canonical form and, if so, estimates the compression ratio (often by compressing a ~1%
//! sample). There is no dynamic registry — the set of schemes is fixed at build time via
//! [`ALL_SCHEMES`].
//!
//! Schemes can produce arrays that are themselves further compressed (e.g. FoR then BitPacking),
//! up to [`MAX_CASCADE`] (3) layers deep. Descendant exclusion rules for of [`SchemeId`] prevents
//! the same scheme from being applied twice in a chain.
//!
//! # Example
//!
//! ```rust
//! use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, Scheme, SchemeExt};
//! use vortex_btrblocks::schemes::integer::IntDictScheme;
//!
//! // Default compressor with all schemes enabled.
//! let compressor = BtrBlocksCompressor::default();
//!
//! // Configure with builder to exclude specific schemes.
//! let compressor = BtrBlocksCompressorBuilder::default()
//!     .exclude([IntDictScheme.id()])
//!     .build();
//! ```
//!
//! [BtrBlocks]: https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf

mod builder;
mod canonical_compressor;
/// Compression scheme implementations.
pub mod schemes;

// Re-export framework types from vortex-compressor for backwards compatibility.
// Btrblocks-specific exports.
pub use builder::ALL_SCHEMES;
pub use builder::BtrBlocksCompressorBuilder;
pub use builder::default_excluded;
pub use canonical_compressor::BtrBlocksCompressor;
pub use schemes::patches::compress_patches;
pub use vortex_compressor::CascadingCompressor;
pub use vortex_compressor::builtins::integer_dictionary_encode;
pub use vortex_compressor::ctx::CompressorContext;
pub use vortex_compressor::ctx::MAX_CASCADE;
pub use vortex_compressor::scheme::Scheme;
pub use vortex_compressor::scheme::SchemeExt;
pub use vortex_compressor::scheme::SchemeId;
pub use vortex_compressor::scheme::estimate_compression_ratio_with_sampling;
pub use vortex_compressor::stats::ArrayAndStats;
pub use vortex_compressor::stats::BoolStats;
pub use vortex_compressor::stats::FloatStats;
pub use vortex_compressor::stats::GenerateStatsOptions;
pub use vortex_compressor::stats::IntegerStats;
pub use vortex_compressor::stats::StringStats;
