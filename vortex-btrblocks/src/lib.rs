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
//! - **Adaptive Compression**: Automatically selects the best compression scheme based on data patterns
//! - **Type-Specific Compressors**: Specialized compression for integers, floats, strings, and temporal data
//! - **Cascaded Encoding**: Multiple compression layers can be applied for optimal results
//! - **Statistical Analysis**: Uses data sampling and statistics to predict compression ratios
//! - **Recursive Structure Handling**: Compresses nested structures like structs and lists
//!
//! # How It Works
//!
//! [`BtrBlocksCompressor::compress()`] takes an `&ArrayRef` and returns an `ArrayRef` that may
//! use a different encoding. It first canonicalizes the input, then dispatches by type.
//! Primitives go to a type-specific `Compressor` (integer, float, or string). Compound types
//! like structs and lists recurse into their fields and elements.
//!
//! Each type-specific compressor holds a static list of `Scheme` implementations (e.g.
//! BitPacking, ALP, Dict). There is no dynamic registry. The compressor evaluates each scheme by
//! compressing a ~1% sample and measuring the ratio, then picks the best. See `SchemeExt` for
//! details on how sampling works.
//!
//! Schemes can produce arrays that are themselves further compressed (e.g. FoR then BitPacking),
//! up to `MAX_CASCADE` (3) layers deep. An `Excludes` set prevents the same scheme from being
//! applied twice in a chain.
//!
//! # Example
//!
//! ```rust
//! use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, IntCode};
//! use vortex_array::DynArray;
//!
//! // Default compressor with all schemes enabled
//! let compressor = BtrBlocksCompressor::default();
//!
//! // Configure with builder to exclude specific schemes
//! let compressor = BtrBlocksCompressorBuilder::default()
//!     .exclude_int([IntCode::Dict])
//!     .build();
//! ```
//!
//! [BtrBlocks]: https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf

pub use compressor::float::FloatCode;
use compressor::float::FloatCompressor;
pub use compressor::integer::IntCode;
use compressor::integer::IntCompressor;
pub use compressor::string::StringCode;
use compressor::string::StringCompressor;

mod builder;
mod canonical_compressor;
mod compressor;
mod ctx;
mod sample;
mod scheme;
mod stats;

pub use builder::BtrBlocksCompressorBuilder;
pub use canonical_compressor::BtrBlocksCompressor;
pub use canonical_compressor::CanonicalCompressor;
use compressor::Compressor;
use compressor::CompressorExt;
use compressor::MAX_CASCADE;
pub use compressor::integer::IntegerStats;
pub use compressor::integer::dictionary::dictionary_encode as integer_dictionary_encode;
use ctx::CompressorContext;
use ctx::Excludes;
use scheme::Scheme;
use scheme::SchemeExt;
pub use stats::CompressorStats;
pub use stats::GenerateStatsOptions;
