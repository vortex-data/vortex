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
//! [`BtrBlocksCompressor::compress()`] takes an `&ArrayRef` plus a mutable execution context and
//! returns an `ArrayRef` that may use a different encoding. It first canonicalizes the input, then dispatches by type.
//! Primitives and strings go through `choose_and_compress`, which evaluates every enabled
//! [`Scheme`] and picks the one with the best compression ratio. Compound types like structs
//! and lists recurse into their fields and elements.
//!
//! Each `Scheme` implementation declares whether it [`matches`](Scheme::matches) a given
//! canonical form and, if so, estimates the compression ratio (often by compressing a ~1%
//! sample). The schemes available to a compressor are those registered on its session's
//! [`CompressionSession`], which starts with [`DEFAULT_SCHEMES`].
//! [`BtrBlocksCompressor::from_session`] keeps the registered schemes whose serialized IDs the
//! session's enabled editions permit; [`from_session_no_editions`] keeps them all.
//!
//! [`from_session_no_editions`]: BtrBlocksCompressor::from_session_no_editions
//!
//! Schemes can produce arrays that are themselves further compressed (e.g. FoR then BitPacking),
//! up to [`MAX_CASCADE`] (3) layers deep. Descendant exclusion rules for of [`SchemeId`] prevents
//! the same scheme from being applied twice in a chain.
//!
//! # Example
//!
//! ```rust
//! use vortex_array::{IntoArray, VortexSessionExecute, array_session};
//! use vortex_array::arrays::PrimitiveArray;
//! use vortex_array::validity::Validity;
//! use vortex_btrblocks::BtrBlocksCompressor;
//! use vortex_buffer::buffer;
//!
//! # fn example() -> vortex_error::VortexResult<()> {
//! let session = array_session();
//! let array = PrimitiveArray::new(buffer![42u64; 1024], Validity::NonNullable).into_array();
//!
//! // In memory, with no editions to respect, compress with every registered scheme.
//! let compressor = BtrBlocksCompressor::from_session_no_editions(&session);
//! let compressed = compressor.compress(&array, &mut session.create_execution_ctx())?;
//! assert_eq!(compressed.dtype(), array.dtype());
//! # Ok(())
//! # }
//! ```
//!
//! [BtrBlocks]: https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf

mod canonical_compressor;
/// Compression scheme implementations.
pub mod schemes;
/// Session registry of compression schemes.
pub mod session;
#[cfg(test)]
mod tests;
#[cfg(test)]
#[cfg(not(codspeed))]
mod trace_tests;

// Re-export framework types from vortex-compressor for backwards compatibility.
// Btrblocks-specific exports.
pub use canonical_compressor::BtrBlocksCompressor;
pub use schemes::patches::compress_patches;
#[cfg(feature = "zstd")]
pub use session::COMPACT_SCHEMES;
pub use session::CompressionSession;
pub use session::CompressionSessionExt;
pub use session::DEFAULT_SCHEMES;
pub use session::DELTA_SCHEME;
pub use vortex_compressor::CascadingCompressor;
pub use vortex_compressor::scheme::CompressorContext;
pub use vortex_compressor::scheme::MAX_CASCADE;
pub use vortex_compressor::scheme::Scheme;
pub use vortex_compressor::scheme::SchemeExt;
pub use vortex_compressor::scheme::SchemeId;
pub use vortex_compressor::stats::ArrayAndStats;
pub use vortex_compressor::stats::BoolStats;
pub use vortex_compressor::stats::FloatStats;
pub use vortex_compressor::stats::GenerateStatsOptions;
pub use vortex_compressor::stats::IntegerStats;
pub use vortex_compressor::stats::StringStats;
