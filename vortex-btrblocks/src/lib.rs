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
//! sample). Schemes are registered on a session: [`initialize`] registers [`DEFAULT_SCHEMES`],
//! and [`BtrBlocksCompressor::from_session`] compresses with the registered schemes whose
//! serialized IDs the session's enabled editions permit.
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
//! use vortex_btrblocks::{BtrBlocksCompressor, DEFAULT_SCHEMES};
//! use vortex_buffer::buffer;
//!
//! # fn example() -> vortex_error::VortexResult<()> {
//! let session = array_session();
//! let array = PrimitiveArray::new(buffer![42u64; 1024], Validity::NonNullable).into_array();
//!
//! // In memory, with no editions to respect, compress with the default schemes directly.
//! let compressor = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
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
#[cfg(test)]
mod tests;
#[cfg(test)]
#[cfg(not(codspeed))]
mod trace_tests;

// Re-export framework types from vortex-compressor for backwards compatibility.
// Btrblocks-specific exports.
pub use canonical_compressor::BtrBlocksCompressor;
pub use schemes::patches::compress_patches;
pub use vortex_compressor::CascadingCompressor;
pub use vortex_compressor::scheme::CompressorContext;
pub use vortex_compressor::scheme::MAX_CASCADE;
pub use vortex_compressor::scheme::Scheme;
pub use vortex_compressor::scheme::SchemeExt;
pub use vortex_compressor::scheme::SchemeId;
pub use vortex_compressor::session::CompressionSession;
pub use vortex_compressor::session::CompressionSessionExt;
pub use vortex_compressor::stats::ArrayAndStats;
pub use vortex_compressor::stats::BoolStats;
pub use vortex_compressor::stats::FloatStats;
pub use vortex_compressor::stats::GenerateStatsOptions;
pub use vortex_compressor::stats::IntegerStats;
pub use vortex_compressor::stats::StringStats;
use vortex_session::VortexSession;

use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// The default compression schemes.
///
/// This list is order-sensitive: [`initialize`] registers it in this order and the compressor
/// preserves registration order, so that tie-breaking is deterministic.
pub const DEFAULT_SCHEMES: &[&dyn Scheme] = &[
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Integer schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
    &integer::FoRScheme,
    // NOTE: ZigZag should precede BitPacking because we don't want negative numbers.
    &integer::ZigZagScheme,
    &integer::BitPackingScheme,
    &integer::SparseScheme,
    &integer::IntDictScheme,
    &integer::RunEndScheme,
    &integer::SequenceScheme,
    &integer::IntRLEScheme,
    // Delta is omitted here: see [`DELTA_SCHEME`].
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Float schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &float::ALPScheme,
    &float::ALPRDScheme,
    &float::FloatDictScheme,
    &float::NullDominatedSparseScheme,
    &float::FloatRLEScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // String schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &string::StringDictScheme,
    // Both string-fragmentation schemes are registered; the sample-based
    // selector keeps whichever is smaller per column.
    &string::FSSTScheme,
    &string::OnPairScheme,
    &string::NullDominatedSparseScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Binary schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &binary::BinaryDictScheme,
    &binary::VarBinScheme,
    // Decimal schemes.
    &decimal::DecimalScheme,
    // Temporal schemes.
    &temporal::TemporalScheme,
];

/// Compact schemes (Zstd for strings and binary, Pco for numerics when the `pco` feature is on).
///
/// Not part of [`DEFAULT_SCHEMES`]: they trade decode speed for compression ratio, so callers add
/// them to a compressor's scheme list explicitly.
#[cfg(feature = "zstd")]
pub const COMPACT_SCHEMES: &[&dyn Scheme] = &[
    &string::ZstdScheme,
    &binary::ZstdScheme,
    #[cfg(feature = "pco")]
    &integer::PcoScheme,
    #[cfg(feature = "pco")]
    &float::PcoScheme,
];

/// Delta, kept out of [`DEFAULT_SCHEMES`] because it is slower to decompress than the schemes that
/// would otherwise win. Callers that want it add it to their scheme list and permit
/// `fastlanes.delta`.
///
/// TODO(robert): Return it to [`DEFAULT_SCHEMES`] once we have scheme filtering.
pub static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// Registers [`DEFAULT_SCHEMES`] on `session`, in order.
///
/// Registration is idempotent, so this may run more than once.
pub fn initialize(session: &VortexSession) {
    for scheme in DEFAULT_SCHEMES {
        session.register_scheme(*scheme);
    }
}
