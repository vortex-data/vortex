// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! First-party compression initialization and edition-aware compressor construction.
//!
//! Register encoding packages and their schemes, select editions, then construct
//! [`BtrBlocksCompressor::from_session`]. Registration opts a scheme into compression.
//! Construction removes configurations whose serialized outputs are not permitted.

mod initialize;
pub use initialize::initialize;
pub use initialize::initialize_compact;
mod canonical_compressor;
/// Compression scheme implementations.
pub mod schemes;
mod session;
#[cfg(test)]
#[cfg(not(codspeed))]
mod trace_tests;

// Re-export framework types from vortex-compressor for backwards compatibility.
// Btrblocks-specific exports.
/// Optional delta scheme configuration.
pub static DELTA_SCHEME: schemes::integer::DeltaScheme = schemes::integer::DeltaScheme::new(1.25);
pub use canonical_compressor::BtrBlocksCompressor;
pub use session::CompressionSession;
pub use session::CompressionSessionExt;
pub use vortex_compressor::CascadingCompressor;
pub use vortex_compressor::compress_patches;
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
