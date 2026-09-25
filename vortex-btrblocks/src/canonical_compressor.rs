// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BtrBlocks-specific compressor wrapping the generic [`CascadingCompressor`].

use std::ops::Deref;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressorBuilder;
use crate::CascadingCompressor;

/// The BtrBlocks-style compressor with all built-in schemes pre-registered.
///
/// This is a thin wrapper around [`CascadingCompressor`]. [`from_session`](Self::from_session)
/// uses the schemes registered in the session's [`CompressionSession`](crate::CompressionSession)
/// whose produced encodings the session registers and its enabled editions permit.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{
///     BtrBlocksCompressor, BtrBlocksCompressorBuilder, CompressionSession, Scheme, SchemeExt,
/// };
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
/// use vortex_session::VortexSession;
///
/// let session = VortexSession::empty().with::<CompressionSession>();
///
/// // Compressor with the session's schemes, restricted to the encodings it allows. This session
/// // enables no editions, so every scheme is dropped.
/// let compressor = BtrBlocksCompressor::from_session(&session);
///
/// // Remove specific schemes using the builder.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session)
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor(
    /// The underlying cascading compressor.
    pub CascadingCompressor,
);

impl BtrBlocksCompressor {
    /// Creates a compressor with the schemes registered in the session's
    /// [`CompressionSession`](crate::CompressionSession), keeping only those whose produced
    /// encodings the session registers and its enabled editions permit.
    pub fn from_session(session: &VortexSession) -> Self {
        BtrBlocksCompressorBuilder::from_session(session).build()
    }

    /// Compresses an array using BtrBlocks-inspired compression.
    pub fn compress(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.0.compress(array, ctx)
    }
}

impl Deref for BtrBlocksCompressor {
    type Target = CascadingCompressor;

    fn deref(&self) -> &CascadingCompressor {
        &self.0
    }
}

