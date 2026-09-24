// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BtrBlocks-specific compressor wrapping the generic [`CascadingCompressor`].

use std::ops::Deref;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::CascadingCompressor;
use crate::CompressionSessionExt;

/// The BtrBlocks-style compressor.
///
/// This is a thin wrapper around [`CascadingCompressor`] built from the schemes registered on a
/// session. [`from_session`](Self::from_session) keeps the schemes whose serialized IDs the
/// session's enabled editions permit; [`from_session_no_editions`](Self::from_session_no_editions)
/// keeps every registered scheme, for in-memory compression where no edition applies.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::BtrBlocksCompressor;
///
/// let session = vortex_array::array_session();
/// vortex_btrblocks::initialize(&session);
///
/// // Every registered scheme; this session enables no editions.
/// let compressor = BtrBlocksCompressor::from_session_no_editions(&session);
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor(
    /// The underlying cascading compressor.
    pub CascadingCompressor,
);

impl BtrBlocksCompressor {
    /// Creates a compressor over the schemes registered on `session` whose serialized IDs the
    /// session's enabled editions permit.
    pub fn from_session(session: &VortexSession) -> Self {
        Self(CascadingCompressor::new(session.permitted_schemes()))
    }

    /// Creates a compressor over every scheme registered on `session`, ignoring editions.
    ///
    /// Use for in-memory compression, where no edition restricts what a file may contain.
    pub fn from_session_no_editions(session: &VortexSession) -> Self {
        Self(CascadingCompressor::new(session.registered_schemes()))
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
