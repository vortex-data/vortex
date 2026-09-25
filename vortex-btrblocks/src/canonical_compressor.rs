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
use crate::SchemeExt;
use crate::SchemeId;

/// Options for building a [`BtrBlocksCompressor`] from a session.
#[derive(Clone, Debug)]
pub struct BtrBlocksOptions {
    /// Keep only the registered schemes whose serialized IDs the session's enabled editions
    /// permit, which a file writer requires. Off, every registered scheme is used, for in-memory
    /// compression where no edition applies.
    pub enforce_editions: bool,
    /// Schemes to leave out.
    pub exclude_schemes: Vec<SchemeId>,
}

impl Default for BtrBlocksOptions {
    fn default() -> Self {
        Self {
            enforce_editions: true,
            exclude_schemes: Vec::new(),
        }
    }
}

/// The BtrBlocks-style compressor.
///
/// This is a thin wrapper around [`CascadingCompressor`] built from the schemes registered on a
/// session. [`from_session`](Self::from_session) keeps the schemes whose serialized IDs the
/// session's enabled editions permit; [`from_session_with_options`](Self::from_session_with_options)
/// takes [`BtrBlocksOptions`] to ignore editions or leave schemes out.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::BtrBlocksCompressor;
/// use vortex_btrblocks::BtrBlocksOptions;
///
/// let session = vortex_array::array_session();
///
/// // Every registered scheme; this session enables no editions.
/// let compressor = BtrBlocksCompressor::from_session_with_options(
///     &session,
///     &BtrBlocksOptions {
///         enforce_editions: false,
///         ..Default::default()
///     },
/// );
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor(
    /// The underlying cascading compressor.
    pub CascadingCompressor,
);

impl BtrBlocksCompressor {
    /// A compressor with no schemes, which leaves every array as it is.
    pub fn empty() -> Self {
        Self(CascadingCompressor::new(Vec::new()))
    }

    /// Creates a compressor over the schemes registered on `session` whose serialized IDs the
    /// session's enabled editions permit.
    pub fn from_session(session: &VortexSession) -> Self {
        Self::from_session_with_options(session, &BtrBlocksOptions::default())
    }

    /// Creates a compressor over the schemes registered on `session`, per `options`.
    pub fn from_session_with_options(session: &VortexSession, options: &BtrBlocksOptions) -> Self {
        let mut schemes = if options.enforce_editions {
            session.permitted_schemes()
        } else {
            session.registered_schemes()
        };
        schemes.retain(|scheme| !options.exclude_schemes.contains(&scheme.id()));
        Self(CascadingCompressor::new(schemes))
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
