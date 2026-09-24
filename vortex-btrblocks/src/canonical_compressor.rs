// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BtrBlocks-specific compressor wrapping the generic [`CascadingCompressor`].

use std::ops::Deref;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::CascadingCompressor;
use crate::Scheme;

/// The BtrBlocks-style compressor.
///
/// This is a thin wrapper around [`CascadingCompressor`]. [`from_session`](Self::from_session)
/// compresses with the schemes registered on a session that its enabled editions permit;
/// [`new`](Self::new) takes an explicit scheme list, for example [`DEFAULT_SCHEMES`] for in-memory
/// compression where no edition applies.
///
/// [`DEFAULT_SCHEMES`]: crate::DEFAULT_SCHEMES
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressor, DEFAULT_SCHEMES, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Every default scheme, for in-memory compression.
/// let compressor = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
///
/// // Every default scheme except one.
/// let compressor = BtrBlocksCompressor::new(
///     DEFAULT_SCHEMES
///         .iter()
///         .copied()
///         .filter(|scheme| scheme.id() != IntDictScheme.id())
///         .collect(),
/// );
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor(
    /// The underlying cascading compressor.
    pub CascadingCompressor,
);

impl BtrBlocksCompressor {
    /// Creates a compressor over exactly `schemes`, in order.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        Self(CascadingCompressor::new(schemes))
    }

    /// Creates a compressor over the schemes registered on `session` whose serialized IDs the
    /// session's enabled editions permit.
    pub fn from_session(session: &VortexSession) -> Self {
        Self(CascadingCompressor::from_session(session))
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
