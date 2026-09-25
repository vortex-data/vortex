// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BtrBlocks-specific compressor wrapping the generic [`CascadingCompressor`].

use std::ops::Deref;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::session::ArraySessionExt;
use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::CascadingCompressor;
use crate::CompressionSessionExt;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;

/// Options for building a [`BtrBlocksCompressor`] from a session.
#[derive(Clone, Debug)]
pub struct BtrBlocksOptions {
    /// Keep only the schemes whose serialized IDs the session's enabled editions permit, which a
    /// file writer requires. Off, for in-memory compression where no edition applies, every
    /// scheme whose encodings have a registered plugin is used.
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
    /// Creates a compressor over `schemes` as given.
    ///
    /// Prefer [`from_session`](Self::from_session) for writing files: it only keeps schemes the
    /// session can serialize and its editions permit, which this constructor does not check.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        Self(CascadingCompressor::new(schemes))
    }

    /// A compressor with no schemes, which leaves every array as it is.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Creates a compressor over the schemes registered on `session` whose serialized IDs the
    /// session's enabled editions permit.
    pub fn from_session(session: &VortexSession) -> Self {
        Self::from_session_with_options(session, &BtrBlocksOptions::default())
    }

    /// Creates a compressor over the schemes registered on `session`, per `options`.
    ///
    /// Only schemes the session can write are kept: every encoding they produce has a registered
    /// array plugin and, with `enforce_editions`, is permitted by the enabled editions. This is
    /// the rule the file writer applies to the arrays it serializes.
    pub fn from_session_with_options(session: &VortexSession, options: &BtrBlocksOptions) -> Self {
        let mut schemes = writable_schemes(session, options.enforce_editions);
        schemes.retain(|scheme| !options.exclude_schemes.contains(&scheme.id()));
        Self::new(schemes)
    }

    /// Compresses an array using BtrBlocks-inspired compression.
    pub fn compress(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.0.compress(array, ctx)
    }
}

/// The registered schemes whose produced encodings all have a registered array plugin and, with
/// `enforce_editions`, are permitted by the enabled editions.
fn writable_schemes(session: &VortexSession, enforce_editions: bool) -> Vec<&'static dyn Scheme> {
    let registered: HashSet<ArrayId> = session
        .arrays()
        .registry()
        .read(|registry| registry.keys().copied().collect());
    let allowed: HashSet<ArrayId> = if enforce_editions {
        session
            .enabled_component_ids(ComponentKind::Array)
            .into_iter()
            .filter(|id| registered.contains(id))
            .collect()
    } else {
        registered
    };
    session
        .compression()
        .schemes()
        .iter()
        .copied()
        .filter(|scheme| {
            scheme
                .produced_encodings()
                .iter()
                .all(|id| allowed.contains(id))
        })
        .collect()
}

impl Deref for BtrBlocksCompressor {
    type Target = CascadingCompressor;

    fn deref(&self) -> &CascadingCompressor {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array_session;

    use super::*;
    use crate::SchemeId;
    use crate::schemes::integer::FoRScheme;
    use crate::schemes::integer::IntDictScheme;

    fn ids(schemes: &[&'static dyn Scheme]) -> Vec<SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    /// Without enabled editions no serialized ID is permitted, so nothing survives.
    #[test]
    fn no_editions_permit_nothing() {
        assert!(writable_schemes(&array_session(), true).is_empty());
    }

    /// A scheme whose encoding has no registered plugin can never be written, editions or not.
    #[test]
    fn unregistered_encodings_are_never_writable() {
        let writable = ids(&writable_schemes(&array_session(), false));
        assert!(writable.contains(&IntDictScheme.id()));
        assert!(!writable.contains(&FoRScheme.id()));
    }
}
