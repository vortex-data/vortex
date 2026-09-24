// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry of compression schemes.
//!
//! A session's [`CompressionSession`] holds the schemes available to compressors built from it
//! with [`BtrBlocksCompressor::from_session`](crate::BtrBlocksCompressor::from_session). It
//! starts with [`DEFAULT_SCHEMES`]. Whether a registered scheme may write its encodings is
//! decided by the session's enabled editions.

use std::any::Any;

use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// The default compression schemes, registered by [`CompressionSession::default`].
///
/// This list is order-sensitive: the compressor preserves registration order, so that
/// tie-breaking is deterministic.
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

/// The compression schemes registered on a session, in registration order.
///
/// Registration order is the compressor's tie-break order between equally good schemes, so
/// sessions that register the same schemes in the same order compress identically.
/// [`Default`] registers [`DEFAULT_SCHEMES`]; [`empty`](Self::empty) registers none.
#[derive(Clone, Debug)]
pub struct CompressionSession {
    /// Registered schemes in registration order.
    schemes: Vec<&'static dyn Scheme>,
}

impl CompressionSession {
    /// A registry with no schemes.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
        }
    }

    /// Registers a scheme.
    ///
    /// Registering a [`SchemeId`](crate::SchemeId) that is already present is a no-op.
    pub fn register(&mut self, scheme: &'static dyn Scheme) {
        if !self.schemes.iter().any(|s| s.id() == scheme.id()) {
            self.schemes.push(scheme);
        }
    }

    /// The registered schemes in registration order.
    pub fn schemes(&self) -> &[&'static dyn Scheme] {
        &self.schemes
    }
}

impl Default for CompressionSession {
    fn default() -> Self {
        Self {
            schemes: DEFAULT_SCHEMES.to_vec(),
        }
    }
}

impl SessionVar for CompressionSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Session access to the compression scheme registry.
pub trait CompressionSessionExt: SessionExt {
    /// Returns the compression scheme registry.
    fn compression(&self) -> SessionGuard<'_, CompressionSession> {
        self.get::<CompressionSession>()
    }

    /// Registers a compression scheme, see [`CompressionSession::register`].
    fn register_scheme(&self, scheme: &'static dyn Scheme) {
        self.get_mut::<CompressionSession>().register(scheme);
    }

    /// The registered compression schemes in registration order.
    fn registered_schemes(&self) -> Vec<&'static dyn Scheme> {
        self.compression().schemes().to_vec()
    }

    /// The registered schemes whose serialized IDs the enabled editions all permit.
    fn permitted_schemes(&self) -> Vec<&'static dyn Scheme> {
        self.permit(self.registered_schemes())
    }

    /// Keeps the schemes in `schemes` whose serialized IDs the enabled editions all permit.
    fn permit(&self, schemes: Vec<&'static dyn Scheme>) -> Vec<&'static dyn Scheme> {
        let allowed: HashSet<_> = self
            .enabled_component_ids(ComponentKind::Array)
            .into_iter()
            .collect();
        schemes
            .into_iter()
            .filter(|scheme| {
                scheme
                    .produced_encodings()
                    .iter()
                    .all(|id| allowed.contains(id))
            })
            .collect()
    }
}

impl<S: SessionExt> CompressionSessionExt for S {}

#[cfg(test)]
mod tests {
    use vortex_array::array_session;

    use super::*;
    use crate::SchemeId;
    use crate::schemes::float::FloatDictScheme;
    use crate::schemes::integer::IntDictScheme;

    fn ids(schemes: &[&'static dyn Scheme]) -> Vec<SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    #[test]
    fn default_registers_default_schemes() {
        let session = array_session();
        assert_eq!(ids(&session.registered_schemes()), ids(DEFAULT_SCHEMES));
    }

    #[test]
    fn registration_keeps_order_and_is_idempotent() {
        let session = array_session().with_some(CompressionSession::empty());
        session.register_scheme(&IntDictScheme);
        session.register_scheme(&FloatDictScheme);
        session.register_scheme(&IntDictScheme);
        assert_eq!(
            ids(&session.registered_schemes()),
            vec![IntDictScheme.id(), FloatDictScheme.id()]
        );
    }

    /// Without enabled editions no serialized ID is permitted, so nothing survives.
    #[test]
    fn no_editions_permit_nothing() {
        let session = array_session();
        assert!(session.permitted_schemes().is_empty());
    }
}
