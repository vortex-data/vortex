// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry of compression schemes.
//!
//! Registering a scheme makes it available to compressors built from the session. Whether a
//! registered scheme may actually write its encodings is decided separately, by the serialized IDs
//! the compressor permits.

use std::any::Any;

use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// The default compression schemes.
///
/// This list is order-sensitive: [`CompressionSession`] registers it in this order by default and
/// the builder preserves registration order when constructing the final scheme list, so that
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

/// The compression schemes registered on a session, in registration order.
///
/// The default registers [`DEFAULT_SCHEMES`], so a session gets the default schemes the first time
/// its registry is used. Registration order is the compressor's tie-break order between equally
/// good schemes, so sessions that register the same crates in the same order compress identically.
#[derive(Clone, Debug)]
pub struct CompressionSession {
    /// Registered schemes in registration order.
    schemes: Vec<&'static dyn Scheme>,
}

impl Default for CompressionSession {
    fn default() -> Self {
        Self {
            schemes: DEFAULT_SCHEMES.to_vec(),
        }
    }
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
    /// Registering a [`SchemeId`](crate::SchemeId) that is already present is a no-op, so
    /// initializers may run more than once.
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
    /// Returns the compression scheme registry, registering the default schemes if absent.
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
}

impl<S: SessionExt> CompressionSessionExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DELTA_SCHEME;

    fn ids(schemes: &[&'static dyn Scheme]) -> Vec<crate::SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    #[test]
    fn default_registers_all_schemes_in_order() {
        let session = vortex_array::array_session();
        assert_eq!(ids(&session.registered_schemes()), ids(DEFAULT_SCHEMES));
    }

    #[test]
    fn registration_is_idempotent_and_appends() {
        let session = vortex_array::array_session();
        session.register_scheme(&DELTA_SCHEME);
        session.register_scheme(&DELTA_SCHEME);
        session.register_scheme(DEFAULT_SCHEMES[0]);
        let mut expected = ids(DEFAULT_SCHEMES);
        expected.push(DELTA_SCHEME.id());
        assert_eq!(ids(&session.registered_schemes()), expected);
    }
}
