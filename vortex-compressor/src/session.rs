// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry of compression schemes.
//!
//! Registering a scheme makes it available to compressors built from the session with
//! [`CascadingCompressor::from_session`](crate::CascadingCompressor::from_session). Whether a
//! registered scheme may write its encodings is decided by the session's enabled editions.

use std::any::Any;

use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_set::HashSet;

use crate::scheme::Scheme;
use crate::scheme::SchemeExt;

/// The compression schemes registered on a session, in registration order.
///
/// Registration order is the compressor's tie-break order between equally good schemes, so
/// sessions that register the same crates in the same order compress identically.
#[derive(Clone, Debug, Default)]
pub struct CompressionSession {
    /// Registered schemes in registration order.
    schemes: Vec<&'static dyn Scheme>,
}

impl CompressionSession {
    /// Registers a scheme.
    ///
    /// Registering a [`SchemeId`](crate::scheme::SchemeId) that is already present is a no-op, so
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
    use crate::builtins::FloatDictScheme;
    use crate::builtins::IntDictScheme;

    fn ids(schemes: &[&'static dyn Scheme]) -> Vec<crate::scheme::SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    #[test]
    fn registration_keeps_order_and_is_idempotent() {
        let session = array_session();
        assert!(session.registered_schemes().is_empty());
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
        session.register_scheme(&IntDictScheme);
        assert!(session.permitted_schemes().is_empty());
    }
}
