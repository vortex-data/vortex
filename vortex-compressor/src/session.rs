// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Schemes explicitly registered for compression in a session.

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;

/// An ordered registry. Clones share registrations and registration is atomic.
#[derive(Clone, Debug, Default)]
pub struct CompressionSession {
    /// Shared registration order, protected across deduplication and insertion.
    schemes: Arc<RwLock<Vec<&'static dyn Scheme>>>,
}

impl CompressionSession {
    /// Create an empty registry.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Register a scheme. An already registered ID keeps its original instance and position.
    pub fn register(&self, scheme: &'static dyn Scheme) {
        let mut schemes = self.schemes.write();
        if !schemes.iter().any(|s| s.id() == scheme.id()) {
            schemes.push(scheme);
        }
    }

    /// Remove a registered scheme from future compressor construction.
    pub fn unregister(&self, id: SchemeId) {
        self.schemes.write().retain(|s| s.id() != id);
    }

    /// Snapshot the registered schemes in registration order.
    pub fn schemes(&self) -> Vec<&'static dyn Scheme> {
        self.schemes.read().clone()
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

/// Access to the session's compression registry.
pub trait CompressionSessionExt: SessionExt {
    /// Return the registry, initially empty.
    fn compression(&self) -> SessionGuard<'_, CompressionSession> {
        self.get::<CompressionSession>()
    }

    /// Register a scheme for future compression.
    fn register_scheme(&self, scheme: &'static dyn Scheme) {
        self.compression().register(scheme);
    }

    /// Fork session configuration with an independent copy of the compression registry.
    /// Other registered services remain shared. Use for per-operation scheme registration.
    fn fork_compression(&self) -> vortex_session::VortexSession {
        let session = self.session().fork();
        let registry = CompressionSession::empty();
        for scheme in self.registered_schemes() {
            registry.register(scheme);
        }
        session.register(registry);
        session
    }

    /// Snapshot the schemes selected by registration.
    fn registered_schemes(&self) -> Vec<&'static dyn Scheme> {
        self.compression().schemes()
    }
}
impl<S: SessionExt> CompressionSessionExt for S {}

#[cfg(test)]
mod tests {
    use vortex_session::VortexSession;

    use super::CompressionSessionExt;
    use crate::builtins::FloatDictScheme;
    use crate::builtins::IntDictScheme;
    use crate::builtins::StringDictScheme;
    use crate::scheme::Scheme;
    use crate::scheme::SchemeExt;

    #[test]
    fn registration_is_explicit_and_atomic() {
        let session = VortexSession::empty();
        assert!(session.registered_schemes().is_empty());
        let schemes: &[&'static dyn Scheme] =
            &[&IntDictScheme, &FloatDictScheme, &StringDictScheme];
        std::thread::scope(|scope| {
            for &scheme in schemes {
                let session = session.clone();
                scope.spawn(move || {
                    for _ in 0..32 {
                        session.register_scheme(scheme);
                    }
                });
            }
        });
        let registered = session.registered_schemes();
        assert_eq!(registered.len(), schemes.len());
        assert!(
            schemes
                .iter()
                .all(|expected| registered.iter().any(|actual| actual.id() == expected.id()))
        );
    }

    #[test]
    fn per_operation_registration_is_isolated() {
        let session = VortexSession::empty();
        session.register_scheme(&IntDictScheme);
        let fork = session.fork_compression();
        fork.register_scheme(&FloatDictScheme);
        fork.compression().unregister(IntDictScheme.id());
        assert_eq!(session.registered_schemes()[0].id(), IntDictScheme.id());
        assert_eq!(fork.registered_schemes()[0].id(), FloatDictScheme.id());
    }
}
