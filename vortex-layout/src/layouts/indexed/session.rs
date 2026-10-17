//! Session registry of index kinds.

use std::any::Any;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::layouts::indexed::index::IndexId;
use crate::layouts::indexed::index::IndexVTableRef;

/// Registry of index kinds, keyed by their stable [`IndexId`].
type IndexRegistry = ArcSwapMap<Id, IndexVTableRef>;

/// Session state holding the registered index kinds.
///
/// Empty by default: index kinds live in their own crates and are registered explicitly, the same
/// way layout encodings are. A kind that is not registered is simply ignored on read — its child is
/// dropped from consideration and the data child answers the query directly.
#[derive(Clone, Debug, Default)]
pub struct IndexSession {
    registry: IndexRegistry,
}

impl IndexSession {
    /// Register an index kind, replacing any existing kind with the same id.
    pub fn register(&self, index: IndexVTableRef) {
        self.registry.insert(index.id(), index);
    }

    /// Find a registered index kind by id.
    pub fn find(&self, id: &IndexId) -> Option<IndexVTableRef> {
        self.registry.get(id)
    }

    /// The underlying registry.
    pub fn registry(&self) -> &IndexRegistry {
        &self.registry
    }
}

impl SessionVar for IndexSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for reaching the index-kind registry from a session.
pub trait IndexSessionExt: SessionExt {
    /// Returns the index-kind registry.
    fn indexes(&self) -> SessionGuard<'_, IndexSession> {
        self.get::<IndexSession>()
    }
}

impl<S: SessionExt> IndexSessionExt for S {}
