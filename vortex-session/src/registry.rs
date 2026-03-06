// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Many session types use a registry of objects that can be looked up by name to construct
//! contexts. This module provides a generic registry type for that purpose.

use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use arcref::ArcRef;
use parking_lot::RwLock;
use vortex_error::VortexExpect;
use vortex_utils::aliases::dash_map::DashMap;

/// An identifier for an item in a registry.
pub type Id = ArcRef<str>;

/// A registry of items that are keyed by a string identifier.
#[derive(Clone, Debug)]
pub struct Registry<T>(Arc<DashMap<Id, T>>);

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Clone> Registry<T> {
    pub fn empty() -> Self {
        Self(Default::default())
    }

    /// List the IDs in the registry.
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.0.iter().map(|i| i.key().clone())
    }

    /// List the items in the registry.
    pub fn items(&self) -> impl Iterator<Item = T> + '_ {
        self.0.iter().map(|i| i.value().clone())
    }

    /// Return the items with the given IDs.
    pub fn find_many<'a>(
        &self,
        ids: impl IntoIterator<Item = &'a Id>,
    ) -> impl Iterator<Item = Option<impl Deref<Target = T>>> {
        ids.into_iter().map(|id| self.0.get(id))
    }

    /// Find the item with the given ID.
    pub fn find(&self, id: &Id) -> Option<T> {
        self.0.get(id).as_deref().cloned()
    }

    /// Register a new item, replacing any existing item with the same ID.
    pub fn register(&self, id: impl Into<Id>, item: impl Into<T>) {
        self.0.insert(id.into(), item.into());
    }

    /// Register a new item, replacing any existing item with the same ID, and return self for
    pub fn with(self, id: impl Into<Id>, item: impl Into<T>) -> Self {
        self.register(id, item.into());
        self
    }
}

/// A [`ReadContext`] holds a set of interned IDs for use during deserialization, mapping
/// u16 indices to IDs.
#[derive(Clone, Debug)]
pub struct ReadContext {
    ids: Arc<[Id]>,
}

impl ReadContext {
    /// Create a context with the given initial IDs.
    pub fn new(ids: impl Into<Arc<[Id]>>) -> Self {
        Self { ids: ids.into() }
    }

    /// Resolve an interned ID by its index.
    pub fn resolve(&self, idx: u16) -> Option<Id> {
        self.ids.get(idx as usize).cloned()
    }

    pub fn ids(&self) -> &[Id] {
        &self.ids
    }
}

/// A [`Context`] holds a set of interned IDs for use during serialization/deserialization, mapping
/// IDs to u16 indices.
///
/// ## Upcoming Changes
///
/// 1. This object holds an Arc of RwLock internally because we need concurrent access from the
///    layout writer code path. We should update SegmentSink to take an Array rather than
///    ByteBuffer such that serializing arrays is done sequentially.
/// 2. The name is terrible. `Interner<T>` is better, but I want to minimize breakage for now.
#[derive(Clone, Debug)]
pub struct Context<T> {
    // TODO(ngates): it's a long story, but if we make SegmentSink and SegmentSource take an
    //  enum of Segment { Array, DType, Buffer } then we don't actually need a mutable context
    //  in the LayoutWriter, therefore we don't need a RwLock here and everyone is happier.
    ids: Arc<RwLock<Vec<Id>>>,
    // Optional registry used to filter the permissible interned items.
    registry: Option<Registry<T>>,
}

impl<T> Default for Context<T> {
    fn default() -> Self {
        Self {
            ids: Arc::new(RwLock::new(Vec::new())),
            registry: None,
        }
    }
}

impl<T: Clone> Context<T> {
    /// Create a context with the given initial IDs.
    pub fn new(ids: Vec<Id>) -> Self {
        Self {
            ids: Arc::new(RwLock::new(ids)),
            registry: None,
        }
    }

    /// Create an empty context.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Configure a registry to restrict the permissible set of interned items.
    pub fn with_registry(mut self, registry: Registry<T>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Intern an ID, returning its index.
    pub fn intern(&self, id: &Id) -> Option<u16> {
        if let Some(registry) = &self.registry
            && registry.find(id).is_none()
        {
            // ID not in registry, cannot intern.
            return None;
        }

        let mut ids = self.ids.write();
        if let Some(idx) = ids.iter().position(|e| e == id) {
            return Some(u16::try_from(idx).vortex_expect("Cannot have more than u16::MAX items"));
        }

        let idx = ids.len();
        assert!(
            idx < u16::MAX as usize,
            "Cannot have more than u16::MAX items"
        );
        ids.push(id.clone());
        Some(u16::try_from(idx).vortex_expect("checked already"))
    }

    /// Get the list of interned IDs.
    pub fn to_ids(&self) -> Vec<Id> {
        self.ids.read().clone()
    }
}
