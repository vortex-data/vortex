// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Many session types use a registry of objects that can be looked up by name to construct
//! contexts. This module provides a generic registry type for that purpose.

use std::ops::Deref;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_utils::aliases::dash_map::DashMap;

/// A registry of items that are keyed by a string identifier.
// TODO(ngates): define a RegistryItem trait that has a custom key to avoid to_string calls.
#[derive(Clone, Debug)]
pub struct Registry<T>(Arc<DashMap<ArcRef<str>, T>>);

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
    pub fn ids(&self) -> impl Iterator<Item = ArcRef<str>> + '_ {
        self.0.iter().map(|i| i.key().clone())
    }

    /// List the items in the registry.
    pub fn items(&self) -> impl Iterator<Item = T> + '_ {
        self.0.iter().map(|i| i.value().clone())
    }

    /// Return the items with the given IDs.
    pub fn find_many<'a>(
        &self,
        ids: impl IntoIterator<Item = &'a ArcRef<str>>,
    ) -> impl Iterator<Item = Option<impl Deref<Target = T>>> {
        ids.into_iter().map(|id| self.0.get(id))
    }

    /// Find the item with the given ID.
    pub fn find(&self, id: &ArcRef<str>) -> Option<T> {
        self.0.get(id).as_deref().cloned()
    }

    /// Register a new item, replacing any existing item with the same ID.
    pub fn register(&self, id: impl Into<ArcRef<str>>, item: impl Into<T>) {
        self.0.insert(id.into(), item.into());
    }

    /// Register a new item, replacing any existing item with the same ID, and return self for
    pub fn with(self, id: impl Into<ArcRef<str>>, item: impl Into<T>) -> Self {
        self.register(id, item.into());
        self
    }
}
