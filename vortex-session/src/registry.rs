// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Many session types use a registry of objects that can be looked up by name to construct
//! contexts. This module provides a generic registry type for that purpose.

use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_error::VortexExpect;
use vortex_utils::aliases::dash_map::DashMap;

/// A registry of items that are keyed by a string identifier.
#[derive(Clone, Debug)]
pub struct Registry<T>(Arc<DashMap<String, T>>);

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Display + Clone + Eq> Registry<T> {
    pub fn empty() -> Self {
        Self(Default::default())
    }

    /// List the items in the registry.
    pub fn items(&self) -> impl Iterator<Item = T> + '_ {
        self.0.iter().map(|i| i.value().clone())
    }

    /// Return the items with the given IDs.
    pub fn find_many<'a>(
        &self,
        ids: impl IntoIterator<Item = &'a str>,
    ) -> impl Iterator<Item = Option<impl Deref<Target = T>>> {
        ids.into_iter().map(|id| self.0.get(id))
    }

    /// Find the item with the given ID.
    pub fn find(&self, id: &str) -> Option<T> {
        self.0.get(id).as_deref().cloned()
    }

    /// Register a new item, replacing any existing item with the same ID.
    pub fn register(&self, item: T) {
        self.0.insert(item.to_string(), item);
    }

    /// Register a new item, replacing any existing item with the same ID.
    pub fn register_many<I: IntoIterator<Item = T>>(&self, items: I) {
        for item in items {
            self.0.insert(item.to_string(), item);
        }
    }
}

/// A set of ordered items from a registry, addressed by u16 integers.
/// This can be used to create a dictionary-encoded set of identifiable items.
#[derive(Clone, Debug)]
pub struct RegistrySet<T> {
    registry: Registry<T>,
    ids: Arc<RwLock<Vec<String>>>,
}

impl<T: Display + Clone + Eq> RegistrySet<T> {
    pub fn new(registry: Registry<T>, ids: Vec<String>) -> Self {
        Self {
            registry,
            ids: Arc::new(RwLock::new(ids)),
        }
    }

    pub fn empty(registry: Registry<T>) -> Self {
        Self {
            registry,
            ids: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn get(&self, position: u16) -> Option<T> {
        let ids = self.ids.read();
        let id = ids.get(position as usize)?;
        self.registry.find(id)
    }

    pub fn position(&self, id: &str) -> Option<u16> {
        {
            let ids = self.ids.read();
            if let Some(idx) = ids
                .iter()
                .position(|arc_str| arc_str.deref() == id)
                .map(|idx| u16::try_from(idx).vortex_expect("maximum u16 items exceeded"))
            {
                return Some(idx);
            }
        }

        // Otherwise, try to find in the registry and add it
        let encoding = self.registry.find(id)?;
        let mut ids = self.ids.write();
        let idx = ids.len();
        ids.push(encoding.to_string());
        Some(u16::try_from(idx).vortex_expect("maximum u16 items exceeded"))
    }
}
