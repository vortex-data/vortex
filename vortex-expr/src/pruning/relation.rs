// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_utils::aliases::hash_map::{HashMap, IntoIter};
use vortex_utils::aliases::hash_set::HashSet;

#[derive(Debug, Clone)]
pub struct Relation<K, V> {
    map: HashMap<K, HashSet<V>>,
}

impl<K: Hash + Eq, V: Hash + Eq> Default for Relation<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq, V: Hash + Eq> Relation<K, V> {
    pub fn new() -> Self {
        Relation {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, k: K, v: V) {
        self.map.entry(k).or_default().insert(v);
    }

    pub fn map(&self) -> &HashMap<K, HashSet<V>> {
        &self.map
    }
}

impl<K, V> From<HashMap<K, HashSet<V>>> for Relation<K, V> {
    fn from(value: HashMap<K, HashSet<V>>) -> Self {
        Self { map: value }
    }
}

impl<K, V> IntoIterator for Relation<K, V> {
    type Item = (K, HashSet<V>);
    type IntoIter = IntoIter<K, HashSet<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}
