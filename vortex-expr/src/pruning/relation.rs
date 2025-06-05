use std::hash::Hash;

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;

#[derive(Debug, Clone)]
pub(super) struct Relation<K, V> {
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

    pub fn extend(&mut self, other: Relation<K, V>) {
        for (l, rs) in other.map.into_iter() {
            self.map.entry(l).or_default().extend(rs.into_iter())
        }
    }

    pub fn insert(&mut self, k: K, v: V) {
        self.map.entry(k).or_default().insert(v);
    }

    pub fn map(&self) -> &HashMap<K, HashSet<V>> {
        &self.map
    }
}
