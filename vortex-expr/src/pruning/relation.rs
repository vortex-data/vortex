use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools as _;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;

#[derive(Debug, Clone)]
pub struct Relation<K, V> {
    map: HashMap<K, HashSet<V>>,
}

impl<K: Display, V: Display> Display for Relation<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.map.iter().format_with(",", |(k, v), fmt| {
                fmt(&format_args!("{k}: {{{}}}", v.iter().format(",")))
            })
        )
    }
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

    #[allow(dead_code)]
    pub fn union(mut iter: impl Iterator<Item = Relation<K, V>>) -> Relation<K, V> {
        if let Some(mut x) = iter.next() {
            for y in iter {
                x.extend(y)
            }
            x
        } else {
            Relation::new()
        }
    }

    #[allow(dead_code)]
    pub fn extend(&mut self, other: Relation<K, V>) {
        for (l, rs) in other.map.into_iter() {
            self.map.entry(l).or_default().extend(rs.into_iter())
        }
    }

    pub fn insert(&mut self, k: K, v: V) {
        self.map.entry(k).or_default().insert(v);
    }

    #[allow(dead_code)]
    pub fn into_map(self) -> HashMap<K, HashSet<V>> {
        self.map
    }

    pub fn map(&self) -> &HashMap<K, HashSet<V>> {
        &self.map
    }
}
