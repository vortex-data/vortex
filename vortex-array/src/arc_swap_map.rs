// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A concurrent, copy-on-write map backed by an [`ArcSwap`].

use std::borrow::Borrow;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use arc_swap::ArcSwap;
use vortex_utils::aliases::hash_map::HashMap;

/// A concurrent [`HashMap`] backed by an [`ArcSwap`], offering lock-free reads
/// and copy-on-write writes.
///
/// Reads load the current snapshot without blocking writers. Writes clone the
/// whole map, apply their change, and atomically publish the new version, so a
/// reader always observes a consistent snapshot and writers never block readers.
///
/// This is the shared building block behind the session-scoped registries (the
/// optimizer-kernel and aggregate-function registries). Because every write
/// clones the entire map, it is intended for maps that are written rarely
/// (typically only while a session is being configured) and read often.
pub(crate) struct ArcSwapMap<K, V> {
    inner: ArcSwap<HashMap<K, V>>,
}

impl<K, V> Default for ArcSwapMap<K, V> {
    fn default() -> Self {
        Self {
            inner: ArcSwap::from_pointee(HashMap::default()),
        }
    }
}

impl<K: Debug, V: Debug> Debug for ArcSwapMap<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.read(|map| f.debug_tuple("ArcSwapMap").field(map).finish())
    }
}

impl<K, V> ArcSwapMap<K, V> {
    /// Read the current snapshot, passing it to `f`.
    ///
    /// Every lookup inside `f` observes the same snapshot, which matters when a
    /// single logical read consults more than one key.
    pub(crate) fn read<R>(&self, f: impl FnOnce(&HashMap<K, V>) -> R) -> R {
        f(&self.inner.load())
    }

    /// Replace the map with the result of applying `f` to a private copy.
    ///
    /// Writes are copy-on-write via [`ArcSwap::rcu`], so `f` may run more than
    /// once under contention and must not move out of its captures.
    fn modify(&self, f: impl Fn(&mut HashMap<K, V>))
    where
        K: Clone,
        V: Clone,
    {
        self.inner.rcu(|existing| {
            let mut map = existing.as_ref().clone();
            f(&mut map);
            map
        });
    }
}

impl<K: Eq + Hash, V: Clone> ArcSwapMap<K, V> {
    /// Return a clone of the value stored under `key`, if present.
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.inner.load().get(key).cloned()
    }

    /// Insert `value` under `key`, replacing any existing value.
    pub(crate) fn insert(&self, key: K, value: V)
    where
        K: Clone,
    {
        self.modify(|map| {
            map.insert(key.clone(), value.clone());
        });
    }
}

impl<K: Eq + Hash + Clone, T: Clone> ArcSwapMap<K, Arc<[T]>> {
    /// Append `values` to the list stored under `key`, creating it if absent.
    ///
    /// Each key maps to an immutable `Arc<[T]>`; appending rebuilds that slice
    /// copy-on-write so existing readers keep their previous snapshot.
    pub(crate) fn extend(&self, key: K, values: &[T]) {
        self.modify(|map| {
            let merged: Arc<[T]> = match map.get(&key) {
                Some(existing) => existing.iter().chain(values).cloned().collect(),
                None => values.into(),
            };
            map.insert(key.clone(), merged);
        });
    }

    /// Append a single `value` to the list stored under `key`, creating it if
    /// absent.
    pub(crate) fn push(&self, key: K, value: T) {
        self.extend(key, &[value]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_and_insert() {
        let map = ArcSwapMap::<u32, i32>::default();
        assert_eq!(map.get(&1), None);
        map.insert(1, 10);
        map.insert(1, 20);
        assert_eq!(map.get(&1), Some(20));
    }

    #[test]
    fn extend_appends_per_key() {
        let map = ArcSwapMap::<u32, Arc<[i32]>>::default();
        map.extend(1, &[1, 2]);
        map.extend(1, &[3]);
        map.extend(2, &[4]);
        assert_eq!(map.get(&1).as_deref(), Some([1, 2, 3].as_slice()));
        assert_eq!(map.get(&2).as_deref(), Some([4].as_slice()));
    }

    #[test]
    fn push_appends_single_values() {
        let map = ArcSwapMap::<u32, Arc<[i32]>>::default();
        map.push(1, 1);
        map.push(1, 2);
        assert_eq!(map.get(&1).as_deref(), Some([1, 2].as_slice()));
    }

    #[test]
    fn read_observes_a_single_snapshot() {
        let map = ArcSwapMap::<u32, i32>::default();
        map.insert(1, 1);
        map.insert(2, 2);
        assert_eq!(map.read(|m| m.values().sum::<i32>()), 3);
    }
}
