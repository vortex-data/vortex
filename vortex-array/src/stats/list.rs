// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Append-only, write-once storage for array statistics, keyed by aggregate function.
//!
//! ```text
//! seeded:   [min][max][is_sorted]              known at construction, never changed
//! computed: head ──▶ [null_count] ──▶ [mean] ──▶ null
//!                     newest                     immutable once published
//! ```
//!
//! Seeded stats sit in one vector. Stats computed later are prepended as nodes with a
//! compare-and-swap. Readers never lock: they scan the seeded stats, then follow `next`
//! pointers. Nothing is removed while the list is alive, so a reference into it lives as long
//! as the list. An inexact value may be joined by an exact one for the same key; readers prefer
//! the exact one.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering;

use arcref::ArcRef;

use crate::aggregate_fn::AggregateFnRef;
use crate::expr::stats::Precision;
use crate::scalar::ScalarValue;
use crate::stats::static_key;

/// The aggregate an entry is the result of.
///
/// Built-in aggregates use their static instance, so storing them costs no reference counting.
pub(crate) type Key = ArcRef<AggregateFnRef>;

/// One stored statistic: the result of `key` over the array, exact or a bound.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) key: Key,
    pub(crate) value: Precision<ScalarValue>,
    /// True if `key` is a built-in, stored as its static instance. Built-in keys compare by
    /// pointer, and never approximate one another: implied results are stored explicitly.
    pub(crate) builtin: bool,
}

impl Entry {
    /// An entry for `agg`, using its static instance if it is a built-in.
    pub(crate) fn new(agg: &AggregateFnRef, value: Precision<ScalarValue>) -> Self {
        match static_key(agg) {
            Some(key) => Self::builtin(key, value),
            None => Self {
                key: Key::from(agg.clone()),
                value,
                builtin: false,
            },
        }
    }

    /// An entry for a built-in aggregate, given as its static instance.
    pub(crate) fn builtin(key: &'static AggregateFnRef, value: Precision<ScalarValue>) -> Self {
        Self {
            key: Key::from(key),
            value,
            builtin: true,
        }
    }

    /// Returns true if this entry is the result of `key`. `key_is_builtin` is whether `key` is
    /// a static built-in instance, so callers resolve it once per lookup.
    pub(crate) fn is_for(&self, key: &AggregateFnRef, key_is_builtin: bool) -> bool {
        if self.builtin && key_is_builtin {
            return self.key.ptr_eq(key);
        }
        self.key.ptr_eq(key) || *self.key == *key
    }

    /// Returns true if `self`, already stored, makes storing `new` redundant.
    ///
    /// An exact value blocks everything for its key. An inexact value blocks only other
    /// inexact values, so a bound can be upgraded to an exact value once.
    fn blocks(&self, new: &Entry) -> bool {
        self.is_for(&new.key, new.builtin) && (self.value.is_exact() || !new.value.is_exact())
    }
}

struct Node {
    entry: Entry,
    // Set before the node is published and never changed after.
    next: *const Node,
}

/// Append-only, write-once list of statistics.
#[derive(Default)]
pub(super) struct StatsList {
    // Known when the list was built and never changed after. Empty lists, e.g. one shared
    // before execution, do not allocate for it.
    seeded: Vec<Entry>,
    // Null, or the newest computed node. Every node is owned by the list until drop.
    head: AtomicPtr<Node>,
}

// SAFETY: the list owns its nodes like a `Box<Node>` chain. Nodes are only read through shared
// references after publication, so the list is `Send`/`Sync` when `Entry` is.
unsafe impl Send for StatsList where Entry: Send + Sync {}
// SAFETY: see `Send` above.
unsafe impl Sync for StatsList where Entry: Send + Sync {}

impl StatsList {
    /// Builds a list holding `seeded`, known when the array is built.
    pub(super) fn from_seed(seeded: Vec<Entry>) -> Self {
        Self {
            seeded,
            head: AtomicPtr::default(),
        }
    }

    /// Iterates all entries.
    pub(super) fn iter(&self) -> impl Iterator<Item = &Entry> {
        let computed = Iter {
            next: self.head.load(Ordering::Acquire),
            _list: PhantomData,
        };
        self.seeded.iter().chain(computed)
    }

    /// Stores `entry` unless an existing entry blocks it. Returns true if it was stored.
    pub(super) fn insert(&self, entry: Entry) -> bool {
        if self.seeded.iter().any(|stored| stored.blocks(&entry)) {
            return false;
        }

        // Check before allocating: most writes of a known value are no-ops
        let mut head = self.head.load(Ordering::Acquire);
        if Self::blocked(head, ptr::null(), &entry) {
            return false;
        }

        let node = Box::into_raw(Box::new(Node {
            entry,
            next: ptr::null(),
        }));
        loop {
            // SAFETY: `node` is not published yet, so we have exclusive access.
            unsafe { (*node).next = head };
            match self
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(current) => {
                    // Only nodes pushed since `head` still need checking
                    // SAFETY: `node` is not published yet, so we have exclusive access.
                    if Self::blocked(current, head, unsafe { &(*node).entry }) {
                        // SAFETY: `node` was never published, so we still own it.
                        drop(unsafe { Box::from_raw(node) });
                        return false;
                    }
                    head = current;
                }
            }
        }
    }

    /// Returns true if a node from `from` up to, not including, `until` blocks `entry`.
    fn blocked(from: *const Node, until: *const Node, entry: &Entry) -> bool {
        let mut cursor = from;
        while cursor != until {
            // SAFETY: published nodes stay alive and immutable until the list drops.
            let current = unsafe { &*cursor };
            if current.entry.blocks(entry) {
                return true;
            }
            cursor = current.next;
        }
        false
    }
}

impl Drop for StatsList {
    fn drop(&mut self) {
        let mut cursor = *self.head.get_mut();
        while !cursor.is_null() {
            // SAFETY: we have exclusive access and every node came from `Box::into_raw`.
            let node = unsafe { Box::from_raw(cursor) };
            cursor = node.next.cast_mut();
        }
    }
}

/// Iterator over the computed nodes of a [`StatsList`], newest first.
struct Iter<'a> {
    next: *const Node,
    _list: PhantomData<&'a StatsList>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Entry;

    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: published nodes stay alive and immutable while the list is borrowed.
        let node = unsafe { self.next.as_ref() }?;
        self.next = node.next;
        Some(&node.entry)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;
    use crate::expr::stats::Stat;

    fn stat(stat: Stat, value: Precision<ScalarValue>) -> Entry {
        Entry::builtin(stat.aggregate_fn(), value)
    }

    fn values(list: &StatsList, of: Stat) -> Vec<Precision<ScalarValue>> {
        list.iter()
            .filter(|entry| entry.is_for(of.aggregate_fn(), true))
            .map(|entry| entry.value.clone())
            .collect()
    }

    #[test]
    fn write_once() {
        let list = StatsList::default();
        assert!(list.insert(stat(Stat::Min, Precision::exact(1u32))));
        assert!(!list.insert(stat(Stat::Min, Precision::exact(2u32))));
        assert!(!list.insert(stat(Stat::Min, Precision::inexact(0u32))));
        assert_eq!(values(&list, Stat::Min), vec![Precision::exact(1u32)]);
    }

    #[test]
    fn inexact_upgrades_once() {
        let list = StatsList::default();
        assert!(list.insert(stat(Stat::Max, Precision::inexact(10u32))));
        assert!(!list.insert(stat(Stat::Max, Precision::inexact(9u32))));
        assert!(list.insert(stat(Stat::Max, Precision::exact(8u32))));
        assert!(!list.insert(stat(Stat::Max, Precision::exact(8u32))));

        // The bound stays next to the exact value, which readers prefer
        let stored = values(&list, Stat::Max);
        assert_eq!(stored.len(), 2);
        assert!(stored.contains(&Precision::inexact(10u32)));
        assert!(stored.contains(&Precision::exact(8u32)));
    }

    #[test]
    fn concurrent_writers_store_one_value() {
        let list = Arc::new(StatsList::default());
        let stored: usize = (0..8u32)
            .map(|i| {
                let list = Arc::clone(&list);
                thread::spawn(move || {
                    usize::from(list.insert(stat(Stat::Sum, Precision::exact(i))))
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap_or_default())
            .sum();

        assert_eq!(stored, 1);
        assert_eq!(values(&list, Stat::Sum).len(), 1);
    }

    const SIX_STATS: [Stat; 6] = [
        Stat::Min,
        Stat::Max,
        Stat::Sum,
        Stat::NullCount,
        Stat::NaNCount,
        Stat::IsConstant,
    ];

    #[test]
    fn seeded_stats_block_computed_ones() {
        let list = StatsList::from_seed(vec![stat(Stat::Min, Precision::exact(1u32))]);
        assert!(!list.insert(stat(Stat::Min, Precision::exact(2u32))));
        assert!(list.insert(stat(Stat::Max, Precision::exact(3u32))));
        assert_eq!(values(&list, Stat::Min), vec![Precision::exact(1u32)]);
        assert_eq!(values(&list, Stat::Max), vec![Precision::exact(3u32)]);
    }

    #[test]
    fn stores_many_entries() {
        let list = StatsList::default();
        for key in SIX_STATS {
            assert!(list.insert(stat(key, Precision::exact(1u32))));
        }
        for key in SIX_STATS {
            assert!(!list.insert(stat(key, Precision::exact(2u32))));
        }
        assert_eq!(list.iter().count(), SIX_STATS.len());
    }

    #[test]
    fn concurrent_writers_store_each_key_once() {
        let list = Arc::new(StatsList::default());
        let writers = (0..4)
            .map(|_| {
                let list = Arc::clone(&list);
                thread::spawn(move || {
                    for key in SIX_STATS {
                        list.insert(stat(key, Precision::exact(1u32)));
                    }
                })
            })
            .collect::<Vec<_>>();
        for writer in writers {
            writer.join().unwrap_or_default();
        }

        assert_eq!(list.iter().count(), SIX_STATS.len());
        for key in SIX_STATS {
            assert_eq!(values(&list, key).len(), 1);
        }
    }

    #[test]
    fn readers_see_published_entries_during_writes() {
        let list = Arc::new(StatsList::default());
        let writer = {
            let list = Arc::clone(&list);
            thread::spawn(move || {
                for key in [Stat::Min, Stat::Max, Stat::Sum, Stat::NullCount] {
                    list.insert(stat(key, Precision::exact(1u32)));
                }
            })
        };

        // Every entry a reader reaches is fully initialized
        for _ in 0..16 {
            for entry in list.iter() {
                assert!(entry.value.is_exact());
            }
        }

        writer.join().unwrap_or_default();
        assert_eq!(list.iter().count(), 4);
    }
}
