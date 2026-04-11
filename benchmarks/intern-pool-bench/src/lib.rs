// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Intern pool lookup functions and optimized data structures.
//!
//! Contains `#[inline(never)]` wrappers for ASM inspection via `cargo asm`,
//! plus a `CompactPool` that eliminates key comparison for maximum lookup speed.

#![expect(clippy::cast_possible_truncation)]
#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::hash::BuildHasher;

use rustc_hash::FxHashMap;

/// Type aliases for the hash map variants we're comparing.
pub type FoldhashMap<K, V> = hashbrown::HashMap<K, V>;
pub type AhashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;

// ─── Hot-path lookup functions for ASM inspection ────────────────────────────

/// Lookup in std HashMap (SipHash).
#[inline(never)]
pub fn lookup_siphash(map: &HashMap<&str, u64>, key: &str) -> Option<u64> {
    map.get(key).copied()
}

/// Lookup in FxHashMap.
#[inline(never)]
pub fn lookup_fxhash(map: &FxHashMap<&str, u64>, key: &str) -> Option<u64> {
    map.get(key).copied()
}

/// Lookup in hashbrown HashMap (foldhash).
#[inline(never)]
pub fn lookup_foldhash(map: &FoldhashMap<&str, u64>, key: &str) -> Option<u64> {
    map.get(key).copied()
}

/// Lookup in ahash HashMap.
#[inline(never)]
pub fn lookup_ahash(map: &AhashMap<&str, u64>, key: &str) -> Option<u64> {
    map.get(key).copied()
}

/// Binary search on sorted slice.
#[inline(never)]
pub fn lookup_binary_search(table: &[(&str, u64)], key: &str) -> Option<u64> {
    table
        .binary_search_by_key(&key, |(k, _)| *k)
        .ok()
        .map(|i| table[i].1)
}

// ─── CompactPool: hash-only lookup, no key comparison ────────────────────────
//
// For ~200 entries with 64-bit hashes, hash collisions have probability ~10^-14.
// We can safely compare hashes only, eliminating the `bcmp` call from the hot path.
//
// Layout: flat array of (hash, value) pairs with open addressing.
// Empty slots marked by hash=0.

/// A minimal hash table optimized for small, static string pools.
///
/// Trades correctness guarantees for speed: compares hashes only, never compares keys.
/// Safe when the number of entries is small (< 10K) and hash function is 64-bit.
pub struct CompactPool {
    table: Box<[(u64, u64)]>,
    mask: usize,
    build_hasher: foldhash::fast::FixedState,
}

impl CompactPool {
    /// Build from an iterator of (key, value) pairs.
    ///
    /// Panics if two keys produce the same hash (astronomically unlikely for < 10K entries).
    pub fn new(entries: impl IntoIterator<Item = (&'static str, u64)>) -> Self {
        let entries: Vec<_> = entries.into_iter().collect();
        // 4x overallocation → ~25% load factor → almost always 1 probe
        let capacity = (entries.len() * 4).next_power_of_two();
        let mask = capacity - 1;
        let build_hasher = foldhash::fast::FixedState::with_seed(0);
        let mut table = vec![(0u64, 0u64); capacity];

        for (key, value) in &entries {
            let hash = Self::hash_str(&build_hasher, key);
            let mut idx = hash as usize & mask;
            loop {
                let slot = &mut table[idx];
                assert!(slot.0 != hash, "hash collision in CompactPool");
                if slot.0 == 0 {
                    *slot = (hash, *value);
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self {
            table: table.into_boxed_slice(),
            mask,
            build_hasher,
        }
    }

    /// Lookup by string key. Hashes the key and compares hash only.
    #[inline(never)]
    pub fn get(&self, key: &str) -> Option<u64> {
        let hash = Self::hash_str(&self.build_hasher, key);
        self.get_by_hash(hash)
    }

    /// Lookup by pre-computed hash. Skips hashing entirely.
    #[inline(never)]
    pub fn get_by_hash(&self, hash: u64) -> Option<u64> {
        let mut idx = hash as usize & self.mask;
        loop {
            // SAFETY: idx is always masked to table bounds.
            let &(stored_hash, value) = unsafe { self.table.get_unchecked(idx) };
            if stored_hash == hash {
                return Some(value);
            }
            if stored_hash == 0 {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Pre-compute the hash for a key (for batched pre-hashed lookups).
    pub fn hash_key(&self, key: &str) -> u64 {
        Self::hash_str(&self.build_hasher, key)
    }

    fn hash_str(build_hasher: &foldhash::fast::FixedState, key: &str) -> u64 {
        let hash = build_hasher.hash_one(key);
        // Reserve 0 as the empty sentinel.
        if hash == 0 { 1 } else { hash }
    }
}
