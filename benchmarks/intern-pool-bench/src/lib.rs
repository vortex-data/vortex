// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Intern pool lookup functions and optimized data structures.
//!
//! Contains `#[inline(never)]` wrappers for ASM inspection via `cargo asm`,
//! plus a [`CompactPool`] that eliminates key comparison for maximum lookup speed.

#![expect(clippy::cast_possible_truncation)]
#![allow(clippy::disallowed_types)]
// Hash functions conventionally use single-char names (a, b, h, etc.)
#![allow(clippy::many_single_char_names)]

use std::collections::HashMap;

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

// ─── Const-evaluable hash function ──────────────────────────────────────────
//
// Uses wyhash-style widening multiply for high-quality mixing.
// Handles all string lengths but optimized for short keys (< 16 bytes).

/// Widening multiply + xor-fold. Core mixing function from wyhash/foldhash.
#[inline(always)]
const fn wymix(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[inline(always)]
const fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[inline(always)]
const fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

/// Const-evaluable hash function for string keys.
///
/// Deterministic (no runtime seed). Uses wyhash-style mixing for quality.
/// Produces the same hash at compile time and runtime.
#[inline(always)]
const fn const_hash(s: &str) -> u64 {
    const SEED_A: u64 = 0x9E37_79B9_7F4A_7C15; // golden ratio
    const SEED_B: u64 = 0x517C_C1B7_2722_0A95;

    let bytes = s.as_bytes();
    let len = bytes.len();

    let (a, b) = if len == 0 {
        (0u64, 0u64)
    } else if len <= 3 {
        // 1-3 bytes: read first, middle, last
        let x = bytes[0] as u64;
        let y = bytes[len >> 1] as u64;
        let z = bytes[len - 1] as u64;
        (x | (y << 8) | (z << 16), len as u64)
    } else if len <= 8 {
        // 4-8 bytes: read first 4 + last 4 (may overlap)
        let lo = read_u32_le(bytes, 0) as u64;
        let hi = read_u32_le(bytes, len - 4) as u64;
        (lo | (hi << 32), len as u64)
    } else if len <= 16 {
        // 8-16 bytes: read first 8 + last 8 (may overlap)
        (read_u64_le(bytes, 0), read_u64_le(bytes, len - 8))
    } else {
        // 16+ bytes: chain through the input
        let mut h = SEED_A;
        let mut i = 0;
        while i + 16 <= len {
            let va = read_u64_le(bytes, i);
            let vb = read_u64_le(bytes, i + 8);
            h = wymix(h ^ va, SEED_B ^ vb);
            i += 16;
        }
        let va = read_u64_le(bytes, len - 16);
        let vb = read_u64_le(bytes, len - 8);
        (h ^ va, vb)
    };

    wymix(a ^ SEED_A, b ^ SEED_B ^ (len as u64))
}

/// Const-evaluable hash, with 0 remapped to 1 (0 is the empty sentinel).
#[inline(always)]
const fn const_hash_nonzero(s: &str) -> u64 {
    let h = const_hash(s);
    if h == 0 { 1 } else { h }
}

// ─── StringId: compile-time pre-computed hash handle ─────────────────────────

/// A pre-computed hash of a string key, for O(1) lookup without re-hashing.
///
/// Can be created at **compile time** via [`StringId::of`], making the hot-path
/// lookup (`pool.resolve(id)`) a single array probe with zero hashing.
///
/// ```
/// # use intern_pool_bench::{CompactPool, StringId};
/// // Compute at compile time:
/// const BOOL_ID: StringId = StringId::of("bool");
/// const PRIM_ID: StringId = StringId::of("primitive");
///
/// let pool = CompactPool::new([("bool", 0), ("primitive", 1)]);
///
/// // Resolve on hot path — zero cost:
/// assert_eq!(pool.resolve(BOOL_ID), Some(0));
/// assert_eq!(pool.resolve(PRIM_ID), Some(1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(u64);

impl StringId {
    /// Compute the hash of a string key at compile time.
    ///
    /// The returned `StringId` can be used with [`CompactPool::resolve`].
    pub const fn of(s: &str) -> Self {
        Self(const_hash_nonzero(s))
    }
}

// ─── CompactPool: hash-only lookup, no key comparison ────────────────────────

/// A minimal hash table optimized for small, static string pools.
///
/// Uses hash-only comparison (no `bcmp` key compare), which is safe when
/// the entry count is small (< 10K) and the hash function is 64-bit
/// (collision probability ~10^-14 for 200 entries).
///
/// Layout: flat `(hash, value)` array with open addressing and ~25% load factor.
pub struct CompactPool {
    table: Box<[(u64, u64)]>,
    mask: usize,
}

impl CompactPool {
    /// Build from an iterator of (key, value) pairs.
    ///
    /// Panics if two keys produce the same 64-bit hash.
    pub fn new(entries: impl IntoIterator<Item = (&'static str, u64)>) -> Self {
        let entries: Vec<_> = entries.into_iter().collect();
        // 4x overallocation → ~25% load factor → almost always 1 probe
        let capacity = (entries.len() * 4).next_power_of_two();
        let mask = capacity - 1;
        let mut table = vec![(0u64, 0u64); capacity];

        for (key, value) in &entries {
            let hash = const_hash_nonzero(key);
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
        }
    }

    /// Pre-compute a [`StringId`] for a key at runtime.
    ///
    /// Prefer [`StringId::of`] when the key is a string literal (compiles to a constant).
    pub fn id(&self, key: &str) -> StringId {
        StringId::of(key)
    }

    /// Resolve a pre-computed [`StringId`] to its value. **No hashing, no key comparison.**
    ///
    /// This is the fastest possible lookup — a single array probe (~2ns).
    #[inline]
    pub fn resolve(&self, id: StringId) -> Option<u64> {
        self.get_by_hash(id.0)
    }

    /// Lookup by string key. Hashes the key, then probes with hash-only comparison.
    #[inline]
    pub fn get(&self, key: &str) -> Option<u64> {
        self.get_by_hash(const_hash_nonzero(key))
    }

    /// Lookup by raw pre-computed hash.
    #[inline]
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
}
