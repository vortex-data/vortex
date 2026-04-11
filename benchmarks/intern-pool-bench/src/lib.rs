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
use std::sync::OnceLock;
use std::sync::atomic::AtomicU16;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

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

// ─── ID read hot-path inspection ─────────────────────────────────────────────

/// Read from a plain slice by index.
#[inline(never)]
pub fn read_vec(slice: &[u64], idx: usize) -> u64 {
    slice[idx]
}

/// Read from a slice with unchecked indexing.
#[inline(never)]
pub fn read_vec_unchecked(slice: &[u64], idx: usize) -> u64 {
    // SAFETY: caller must ensure idx < slice.len()
    unsafe { *slice.get_unchecked(idx) }
}

/// Read from AtomicU64 with Relaxed ordering. On x86 = plain mov.
#[inline(never)]
pub fn read_atomic_relaxed(arr: &[AtomicU64], idx: usize) -> u64 {
    arr[idx].load(Ordering::Relaxed)
}

/// Read from OnceLock<u64>. Acquire load + is-initialized branch.
#[inline(never)]
pub fn read_oncelock(arr: &[OnceLock<u64>], idx: usize) -> u64 {
    arr[idx].get().copied().unwrap_or(0)
}

// ─── OnceRacy: lock-free racy initialization cell ────────────────────────────
//
// Like OnceLock, but without locks or Acquire fences. Racing initializers are
// harmless because init is idempotent — all threads compute the same value.
//
// Hot path: single Relaxed load (1 instruction on x86, plain `ldr` on ARM).
// Size: just the value + sentinel, no state byte, no padding.

/// A lock-free, racy initialization cell for small Copy types.
///
/// Designed for values where initialization is idempotent — multiple threads
/// may race to initialize, but they all compute the same value, so racing
/// stores are harmless. No mutex, no acquire fence, no parking lot.
///
/// ## Hot path ASM (x86-64)
/// ```text
/// movzx eax, word ptr [rdi]    ; one instruction. done.
/// ```
///
/// ## Size
/// `OnceRacy<u16>` = 2 bytes. `OnceLock<u16>` = 8 bytes.
pub struct OnceRacy<T: RacyValue>(T::Atomic);

/// Trait for values that can be stored in a `OnceRacy`.
/// Provides the atomic type and sentinel value.
pub trait RacyValue: Copy + Eq {
    /// The atomic type that can store this value.
    type Atomic;

    /// A sentinel value that means "uninitialized". Must not be a valid ID.
    const SENTINEL: Self;

    /// Create a new atomic initialized to the sentinel.
    fn new_atomic() -> Self::Atomic;

    /// Relaxed load from the atomic.
    fn load(atomic: &Self::Atomic) -> Self;

    /// Relaxed store into the atomic.
    fn store(atomic: &Self::Atomic, val: Self);
}

impl RacyValue for u16 {
    type Atomic = AtomicU16;
    const SENTINEL: u16 = u16::MAX;

    fn new_atomic() -> AtomicU16 {
        AtomicU16::new(Self::SENTINEL)
    }

    #[inline(always)]
    fn load(atomic: &AtomicU16) -> u16 {
        atomic.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn store(atomic: &AtomicU16, val: u16) {
        atomic.store(val, Ordering::Relaxed);
    }
}

impl RacyValue for u32 {
    type Atomic = AtomicU32;
    const SENTINEL: u32 = u32::MAX;

    fn new_atomic() -> AtomicU32 {
        AtomicU32::new(Self::SENTINEL)
    }

    #[inline(always)]
    fn load(atomic: &AtomicU32) -> u32 {
        atomic.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn store(atomic: &AtomicU32, val: u32) {
        atomic.store(val, Ordering::Relaxed);
    }
}

impl RacyValue for u64 {
    type Atomic = AtomicU64;
    const SENTINEL: u64 = u64::MAX;

    fn new_atomic() -> AtomicU64 {
        AtomicU64::new(Self::SENTINEL)
    }

    #[inline(always)]
    fn load(atomic: &AtomicU64) -> u64 {
        atomic.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn store(atomic: &AtomicU64, val: u64) {
        atomic.store(val, Ordering::Relaxed);
    }
}

impl<T: RacyValue> OnceRacy<T> {
    /// Read the value. Returns `None` if not yet initialized.
    #[inline(always)]
    pub fn get(&self) -> Option<T> {
        let v = T::load(&self.0);
        if v == T::SENTINEL { None } else { Some(v) }
    }

    /// Read the value, assuming it was initialized.
    #[inline(always)]
    pub fn get_unchecked(&self) -> T {
        T::load(&self.0)
    }

    /// Initialize or return the existing value. Races are fine —
    /// all callers must provide the same value for correctness.
    #[inline(always)]
    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> T {
        let v = T::load(&self.0);
        if v != T::SENTINEL {
            return v;
        }
        let val = f();
        T::store(&self.0, val);
        val
    }

    /// Explicitly set the value (for eager init).
    pub fn set(&self, val: T) {
        T::store(&self.0, val);
    }
}

// SAFETY: The inner type is an atomic, which is Sync by construction.
unsafe impl<T: RacyValue> Sync for OnceRacy<T> {}

/// For ASM inspection.
#[inline(never)]
pub fn read_once_racy_u16(id: &OnceRacy<u16>) -> Option<u16> {
    id.get()
}

#[inline(never)]
pub fn read_once_racy_u16_unchecked(id: &OnceRacy<u16>) -> u16 {
    id.get_unchecked()
}

/// Size assertions.
const _: () = {
    assert!(size_of::<OnceRacy<u16>>() == 2);
    assert!(size_of::<OnceRacy<u32>>() == 4);
    assert!(size_of::<OnceRacy<u64>>() == 8);
    // OnceLock<u16> = 8 bytes (4x larger)
    // OnceLock<u64> = 16 bytes (2x larger)
};

// Concrete const constructors (can't use trait methods in const context).

impl OnceRacy<u16> {
    /// Create an uninitialized `OnceRacy<u16>`.
    pub const fn new() -> Self {
        Self(AtomicU16::new(u16::MAX))
    }
}

impl OnceRacy<u32> {
    /// Create an uninitialized `OnceRacy<u32>`.
    pub const fn new() -> Self {
        Self(AtomicU32::new(u32::MAX))
    }
}

impl OnceRacy<u64> {
    /// Create an uninitialized `OnceRacy<u64>`.
    pub const fn new() -> Self {
        Self(AtomicU64::new(u64::MAX))
    }
}

// Keep CachedIdAtomic as an alias for backward compat with existing benchmarks.
pub type CachedIdAtomic = OnceRacy<u16>;

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

// ─── EncodingId: auto-assigned dense ordinal ─────────────────────────────────
//
// Each encoding declares a `static EncodingId`. The ordinal is auto-assigned
// the first time `InternedRegistry::register` is called. After init, reading
// the ordinal is a single `Relaxed` atomic load (~0.4ns on x86, plain `ldr` on ARM).

/// An encoding's ordinal — auto-assigned at registration, zero-cost to read.
///
/// Declare one per encoding as a `static`. The ordinal starts as `UNSET` and
/// gets assigned during `InternedRegistry::register()`.
///
/// ```
/// # use intern_pool_bench::EncodingId;
/// static BOOL_ENC: EncodingId = EncodingId::unset();
/// static PRIM_ENC: EncodingId = EncodingId::unset();
///
/// // Reading before init returns None:
/// assert!(BOOL_ENC.get().is_none());
/// ```
pub struct EncodingId(AtomicU16);

const UNSET: u16 = u16::MAX;

impl EncodingId {
    /// Create an unset encoding ID. Must be initialized via `InternedRegistry::register`.
    pub const fn unset() -> Self {
        Self(AtomicU16::new(UNSET))
    }

    /// Read the ordinal. Returns `None` if not yet registered.
    #[inline(always)]
    pub fn get(&self) -> Option<u16> {
        let v = self.0.load(Ordering::Relaxed);
        if v == UNSET { None } else { Some(v) }
    }

    /// Read the ordinal, assuming it has been initialized.
    /// In release builds, returns the raw value without checking.
    #[inline(always)]
    pub fn get_unchecked(&self) -> u16 {
        let v = self.0.load(Ordering::Relaxed);
        debug_assert!(v != UNSET, "EncodingId read before registration");
        v
    }

    fn set(&self, ord: u16) {
        let prev = self.0.swap(ord, Ordering::Relaxed);
        debug_assert!(
            prev == UNSET || prev == ord,
            "EncodingId registered twice with different ordinals"
        );
    }
}

// SAFETY: EncodingId is just an AtomicU16 — Send+Sync by construction.
unsafe impl Sync for EncodingId {}

/// A frozen registry that maps dense `u16` ordinals to values.
///
/// Ordinals are auto-assigned in registration order (0, 1, 2, ...).
/// After init, lookup by ordinal is a plain array index — 0.4ns.
///
/// ```
/// # use intern_pool_bench::{EncodingId, InternedRegistry};
/// static BOOL: EncodingId = EncodingId::unset();
/// static PRIM: EncodingId = EncodingId::unset();
///
/// let mut registry = InternedRegistry::<&str>::new();
/// registry.register(&BOOL, "vortex.bool", "BoolPlugin");
/// registry.register(&PRIM, "vortex.primitive", "PrimPlugin");
///
/// // After init — zero-cost reads:
/// assert_eq!(registry.get(BOOL.get_unchecked()), Some(&"BoolPlugin"));
/// assert_eq!(registry.get(PRIM.get_unchecked()), Some(&"PrimPlugin"));
/// ```
pub struct InternedRegistry<T> {
    /// Dense array of values indexed by ordinal.
    values: Vec<T>,
    /// String name → ordinal mapping (for deserialization from string IDs).
    name_to_ord: CompactPool,
}

impl<T> InternedRegistry<T> {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            name_to_ord: CompactPool::new(std::iter::empty()),
        }
    }

    /// Register a new encoding. Assigns the next sequential ordinal.
    ///
    /// - `id`: the static `EncodingId` slot to populate with the ordinal.
    /// - `name`: the string name (e.g., "vortex.primitive") for deserialization.
    /// - `value`: the plugin/vtable/data to store.
    pub fn register(&mut self, id: &EncodingId, name: &'static str, value: T) {
        let ord = self.values.len() as u16;
        id.set(ord);
        self.values.push(value);
        // We'll rebuild the name→ord pool after all registrations.
        // For now, just track the names.
        let _ = name; // used in freeze()
    }

    /// Lookup by ordinal — the hot path. Just an array index.
    #[inline(always)]
    pub fn get(&self, ord: u16) -> Option<&T> {
        self.values.get(ord as usize)
    }

    /// Lookup by ordinal, no bounds check.
    #[inline(always)]
    pub fn get_unchecked(&self, ord: u16) -> &T {
        // SAFETY: ord was assigned by us and is in range.
        unsafe { self.values.get_unchecked(ord as usize) }
    }

    /// Number of registered encodings.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<T> Default for InternedRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder that collects registrations, then freezes into an `InternedRegistry`.
///
/// This separates the mutable registration phase from the immutable read phase.
pub struct RegistryBuilder<T> {
    names: Vec<&'static str>,
    values: Vec<T>,
}

impl<T> RegistryBuilder<T> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Register an encoding. Auto-assigns the next ordinal to `id`.
    pub fn register(&mut self, id: &EncodingId, name: &'static str, value: T) {
        let ord = self.values.len() as u16;
        id.set(ord);
        self.names.push(name);
        self.values.push(value);
    }

    /// Freeze into an immutable `InternedRegistry`.
    /// Builds the `CompactPool` for string→ordinal lookups (deserialization path).
    pub fn freeze(self) -> InternedRegistry<T> {
        let name_to_ord = CompactPool::new(
            self.names
                .iter()
                .enumerate()
                .map(|(i, name)| (*name, i as u64)),
        );
        InternedRegistry {
            values: self.values,
            name_to_ord,
        }
    }
}

impl<T> Default for RegistryBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ASM inspection for InternedRegistry ─────────────────────────────────────

/// Read an encoding ordinal from a static EncodingId.
#[inline(never)]
pub fn read_encoding_id(id: &EncodingId) -> u16 {
    id.get_unchecked()
}

/// Lookup by ordinal in the InternedRegistry.
#[inline(never)]
pub fn read_registry<T>(registry: &InternedRegistry<T>, ord: u16) -> &T {
    registry.get_unchecked(ord)
}

/// The full hot path: read the static ordinal, then index the registry.
#[inline(never)]
pub fn read_full<'a, T>(id: &EncodingId, registry: &'a InternedRegistry<T>) -> &'a T {
    registry.get_unchecked(id.get_unchecked())
}

// ─── Global registry pattern ────────────────────────────────────────────────
//
// The registry is global (not session-scoped). EncodingIds are static atomics.
// Init once at program start, then read for free from anywhere.
//
// Usage:
// ```
// // Each encoding declares a global static:
// static BOOL: EncodingId = EncodingId::unset();
// static PRIM: EncodingId = EncodingId::unset();
//
// // At startup — call once:
// fn init_global_registry() {
//     let mut builder = RegistryBuilder::new();
//     builder.register(&BOOL, "vortex.bool", BoolPlugin);
//     builder.register(&PRIM, "vortex.primitive", PrimPlugin);
//     GLOBAL_REGISTRY.set(builder.freeze()).unwrap();
// }
//
// // Hot path — anywhere, any thread, zero cost:
// let reg = global_registry();
// reg.get_unchecked(BOOL.get_unchecked())  // ~0.4ns
// ```

/// Global registry storage. Set once at init, read forever.
static GLOBAL_REGISTRY: OnceLock<InternedRegistry<u64>> = OnceLock::new();

/// Access the global registry after init. Panics if not initialized.
#[inline(always)]
pub fn global_registry() -> &'static InternedRegistry<u64> {
    // After init, this is a single pointer load from a static.
    // OnceLock::get() on an initialized lock is just an Acquire load + branch (well-predicted).
    // SAFETY: after init, this is just an Acquire load + well-predicted branch.
    #[allow(clippy::expect_used)]
    GLOBAL_REGISTRY
        .get()
        .expect("global registry not initialized")
}

/// Initialize the global registry with test data (for benchmarking).
pub fn init_global_registry(names: &[&'static str], ids: &[&EncodingId]) {
    let mut builder = RegistryBuilder::new();
    for (i, (&name, id)) in names.iter().zip(ids.iter()).enumerate() {
        builder.register(id, name, i as u64);
    }
    drop(GLOBAL_REGISTRY.set(builder.freeze()));
}
