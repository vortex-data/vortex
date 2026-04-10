// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Global string interning pool for encoding and scalar function IDs.
//!
//! Both [`ArrayId`] and [`ScalarFnId`] are `ArcRef<str>`. Interning maps each unique
//! string to a dense sequential [`u32`], enabling `O(1)` dispatch table lookups via
//! [`Vec`] indexing instead of hashing.

use std::sync::OnceLock;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use parking_lot::RwLock;
use vortex_utils::aliases::hash_map::HashMap;

static POOL: OnceLock<RwLock<HashMap<Box<str>, u32>>> = OnceLock::new();
static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Intern a string, returning a stable dense sequential index.
///
/// Repeated calls with the same string always return the same index.
/// Indices are assigned starting from `0` in order of first insertion.
pub fn intern(s: &str) -> u32 {
    let pool = POOL.get_or_init(|| RwLock::new(HashMap::default()));
    {
        let guard = pool.read();
        if let Some(&idx) = guard.get(s) {
            return idx;
        }
    }
    let mut guard = pool.write();
    // Re-check under write lock (another thread may have inserted).
    *guard
        .entry(Box::from(s))
        .or_insert_with(|| COUNTER.fetch_add(1, Ordering::Relaxed))
}
