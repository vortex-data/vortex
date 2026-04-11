// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark: full child-iteration loop with N children.
//!
//! Scenario: Cast(Chunked(N chunks of Primitive))
//! The executor iterates N+1 slots (chunk_offsets + N chunks), calling
//! reduce_parent and execute_parent on each. Most calls are wasted
//! because the chunk children (Primitive) don't have rules that match
//! the Cast parent's encoding_id for reduce, and only CastExecuteAdaptor
//! matches for execute.
//!
//! We compare:
//! 1. CURRENT: for each slot, virtual dispatch → linear scan of rules → matches()
//! 2. PROPOSED: for each slot, HashMap<(parent_id, child_id)> lookup

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

/// Build Cast(Chunked(N chunks of Primitive i32))
fn make_cast_of_chunked(nchunks: usize) -> ArrayRef {
    let chunks: Vec<ArrayRef> = (0..nchunks)
        .map(|i| {
            PrimitiveArray::new(Buffer::from(vec![i as i32; 100]), Validity::NonNullable)
                .into_array()
        })
        .collect();
    let len = nchunks * 100;
    let chunked = unsafe {
        ChunkedArray::new_unchecked(
            chunks,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
    }
    .into_array();

    Cast.try_new_array(
        len,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        [chunked],
    )
    .unwrap()
}

/// Pre-built registry: (parent_encoding_id, child_encoding_id) -> has_rules
/// Uses &str keys (no allocation in lookup path).
struct FlatRegistry {
    map: HashMap<(&'static str, &'static str), bool>,
}

impl FlatRegistry {
    fn new() -> Self {
        let mut map = HashMap::new();
        // Realistic entries: which (parent, child) pairs have reduce or execute rules
        let pairs: &[(&str, &str)] = &[
            // Chunked child has reduce rules for these parents:
            ("vortex.cast", "vortex.chunked"),
            ("vortex.fill_null", "vortex.chunked"),
            // Chunked child has execute kernels for these parents:
            ("vortex.filter", "vortex.chunked"),
            ("vortex.mask", "vortex.chunked"),
            ("vortex.slice", "vortex.chunked"),
            ("vortex.take", "vortex.chunked"),
            ("vortex.zip", "vortex.chunked"),
            // Primitive child rules:
            ("vortex.masked", "vortex.primitive"),
            ("vortex.mask", "vortex.primitive"),
            ("vortex.slice", "vortex.primitive"),
            ("vortex.cast", "vortex.primitive"),
            ("vortex.between", "vortex.primitive"),
            ("vortex.fill_null", "vortex.primitive"),
            ("vortex.take", "vortex.primitive"),
            ("vortex.binary", "vortex.primitive"),
            // chunk_offsets (also Primitive) — same entries apply
        ];
        for (p, c) in pairs {
            map.insert((*p, *c), true);
        }
        Self { map }
    }

    #[inline]
    fn has_rules(&self, parent_id: &str, child_id: &str) -> bool {
        self.map.contains_key(&(parent_id, child_id))
    }
}

/// Two-level registry: parent_id -> { child_id -> has_rules }
struct TwoLevelRegistry {
    map: HashMap<&'static str, HashMap<&'static str, bool>>,
}

impl TwoLevelRegistry {
    fn new() -> Self {
        let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();
        let entries: &[(&str, &str)] = &[
            ("vortex.cast", "vortex.chunked"),
            ("vortex.cast", "vortex.primitive"),
            ("vortex.fill_null", "vortex.chunked"),
            ("vortex.fill_null", "vortex.primitive"),
            ("vortex.filter", "vortex.chunked"),
            ("vortex.mask", "vortex.chunked"),
            ("vortex.mask", "vortex.primitive"),
            ("vortex.slice", "vortex.chunked"),
            ("vortex.slice", "vortex.primitive"),
            ("vortex.take", "vortex.chunked"),
            ("vortex.take", "vortex.primitive"),
            ("vortex.zip", "vortex.chunked"),
            ("vortex.masked", "vortex.primitive"),
            ("vortex.between", "vortex.primitive"),
            ("vortex.binary", "vortex.primitive"),
        ];
        for (p, c) in entries {
            map.entry(p).or_default().insert(c, true);
        }
        Self { map }
    }
}

const NCHUNKS: &[usize] = &[10, 100];

// ============================================================================
// CURRENT: the real reduce_parent + execute_parent child loop
// ============================================================================

/// Current approach: iterate all slots, call reduce_parent on each.
/// This does virtual dispatch → ParentRuleSet linear scan → matches() per rule.
#[divan::bench(args = NCHUNKS)]
fn current_reduce_parent_loop(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    bencher.bench(|| {
        let parent = black_box(&parent);
        for (slot_idx, slot) in parent.slots().iter().enumerate() {
            if let Some(child) = slot {
                black_box(child.reduce_parent(parent, slot_idx)).unwrap();
            }
        }
    });
}

/// Current approach: iterate all slots, call execute_parent on each.
#[divan::bench(args = NCHUNKS)]
fn current_execute_parent_loop(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    bencher.bench(|| {
        let parent = black_box(&parent);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for (slot_idx, slot) in parent.slots().iter().enumerate() {
            if let Some(child) = slot {
                black_box(child.execute_parent(parent, slot_idx, &mut ctx)).unwrap();
            }
        }
    });
}

// ============================================================================
// PROPOSED: HashMap lookup, no virtual dispatch for matching
// ============================================================================

/// Proposed: two-level HashMap lookup for reduce_parent.
/// Outer lookup by parent_id (one miss = skip everything).
/// Inner lookup by child_id per slot.
#[divan::bench(args = NCHUNKS)]
fn proposed_reduce_parent_loop(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    let registry = TwoLevelRegistry::new();
    bencher.bench(|| {
        let parent = black_box(&parent);
        let parent_id_owned = parent.encoding_id();
        let parent_id: &str = parent_id_owned.as_ref();
        if let Some(child_map) = registry.map.get(parent_id) {
            for slot in parent.slots().iter() {
                if let Some(child) = slot {
                    let child_id_owned = child.encoding_id();
                    let child_id: &str = child_id_owned.as_ref();
                    black_box(child_map.get(child_id));
                }
            }
        }
    });
}

/// Proposed: two-level HashMap lookup for execute_parent.
#[divan::bench(args = NCHUNKS)]
fn proposed_execute_parent_loop(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    let registry = TwoLevelRegistry::new();
    bencher.bench(|| {
        let parent = black_box(&parent);
        let parent_id_owned = parent.encoding_id();
        let parent_id: &str = parent_id_owned.as_ref();
        if let Some(child_map) = registry.map.get(parent_id) {
            for slot in parent.slots().iter() {
                if let Some(child) = slot {
                    let child_id_owned = child.encoding_id();
                    let child_id: &str = child_id_owned.as_ref();
                    black_box(child_map.get(child_id));
                }
            }
        }
    });
}
