// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Precise benchmark: matching overhead for Cast(Chunked(N × Primitive)).
//!
//! The Cast ScalarFn parent has N+1 child slots (chunk_offsets + N chunks).
//! For each child slot, the current code does:
//!   reduce_parent: vtable dispatch → 3 matches() calls (Masked, Mask, Slice) → all miss
//!   execute_parent: vtable dispatch → 4 matches() calls → Cast hits → runs kernel
//!
//! We measure two things separately:
//! 1. "match_only": just the matching/dispatch overhead (no rule fires)
//!    - For reduce_parent: parent is Cast, child is Primitive → 3 matches(), all miss
//!    - For the proposed: HashMap lookup → miss or hit, no rule execution
//! 2. "match_and_execute": matching + the rule that fires
//!    - For execute_parent: CastExecuteAdaptor matches and runs

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

/// Two-level registry keyed by &str (no allocation on lookup).
fn build_registry() -> HashMap<&'static str, HashMap<&'static str, bool>> {
    let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();
    // Realistic: parent_ids that have rules, and which child_ids match
    for (p, c) in [
        ("vortex.cast", "vortex.chunked"),
        ("vortex.cast", "vortex.primitive"),
        ("vortex.fill_null", "vortex.chunked"),
        ("vortex.fill_null", "vortex.primitive"),
        ("vortex.filter", "vortex.chunked"),
        ("vortex.filter", "vortex.primitive"),
        ("vortex.mask", "vortex.chunked"),
        ("vortex.mask", "vortex.primitive"),
        ("vortex.slice", "vortex.chunked"),
        ("vortex.slice", "vortex.primitive"),
        ("vortex.masked", "vortex.primitive"),
        ("vortex.between", "vortex.primitive"),
        ("vortex.binary", "vortex.primitive"),
        ("vortex.take", "vortex.primitive"),
        ("vortex.take", "vortex.chunked"),
        ("vortex.zip", "vortex.chunked"),
    ] {
        map.entry(p).or_default().insert(c, true);
    }
    map
}

const NCHUNKS: &[usize] = &[1, 10, 100];

// ============================================================================
// MATCH ONLY: reduce_parent where NO rule fires (all matches() miss)
//
// Current: for each of N+1 slots:
//   - virtual dispatch to child VTable
//   - ArrayView construction
//   - iterate 3 rules, call matches() on each (all return false)
//
// Proposed: for each of N+1 slots:
//   - read child.encoding_id() (cheap)
//   - HashMap.get(child_id) — hit, but no rule execution
// ============================================================================

#[divan::bench(args = NCHUNKS)]
fn match_only_current_reduce_parent(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    // reduce_parent on Primitive child with Cast parent:
    // Primitive has 3 reduce rules (Masked, Mask, Slice). None match Cast.
    // So this measures: N+1 × (vtable dispatch + 3 × matches() miss)
    bencher.bench(|| {
        let parent = black_box(&parent);
        for (slot_idx, slot) in parent.slots().iter().enumerate() {
            if let Some(child) = slot {
                black_box(child.reduce_parent(parent, slot_idx)).unwrap();
            }
        }
    });
}

#[divan::bench(args = NCHUNKS)]
fn match_only_proposed_reduce_parent(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    let registry = build_registry();
    // Proposed: look up (parent_id, child_id) in HashMap.
    // parent_id = "vortex.cast" → outer hit.
    // child_id = "vortex.primitive" → inner hit (but we just look up, don't execute).
    // child_id = "vortex.chunked" → inner hit.
    // This measures: encoding_id() calls + HashMap lookups.
    bencher.bench(|| {
        let parent = black_box(&parent);
        let pid = parent.encoding_id();
        if let Some(child_map) = registry.get(pid.as_ref()) {
            for slot in parent.slots().iter() {
                if let Some(child) = slot {
                    let cid = child.encoding_id();
                    black_box(child_map.get(cid.as_ref()));
                }
            }
        }
    });
}

// ============================================================================
// MATCH + EXECUTE: execute_parent where CastExecuteAdaptor fires
//
// Current: for each of N+1 slots:
//   - virtual dispatch to child VTable
//   - ArrayView construction
//   - iterate 4 kernels: Between(miss), Cast(hit → runs kernel), ...
//   - first child that returns Some wins
//
// Proposed: for each of N+1 slots:
//   - HashMap lookup → get rule list
//   - run matched rules only (no scanning non-matching rules)
//   - still need to call the actual rule function
//
// NOTE: execute_parent actually EXECUTES the cast for Primitive children,
// producing a real result. We include this real compute to show total cost.
// ============================================================================

#[divan::bench(args = NCHUNKS)]
fn match_and_exec_current_execute_parent(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    // execute_parent on Primitive child with Cast parent:
    // Primitive has 4 execute kernels (Between, Cast, FillNull, Take).
    // Between misses, Cast hits and runs the kernel.
    // The Chunked child at slot 0 also has kernels but Cast is not one of them
    // (Chunked kernels: Filter, Mask, Slice, Take, Zip) → all miss.
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

#[divan::bench(args = NCHUNKS)]
fn match_and_exec_proposed_execute_parent(bencher: divan::Bencher, nchunks: usize) {
    let parent = make_cast_of_chunked(nchunks);
    let registry = build_registry();
    // Proposed: HashMap lookup finds the rules, then we still call execute_parent
    // but ONLY on children that have matching rules.
    // This avoids the vtable dispatch + matches() scan for non-matching children.
    bencher.bench(|| {
        let parent = black_box(&parent);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let pid = parent.encoding_id();
        if let Some(child_map) = registry.get(pid.as_ref()) {
            for (slot_idx, slot) in parent.slots().iter().enumerate() {
                if let Some(child) = slot {
                    let cid = child.encoding_id();
                    if child_map.get(cid.as_ref()).is_some() {
                        // Rule exists — call the real execute_parent
                        black_box(child.execute_parent(parent, slot_idx, &mut ctx)).unwrap();
                    }
                }
            }
        }
    });
}
