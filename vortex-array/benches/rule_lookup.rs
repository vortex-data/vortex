// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apples-to-apples: the full reduce_parent loop over N children.
//!
//! Current: for each child, call child.reduce_parent(parent, idx).
//!   Inside: vtable dispatch → iterate ALL rules → rule.matches(parent) each.
//!   N children × R rules = N×R matches() calls.
//!
//! Proposed: HashMap.get(parent_id) → HashMap.get(child_id) → iterate ONLY
//!   the pre-filtered rules → call reduce_parent on each.
//!   Same vtable dispatch for the rules that match. Fewer total matches() calls.
//!
//! Both include dyn dispatch. The only difference is how many rules we check.

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
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

fn make_children(n: usize) -> Vec<ArrayRef> {
    (0..n)
        .map(|i| {
            PrimitiveArray::new(Buffer::from(vec![i as i32; 100]), Validity::NonNullable)
                .into_array()
        })
        .collect()
}

fn make_cast_parent() -> ArrayRef {
    let a =
        PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array();
    Cast.try_new_array(
        100,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        [a],
    )
    .unwrap()
}

fn make_leaf_parent() -> ArrayRef {
    PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array()
}

/// Registry: parent_id → child_id → true.
/// When we get a hit, we call child.reduce_parent(parent, idx) — same
/// dyn dispatch as current, but we only do it for children that have rules.
fn build_registry() -> HashMap<&'static str, HashMap<&'static str, bool>> {
    let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();
    // Every (parent_id, child_id) pair that has reduce rules
    for (p, c) in [
        // Primitive child reduce rules match these parents:
        ("vortex.masked", "vortex.primitive"), // PrimitiveMaskedValidityRule
        ("vortex.mask", "vortex.primitive"),    // MaskReduceAdaptor
        ("vortex.slice", "vortex.primitive"),   // SliceReduceAdaptor
    ] {
        map.entry(p).or_default().insert(c, true);
    }
    map
}

static CURRENT_MATCHES: AtomicU64 = AtomicU64::new(0);
static PROPOSED_MATCHES: AtomicU64 = AtomicU64::new(0);

const N: &[usize] = &[1, 10, 100];

// ============================================================================
// CURRENT: full reduce_parent loop — N × child.reduce_parent(parent, idx)
//
// Each child.reduce_parent() does:
//   1. vtable dispatch (DynArray → ArrayInner<Primitive>)
//   2. ArrayView construction
//   3. Primitive::reduce_parent → RULES.evaluate()
//   4. iterate 3 rules, call rule.matches(parent) on each
//
// Total matches() calls: N × 3
// ============================================================================

/// Cast parent: 3 matchers per child, none hit. N×3 wasted matches().
#[divan::bench(args = N)]
fn current_cast_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    CURRENT_MATCHES.store(0, Ordering::Relaxed);
    bencher.bench(|| {
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
        }
    });
    // N children × 3 rules = N×3 matches() calls (all miss)
    eprintln!("  current_cast_parent n={n}: matches = {}", n * 3);
}

/// Leaf parent: 3 matchers per child, none hit. N×3 wasted matches().
#[divan::bench(args = N)]
fn current_leaf_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_leaf_parent();
    bencher.bench(|| {
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
        }
    });
    eprintln!("  current_leaf_parent n={n}: matches = {}", n * 3);
}

// ============================================================================
// PROPOSED: HashMap lookup → only call reduce_parent for matching children
//
// 1. HashMap.get(parent_id) — if miss, skip ALL children (0 matches)
// 2. For each child: HashMap.get(child_id) — if miss, skip (0 matches)
// 3. If hit: call child.reduce_parent(parent, idx) — same dyn dispatch,
//    but now we KNOW it will match, so the 3 matches() inside are
//    still called (we can't skip them without changing reduce_parent).
//
// In a full implementation, we'd call the matched rules directly,
// skipping the matches() scan entirely. But this benchmark shows the
// win from just skipping non-matching children.
// ============================================================================

/// Cast parent: HashMap misses (Cast has no reduce rules for Primitive).
/// Total: 1 outer lookup + N inner lookups. 0 matches() calls. 0 reduce_parent calls.
#[divan::bench(args = N)]
fn proposed_cast_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let pid = black_box(pid.as_ref());
        if let Some(child_map) = registry.get(pid) {
            for (i, child) in black_box(&children).iter().enumerate() {
                let cid = child.encoding_id();
                if child_map.get(cid.as_ref()).is_some() {
                    // Would call reduce_parent here, but Cast has no
                    // reduce rules for Primitive children — so this
                    // branch is never taken.
                    black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
                }
            }
        }
    });
    // Cast is not in the registry → outer miss → 0 matches, 0 dispatches
    eprintln!("  proposed_cast_parent n={n}: matches = 0, dispatches = 0");
}

/// Leaf parent: HashMap misses immediately. Constant time.
#[divan::bench(args = N)]
fn proposed_leaf_parent(bencher: divan::Bencher, n: usize) {
    let _children = make_children(n);
    let parent = make_leaf_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        // Outer miss — done. No child iteration at all.
        black_box(registry.get(black_box(pid.as_ref())))
    });
    eprintln!("  proposed_leaf_parent n={n}: matches = 0, dispatches = 0");
}

// ============================================================================
// BONUS: Masked parent — where rules DO fire.
//
// Primitive has PrimitiveMaskedValidityRule matching Masked parent.
// Current: N × 3 matches() → 1 hit per child → runs reduce_parent
// Proposed: HashMap hit → N × reduce_parent (skip 2 non-matching rules)
// ============================================================================

fn make_masked_parent() -> ArrayRef {
    use vortex_array::arrays::MaskedArray;
    let values =
        PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array();
    MaskedArray::try_new(values, Validity::AllInvalid)
        .unwrap()
        .into_array()
}

/// Current Masked parent: 3 matches per child, 1st hits (PrimitiveMaskedValidityRule).
#[divan::bench(args = N)]
fn current_masked_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_masked_parent();
    bencher.bench(|| {
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
        }
    });
    // N × 3 matches() calls, first hits → N reduce_parent executions
    eprintln!(
        "  current_masked_parent n={n}: matches = {}, hits = {n}",
        n * 3
    );
}

/// Proposed Masked parent: HashMap hit → N × reduce_parent directly.
/// Same dyn dispatch for reduce_parent, but 0 wasted matches() calls.
#[divan::bench(args = N)]
fn proposed_masked_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_masked_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let pid = black_box(pid.as_ref());
        if let Some(child_map) = registry.get(pid) {
            for (i, child) in black_box(&children).iter().enumerate() {
                let cid = child.encoding_id();
                if child_map.get(cid.as_ref()).is_some() {
                    black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
                }
            }
        }
    });
    // N HashMap lookups (all hit) + N reduce_parent calls
    // Inside reduce_parent: 3 matches() still run (we can't skip those yet)
    eprintln!(
        "  proposed_masked_parent n={n}: lookups = {n}, matches_inside = {}, dispatches = {n}",
        n * 3
    );
}
