// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures precisely:
//!
//! reduce (match only, nothing fires):
//!   current:  N × 4 × rule.matches(parent)   — each is as_any().is::<>()
//!   proposed: 1 × HashMap.get(parent_id) + N × HashMap.get(child_id)
//!
//! execute (match scan + kernel runs):
//!   current:  N × child.execute_parent(parent) — matches scan + kernel
//!   proposed: N × (HashMap.get + child.execute_parent) — skip scan

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::rules::DynArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

// 4 matchers — same public adaptors Bool uses (minus pub(crate) BoolMaskedValidityRule).
// Parent types matched: ExactScalarFn<Cast>, ExactScalarFn<Mask>, Slice, Filter.
// For a Cast parent: CastReduceAdaptor hits, others miss.
// For a Primitive parent: ALL miss.
static RULES: [&dyn DynArrayParentReduceRule<Bool>; 4] = [
    ParentRuleSet::lift(&CastReduceAdaptor(Bool)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Bool)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Bool)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Bool)),
];

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

fn build_registry() -> HashMap<&'static str, HashMap<&'static str, bool>> {
    let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();
    for (p, c) in [
        ("vortex.cast", "vortex.primitive"),
        ("vortex.cast", "vortex.bool"),
        ("vortex.mask", "vortex.primitive"),
        ("vortex.mask", "vortex.bool"),
        ("vortex.slice", "vortex.primitive"),
        ("vortex.slice", "vortex.bool"),
        ("vortex.filter", "vortex.primitive"),
        ("vortex.filter", "vortex.bool"),
        ("vortex.masked", "vortex.primitive"),
        ("vortex.between", "vortex.primitive"),
        ("vortex.binary", "vortex.primitive"),
    ] {
        map.entry(p).or_default().insert(c, true);
    }
    map
}

const N: &[usize] = &[1, 10, 100];

// ============================================================================
// REDUCE MATCH ONLY: Cast parent, all 4 matchers miss except Cast
// ============================================================================

/// N × 4 × rule.matches(parent). Pure matcher cost.
/// Cast parent: CastReduceAdaptor matches (ExactScalarFn<Cast>), others miss.
/// Each matches() = vtable call on DynArrayParentReduceRule → as_any().is::<>().
#[divan::bench(args = N)]
fn reduce_matcher_scan_cast_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    bencher.bench(|| {
        for _child in black_box(&children).iter() {
            for rule in &RULES {
                black_box(rule.matches(black_box(&parent)));
            }
        }
    });
}

/// N × 4 × rule.matches(parent). Leaf parent: ALL miss.
#[divan::bench(args = N)]
fn reduce_matcher_scan_leaf_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_leaf_parent();
    bencher.bench(|| {
        for _child in black_box(&children).iter() {
            for rule in &RULES {
                black_box(rule.matches(black_box(&parent)));
            }
        }
    });
}

/// Proposed: 1 outer HashMap.get(parent_id) + N inner HashMap.get(child_id).
/// Cast parent: outer hits, N inner lookups.
#[divan::bench(args = N)]
fn reduce_hashmap_lookup_cast_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let child_map = registry.get(black_box(pid.as_ref()));
        for child in black_box(&children).iter() {
            let cid = child.encoding_id();
            if let Some(cm) = child_map {
                black_box(cm.get(cid.as_ref()));
            }
        }
    });
}

/// Proposed: 1 outer HashMap.get(parent_id) misses → skip all children.
/// Leaf parent: constant time regardless of N.
#[divan::bench(args = N)]
fn reduce_hashmap_lookup_leaf_parent(bencher: divan::Bencher, n: usize) {
    let _children = make_children(n);
    let parent = make_leaf_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        // Outer miss = done. No child iteration.
        black_box(registry.get(black_box(pid.as_ref())))
    });
}

// ============================================================================
// EXECUTE MATCH + KERNEL: Cast parent, kernel fires on every child
//
// Current: N × child.execute_parent() — includes matches() scan + kernel
// Proposed: N × (HashMap.get + child.execute_parent()) — HashMap before dispatch
// The kernel cost dominates so we expect similar numbers.
// ============================================================================

/// Current: N × child.execute_parent(parent, i, ctx).
/// Includes vtable dispatch → 4 kernel.matches() → Cast hits → runs kernel.
#[divan::bench(args = N)]
fn execute_current(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    bencher.bench(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.execute_parent(black_box(&parent), i, &mut ctx)).unwrap();
        }
    });
}

/// Proposed: HashMap.get(child_id) → if hit, call child.execute_parent().
/// In reality we'd call the kernel directly; here we still go through dispatch
/// to show the best-case overhead of the HashMap gate.
#[divan::bench(args = N)]
fn execute_proposed(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let child_map = registry.get(black_box(pid.as_ref()));
        for (i, child) in black_box(&children).iter().enumerate() {
            let cid = child.encoding_id();
            if let Some(cm) = child_map {
                if cm.get(cid.as_ref()).is_some() {
                    black_box(child.execute_parent(black_box(&parent), i, &mut ctx)).unwrap();
                }
            }
        }
    });
}
