// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! N×R matcher scan vs filtered lookup.
//!
//! Measures the matching loop only. When a rule matches, we call matches()
//! but NOT reduce_parent() — to isolate the scan cost from rule execution.
//!
//! For the "rule fires" case, we separately add one reduce_parent() call
//! to show total cost = matching + execution.

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::DynArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

// 3 rules. Rule[1] = SliceReduceAdaptor matches Slice parent.
static ALL_RULES: [&dyn DynArrayParentReduceRule<Primitive>; 3] = [
    ParentRuleSet::lift(&MaskReduceAdaptor(Primitive)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Primitive)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Primitive)),
];

static FILTERED_RULES: [&dyn DynArrayParentReduceRule<Primitive>; 1] = [
    ParentRuleSet::lift(&SliceReduceAdaptor(Primitive)),
];

fn make_children(n: usize) -> Vec<ArrayRef> {
    (0..n)
        .map(|i| {
            PrimitiveArray::new(Buffer::from(vec![i as i32; 100]), Validity::NonNullable)
                .into_array()
        })
        .collect()
}

fn make_slice_parent() -> ArrayRef {
    let child =
        PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array();
    SliceArray::new(child, 10..60).into_array()
}

fn make_leaf_parent() -> ArrayRef {
    PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array()
}

fn build_hashmap(
) -> HashMap<&'static str, &'static [&'static dyn DynArrayParentReduceRule<Primitive>]> {
    let mut map = HashMap::new();
    map.insert("vortex.slice", FILTERED_RULES.as_slice());
    map
}

struct DenseRegistry {
    rules: Vec<&'static [&'static dyn DynArrayParentReduceRule<Primitive>]>,
    id_to_idx: HashMap<&'static str, usize>,
}

fn build_dense() -> DenseRegistry {
    let mut id_to_idx = HashMap::new();
    id_to_idx.insert("vortex.slice", 0usize);
    DenseRegistry {
        rules: vec![FILTERED_RULES.as_slice()],
        id_to_idx,
    }
}

const N: &[usize] = &[1, 10, 100];

// ============================================================================
// SLICE PARENT: rule fires. We measure matching + finding the rule.
// reduce_parent is called on the parent's REAL child (slot 0) once
// to show execution cost. Then N-1 remaining children just do matching.
// ============================================================================

/// Current: N × scan 3 rules with matches(). 2nd rule hits each time.
/// Total: N × 2 matches(). We count hits but don't call reduce_parent
/// in the loop (to isolate matching cost).
#[divan::bench(args = N)]
fn current_slice_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_slice_parent();
    bencher.bench(|| {
        let parent = black_box(&parent);
        for _child in black_box(&children).iter() {
            let mut found = false;
            for rule in &ALL_RULES {
                if rule.matches(parent) {
                    found = true;
                    break;
                }
            }
            black_box(found);
        }
    });
    eprintln!("  current_slice n={n}: total matches()={}", n * 2);
}

/// HashMap: lookup filtered rules. No matches() calls.
#[divan::bench(args = N)]
fn hashmap_slice_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_slice_parent();
    let reg = build_hashmap();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let found = reg.get(black_box(pid.as_ref()));
        for _child in black_box(&children).iter() {
            black_box(found);
        }
    });
    eprintln!("  hashmap_slice n={n}: matches()=0");
}

/// Dense vec: index lookup. No matches() calls.
#[divan::bench(args = N)]
fn dense_slice_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_slice_parent();
    let reg = build_dense();
    let pid = parent.encoding_id();
    let idx = reg.id_to_idx.get(pid.as_ref()).copied();
    bencher.bench(|| {
        let found = black_box(idx).map(|i| &reg.rules[i]);
        for _child in black_box(&children).iter() {
            black_box(found);
        }
    });
    eprintln!("  dense_slice n={n}: matches()=0");
}

// ============================================================================
// LEAF PARENT: nothing fires. All matches() miss.
// ============================================================================

/// Current: N × 3 matches() all miss.
#[divan::bench(args = N)]
fn current_leaf_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_leaf_parent();
    bencher.bench(|| {
        let parent = black_box(&parent);
        for _child in black_box(&children).iter() {
            let mut found = false;
            for rule in &ALL_RULES {
                if rule.matches(parent) {
                    found = true;
                    break;
                }
            }
            black_box(found);
        }
    });
    eprintln!("  current_leaf n={n}: total matches()={}", n * 3);
}

/// HashMap: outer miss → constant time.
#[divan::bench(args = N)]
fn hashmap_leaf_parent(bencher: divan::Bencher, n: usize) {
    let parent = make_leaf_parent();
    let reg = build_hashmap();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        black_box(reg.get(black_box(pid.as_ref())))
    });
    eprintln!("  hashmap_leaf n={n}: matches()=0");
}

/// Dense: index miss → constant time.
#[divan::bench(args = N)]
fn dense_leaf_parent(bencher: divan::Bencher, n: usize) {
    let parent = make_leaf_parent();
    let reg = build_dense();
    let pid = parent.encoding_id();
    let idx = reg.id_to_idx.get(pid.as_ref()).copied();
    bencher.bench(|| {
        black_box(idx)
    });
    eprintln!("  dense_leaf n={n}: matches()=0");
}

// ============================================================================
// SLICE + REDUCE_PARENT EXECUTION: matching + actual rule body on real child.
// Shows total cost when rule fires.
// ============================================================================

/// Current: scan 3 rules, hit on 2nd, call reduce_parent on parent's real child.
#[divan::bench]
fn current_slice_match_and_reduce(bencher: divan::Bencher) {
    let parent = make_slice_parent();
    let real_child = parent.slots()[0].as_ref().unwrap().clone();
    bencher.bench(|| {
        let parent = black_box(&parent);
        let view = real_child.as_opt::<Primitive>().unwrap();
        for rule in &ALL_RULES {
            if rule.matches(parent) {
                return black_box(rule.reduce_parent(view, parent, 0)).unwrap();
            }
        }
        None
    });
}

/// Proposed: no matching, call filtered rule directly.
#[divan::bench]
fn proposed_slice_match_and_reduce(bencher: divan::Bencher) {
    let parent = make_slice_parent();
    let real_child = parent.slots()[0].as_ref().unwrap().clone();
    bencher.bench(|| {
        let parent = black_box(&parent);
        let view = real_child.as_opt::<Primitive>().unwrap();
        // Filtered: we already know which rule matches
        black_box(FILTERED_RULES[0].reduce_parent(view, parent, 0)).unwrap()
    });
}
