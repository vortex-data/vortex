// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Realistic benchmark: Cast(Chunked(N × Dict))
//!
//! After ChunkedUnaryScalarFnPushDownRule pushes Cast into chunks,
//! each chunk is Cast(Dict). Dict has 7 reduce rules. The optimizer
//! scans all 7 per Dict child.
//!
//! Current:  N × 7 × rule.matches(parent)  +  matching rules fire
//! Proposed: N × 1 lookup  +  only matching rules fire

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
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
use vortex_array::scalar_fn::fns::like::LikeReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

// Dict's 7 reduce rules (2 AnyScalarFn ones are pub(crate), so we sub with
// more public adaptors — same matching cost per rule).
static DICT_RULES: [&dyn DynArrayParentReduceRule<Dict>; 7] = [
    ParentRuleSet::lift(&FilterReduceAdaptor(Dict)),   // matches Filter
    ParentRuleSet::lift(&CastReduceAdaptor(Dict)),     // matches ExactScalarFn<Cast> ← HITS
    ParentRuleSet::lift(&MaskReduceAdaptor(Dict)),     // matches ExactScalarFn<Mask>
    ParentRuleSet::lift(&LikeReduceAdaptor(Dict)),     // matches ExactScalarFn<Like>
    ParentRuleSet::lift(&MaskReduceAdaptor(Dict)),     // stand-in for AnyScalarFn rule
    ParentRuleSet::lift(&SliceReduceAdaptor(Dict)),    // matches Slice
    ParentRuleSet::lift(&FilterReduceAdaptor(Dict)),   // stand-in for AnyScalarFn rule
];

// Pre-filtered: only the rule that matches Cast parent.
static DICT_CAST_RULES: [&dyn DynArrayParentReduceRule<Dict>; 1] = [
    ParentRuleSet::lift(&CastReduceAdaptor(Dict)),
];

fn make_dict_children(n: usize) -> Vec<ArrayRef> {
    (0..n)
        .map(|_| {
            let codes = PrimitiveArray::new(
                Buffer::from(vec![0u8, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
                Validity::NonNullable,
            )
            .into_array();
            let values = PrimitiveArray::new(
                Buffer::from(vec![100i32, 200, 300]),
                Validity::NonNullable,
            )
            .into_array();
            DictArray::try_new(codes, values).unwrap().into_array()
        })
        .collect()
}

fn make_cast_parent() -> ArrayRef {
    let child = PrimitiveArray::new(
        Buffer::from(vec![1i32; 10]),
        Validity::NonNullable,
    )
    .into_array();
    Cast.try_new_array(
        10,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        [child],
    )
    .unwrap()
}

fn make_leaf_parent() -> ArrayRef {
    PrimitiveArray::new(Buffer::from(vec![1i32; 10]), Validity::NonNullable).into_array()
}

fn build_hashmap(
) -> HashMap<&'static str, &'static [&'static dyn DynArrayParentReduceRule<Dict>]> {
    let mut map = HashMap::new();
    map.insert("vortex.cast", DICT_CAST_RULES.as_slice());
    map
}

const N: &[usize] = &[1, 10, 100];

// ============================================================================
// MATCHING ONLY: Cast parent, Dict children, 7 rules
// ============================================================================

/// Current: N × scan 7 rules. Cast hits on 2nd → 2 matches() per child.
#[divan::bench(args = N)]
fn dict_current_cast(bencher: divan::Bencher, n: usize) {
    let children = make_dict_children(n);
    let parent = make_cast_parent();
    bencher.bench(|| {
        let parent = black_box(&parent);
        for _child in black_box(&children).iter() {
            for rule in &DICT_RULES {
                if rule.matches(parent) {
                    break;
                }
            }
        }
    });
    eprintln!("  dict_current_cast n={n}: matches={}", n * 2);
}

/// Current: N × scan 7 rules, leaf parent → all 7 miss.
#[divan::bench(args = N)]
fn dict_current_leaf(bencher: divan::Bencher, n: usize) {
    let children = make_dict_children(n);
    let parent = make_leaf_parent();
    bencher.bench(|| {
        let parent = black_box(&parent);
        for _child in black_box(&children).iter() {
            for rule in &DICT_RULES {
                if rule.matches(parent) {
                    break;
                }
            }
        }
    });
    eprintln!("  dict_current_leaf n={n}: matches={}", n * 7);
}

/// Proposed HashMap: 1 lookup → pre-filtered rules. 0 matches().
#[divan::bench(args = N)]
fn dict_hashmap_cast(bencher: divan::Bencher, n: usize) {
    let children = make_dict_children(n);
    let parent = make_cast_parent();
    let reg = build_hashmap();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let found = reg.get(black_box(pid.as_ref()));
        for _child in black_box(&children).iter() {
            black_box(found);
        }
    });
    eprintln!("  dict_hashmap_cast n={n}: matches=0");
}

/// Proposed HashMap: leaf miss → 0 work.
#[divan::bench(args = N)]
fn dict_hashmap_leaf(bencher: divan::Bencher, n: usize) {
    let parent = make_leaf_parent();
    let reg = build_hashmap();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        black_box(reg.get(black_box(pid.as_ref())))
    });
    eprintln!("  dict_hashmap_leaf n={n}: matches=0");
}

// ============================================================================
// MATCHING + REDUCE_PARENT: Dict child, Cast parent, rule fires
// ============================================================================

/// Current: scan 7 rules, hit on 2nd, call reduce_parent on real Dict child.
#[divan::bench]
fn dict_match_and_reduce_current(bencher: divan::Bencher) {
    let parent = make_cast_parent();
    let dict_child = make_dict_children(1).pop().unwrap();
    bencher.bench(|| {
        let parent = black_box(&parent);
        let view = dict_child.as_opt::<Dict>().unwrap();
        for rule in &DICT_RULES {
            if rule.matches(parent) {
                return black_box(rule.reduce_parent(view, parent, 0)).unwrap();
            }
        }
        None
    });
    eprintln!("  dict_match_and_reduce_current: matches=2 (1 miss + 1 hit)");
}

/// Proposed: skip scan, call pre-filtered rule directly.
#[divan::bench]
fn dict_match_and_reduce_proposed(bencher: divan::Bencher) {
    let parent = make_cast_parent();
    let dict_child = make_dict_children(1).pop().unwrap();
    bencher.bench(|| {
        let parent = black_box(&parent);
        let view = dict_child.as_opt::<Dict>().unwrap();
        black_box(DICT_CAST_RULES[0].reduce_parent(view, parent, 0)).unwrap()
    });
    eprintln!("  dict_match_and_reduce_proposed: matches=0");
}
