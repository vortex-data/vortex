// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark: current linear-scan matches() vs HashMap (parent_id, child_id) lookup.
//!
//! What we measure precisely:
//! - "current": the real child.reduce_parent(parent, idx) call, which does
//!   virtual dispatch → iterate rules → rule.matches(parent) via as_any().is::<T>()
//! - "proposed": HashMap::get with (parent_encoding_id, child_encoding_id) key
//!
//! We do NOT measure rule execution (reduce_parent body). Only the matching/lookup.

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

fn make_binary_expr() -> ArrayRef {
    let a = PrimitiveArray::new(Buffer::from(vec![1i32; 1024]), Validity::NonNullable).into_array();
    let b = PrimitiveArray::new(Buffer::from(vec![2i32; 1024]), Validity::NonNullable).into_array();
    Binary
        .try_new_array(1024, Operator::Lt, [a, b])
        .unwrap()
}

fn make_cast_expr() -> ArrayRef {
    let a = PrimitiveArray::new(Buffer::from(vec![1i32; 1024]), Validity::NonNullable).into_array();
    Cast.try_new_array(
        1024,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        [a],
    )
    .unwrap()
}

fn make_leaf() -> ArrayRef {
    PrimitiveArray::new(Buffer::from(vec![1i32; 1024]), Validity::NonNullable).into_array()
}

fn build_registry() -> HashMap<(String, String), bool> {
    let mut map = HashMap::new();
    let child_ids = [
        "vortex.primitive", "vortex.bool", "vortex.dict", "vortex.chunked",
        "vortex.constant", "vortex.varbin", "vortex.varbinview", "vortex.struct",
        "vortex.extension", "vortex.null", "vortex.masked", "vortex.list",
        "vortex.listview", "vortex.fixed_size_list", "vortex.decimal",
    ];
    let parent_ids = [
        "vortex.cast", "vortex.binary", "vortex.between", "vortex.mask",
        "vortex.fill_null", "vortex.not", "vortex.like", "vortex.filter",
        "vortex.slice", "vortex.masked",
    ];
    for p in &parent_ids {
        for c in &child_ids {
            map.insert((p.to_string(), c.to_string()), true);
        }
    }
    map
}

fn build_two_level() -> HashMap<String, HashMap<String, bool>> {
    let mut map: HashMap<String, HashMap<String, bool>> = HashMap::new();
    let child_ids = [
        "vortex.primitive", "vortex.bool", "vortex.dict", "vortex.chunked", "vortex.constant",
    ];
    let parent_ids = [
        "vortex.cast", "vortex.binary", "vortex.between", "vortex.mask",
        "vortex.filter", "vortex.slice", "vortex.masked",
    ];
    for p in &parent_ids {
        let inner = map.entry(p.to_string()).or_default();
        for c in &child_ids {
            inner.insert(c.to_string(), true);
        }
    }
    map
}

// ============================================================================
// CURRENT: real reduce_parent dispatch (virtual dispatch + linear matches())
// ============================================================================

/// Leaf parent (Primitive). Child is Primitive with ~5 rules, all miss.
/// This is the common case: parent is not interesting.
#[divan::bench]
fn current_leaf_parent_one_child(bencher: divan::Bencher) {
    let parent = make_leaf();
    let child = make_leaf();
    bencher.bench(|| {
        black_box(child.reduce_parent(black_box(&parent), 0)).unwrap()
    });
}

/// Cast parent. Child is Primitive — CastReduceAdaptor matches.
#[divan::bench]
fn current_cast_parent_one_child(bencher: divan::Bencher) {
    let parent = make_cast_expr();
    let child = make_leaf();
    bencher.bench(|| {
        black_box(child.reduce_parent(black_box(&parent), 0)).unwrap()
    });
}

/// Binary parent. Iterate both Primitive children.
#[divan::bench]
fn current_binary_parent_two_children(bencher: divan::Bencher) {
    let parent = make_binary_expr();
    bencher.bench(|| {
        for (i, slot) in black_box(&parent).slots().iter().enumerate() {
            if let Some(child) = slot {
                black_box(child.reduce_parent(&parent, i)).unwrap();
            }
        }
    });
}

// ============================================================================
// PROPOSED: single HashMap<(parent_id, child_id), _> lookup
// ============================================================================

/// Leaf parent — HashMap miss.
#[divan::bench]
fn proposed_flat_leaf_parent_one_child(bencher: divan::Bencher) {
    let registry = build_registry();
    let parent = make_leaf();
    let child = make_leaf();
    // Pre-extract the IDs (this is what we'd cache on the array)
    let pid = parent.encoding_id().to_string();
    let cid = child.encoding_id().to_string();
    bencher.bench(|| {
        black_box(registry.get(&(black_box(&pid).clone(), black_box(&cid).clone())))
    });
}

/// Cast parent — HashMap hit.
#[divan::bench]
fn proposed_flat_cast_parent_one_child(bencher: divan::Bencher) {
    let registry = build_registry();
    let parent = make_cast_expr();
    let child = make_leaf();
    let pid = parent.encoding_id().to_string();
    let cid = child.encoding_id().to_string();
    bencher.bench(|| {
        black_box(registry.get(&(black_box(&pid).clone(), black_box(&cid).clone())))
    });
}

/// Binary parent — two lookups.
#[divan::bench]
fn proposed_flat_binary_parent_two_children(bencher: divan::Bencher) {
    let registry = build_registry();
    let parent = make_binary_expr();
    let children: Vec<_> = parent
        .slots()
        .iter()
        .filter_map(|s| s.as_ref())
        .map(|c| c.encoding_id().to_string())
        .collect();
    let pid = parent.encoding_id().to_string();
    bencher.bench(|| {
        let pid = black_box(&pid);
        for cid in black_box(&children) {
            black_box(registry.get(&(pid.clone(), cid.clone())));
        }
    });
}

// ============================================================================
// PROPOSED: two-level HashMap<parent_id, HashMap<child_id, _>>
// ============================================================================

/// Leaf parent — outer miss, no inner lookup.
#[divan::bench]
fn proposed_two_level_leaf_parent(bencher: divan::Bencher) {
    let registry = build_two_level();
    let parent = make_leaf();
    let pid = parent.encoding_id().to_string();
    bencher.bench(|| {
        black_box(registry.get(black_box(&pid)))
    });
}

/// Binary parent — outer hit + two inner lookups.
#[divan::bench]
fn proposed_two_level_binary_parent_two_children(bencher: divan::Bencher) {
    let registry = build_two_level();
    let parent = make_binary_expr();
    let children: Vec<_> = parent
        .slots()
        .iter()
        .filter_map(|s| s.as_ref())
        .map(|c| c.encoding_id().to_string())
        .collect();
    let pid = parent.encoding_id().to_string();
    bencher.bench(|| {
        if let Some(child_map) = registry.get(black_box(&pid)) {
            for cid in black_box(&children) {
                black_box(child_map.get(cid));
            }
        }
    });
}
