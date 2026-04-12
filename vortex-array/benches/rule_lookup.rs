// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Complete matching model: extracts matching logic, runs no transformations.
//!
//! For each child encoding, defines the list of matchers (parent_id strings).
//! For the proposed approach, builds a HashMap<(parent_id, child_id), bool>.
//!
//! Walks several realistic array trees and counts:
//!   - matches() calls (current approach: linear scan)
//!   - HashMap lookups (proposed approach)
//! and measures the time for each.
//!
//! No rule body executes — we only measure the FIND-MATCHING-RULE step.

#![expect(clippy::unwrap_used)]

use std::any::TypeId;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Null;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::ScalarFnVTable;
use vortex_array::matcher::Matcher as VortexMatcher;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

// ============================================================================
// REAL MATCHERS — use as_any().is::<>() via VortexMatcher::matches
// ============================================================================

/// A matcher predicate using REAL TypeId comparison (as_any().is::<>()).
/// This is what the actual code does in rule.matches(parent).
type MatchFn = fn(&ArrayRef) -> bool;

// Each matcher is a fn pointer that calls the real Matcher trait.
// fn pointer call ≈ vtable call ≈ what the real `dyn Rule.matches()` does.
fn match_masked(p: &ArrayRef) -> bool {
    <Masked as VortexMatcher>::matches(p)
}
fn match_slice(p: &ArrayRef) -> bool {
    <Slice as VortexMatcher>::matches(p)
}
fn match_filter(p: &ArrayRef) -> bool {
    <Filter as VortexMatcher>::matches(p)
}
fn match_dict(p: &ArrayRef) -> bool {
    <Dict as VortexMatcher>::matches(p)
}
// For ExactScalarFn<F> we just use AnyScalarFn — checking the exact F is
// a second downcast inside try_match. matches() for ExactScalarFn is the
// same cost as AnyScalarFn (one as_any().is::<ScalarFnVTable>() check).
fn match_any_scalar_fn(p: &ArrayRef) -> bool {
    p.is::<ScalarFnVTable>()
}

/// All known scalar fn IDs (for AnyScalarFn matchers in the proposed lookup).
const ALL_SCALAR_FN_IDS: &[&str] = &[
    "vortex.cast",
    "vortex.binary",
    "vortex.between",
    "vortex.mask",
    "vortex.fill_null",
    "vortex.not",
    "vortex.like",
    "vortex.zip",
    "vortex.list.contains",
    "vortex.get_item",
    "vortex.is_null",
    "vortex.select",
    "vortex.case_when",
    "vortex.merge",
    "vortex.dynamic",
    "vortex.root",
    "vortex.literal",
    "vortex.array",
    "vortex.pack",
];

/// Look up the matcher list for a child encoding by ENCODING ID.
/// Each returned fn does a REAL `as_any().is::<>()` check (TypeId compare).
#[inline(never)]
fn rules_for_child(child_id: &str) -> &'static [MatchFn] {
    match child_id {
        "vortex.primitive" => &[match_masked, match_any_scalar_fn, match_slice],
        "vortex.bool" => &[
            match_masked,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
            match_filter,
        ],
        "vortex.dict" => &[
            match_filter,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
        ],
        "vortex.chunked" => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
        ],
        "vortex.constant" => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_filter,
            match_any_scalar_fn,
            match_filter,
            match_any_scalar_fn,
            match_slice,
            match_dict,
        ],
        "vortex.varbin" | "vortex.varbinview" | "vortex.list" | "vortex.fixed_size_list" => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
        ],
        "vortex.struct" => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
            match_dict,
        ],
        "vortex.ext" => &[
            match_filter,
            match_any_scalar_fn,
            match_filter,
            match_any_scalar_fn,
            match_slice,
        ],
        "vortex.null" | "vortex.listview" => &[
            match_filter,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
            match_dict,
        ],
        "vortex.decimal" => &[match_masked, match_any_scalar_fn, match_slice],
        "vortex.slice" => &[match_slice],
        "vortex.filter" => &[match_filter],
        "vortex.masked" => &[match_filter, match_any_scalar_fn, match_slice, match_dict],
        _ => &[],
    }
}

/// Two-level lookup: parent_id → child_id → bool.
/// Built once at startup. Pre-resolves AnyScalarFn into all scalar fn IDs.
fn build_two_level() -> HashMap<&'static str, HashMap<&'static str, bool>> {
    let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();

    // For each child encoding, list (parent encoding ids that match)
    let entries: &[(&str, &[&str])] = &[
        ("vortex.primitive", &["vortex.masked", "vortex.slice"]),
        (
            "vortex.bool",
            &["vortex.masked", "vortex.slice", "vortex.filter"],
        ),
        (
            "vortex.dict",
            &["vortex.filter", "vortex.slice"],
        ),
        ("vortex.chunked", &[]),
        (
            "vortex.constant",
            &["vortex.filter", "vortex.slice", "vortex.dict"],
        ),
        ("vortex.varbin", &["vortex.slice"]),
        ("vortex.varbinview", &["vortex.slice"]),
        ("vortex.list", &["vortex.slice"]),
        ("vortex.fixed_size_list", &["vortex.slice"]),
        ("vortex.struct", &["vortex.slice", "vortex.dict"]),
        (
            "vortex.ext",
            &["vortex.filter", "vortex.slice"],
        ),
        (
            "vortex.null",
            &["vortex.filter", "vortex.slice", "vortex.dict"],
        ),
        (
            "vortex.listview",
            &["vortex.filter", "vortex.slice", "vortex.dict"],
        ),
        ("vortex.decimal", &["vortex.masked", "vortex.slice"]),
        ("vortex.slice", &["vortex.slice"]),
        ("vortex.filter", &["vortex.filter"]),
        (
            "vortex.masked",
            &["vortex.filter", "vortex.slice", "vortex.dict"],
        ),
    ];

    // Add specific exact entries.
    for (child_id, parents) in entries {
        for p in *parents {
            map.entry(*p).or_default().insert(*child_id, true);
        }
    }

    // Add AnyScalarFn entries: for each scalar fn id, the children with
    // an AnyScalarFn rule (Primitive, Bool, Dict, Chunked, etc).
    let scalar_fn_children: &[&str] = &[
        "vortex.primitive",
        "vortex.bool",
        "vortex.dict",
        "vortex.chunked",
        "vortex.constant",
        "vortex.varbin",
        "vortex.varbinview",
        "vortex.struct",
        "vortex.ext",
        "vortex.null",
        "vortex.listview",
        "vortex.decimal",
        "vortex.masked",
    ];
    for sf in ALL_SCALAR_FN_IDS {
        for c in scalar_fn_children {
            map.entry(*sf).or_default().insert(*c, true);
        }
    }

    map
}

// ============================================================================
// COUNTERS
// ============================================================================

static MATCH_CALLS: AtomicU64 = AtomicU64::new(0);
static LOOKUP_CALLS: AtomicU64 = AtomicU64::new(0);
static MATCHED: AtomicU64 = AtomicU64::new(0);

fn reset_counters() {
    MATCH_CALLS.store(0, Ordering::Relaxed);
    LOOKUP_CALLS.store(0, Ordering::Relaxed);
    MATCHED.store(0, Ordering::Relaxed);
}

// ============================================================================
// MATCHING FUNCTIONS — what we actually measure
// ============================================================================

/// Current approach: dispatch by child encoding_id → scan rules → call matches() on each.
/// Each matches() does as_any().is::<>() — REAL TypeId compare via fn pointer.
#[inline(never)]
fn current_check(parent: &ArrayRef, child_id: &str) -> bool {
    let matchers = rules_for_child(child_id);
    for m in matchers {
        MATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        if m(parent) {
            MATCHED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Proposed two-level: outer HashMap<parent_id, _> + inner HashMap<child_id, _>.
/// The outer lookup can short-circuit when the parent has no rules at all.
#[inline(never)]
fn proposed_two_level_check(
    lookup: &HashMap<&'static str, HashMap<&'static str, bool>>,
    parent_id: &str,
    child_id: &str,
) -> bool {
    LOOKUP_CALLS.fetch_add(1, Ordering::Relaxed);
    if let Some(child_map) = lookup.get(parent_id) {
        if child_map.contains_key(child_id) {
            MATCHED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

// Versions WITHOUT counters for the actual timing benchmarks
// (counters add atomic ops which would distort timings).

#[inline(never)]
fn current_check_fast(parent: &ArrayRef, child_id: &str) -> bool {
    for m in rules_for_child(child_id) {
        if m(parent) {
            return true;
        }
    }
    false
}

#[inline(never)]
fn proposed_two_level_fast(
    lookup: &HashMap<&'static str, HashMap<&'static str, bool>>,
    parent_id: &str,
    child_id: &str,
) -> bool {
    if let Some(child_map) = lookup.get(parent_id) {
        return child_map.contains_key(child_id);
    }
    false
}

// ============================================================================
// TREE WALKER
// ============================================================================

fn walk_current_dyn(parent: &ArrayRef) {
    for slot in parent.slots() {
        if let Some(child) = slot {
            let cid = child.encoding_id();
            let _ = current_check_fast(parent, cid.as_ref());
            walk_current_dyn(child);
        }
    }
}

fn walk_proposed_two_level(
    lookup: &HashMap<&'static str, HashMap<&'static str, bool>>,
    parent: &ArrayRef,
) {
    // Hoist the outer lookup out of the children loop.
    let pid = parent.encoding_id();
    let parent_rules = lookup.get(pid.as_ref());
    for slot in parent.slots() {
        if let Some(child) = slot {
            // Inner lookup only if outer hit
            if let Some(child_map) = parent_rules {
                let cid = child.encoding_id();
                let _ = child_map.contains_key(cid.as_ref());
            }
            walk_proposed_two_level(lookup, child);
        }
    }
}

// ============================================================================
// TREE CONSTRUCTORS
// ============================================================================

fn primitive(n: usize) -> ArrayRef {
    PrimitiveArray::new(Buffer::from(vec![1i32; n]), Validity::NonNullable).into_array()
}

fn dict(n_codes: usize, n_values: usize) -> ArrayRef {
    let codes = PrimitiveArray::new(
        Buffer::from((0..n_codes).map(|i| (i % n_values) as u8).collect::<Vec<u8>>()),
        Validity::NonNullable,
    )
    .into_array();
    let values = PrimitiveArray::new(
        Buffer::from((0..n_values).map(|i| i as i32).collect::<Vec<i32>>()),
        Validity::NonNullable,
    )
    .into_array();
    DictArray::try_new(codes, values).unwrap().into_array()
}

fn chunked_of_primitive(n_chunks: usize) -> ArrayRef {
    let chunks: Vec<ArrayRef> = (0..n_chunks).map(|_| primitive(100)).collect();
    unsafe {
        ChunkedArray::new_unchecked(
            chunks,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
    }
    .into_array()
}

fn chunked_of_dict(n_chunks: usize) -> ArrayRef {
    let chunks: Vec<ArrayRef> = (0..n_chunks).map(|_| dict(100, 10)).collect();
    unsafe {
        ChunkedArray::new_unchecked(
            chunks,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
    }
    .into_array()
}

fn cast(child: ArrayRef) -> ArrayRef {
    let len = child.len();
    Cast.try_new_array(
        len,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        [child],
    )
    .unwrap()
}

fn slice(child: ArrayRef) -> ArrayRef {
    let len = child.len();
    SliceArray::new(child, 0..len.min(50)).into_array()
}

// ============================================================================
// BENCHMARKS — measure walking time for various trees
// ============================================================================

const TREE_NAMES: &[&str] = &[
    "primitive",
    "cast_primitive",
    "slice_primitive",
    "chunked_100_primitive",
    "cast_chunked_100_primitive",
    "chunked_100_dict",
    "cast_chunked_100_dict",
    "slice_chunked_100_dict",
    "chunked_1000_dict",
    "cast_chunked_1000_dict",
    "deep_nested",
];

fn make_tree(name: &str) -> ArrayRef {
    match name {
        "primitive" => primitive(100),
        "cast_primitive" => cast(primitive(100)),
        "slice_primitive" => slice(primitive(100)),
        "chunked_100_primitive" => chunked_of_primitive(100),
        "cast_chunked_100_primitive" => cast(chunked_of_primitive(100)),
        "chunked_100_dict" => chunked_of_dict(100),
        "cast_chunked_100_dict" => cast(chunked_of_dict(100)),
        "slice_chunked_100_dict" => slice(chunked_of_dict(100)),
        "chunked_1000_dict" => chunked_of_dict(1000),
        "cast_chunked_1000_dict" => cast(chunked_of_dict(1000)),
        "deep_nested" => cast(slice(cast(chunked_of_dict(100)))),
        _ => panic!(),
    }
}

#[divan::bench(args = TREE_NAMES)]
fn current_walker(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    bencher.bench(|| {
        walk_current_dyn(black_box(&tree));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn proposed_walker(bencher: divan::Bencher, name: &str) {
    let lookup = build_two_level();
    let tree = make_tree(name);
    bencher.bench(|| {
        walk_proposed_two_level(&lookup, black_box(&tree));
    });
}

// ============================================================================
// COUNT-ONLY: report total matches() and lookups for each tree
// ============================================================================

#[divan::bench(args = TREE_NAMES)]
fn count_current(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    reset_counters();
    bencher.bench(|| {
        walk_count_current(&tree);
    });
    let calls = MATCH_CALLS.load(Ordering::Relaxed);
    let matched = MATCHED.load(Ordering::Relaxed);
    eprintln!("  {name}: matches() calls = {calls}, matched = {matched}");
}

fn walk_count_current(parent: &ArrayRef) {
    for slot in parent.slots() {
        if let Some(child) = slot {
            let cid = child.encoding_id();
            let _ = current_check(parent, cid.as_ref());
            walk_count_current(child);
        }
    }
}

#[divan::bench(args = TREE_NAMES)]
fn count_proposed(bencher: divan::Bencher, name: &str) {
    let lookup = build_two_level();
    let tree = make_tree(name);
    reset_counters();
    bencher.bench(|| {
        walk_count_proposed(&lookup, &tree);
    });
    let calls = LOOKUP_CALLS.load(Ordering::Relaxed);
    let matched = MATCHED.load(Ordering::Relaxed);
    eprintln!("  {name}: lookups = {calls}, matched = {matched}");
}

fn walk_count_proposed(
    lookup: &HashMap<&'static str, HashMap<&'static str, bool>>,
    parent: &ArrayRef,
) {
    let pid = parent.encoding_id();
    let parent_rules = lookup.get(pid.as_ref());
    for slot in parent.slots() {
        if let Some(child) = slot {
            LOOKUP_CALLS.fetch_add(1, Ordering::Relaxed);
            if let Some(child_map) = parent_rules {
                let cid = child.encoding_id();
                if child_map.contains_key(cid.as_ref()) {
                    MATCHED.fetch_add(1, Ordering::Relaxed);
                }
            }
            walk_count_proposed(lookup, child);
        }
    }
}
