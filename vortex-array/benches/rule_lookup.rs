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

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
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

// ============================================================================
// EXTRACTED MATCHER MODEL — duplicated from real rules, no transformations
// ============================================================================

/// All known scalar fn IDs (for AnyScalarFn matchers).
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

/// A matcher predicate: given a parent encoding_id, does this rule match?
#[derive(Clone, Copy)]
enum Matcher {
    /// Matches a single specific encoding_id.
    Exact(&'static str),
    /// Matches ANY scalar fn id.
    AnyScalarFn,
}

impl Matcher {
    #[inline]
    fn matches(&self, parent_id: &str) -> bool {
        match self {
            Matcher::Exact(id) => *id == parent_id,
            Matcher::AnyScalarFn => ALL_SCALAR_FN_IDS.contains(&parent_id),
        }
    }
}

/// Look up the matcher list for a child encoding via match (mirrors vtable dispatch).
/// In the real code this is `child.reduce_parent` → vtable → `RULES.evaluate()` →
/// returns the static slice. We model that as a single `match` here, which has
/// equivalent cost to a vtable dispatch.
#[inline(never)]
fn rules_for_child(child_id: &str) -> &'static [Matcher] {
    match child_id {
        "vortex.primitive" => &[
            Matcher::Exact("vortex.masked"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
        ],
        "vortex.bool" => &[
            Matcher::Exact("vortex.masked"),
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
            Matcher::Exact("vortex.filter"),
        ],
        "vortex.dict" => &[
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.like"),
            Matcher::AnyScalarFn,
            Matcher::AnyScalarFn,
            Matcher::Exact("vortex.slice"),
        ],
        "vortex.chunked" => &[
            Matcher::Exact("vortex.cast"),
            Matcher::AnyScalarFn,
            Matcher::AnyScalarFn,
            Matcher::Exact("vortex.fill_null"),
        ],
        "vortex.constant" => &[
            Matcher::Exact("vortex.between"),
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.fill_null"),
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.not"),
            Matcher::Exact("vortex.slice"),
            Matcher::Exact("vortex.dict"),
        ],
        "vortex.varbin" | "vortex.varbinview" | "vortex.list" | "vortex.fixed_size_list" => &[
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
        ],
        "vortex.struct" => &[
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.get_item"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
            Matcher::Exact("vortex.dict"),
        ],
        "vortex.ext" => &[
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
        ],
        "vortex.null" | "vortex.listview" => &[
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.cast"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
            Matcher::Exact("vortex.dict"),
        ],
        "vortex.decimal" => &[
            Matcher::Exact("vortex.masked"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
        ],
        "vortex.slice" => &[Matcher::Exact("vortex.slice")],
        "vortex.filter" => &[Matcher::Exact("vortex.filter")],
        "vortex.masked" => &[
            Matcher::Exact("vortex.filter"),
            Matcher::Exact("vortex.mask"),
            Matcher::Exact("vortex.slice"),
            Matcher::Exact("vortex.dict"),
        ],
        _ => &[],
    }
}

/// Old API kept for the build_two_level constructor that needs all (child, rules).
fn build_child_rules() -> HashMap<&'static str, Vec<Matcher>> {
    let mut m = HashMap::new();
    for child_id in &[
        "vortex.primitive", "vortex.bool", "vortex.dict", "vortex.chunked",
        "vortex.constant", "vortex.varbin", "vortex.varbinview", "vortex.struct",
        "vortex.ext", "vortex.null", "vortex.listview", "vortex.list",
        "vortex.fixed_size_list", "vortex.decimal", "vortex.slice", "vortex.filter",
        "vortex.masked",
    ] {
        m.insert(*child_id, rules_for_child(child_id).to_vec());
    }
    m
}

/// Two-level lookup: parent_id → child_id → bool.
fn build_two_level(
    rules: &HashMap<&'static str, Vec<Matcher>>,
) -> HashMap<&'static str, HashMap<&'static str, bool>> {
    let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();
    for (child_id, matchers) in rules {
        for m in matchers {
            match m {
                Matcher::Exact(parent_id) => {
                    map.entry(parent_id).or_default().insert(child_id, true);
                }
                Matcher::AnyScalarFn => {
                    for sf in ALL_SCALAR_FN_IDS {
                        map.entry(sf).or_default().insert(child_id, true);
                    }
                }
            }
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

/// Current approach: vtable-style dispatch (match on child_id) → scan matchers.
#[inline(never)]
fn current_check(parent_id: &str, child_id: &str) -> bool {
    let matchers = rules_for_child(child_id);
    for m in matchers {
        MATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        if m.matches(parent_id) {
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
fn current_check_fast(parent_id: &str, child_id: &str) -> bool {
    for m in rules_for_child(child_id) {
        if m.matches(parent_id) {
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

fn walk_current_dyn(parent: &ArrayRef, parent_id: &str) {
    for slot in parent.slots() {
        if let Some(child) = slot {
            let cid = child.encoding_id();
            let child_id_str = cid.as_ref().to_string();
            let _ = current_check_fast(parent_id, &child_id_str);
            walk_current_dyn(child, &child_id_str);
        }
    }
}

fn walk_proposed_two_level(
    lookup: &HashMap<&'static str, HashMap<&'static str, bool>>,
    parent: &ArrayRef,
    parent_id: &str,
) {
    // Hoist the outer lookup out of the children loop.
    let parent_rules = lookup.get(parent_id);
    for slot in parent.slots() {
        if let Some(child) = slot {
            let cid = child.encoding_id();
            let child_id_str = cid.as_ref().to_string();
            // Inner lookup only if outer hit
            if let Some(child_map) = parent_rules {
                let _ = child_map.contains_key(child_id_str.as_str());
            }
            walk_proposed_two_level(lookup, child, &child_id_str);
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
    let root_id = tree.encoding_id().as_ref().to_string();
    bencher.bench(|| {
        walk_current_dyn(black_box(&tree), black_box(&root_id));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn proposed_walker(bencher: divan::Bencher, name: &str) {
    let rules = build_child_rules();
    let lookup = build_two_level(&rules);
    let tree = make_tree(name);
    let root_id = tree.encoding_id().as_ref().to_string();
    bencher.bench(|| {
        walk_proposed_two_level(&lookup, black_box(&tree), black_box(&root_id));
    });
}

// ============================================================================
// COUNT-ONLY: report total matches() and lookups for each tree
// ============================================================================

#[divan::bench(args = TREE_NAMES)]
fn count_current(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let root_id = tree.encoding_id().as_ref().to_string();
    reset_counters();
    bencher.bench(|| {
        walk_count_current(&tree, &root_id);
    });
    let calls = MATCH_CALLS.load(Ordering::Relaxed);
    let matched = MATCHED.load(Ordering::Relaxed);
    eprintln!("  {name}: matches() calls = {calls}, matched = {matched}");
}

fn walk_count_current(parent: &ArrayRef, parent_id: &str) {
    for slot in parent.slots() {
        if let Some(child) = slot {
            let cid = child.encoding_id();
            let child_id_str = cid.as_ref().to_string();
            let _ = current_check(parent_id, &child_id_str);
            walk_count_current(child, &child_id_str);
        }
    }
}

#[divan::bench(args = TREE_NAMES)]
fn count_proposed(bencher: divan::Bencher, name: &str) {
    let rules = build_child_rules();
    let lookup = build_two_level(&rules);
    let tree = make_tree(name);
    let root_id = tree.encoding_id().as_ref().to_string();
    reset_counters();
    bencher.bench(|| {
        walk_count_proposed(&lookup, &tree, &root_id);
    });
    let calls = LOOKUP_CALLS.load(Ordering::Relaxed);
    let matched = MATCHED.load(Ordering::Relaxed);
    eprintln!("  {name}: lookups = {calls}, matched = {matched}");
}

fn walk_count_proposed(
    lookup: &HashMap<&'static str, HashMap<&'static str, bool>>,
    parent: &ArrayRef,
    parent_id: &str,
) {
    for slot in parent.slots() {
        if let Some(child) = slot {
            let cid = child.encoding_id();
            let child_id_str = cid.as_ref().to_string();
            let _ = proposed_two_level_check(lookup, parent_id, &child_id_str);
            walk_count_proposed(lookup, child, &child_id_str);
        }
    }
}
