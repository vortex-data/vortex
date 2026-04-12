// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matching-only model with u64 encoding codes.
//!
//! Models a future world where each encoding has a stable u64 code (instead
//! of an &str). For each tree, we pre-walk to collect (parent_code, child_code)
//! pairs, then benchmark iterating those pairs with both approaches.
//!
//! This isolates the MATCHING cost from the tree-walking cost.

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
// u64 encoding codes (would be assigned at registration time in real impl)
// ============================================================================

const C_PRIMITIVE: u64 = 1;
const C_BOOL: u64 = 2;
const C_DICT: u64 = 3;
const C_CHUNKED: u64 = 4;
const C_CONSTANT: u64 = 5;
const C_VARBIN: u64 = 6;
const C_VARBINVIEW: u64 = 7;
const C_STRUCT: u64 = 8;
const C_EXT: u64 = 9;
const C_NULL: u64 = 10;
const C_LISTVIEW: u64 = 11;
const C_LIST: u64 = 12;
const C_FIXEDSIZELIST: u64 = 13;
const C_DECIMAL: u64 = 14;
const C_SLICE: u64 = 15;
const C_FILTER: u64 = 16;
const C_MASKED: u64 = 17;

// Scalar fn codes (each scalar fn has its own encoding ID = scalar fn ID)
const C_CAST: u64 = 100;
const C_BINARY: u64 = 101;
const C_BETWEEN: u64 = 102;
const C_MASK: u64 = 103;
const C_FILL_NULL: u64 = 104;
const C_NOT: u64 = 105;
const C_LIKE: u64 = 106;
const C_ZIP: u64 = 107;
const C_LIST_CONTAINS: u64 = 108;
const C_GET_ITEM: u64 = 109;
const C_IS_NULL: u64 = 110;
const C_SELECT: u64 = 111;
const C_CASE_WHEN: u64 = 112;
const C_MERGE: u64 = 113;
const C_DYNAMIC: u64 = 114;
const C_ROOT: u64 = 115;
const C_LITERAL: u64 = 116;
const C_ARRAY: u64 = 117;
const C_PACK: u64 = 118;

const ALL_SCALAR_FN_CODES: &[u64] = &[
    C_CAST, C_BINARY, C_BETWEEN, C_MASK, C_FILL_NULL, C_NOT, C_LIKE, C_ZIP,
    C_LIST_CONTAINS, C_GET_ITEM, C_IS_NULL, C_SELECT, C_CASE_WHEN, C_MERGE,
    C_DYNAMIC, C_ROOT, C_LITERAL, C_ARRAY, C_PACK,
];

/// Map an encoding_id string to a u64 code (one-time setup, not in the hot path).
fn encoding_id_to_code(id: &str) -> u64 {
    match id {
        "vortex.primitive" => C_PRIMITIVE,
        "vortex.bool" => C_BOOL,
        "vortex.dict" => C_DICT,
        "vortex.chunked" => C_CHUNKED,
        "vortex.constant" => C_CONSTANT,
        "vortex.varbin" => C_VARBIN,
        "vortex.varbinview" => C_VARBINVIEW,
        "vortex.struct" => C_STRUCT,
        "vortex.ext" => C_EXT,
        "vortex.null" => C_NULL,
        "vortex.listview" => C_LISTVIEW,
        "vortex.list" => C_LIST,
        "vortex.fixed_size_list" => C_FIXEDSIZELIST,
        "vortex.decimal" => C_DECIMAL,
        "vortex.slice" => C_SLICE,
        "vortex.filter" => C_FILTER,
        "vortex.masked" => C_MASKED,
        "vortex.cast" => C_CAST,
        "vortex.binary" => C_BINARY,
        "vortex.between" => C_BETWEEN,
        "vortex.mask" => C_MASK,
        "vortex.fill_null" => C_FILL_NULL,
        "vortex.not" => C_NOT,
        "vortex.like" => C_LIKE,
        "vortex.zip" => C_ZIP,
        "vortex.list.contains" => C_LIST_CONTAINS,
        "vortex.get_item" => C_GET_ITEM,
        "vortex.is_null" => C_IS_NULL,
        "vortex.select" => C_SELECT,
        "vortex.case_when" => C_CASE_WHEN,
        "vortex.merge" => C_MERGE,
        "vortex.dynamic" => C_DYNAMIC,
        "vortex.root" => C_ROOT,
        "vortex.literal" => C_LITERAL,
        "vortex.array" => C_ARRAY,
        "vortex.pack" => C_PACK,
        _ => 0, // unknown
    }
}

// ============================================================================
// CURRENT: linear scan over matchers
//
// Each matcher is a u64 code (the parent code it matches). Iterating and
// comparing u64s is the fastest possible "scan" — anything based on TypeId
// or strings is strictly more expensive.
// ============================================================================

#[derive(Clone, Copy)]
enum CurrentMatcher {
    Exact(u64),
    AnyScalarFn,
}

impl CurrentMatcher {
    #[inline(always)]
    fn matches(&self, parent_code: u64) -> bool {
        match self {
            CurrentMatcher::Exact(c) => *c == parent_code,
            CurrentMatcher::AnyScalarFn => ALL_SCALAR_FN_CODES.contains(&parent_code),
        }
    }
}

/// Look up the matcher list for a child encoding code.
/// Mirrors vtable dispatch — one branch.
#[inline(never)]
fn rules_for_child(child_code: u64) -> &'static [CurrentMatcher] {
    match child_code {
        C_PRIMITIVE => &[
            CurrentMatcher::Exact(C_MASKED),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
        ],
        C_BOOL => &[
            CurrentMatcher::Exact(C_MASKED),
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
            CurrentMatcher::Exact(C_FILTER),
        ],
        C_DICT => &[
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_LIKE),
            CurrentMatcher::AnyScalarFn,
            CurrentMatcher::AnyScalarFn,
            CurrentMatcher::Exact(C_SLICE),
        ],
        C_CHUNKED => &[
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::AnyScalarFn,
            CurrentMatcher::AnyScalarFn,
            CurrentMatcher::Exact(C_FILL_NULL),
        ],
        C_CONSTANT => &[
            CurrentMatcher::Exact(C_BETWEEN),
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_FILL_NULL),
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_NOT),
            CurrentMatcher::Exact(C_SLICE),
            CurrentMatcher::Exact(C_DICT),
        ],
        C_VARBIN | C_VARBINVIEW | C_LIST | C_FIXEDSIZELIST => &[
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
        ],
        C_STRUCT => &[
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_GET_ITEM),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
            CurrentMatcher::Exact(C_DICT),
        ],
        C_EXT => &[
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
        ],
        C_NULL | C_LISTVIEW => &[
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_CAST),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
            CurrentMatcher::Exact(C_DICT),
        ],
        C_DECIMAL => &[
            CurrentMatcher::Exact(C_MASKED),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
        ],
        C_SLICE => &[CurrentMatcher::Exact(C_SLICE)],
        C_FILTER => &[CurrentMatcher::Exact(C_FILTER)],
        C_MASKED => &[
            CurrentMatcher::Exact(C_FILTER),
            CurrentMatcher::Exact(C_MASK),
            CurrentMatcher::Exact(C_SLICE),
            CurrentMatcher::Exact(C_DICT),
        ],
        _ => &[],
    }
}

/// Current matching: for one (parent, child) pair, scan child's matchers.
#[inline(never)]
fn current_check(parent_code: u64, child_code: u64) -> bool {
    for m in rules_for_child(child_code) {
        MATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        if m.matches(parent_code) {
            MATCHED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

#[inline(never)]
fn current_check_fast(parent_code: u64, child_code: u64) -> bool {
    for m in rules_for_child(child_code) {
        if m.matches(parent_code) {
            return true;
        }
    }
    false
}

static MATCH_CALLS: AtomicU64 = AtomicU64::new(0);
static MATCHED: AtomicU64 = AtomicU64::new(0);
static LOOKUP_CALLS: AtomicU64 = AtomicU64::new(0);

// ============================================================================
// PROPOSED: dense Vec<Vec<bool>> indexed by (parent_code, child_code)
//
// With u64 codes assigned sequentially, we can use a 2D array for O(1) lookup
// with no hashing.
// ============================================================================

const MAX_CODE: usize = 200; // larger than any code

struct DenseLookup {
    /// table[parent_code][child_code] = has_rules
    table: Box<[bool; MAX_CODE * MAX_CODE]>,
    /// parent_has_any[parent_code] = any rule for this parent at all
    parent_has_any: [bool; MAX_CODE],
}

impl DenseLookup {
    fn new() -> Self {
        let table = vec![false; MAX_CODE * MAX_CODE].into_boxed_slice();
        let table: Box<[bool; MAX_CODE * MAX_CODE]> = table.try_into().unwrap();
        let mut me = Self {
            table,
            parent_has_any: [false; MAX_CODE],
        };
        me.populate();
        me
    }

    fn set(&mut self, parent: u64, child: u64) {
        self.table[(parent as usize) * MAX_CODE + (child as usize)] = true;
        self.parent_has_any[parent as usize] = true;
    }

    #[inline(always)]
    fn has(&self, parent: u64, child: u64) -> bool {
        self.table[(parent as usize) * MAX_CODE + (child as usize)]
    }

    #[inline(always)]
    fn parent_interesting(&self, parent: u64) -> bool {
        self.parent_has_any[parent as usize]
    }

    fn populate(&mut self) {
        let children_with_any_scalar_fn = [C_DICT, C_CHUNKED];
        let entries: &[(u64, &[u64])] = &[
            (C_PRIMITIVE, &[C_MASKED, C_MASK, C_SLICE]),
            (C_BOOL, &[C_MASKED, C_CAST, C_MASK, C_SLICE, C_FILTER]),
            (C_DICT, &[C_FILTER, C_CAST, C_MASK, C_LIKE, C_SLICE]),
            (C_CHUNKED, &[C_CAST, C_FILL_NULL]),
            (C_CONSTANT, &[C_BETWEEN, C_CAST, C_FILTER, C_FILL_NULL, C_NOT, C_SLICE, C_DICT]),
            (C_VARBIN, &[C_CAST, C_MASK, C_SLICE]),
            (C_VARBINVIEW, &[C_CAST, C_MASK, C_SLICE]),
            (C_LIST, &[C_CAST, C_MASK, C_SLICE]),
            (C_FIXEDSIZELIST, &[C_CAST, C_MASK, C_SLICE]),
            (C_STRUCT, &[C_CAST, C_GET_ITEM, C_MASK, C_SLICE, C_DICT]),
            (C_EXT, &[C_FILTER, C_CAST, C_MASK, C_SLICE]),
            (C_NULL, &[C_FILTER, C_CAST, C_MASK, C_SLICE, C_DICT]),
            (C_LISTVIEW, &[C_FILTER, C_CAST, C_MASK, C_SLICE, C_DICT]),
            (C_DECIMAL, &[C_MASKED, C_MASK, C_SLICE]),
            (C_SLICE, &[C_SLICE]),
            (C_FILTER, &[C_FILTER]),
            (C_MASKED, &[C_FILTER, C_MASK, C_SLICE, C_DICT]),
        ];
        for (child, parents) in entries {
            for p in *parents {
                self.set(*p, *child);
            }
        }
        // AnyScalarFn rules: duplicate into every scalar fn id bucket
        for &child in &children_with_any_scalar_fn {
            for &sf in ALL_SCALAR_FN_CODES {
                self.set(sf, child);
            }
        }
    }
}

/// Proposed matching: single 2D array lookup.
#[inline(never)]
fn proposed_check(lookup: &DenseLookup, parent_code: u64, child_code: u64) -> bool {
    LOOKUP_CALLS.fetch_add(1, Ordering::Relaxed);
    let hit = lookup.has(parent_code, child_code);
    if hit {
        MATCHED.fetch_add(1, Ordering::Relaxed);
    }
    hit
}

#[inline(never)]
fn proposed_check_fast(lookup: &DenseLookup, parent_code: u64, child_code: u64) -> bool {
    lookup.has(parent_code, child_code)
}

// ============================================================================
// TREE HELPERS
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

/// A pre-built tree of u64 codes mirroring the array tree.
#[derive(Default)]
struct CodeNode {
    code: u64,
    children: Vec<CodeNode>,
}

fn build_code_tree(array: &ArrayRef) -> CodeNode {
    let code = encoding_id_to_code(array.encoding_id().as_ref());
    let children: Vec<CodeNode> = array
        .slots()
        .iter()
        .filter_map(|s| s.as_ref().map(build_code_tree))
        .collect();
    CodeNode { code, children }
}

/// Walk the tree and collect (parent_code, child_code) pairs in DFS order.
fn collect_pairs(array: &ArrayRef) -> Vec<(u64, u64)> {
    let tree = build_code_tree(array);
    let mut pairs = Vec::new();
    walk_pairs(&tree, &mut pairs);
    pairs
}

fn walk_pairs(node: &CodeNode, pairs: &mut Vec<(u64, u64)>) {
    for child in &node.children {
        pairs.push((node.code, child.code));
        walk_pairs(child, pairs);
    }
}

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

// ============================================================================
// BENCHMARKS — measure ONLY the matching loop over pre-collected pairs
// ============================================================================

#[divan::bench(args = TREE_NAMES)]
fn current_match(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let pairs = collect_pairs(&tree);
    bencher.bench(|| {
        for (p, c) in black_box(&pairs) {
            black_box(current_check_fast(*p, *c));
        }
    });
}

/// Run with counters once per tree to report match counts.
#[divan::bench(args = TREE_NAMES)]
fn count_current(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let pairs = collect_pairs(&tree);
    MATCH_CALLS.store(0, Ordering::Relaxed);
    MATCHED.store(0, Ordering::Relaxed);
    bencher.bench(|| {
        for (p, c) in black_box(&pairs) {
            black_box(current_check(*p, *c));
        }
    });
    let calls = MATCH_CALLS.load(Ordering::Relaxed);
    let matched = MATCHED.load(Ordering::Relaxed);
    eprintln!(
        "  {name}: pairs={}, matches() calls={}, matched={}",
        pairs.len(),
        calls,
        matched
    );
}

#[divan::bench(args = TREE_NAMES)]
fn proposed_match(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let pairs = collect_pairs(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        for (p, c) in black_box(&pairs) {
            black_box(proposed_check_fast(&lookup, *p, *c));
        }
    });
}

#[divan::bench(args = TREE_NAMES)]
fn count_proposed(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let pairs = collect_pairs(&tree);
    let lookup = DenseLookup::new();
    LOOKUP_CALLS.store(0, Ordering::Relaxed);
    MATCHED.store(0, Ordering::Relaxed);
    bencher.bench(|| {
        for (p, c) in black_box(&pairs) {
            black_box(proposed_check(&lookup, *p, *c));
        }
    });
    let calls = LOOKUP_CALLS.load(Ordering::Relaxed);
    let matched = MATCHED.load(Ordering::Relaxed);
    eprintln!(
        "  {name}: pairs={}, lookups={}, matched={}",
        pairs.len(),
        calls,
        matched
    );
}

// ============================================================================
// TREE WALKERS — same structure as the real optimizer
//
// Walks the tree, at each parent: check if parent is interesting, if so
// iterate children. The "is parent interesting" check is the key win.
// ============================================================================

fn walk_current_tree(node: &CodeNode) -> u64 {
    let mut count = 0u64;
    for child in &node.children {
        // Current: scan rules unconditionally
        if current_check_fast(node.code, child.code) {
            count += 1;
        }
        count += walk_current_tree(child);
    }
    count
}

fn walk_proposed_tree(lookup: &DenseLookup, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    // Hoisted: check once per parent
    if lookup.parent_interesting(node.code) {
        for child in &node.children {
            if lookup.has(node.code, child.code) {
                count += 1;
            }
        }
    }
    // Recurse regardless — children may themselves be interesting parents
    for child in &node.children {
        count += walk_proposed_tree(lookup, child);
    }
    count
}

#[divan::bench(args = TREE_NAMES)]
fn current_walk(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    bencher.bench(|| {
        black_box(walk_current_tree(black_box(&code_tree)));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn proposed_walk(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_proposed_tree(&lookup, black_box(&code_tree)));
    });
}

/// Count how many checks each does for each tree.
#[divan::bench(args = TREE_NAMES)]
fn count_walk(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    let mut current_calls = 0u64;
    let mut proposed_calls = 0u64;
    let mut proposed_hits = 0u64;
    let mut interesting_parents = 0u64;
    let mut total_parents = 0u64;
    fn count_current_calls(node: &CodeNode, calls: &mut u64) {
        *calls += node.children.len() as u64;
        // current_check iterates ALL matchers up to first hit
        for child in &node.children {
            for m in rules_for_child(child.code) {
                *calls += 1;
                if m.matches(node.code) {
                    break;
                }
            }
        }
        for child in &node.children {
            count_current_calls(child, calls);
        }
    }
    fn count_proposed_calls(
        lookup: &DenseLookup,
        node: &CodeNode,
        calls: &mut u64,
        hits: &mut u64,
        interesting: &mut u64,
        total: &mut u64,
    ) {
        *total += 1;
        if lookup.parent_interesting(node.code) {
            *interesting += 1;
            for child in &node.children {
                *calls += 1;
                if lookup.has(node.code, child.code) {
                    *hits += 1;
                }
            }
        }
        for child in &node.children {
            count_proposed_calls(lookup, child, calls, hits, interesting, total);
        }
    }
    count_current_calls(&code_tree, &mut current_calls);
    count_proposed_calls(
        &lookup,
        &code_tree,
        &mut proposed_calls,
        &mut proposed_hits,
        &mut interesting_parents,
        &mut total_parents,
    );
    eprintln!(
        "  {name}: parents={total_parents}, interesting_parents={interesting_parents}, current_matches={current_calls}, proposed_lookups={proposed_calls}, hits={proposed_hits}"
    );
    bencher.bench(|| {
        black_box(walk_proposed_tree(&lookup, black_box(&code_tree)));
    });
}

// Also: HashMap-based proposed for comparison
#[divan::bench(args = TREE_NAMES)]
fn proposed_hashmap_match(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let pairs = collect_pairs(&tree);
    let mut lookup: HashMap<(u64, u64), bool> = HashMap::new();
    let dense = DenseLookup::new();
    for p in 0..MAX_CODE as u64 {
        for c in 0..MAX_CODE as u64 {
            if dense.has(p, c) {
                lookup.insert((p, c), true);
            }
        }
    }
    bencher.bench(|| {
        for (p, c) in black_box(&pairs) {
            black_box(lookup.contains_key(&(*p, *c)));
        }
    });
}
