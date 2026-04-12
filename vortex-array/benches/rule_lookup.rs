// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matching-only model as close to vortex as possible.
//!
//! Three approaches:
//! 1. CURRENT: dispatch by child type → static slice of fn pointers → each fn
//!    calls `<P as Matcher>::matches(parent)` which does `as_any().is::<>()`.
//!    Same cost as real `dyn DynArrayParentReduceRule.matches()`.
//! 2. DENSE: pre-built `Vec<bool>` indexed by `(parent_code, child_code)`.
//!    u64 codes are pre-cached on each tree node (no encoding_id() in hot path).
//! 3. HASHMAP: `HashMap<(u64, u64), bool>` keyed by u64 codes.
//!
//! Reports timing AND memory size for each.

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;
use std::mem::size_of;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Masked;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::ScalarFnVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::matcher::Matcher as VortexMatcher;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

// ============================================================================
// REAL MATCHERS — fn pointers calling VortexMatcher::matches
// Each fn does as_any().is::<>() — the same TypeId compare via vtable that
// real `dyn DynArrayParentReduceRule.matches(parent)` does.
// ============================================================================

type MatcherFn = fn(&ArrayRef) -> bool;

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
fn match_any_scalar_fn(p: &ArrayRef) -> bool {
    p.is::<ScalarFnVTable>()
}

// ============================================================================
// u64 ENCODING CODES — pre-assigned, dense indices
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

// Scalar fn codes
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
        _ => 0,
    }
}

// ============================================================================
// CURRENT: dispatch by child code → static slice of MatcherFn
// Each MatcherFn is a real `as_any().is::<>()` call (same cost as the real
// dyn dispatch chain).
// ============================================================================

#[inline(never)]
fn rules_for_child(child_code: u64) -> &'static [MatcherFn] {
    match child_code {
        C_PRIMITIVE => &[match_masked, match_any_scalar_fn, match_slice],
        C_BOOL => &[
            match_masked,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
            match_filter,
        ],
        C_DICT => &[
            match_filter,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
        ],
        C_CHUNKED => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
        ],
        C_CONSTANT => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_filter,
            match_any_scalar_fn,
            match_filter,
            match_any_scalar_fn,
            match_slice,
            match_dict,
        ],
        C_VARBIN | C_VARBINVIEW | C_LIST | C_FIXEDSIZELIST => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
        ],
        C_STRUCT => &[
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
            match_dict,
        ],
        C_EXT => &[
            match_filter,
            match_any_scalar_fn,
            match_filter,
            match_any_scalar_fn,
            match_slice,
        ],
        C_NULL | C_LISTVIEW => &[
            match_filter,
            match_any_scalar_fn,
            match_any_scalar_fn,
            match_slice,
            match_dict,
        ],
        C_DECIMAL => &[match_masked, match_any_scalar_fn, match_slice],
        C_SLICE => &[match_slice],
        C_FILTER => &[match_filter],
        C_MASKED => &[match_filter, match_any_scalar_fn, match_slice, match_dict],
        _ => &[],
    }
}

/// Current: scan rules calling each fn (vtable-equivalent) on parent.
#[inline(never)]
fn current_check(parent: &ArrayRef, child_code: u64) -> bool {
    for f in rules_for_child(child_code) {
        if f(parent) {
            return true;
        }
    }
    false
}

// ============================================================================
// DENSE 2D Vec<bool>
// ============================================================================

const MAX_CODE: usize = 200;

struct DenseLookup {
    table: Box<[bool; MAX_CODE * MAX_CODE]>,
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

    fn mem_bytes(&self) -> usize {
        size_of::<[bool; MAX_CODE * MAX_CODE]>() + size_of::<[bool; MAX_CODE]>()
    }

    fn populate(&mut self) {
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
        // AnyScalarFn rules: dict and chunked match every scalar fn parent
        let any_scalar_fn_children = [C_DICT, C_CHUNKED];
        for &child in &any_scalar_fn_children {
            for &sf in ALL_SCALAR_FN_CODES {
                self.set(sf, child);
            }
        }
    }
}

// ============================================================================
// HASHMAP with u64 keys
// ============================================================================

struct HashLookup {
    map: HashMap<(u64, u64), bool>,
    parent_has_any: HashMap<u64, bool>,
}

impl HashLookup {
    fn new() -> Self {
        let mut me = Self {
            map: HashMap::new(),
            parent_has_any: HashMap::new(),
        };
        me.populate();
        me
    }

    fn set(&mut self, parent: u64, child: u64) {
        self.map.insert((parent, child), true);
        self.parent_has_any.insert(parent, true);
    }

    #[inline(always)]
    fn has(&self, parent: u64, child: u64) -> bool {
        self.map.contains_key(&(parent, child))
    }

    #[inline(always)]
    fn parent_interesting(&self, parent: u64) -> bool {
        self.parent_has_any.contains_key(&parent)
    }

    fn mem_bytes(&self) -> usize {
        // HashMap overhead: ~48 bytes per entry (key + value + bucket overhead)
        let entry_size = size_of::<((u64, u64), bool)>() + 16; // approximate hashbrown overhead
        let parent_entry = size_of::<(u64, bool)>() + 16;
        self.map.len() * entry_size + self.parent_has_any.len() * parent_entry
    }

    fn populate(&mut self) {
        let dense = DenseLookup::new();
        for p in 0..MAX_CODE as u64 {
            for c in 0..MAX_CODE as u64 {
                if dense.has(p, c) {
                    self.set(p, c);
                }
            }
        }
    }
}

// ============================================================================
// PRE-CACHED CODE TREE — u64 codes computed once at construction
// ============================================================================

struct CodeNode {
    code: u64,
    array: ArrayRef,
    children: Vec<CodeNode>,
}

fn build_code_tree(array: &ArrayRef) -> CodeNode {
    let code = encoding_id_to_code(array.encoding_id().as_ref());
    let children: Vec<CodeNode> = array
        .slots()
        .iter()
        .filter_map(|s| s.as_ref().map(build_code_tree))
        .collect();
    CodeNode {
        code,
        array: array.clone(),
        children,
    }
}

fn count_nodes(node: &CodeNode) -> usize {
    1 + node.children.iter().map(count_nodes).sum::<usize>()
}

// ============================================================================
// TREE WALKERS — three approaches
// ============================================================================

fn walk_current(node: &CodeNode) -> u64 {
    let mut count = 0u64;
    for child in &node.children {
        if current_check(&node.array, child.code) {
            count += 1;
        }
        count += walk_current(child);
    }
    count
}

fn walk_dense(lookup: &DenseLookup, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    if lookup.parent_interesting(node.code) {
        for child in &node.children {
            if lookup.has(node.code, child.code) {
                count += 1;
            }
        }
    }
    for child in &node.children {
        count += walk_dense(lookup, child);
    }
    count
}

fn walk_hashmap(lookup: &HashLookup, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    if lookup.parent_interesting(node.code) {
        for child in &node.children {
            if lookup.has(node.code, child.code) {
                count += 1;
            }
        }
    }
    for child in &node.children {
        count += walk_hashmap(lookup, child);
    }
    count
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
// BENCHMARKS
// ============================================================================

#[divan::bench(args = TREE_NAMES)]
fn current(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    bencher.bench(|| {
        black_box(walk_current(black_box(&code_tree)));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn dense(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_dense(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn hashmap(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = HashLookup::new();
    bencher.bench(|| {
        black_box(walk_hashmap(&lookup, black_box(&code_tree)));
    });
}

// One-shot bench that just reports counts and memory
#[divan::bench(args = TREE_NAMES)]
fn report(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let dense_lookup = DenseLookup::new();
    let hash_lookup = HashLookup::new();
    let nodes = count_nodes(&code_tree);

    // Count current matches() calls
    fn count_current(node: &CodeNode, count: &mut u64) {
        for child in &node.children {
            for f in rules_for_child(child.code) {
                *count += 1;
                if f(&node.array) {
                    break;
                }
            }
            count_current(child, count);
        }
    }
    let mut current_calls = 0u64;
    count_current(&code_tree, &mut current_calls);

    // Count dense lookups
    fn count_dense(lookup: &DenseLookup, node: &CodeNode, count: &mut u64, interesting: &mut u64) {
        if lookup.parent_interesting(node.code) {
            *interesting += 1;
            *count += node.children.len() as u64;
        }
        for child in &node.children {
            count_dense(lookup, child, count, interesting);
        }
    }
    let mut dense_calls = 0u64;
    let mut interesting = 0u64;
    count_dense(&dense_lookup, &code_tree, &mut dense_calls, &mut interesting);

    eprintln!(
        "  {name}: nodes={nodes}, interesting_parents={interesting}, current_matches={current_calls}, dense_lookups={dense_calls}"
    );
    eprintln!(
        "    mem: dense={} bytes, hashmap={} bytes",
        dense_lookup.mem_bytes(),
        hash_lookup.mem_bytes()
    );

    // No-op bench so divan doesn't complain
    bencher.bench(|| black_box(0));
}
