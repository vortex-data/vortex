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
// DENSE: Vec<&[MatcherFn]> indexed by combined (parent, child) id
// HASHMAP: HashMap<u64, &[MatcherFn]> keyed by combined id
//
// Lookup returns the list of matchers that might match this (parent, child)
// pair. The list is pre-filtered — no scanning needed.
// ============================================================================

const MAX_CODE: usize = 200;

#[inline(always)]
fn combine_codes(parent: u64, child: u64) -> usize {
    (parent as usize) * MAX_CODE + (child as usize)
}

#[inline(always)]
fn combine_codes_u64(parent: u64, child: u64) -> u64 {
    (parent << 32) | child
}

/// Returns the (parent, child) → list of matchers entries.
/// Mirrors the real PARENT_RULES from each child encoding's ParentRuleSet.
/// For matching-only, the matcher fns are kept but never called.
fn all_entries() -> Vec<(u64, u64, &'static [MatcherFn])> {
    // For each child encoding, the parents that have rules + the matcher list
    // Each (parent, child) pair gets ALL the matchers that could fire for that pair.
    // For real code, this is what the registry would store.
    let mut out: Vec<(u64, u64, &'static [MatcherFn])> = Vec::new();

    // Primitive child: 3 rules → if parent matches Masked/Mask/Slice
    static PRIM_MASKED: [MatcherFn; 1] = [match_masked];
    static PRIM_MASK: [MatcherFn; 1] = [match_any_scalar_fn];
    static PRIM_SLICE: [MatcherFn; 1] = [match_slice];
    out.push((C_MASKED, C_PRIMITIVE, &PRIM_MASKED));
    out.push((C_MASK, C_PRIMITIVE, &PRIM_MASK));
    out.push((C_SLICE, C_PRIMITIVE, &PRIM_SLICE));

    // Bool child: 5 rules
    static BOOL_M: [MatcherFn; 1] = [match_masked];
    static BOOL_C: [MatcherFn; 1] = [match_any_scalar_fn];
    static BOOL_MASK: [MatcherFn; 1] = [match_any_scalar_fn];
    static BOOL_S: [MatcherFn; 1] = [match_slice];
    static BOOL_F: [MatcherFn; 1] = [match_filter];
    out.push((C_MASKED, C_BOOL, &BOOL_M));
    out.push((C_CAST, C_BOOL, &BOOL_C));
    out.push((C_MASK, C_BOOL, &BOOL_MASK));
    out.push((C_SLICE, C_BOOL, &BOOL_S));
    out.push((C_FILTER, C_BOOL, &BOOL_F));

    // Dict child: 7 rules (4 specific + 2 AnyScalarFn duplicated + 1 slice)
    static DICT_FILTER: [MatcherFn; 1] = [match_filter];
    static DICT_CAST: [MatcherFn; 2] = [match_any_scalar_fn, match_any_scalar_fn]; // CastReduce + DictValuesPushDown
    static DICT_SCALARFN: [MatcherFn; 2] = [match_any_scalar_fn, match_any_scalar_fn]; // 2 AnyScalarFn rules
    static DICT_SLICE: [MatcherFn; 1] = [match_slice];
    out.push((C_FILTER, C_DICT, &DICT_FILTER));
    out.push((C_CAST, C_DICT, &DICT_CAST));
    out.push((C_MASK, C_DICT, &DICT_SCALARFN));
    out.push((C_LIKE, C_DICT, &DICT_SCALARFN));
    out.push((C_SLICE, C_DICT, &DICT_SLICE));
    // Duplicate DictionaryScalarFn rules for every scalar fn parent
    for &sf in ALL_SCALAR_FN_CODES {
        if sf != C_CAST && sf != C_PACK {
            out.push((sf, C_DICT, &DICT_SCALARFN));
        }
    }

    // Chunked child: 4 rules
    static CHK_CAST: [MatcherFn; 3] = [match_any_scalar_fn, match_any_scalar_fn, match_any_scalar_fn];
    static CHK_FN: [MatcherFn; 2] = [match_any_scalar_fn, match_any_scalar_fn];
    static CHK_FILL: [MatcherFn; 1] = [match_any_scalar_fn];
    out.push((C_CAST, C_CHUNKED, &CHK_CAST));
    out.push((C_FILL_NULL, C_CHUNKED, &CHK_FILL));
    for &sf in ALL_SCALAR_FN_CODES {
        if sf != C_CAST && sf != C_FILL_NULL {
            out.push((sf, C_CHUNKED, &CHK_FN));
        }
    }

    // Constant child: 8 rules
    static CONST_RULES: [MatcherFn; 1] = [match_any_scalar_fn];
    static CONST_FILTER: [MatcherFn; 2] = [match_filter, match_filter];
    static CONST_SLICE: [MatcherFn; 1] = [match_slice];
    static CONST_DICT: [MatcherFn; 1] = [match_dict];
    out.push((C_BETWEEN, C_CONSTANT, &CONST_RULES));
    out.push((C_CAST, C_CONSTANT, &CONST_RULES));
    out.push((C_FILTER, C_CONSTANT, &CONST_FILTER));
    out.push((C_FILL_NULL, C_CONSTANT, &CONST_RULES));
    out.push((C_NOT, C_CONSTANT, &CONST_RULES));
    out.push((C_SLICE, C_CONSTANT, &CONST_SLICE));
    out.push((C_DICT, C_CONSTANT, &CONST_DICT));

    // VarBin/VarBinView/List/FixedSizeList: 3 rules each
    static SIMPLE_C: [MatcherFn; 1] = [match_any_scalar_fn];
    static SIMPLE_M: [MatcherFn; 1] = [match_any_scalar_fn];
    static SIMPLE_S: [MatcherFn; 1] = [match_slice];
    for &c in &[C_VARBIN, C_VARBINVIEW, C_LIST, C_FIXEDSIZELIST] {
        out.push((C_CAST, c, &SIMPLE_C));
        out.push((C_MASK, c, &SIMPLE_M));
        out.push((C_SLICE, c, &SIMPLE_S));
    }

    // Struct: 5 rules
    static STRUCT_R: [MatcherFn; 1] = [match_any_scalar_fn];
    static STRUCT_S: [MatcherFn; 1] = [match_slice];
    static STRUCT_D: [MatcherFn; 1] = [match_dict];
    out.push((C_CAST, C_STRUCT, &STRUCT_R));
    out.push((C_GET_ITEM, C_STRUCT, &STRUCT_R));
    out.push((C_MASK, C_STRUCT, &STRUCT_R));
    out.push((C_SLICE, C_STRUCT, &STRUCT_S));
    out.push((C_DICT, C_STRUCT, &STRUCT_D));

    // Ext: 5 rules
    static EXT_F: [MatcherFn; 2] = [match_filter, match_filter];
    static EXT_R: [MatcherFn; 1] = [match_any_scalar_fn];
    static EXT_S: [MatcherFn; 1] = [match_slice];
    out.push((C_FILTER, C_EXT, &EXT_F));
    out.push((C_CAST, C_EXT, &EXT_R));
    out.push((C_MASK, C_EXT, &EXT_R));
    out.push((C_SLICE, C_EXT, &EXT_S));

    // Null + ListView: 5 rules
    static NULL_F: [MatcherFn; 1] = [match_filter];
    static NULL_R: [MatcherFn; 1] = [match_any_scalar_fn];
    static NULL_S: [MatcherFn; 1] = [match_slice];
    static NULL_D: [MatcherFn; 1] = [match_dict];
    for &c in &[C_NULL, C_LISTVIEW] {
        out.push((C_FILTER, c, &NULL_F));
        out.push((C_CAST, c, &NULL_R));
        out.push((C_MASK, c, &NULL_R));
        out.push((C_SLICE, c, &NULL_S));
        out.push((C_DICT, c, &NULL_D));
    }

    // Decimal: 3 rules
    static DEC_M: [MatcherFn; 1] = [match_masked];
    static DEC_MASK: [MatcherFn; 1] = [match_any_scalar_fn];
    static DEC_S: [MatcherFn; 1] = [match_slice];
    out.push((C_MASKED, C_DECIMAL, &DEC_M));
    out.push((C_MASK, C_DECIMAL, &DEC_MASK));
    out.push((C_SLICE, C_DECIMAL, &DEC_S));

    // Slice/Filter/Masked
    static SLICE_R: [MatcherFn; 1] = [match_slice];
    static FILTER_R: [MatcherFn; 1] = [match_filter];
    static MASKED_F: [MatcherFn; 1] = [match_filter];
    static MASKED_M: [MatcherFn; 1] = [match_any_scalar_fn];
    static MASKED_S: [MatcherFn; 1] = [match_slice];
    static MASKED_D: [MatcherFn; 1] = [match_dict];
    out.push((C_SLICE, C_SLICE, &SLICE_R));
    out.push((C_FILTER, C_FILTER, &FILTER_R));
    out.push((C_FILTER, C_MASKED, &MASKED_F));
    out.push((C_MASK, C_MASKED, &MASKED_M));
    out.push((C_SLICE, C_MASKED, &MASKED_S));
    out.push((C_DICT, C_MASKED, &MASKED_D));

    out
}

const EMPTY_RULES: &[MatcherFn] = &[];

// --- DENSE: Vec<&[MatcherFn]> indexed by combine_codes(parent, child) ---
struct DenseLookup {
    /// flat 2D table: rules[parent * MAX + child]
    rules: Box<[&'static [MatcherFn]]>,
    /// parent_has_any[p] = true if any rules for this parent
    parent_has_any: Box<[bool]>,
}

impl DenseLookup {
    fn new() -> Self {
        let mut rules: Vec<&'static [MatcherFn]> = vec![EMPTY_RULES; MAX_CODE * MAX_CODE];
        let mut parent_has_any = vec![false; MAX_CODE];
        for (p, c, list) in all_entries() {
            rules[combine_codes(p, c)] = list;
            parent_has_any[p as usize] = true;
        }
        Self {
            rules: rules.into_boxed_slice(),
            parent_has_any: parent_has_any.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn get(&self, parent: u64, child: u64) -> &'static [MatcherFn] {
        self.rules[combine_codes(parent, child)]
    }

    #[inline(always)]
    fn parent_interesting(&self, parent: u64) -> bool {
        self.parent_has_any[parent as usize]
    }

    fn mem_bytes(&self) -> usize {
        size_of::<&[MatcherFn]>() * self.rules.len() + size_of::<bool>() * self.parent_has_any.len()
    }
}

// --- HASHMAP: HashMap<u64, &[MatcherFn]> keyed by combine_codes_u64 ---
struct HashLookup {
    rules: HashMap<u64, &'static [MatcherFn]>,
    parent_has_any: HashMap<u64, ()>,
}

impl HashLookup {
    fn new() -> Self {
        let mut rules = HashMap::new();
        let mut parent_has_any = HashMap::new();
        for (p, c, list) in all_entries() {
            rules.insert(combine_codes_u64(p, c), list);
            parent_has_any.insert(p, ());
        }
        Self {
            rules,
            parent_has_any,
        }
    }

    #[inline(always)]
    fn get(&self, parent: u64, child: u64) -> &'static [MatcherFn] {
        self.rules
            .get(&combine_codes_u64(parent, child))
            .copied()
            .unwrap_or(EMPTY_RULES)
    }

    #[inline(always)]
    fn parent_interesting(&self, parent: u64) -> bool {
        self.parent_has_any.contains_key(&parent)
    }

    fn mem_bytes(&self) -> usize {
        // approximate hashbrown overhead: 24 bytes per entry
        let main_entry = size_of::<u64>() + size_of::<&[MatcherFn]>() + 16;
        let parent_entry = size_of::<u64>() + 16;
        self.rules.len() * main_entry + self.parent_has_any.len() * parent_entry
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
            // Get pre-filtered list of matchers. Iterate to model the cost
            // of touching each rule (in real code we'd run rule.reduce_parent).
            let rules = lookup.get(node.code, child.code);
            for f in rules {
                count += (*f as usize) as u64 & 1;
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
            let rules = lookup.get(node.code, child.code);
            for f in rules {
                count += (*f as usize) as u64 & 1;
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

// ============================================================================
// OPTIMIZATIONS for dense:
// 1. Pre-flatten the tree into (parent, child) pairs
// 2. Iterative tree walker (no recursion)
// 3. Compact dense table that fits in L1 (smaller code space)
// ============================================================================

/// Flatten the tree to a list of (parent_code, child_code) pairs.
fn flatten_tree(node: &CodeNode) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    fn walk(node: &CodeNode, out: &mut Vec<(u64, u64)>) {
        for child in &node.children {
            out.push((node.code, child.code));
            walk(child, out);
        }
    }
    walk(node, &mut out);
    out
}

/// Compact dense table: assumes codes < 32. Total 32*32 = 1024 entries × 16 bytes = 16 KB.
const COMPACT_MAX: usize = 32;
struct CompactDense {
    rules: [&'static [MatcherFn]; COMPACT_MAX * COMPACT_MAX],
    parent_has_any: u64, // bitmap for codes 0..63
}
impl CompactDense {
    fn new() -> Self {
        let mut rules: [&'static [MatcherFn]; COMPACT_MAX * COMPACT_MAX] =
            [EMPTY_RULES; COMPACT_MAX * COMPACT_MAX];
        let mut parent_has_any = 0u64;
        // Remap u64 codes to compact indices [0..32)
        let remap = |code: u64| -> Option<usize> {
            // Compact: use the existing codes mod 32 (collision-free since codes are < 200)
            // For real benchmark we'd assign sequential codes
            // C_PRIMITIVE..C_MASKED = 1..17 → use directly
            // Scalar fns are at 100..118 → remap to 17..36... too big for 32
            // Just use the 17 array codes + truncate scalar fn ids
            if code < 32 { Some(code as usize) } else { None }
        };
        // Skip scalar fn entries since they don't fit in 32 — we'd need a bigger table
        // For this micro-benchmark just use array codes
        for (p, c, list) in all_entries() {
            if let (Some(pi), Some(ci)) = (remap(p), remap(c)) {
                rules[pi * COMPACT_MAX + ci] = list;
                parent_has_any |= 1u64 << pi;
            }
        }
        Self { rules, parent_has_any }
    }
    #[inline(always)]
    fn get(&self, parent: u64, child: u64) -> &'static [MatcherFn] {
        if parent < 32 && child < 32 {
            self.rules[(parent as usize) * COMPACT_MAX + (child as usize)]
        } else {
            EMPTY_RULES
        }
    }
    #[inline(always)]
    fn parent_interesting(&self, parent: u64) -> bool {
        parent < 64 && (self.parent_has_any >> parent) & 1 == 1
    }
    fn mem_bytes(&self) -> usize {
        size_of::<[&[MatcherFn]; COMPACT_MAX * COMPACT_MAX]>() + size_of::<u64>()
    }
}

fn walk_compact(lookup: &CompactDense, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    if lookup.parent_interesting(node.code) {
        for child in &node.children {
            let rules = lookup.get(node.code, child.code);
            for f in rules {
                count += (*f as usize) as u64 & 1;
            }
        }
    }
    for child in &node.children {
        count += walk_compact(lookup, child);
    }
    count
}

/// Iterative tree walker — uses explicit stack instead of recursion.
fn walk_dense_iterative(lookup: &DenseLookup, root: &CodeNode) -> u64 {
    let mut count = 0u64;
    let mut stack: Vec<&CodeNode> = vec![root];
    while let Some(node) = stack.pop() {
        if lookup.parent_interesting(node.code) {
            for child in &node.children {
                let rules = lookup.get(node.code, child.code);
                for f in rules {
                    count += (*f as usize) as u64 & 1;
                }
            }
        }
        for child in node.children.iter().rev() {
            stack.push(child);
        }
    }
    count
}

#[divan::bench(args = TREE_NAMES)]
fn dense_iterative(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_dense_iterative(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn compact(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = CompactDense::new();
    bencher.bench(|| {
        black_box(walk_compact(&lookup, black_box(&code_tree)));
    });
}

// ============================================================================
// More optimizations
// ============================================================================

/// Walker that hoists the parent's row pointer out of the children loop.
/// One row lookup per parent, then index into the row per child.
fn walk_dense_hoisted(lookup: &DenseLookup, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    if lookup.parent_interesting(node.code) {
        let row_offset = (node.code as usize) * MAX_CODE;
        // SAFETY: row_offset + child.code is guaranteed in-range since child.code < MAX_CODE
        // and the table has MAX_CODE * MAX_CODE entries.
        for child in &node.children {
            let rules = unsafe { *lookup.rules.get_unchecked(row_offset + child.code as usize) };
            for f in rules {
                count += (*f as usize) as u64 & 1;
            }
        }
    }
    for child in &node.children {
        count += walk_dense_hoisted(lookup, child);
    }
    count
}

/// Iterative + hoisted parent row.
fn walk_iter_hoisted(lookup: &DenseLookup, root: &CodeNode) -> u64 {
    let mut count = 0u64;
    let mut stack: Vec<&CodeNode> = Vec::with_capacity(64);
    stack.push(root);
    while let Some(node) = stack.pop() {
        if lookup.parent_interesting(node.code) {
            let row_offset = (node.code as usize) * MAX_CODE;
            for child in &node.children {
                let rules =
                    unsafe { *lookup.rules.get_unchecked(row_offset + child.code as usize) };
                for f in rules {
                    count += (*f as usize) as u64 & 1;
                }
            }
        }
        for child in node.children.iter().rev() {
            stack.push(child);
        }
    }
    count
}

/// Walker that detects runs of same-encoding children and batches them.
/// For chunked arrays where all chunks are the same encoding, this gets the
/// rules ONCE per parent and applies to all children.
fn walk_batched(lookup: &DenseLookup, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    if lookup.parent_interesting(node.code) {
        let row_offset = (node.code as usize) * MAX_CODE;
        let mut last_code = u64::MAX;
        let mut last_rules: &[MatcherFn] = EMPTY_RULES;
        for child in &node.children {
            if child.code != last_code {
                last_rules = unsafe {
                    *lookup.rules.get_unchecked(row_offset + child.code as usize)
                };
                last_code = child.code;
            }
            for f in last_rules {
                count += (*f as usize) as u64 & 1;
            }
        }
    }
    for child in &node.children {
        count += walk_batched(lookup, child);
    }
    count
}

#[divan::bench(args = TREE_NAMES)]
fn dense_hoisted(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_dense_hoisted(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn iter_hoisted(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_iter_hoisted(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = TREE_NAMES)]
fn batched(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_batched(&lookup, black_box(&code_tree)));
    });
}

/// Pre-resolve all rules for the tree into a flat list. No table lookup at iteration.
fn pre_resolve(node: &CodeNode, lookup: &DenseLookup) -> Vec<&'static [MatcherFn]> {
    let mut out = Vec::new();
    fn walk(node: &CodeNode, lookup: &DenseLookup, out: &mut Vec<&'static [MatcherFn]>) {
        if lookup.parent_interesting(node.code) {
            for child in &node.children {
                out.push(lookup.get(node.code, child.code));
            }
        }
        for child in &node.children {
            walk(child, lookup, out);
        }
    }
    walk(node, lookup, &mut out);
    out
}

/// Pre-resolve into a Vec of row pointers per parent (one per node).
/// At iteration, look up child code in the cached row.
struct PreResolvedRows {
    /// For each node in DFS order: (Option<row_ptr>, &[child_codes])
    nodes: Vec<(Option<*const [&'static [MatcherFn]; MAX_CODE]>, Vec<u64>)>,
}

#[divan::bench(args = TREE_NAMES)]
fn pre_resolved_rules(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    let pre = pre_resolve(&code_tree, &lookup);
    bencher.bench(|| {
        let mut count = 0u64;
        for rules in black_box(&pre) {
            for f in *rules {
                count += (*f as usize) as u64 & 1;
            }
        }
        black_box(count);
    });
}

// ============================================================================
// u32 dense: u32 index into a flat rules table (4 bytes per cell instead of 16)
// ============================================================================

struct DenseU32 {
    /// flat 2D table: idx[parent * MAX + child] → rules_table index (0 = empty)
    idx: Box<[u32]>,
    /// rules_table[i] = &'static [MatcherFn]; index 0 is empty
    rules_table: Vec<&'static [MatcherFn]>,
    parent_has_any: Box<[bool]>,
}

impl DenseU32 {
    fn new() -> Self {
        let mut idx = vec![0u32; MAX_CODE * MAX_CODE].into_boxed_slice();
        let mut rules_table: Vec<&'static [MatcherFn]> = vec![EMPTY_RULES]; // index 0 = empty
        let mut parent_has_any = vec![false; MAX_CODE].into_boxed_slice();
        for (p, c, list) in all_entries() {
            let i = rules_table.len() as u32;
            rules_table.push(list);
            idx[combine_codes(p, c)] = i;
            parent_has_any[p as usize] = true;
        }
        Self {
            idx,
            rules_table,
            parent_has_any,
        }
    }

    #[inline(always)]
    fn get(&self, parent: u64, child: u64) -> &'static [MatcherFn] {
        let i = self.idx[combine_codes(parent, child)];
        // SAFETY: idx values are always valid indices into rules_table
        unsafe { *self.rules_table.get_unchecked(i as usize) }
    }

    #[inline(always)]
    fn parent_interesting(&self, parent: u64) -> bool {
        self.parent_has_any[parent as usize]
    }

    fn mem_bytes(&self) -> usize {
        size_of::<u32>() * self.idx.len()
            + size_of::<&[MatcherFn]>() * self.rules_table.capacity()
            + self.parent_has_any.len()
    }
}

fn walk_dense_u32(lookup: &DenseU32, node: &CodeNode) -> u64 {
    let mut count = 0u64;
    if lookup.parent_interesting(node.code) {
        for child in &node.children {
            let rules = lookup.get(node.code, child.code);
            for f in rules {
                count += (*f as usize) as u64 & 1;
            }
        }
    }
    for child in &node.children {
        count += walk_dense_u32(lookup, child);
    }
    count
}

#[divan::bench(args = TREE_NAMES)]
fn dense_u32(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseU32::new();
    bencher.bench(|| {
        black_box(walk_dense_u32(&lookup, black_box(&code_tree)));
    });
}

// ============================================================================
// MIXED CHILD TREE: defeats branch prediction
// Build a Struct with many fields of varied encodings.
// ============================================================================

fn bool_array(n: usize) -> ArrayRef {
    use vortex_array::arrays::BoolArray;
    use vortex_buffer::BitBuffer;
    BoolArray::new(BitBuffer::collect_bool(n, |_| true), Validity::NonNullable).into_array()
}

fn varbin_array(n: usize) -> ArrayRef {
    use vortex_array::arrays::VarBinViewArray;
    let strs: Vec<&str> = (0..n).map(|_| "x").collect();
    VarBinViewArray::from_iter_str(strs).into_array()
}

fn make_mixed_chunked(nchunks: usize) -> ArrayRef {
    // Rotate through many encoding types so each chunk has a different code.
    let chunks: Vec<ArrayRef> = (0..nchunks)
        .map(|i| match i % 5 {
            0 => primitive(100),
            1 => dict(100, 10),
            2 => bool_array(100),
            3 => varbin_array(100),
            _ => primitive(100),
        })
        .collect();
    unsafe {
        ChunkedArray::new_unchecked(
            chunks,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
    }
    .into_array()
}

const MIXED_TREE_NAMES: &[&str] = &[
    "mixed_chunked_1k",
    "mixed_chunked_100k",
    "cast_mixed_100k",
];

fn make_mixed_tree(name: &str) -> ArrayRef {
    match name {
        "mixed_chunked_1k" => make_mixed_chunked(1_000),
        "mixed_chunked_100k" => make_mixed_chunked(100_000),
        "cast_mixed_100k" => cast(make_mixed_chunked(100_000)),
        _ => panic!(),
    }
}

#[divan::bench(args = MIXED_TREE_NAMES)]
fn mixed_current(bencher: divan::Bencher, name: &str) {
    let tree = make_mixed_tree(name);
    let code_tree = build_code_tree(&tree);
    bencher.bench(|| {
        black_box(walk_current(black_box(&code_tree)));
    });
}

#[divan::bench(args = MIXED_TREE_NAMES)]
fn mixed_dense(bencher: divan::Bencher, name: &str) {
    let tree = make_mixed_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        black_box(walk_dense(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = MIXED_TREE_NAMES)]
fn mixed_dense_u32(bencher: divan::Bencher, name: &str) {
    let tree = make_mixed_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = DenseU32::new();
    bencher.bench(|| {
        black_box(walk_dense_u32(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = MIXED_TREE_NAMES)]
fn mixed_hashmap(bencher: divan::Bencher, name: &str) {
    let tree = make_mixed_tree(name);
    let code_tree = build_code_tree(&tree);
    let lookup = HashLookup::new();
    bencher.bench(|| {
        black_box(walk_hashmap(&lookup, black_box(&code_tree)));
    });
}

#[divan::bench(args = MIXED_TREE_NAMES)]
fn mixed_report(bencher: divan::Bencher, name: &str) {
    let tree = make_mixed_tree(name);
    let code_tree = build_code_tree(&tree);
    let dense = DenseLookup::new();
    let dense_u32 = DenseU32::new();
    let hash = HashLookup::new();
    let nodes = count_nodes(&code_tree);

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

    eprintln!(
        "  {name}: nodes={nodes}, current_matches={current_calls}, mem: dense={}KB, dense_u32={}KB, hashmap={}B",
        dense.mem_bytes() / 1024,
        dense_u32.mem_bytes() / 1024,
        hash.mem_bytes()
    );
    bencher.bench(|| black_box(0));
}

#[divan::bench(args = TREE_NAMES)]
fn flat_list(bencher: divan::Bencher, name: &str) {
    let tree = make_tree(name);
    let code_tree = build_code_tree(&tree);
    let pairs = flatten_tree(&code_tree);
    let lookup = DenseLookup::new();
    bencher.bench(|| {
        let mut count = 0u64;
        for (p, c) in black_box(&pairs) {
            let rules = lookup.get(*p, *c);
            for f in rules {
                count += (*f as usize) as u64 & 1;
            }
        }
        black_box(count);
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

    // Count current matches() calls (linear scan)
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

    // Count proposed lookups + total rules iterated from lookup result
    fn count_dense(
        lookup: &DenseLookup,
        node: &CodeNode,
        lookups: &mut u64,
        rules_iterated: &mut u64,
        interesting: &mut u64,
    ) {
        if lookup.parent_interesting(node.code) {
            *interesting += 1;
            for child in &node.children {
                *lookups += 1;
                let rules = lookup.get(node.code, child.code);
                *rules_iterated += rules.len() as u64;
            }
        }
        for child in &node.children {
            count_dense(lookup, child, lookups, rules_iterated, interesting);
        }
    }
    let mut dense_lookups = 0u64;
    let mut rules_iterated = 0u64;
    let mut interesting = 0u64;
    count_dense(
        &dense_lookup,
        &code_tree,
        &mut dense_lookups,
        &mut rules_iterated,
        &mut interesting,
    );

    eprintln!(
        "  {name}: nodes={nodes}, interesting_parents={interesting}, current_matches={current_calls}, dense_lookups={dense_lookups}, rules_iterated_from_lookup={rules_iterated}"
    );
    eprintln!(
        "    mem: dense={} bytes, hashmap={} bytes",
        dense_lookup.mem_bytes(),
        hash_lookup.mem_bytes()
    );

    bencher.bench(|| black_box(0));
}
