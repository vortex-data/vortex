// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! Wavelet-tree builders (`Huffman` and `OrderPreserving`) + maximal flat-subtree detection +
//! compact wire serialization for `vortex-pivco`.

#![expect(clippy::cast_possible_truncation)]

use std::cmp::Reverse;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::pow2_bin::EBITS_TABLE;
pub use crate::pow2_bin::MAX_SYMS;

/// Computes `max_ebits` across all leaf symbols of `tree`.
pub(crate) fn compute_tree_max_ebits(tree: &PivcoTree) -> u8 {
    tree.rank_to_sym
        .iter()
        .map(|&s| EBITS_TABLE[usize::from(s)])
        .max()
        .unwrap_or(0)
}

/// Maximum tree depth (`<= 11`, matching `pivco-huffman`'s `PIVCO_MAX_CODE_LEN = 11`).
pub(crate) const MAX_TREE_DEPTH: u8 = 11;

/// Writer-side tree construction strategy (not serialized on the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PivcoTreeMode {
    /// Frequency-optimal flat-aware Huffman tree (`pivco-huffman` `PIVCO_TREE_MODE_OPTIMIZED`).
    Huffman,
    /// Alphabet-ordered probability-bisected tree (`max(left) < min(right)` at every node),
    /// enabling direct `<` / `<=` / `>` / `>=` range predicate pushdown on node bitmaps.
    OrderPreserving,
}

/// Execution schedule node in DFS pre-order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PivcoNode {
    /// Single-symbol chunk (`root` is a leaf).
    SingleLeaf { sym: u8 },
    /// Equal-depth maximal flat subtree of depth `d in 2..=6` (`2^d` leaves at ranks
    /// `base_rank .. base_rank + (1 << d)`).
    Flat {
        depth: u8,
        base_rank: u8,
        c2s: Box<[u8; 64]>,
    },
    /// Binary internal split over ranks `[lo_rank..hi_rank)` with split `pivot_rank`:
    /// `left` handles `< pivot_rank`, `right` handles `>= pivot_rank`.
    Internal {
        tree_depth: u8,
        pivot_rank: u8,
        left_child: ChildSpec,
        right_child: ChildSpec,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildSpec {
    /// Child is a single leaf symbol.
    Leaf(u8),
    /// Child is `nodes[idx]` in the [`PivcoTree`].
    Node(u16),
}

/// Compiled chunk-level wavelet tree shared across all blocks in a [`crate::PivcoArray`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PivcoTree {
    /// Derived property: `true` iff `rank_to_sym` is strictly increasing, enabling direct
    /// `<` / `<=` / `>` / `>=` spine predicate pushdown regardless of how the writer built the tree.
    pub(crate) is_order_preserving: bool,
    /// DFS pre-order execution nodes (`nodes[0]` is the root).
    pub(crate) nodes: Vec<PivcoNode>,
    /// Maps `Pow2-Bin` symbol `0..150` -> contiguous DFS leaf rank `0..num_leaves`.
    pub(crate) sym_to_rank: [u8; MAX_SYMS],
    /// Maps contiguous DFS leaf rank `0..num_leaves` -> `Pow2-Bin` symbol.
    pub(crate) rank_to_sym: Vec<u8>,
    /// Precomputed per-node uniform `ebits`: `Some(eb)` iff every leaf under `nodes[i]` has the
    /// exact same `EBITS_TABLE[sym] == eb`, enabling `O(1)` subtree pruning in top-down `scalar_at`.
    pub(crate) node_uniform_ebits: Vec<Option<u8>>,
    /// Maximum number of nested `Internal` nodes on any root-to-leaf path (`0` when root is
    /// `SingleLeaf` or `Flat`), bounding per-level bitmap scratch in `DecodeWorkspace`.
    pub(crate) max_internal_depth: u8,
}

#[derive(Clone, PartialEq, Eq)]
enum RawTree {
    Leaf(u8),
    Branch(Box<RawTree>, Box<RawTree>),
}

impl RawTree {
    fn min_max_depth(&self) -> (u8, u8) {
        match self {
            Self::Leaf(_) => (0, 0),
            Self::Branch(l, r) => {
                let (lmin, lmax) = l.min_max_depth();
                let (rmin, rmax) = r.min_max_depth();
                (lmin.min(rmin) + 1, lmax.max(rmax) + 1)
            }
        }
    }

    fn collect_leaves(&self, out: &mut Vec<u8>) {
        match self {
            Self::Leaf(s) => out.push(*s),
            Self::Branch(l, r) => {
                l.collect_leaves(out);
                r.collect_leaves(out);
            }
        }
    }
}

impl PivcoTree {
    /// Builds a [`PivcoTree`] from chunk symbol counts.
    pub fn build(counts: &[u64; MAX_SYMS], mode: PivcoTreeMode, allow_flat: bool) -> Self {
        let active: Vec<(u8, u64)> = (0..MAX_SYMS)
            .filter(|&s| counts[s] > 0)
            .map(|s| (s as u8, counts[s]))
            .collect();

        if active.is_empty() {
            return Self::single_symbol(0);
        }
        if active.len() == 1 {
            return Self::single_symbol(active[0].0);
        }

        let raw_root = match mode {
            PivcoTreeMode::Huffman => build_huffman_raw(&active),
            PivcoTreeMode::OrderPreserving => build_order_preserving_raw(&active, 0),
        };

        Self::compile_raw_tree(raw_root, allow_flat)
    }

    /// Returns `true` iff leaf symbols are strictly increasing in DFS rank order (`rank_to_sym`),
    /// enabling direct `<` / `<=` / `>` / `>=` spine predicate pushdown.
    #[inline]
    pub fn is_order_preserving(&self) -> bool {
        self.is_order_preserving
    }

    /// Returns the number of active leaf symbols in this tree (`1..=150`).
    #[inline]
    pub fn num_leaves(&self) -> usize {
        self.rank_to_sym.len()
    }

    pub(crate) fn single_symbol(sym: u8) -> Self {
        let mut sym_to_rank = [0u8; MAX_SYMS];
        sym_to_rank[usize::from(sym)] = 0;
        Self {
            is_order_preserving: true,
            nodes: vec![PivcoNode::SingleLeaf { sym }],
            sym_to_rank,
            rank_to_sym: vec![sym],
            node_uniform_ebits: vec![Some(EBITS_TABLE[usize::from(sym)])],
            max_internal_depth: 0,
        }
    }

    fn compile_raw_tree(raw_root: RawTree, allow_flat: bool) -> Self {
        let mut rank_to_sym = Vec::new();
        raw_root.collect_leaves(&mut rank_to_sym);
        let mut sym_to_rank = [0u8; MAX_SYMS];
        for (rank, &sym) in rank_to_sym.iter().enumerate() {
            sym_to_rank[usize::from(sym)] = rank as u8;
        }
        let is_order_preserving = rank_to_sym.windows(2).all(|w| w[0] < w[1]);

        let mut nodes = Vec::new();
        compile_subtree(&raw_root, 0, &mut 0u8, allow_flat, &mut nodes);
        let node_uniform_ebits = compute_node_uniform_ebits(&nodes);
        let max_internal_depth = compute_max_internal_depth(&nodes);
        Self {
            is_order_preserving,
            nodes,
            sym_to_rank,
            rank_to_sym,
            node_uniform_ebits,
            max_internal_depth,
        }
    }

    /// Compactly serializes the tree as pre-order node opcodes (`0x00` = leaf, `0x01` = internal,
    /// `0x02..=0x06` = flat subtree of depth `d` followed by `2^d` symbol bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nodes.len() * 4);
        serialize_node(self, 0, &mut out);
        out
    }

    /// Reconstructs a validated [`PivcoTree`] from serialized pre-order opcode bytes.
    pub fn from_bytes(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(!bytes.is_empty(), "Empty PivcoTree metadata");
        let mut pos = 0usize;
        let mut seen = [false; MAX_SYMS];
        let mut sym_to_rank = [0u8; MAX_SYMS];
        let mut rank_to_sym = Vec::new();
        let mut nodes = Vec::new();

        deserialize_root(
            bytes,
            &mut pos,
            &mut seen,
            &mut sym_to_rank,
            &mut rank_to_sym,
            &mut nodes,
        )?;
        vortex_ensure_eq!(pos, bytes.len(), "Trailing bytes in PivcoTree metadata");

        let is_order_preserving = rank_to_sym.windows(2).all(|w| w[0] < w[1]);
        let node_uniform_ebits = compute_node_uniform_ebits(&nodes);
        let max_internal_depth = compute_max_internal_depth(&nodes);
        Ok(Self {
            is_order_preserving,
            nodes,
            sym_to_rank,
            rank_to_sym,
            node_uniform_ebits,
            max_internal_depth,
        })
    }
}

fn compute_max_internal_depth(nodes: &[PivcoNode]) -> u8 {
    let mut depth = vec![0u8; nodes.len()];
    // Nodes are stored in DFS pre-order so children always have indices > parent.
    for idx in (0..nodes.len()).rev() {
        if let PivcoNode::Internal {
            left_child,
            right_child,
            ..
        } = &nodes[idx]
        {
            let l_depth = match left_child {
                ChildSpec::Leaf(_) => 0u8,
                ChildSpec::Node(c) => depth[usize::from(*c)],
            };
            let r_depth = match right_child {
                ChildSpec::Leaf(_) => 0u8,
                ChildSpec::Node(c) => depth[usize::from(*c)],
            };
            depth[idx] = 1 + l_depth.max(r_depth);
        }
    }
    depth.first().copied().unwrap_or(0)
}

fn compute_node_uniform_ebits(nodes: &[PivcoNode]) -> Vec<Option<u8>> {
    let mut uniform = vec![None; nodes.len()];
    // Nodes are stored in DFS pre-order so children always have indices > parent.
    for idx in (0..nodes.len()).rev() {
        uniform[idx] = match &nodes[idx] {
            PivcoNode::SingleLeaf { sym } => Some(EBITS_TABLE[usize::from(*sym)]),
            PivcoNode::Flat { depth, c2s, .. } => {
                let count = 1usize << *depth;
                let first = EBITS_TABLE[usize::from(c2s[0])];
                c2s[1..count]
                    .iter()
                    .all(|&sym| EBITS_TABLE[usize::from(sym)] == first)
                    .then_some(first)
            }
            PivcoNode::Internal {
                left_child,
                right_child,
                ..
            } => {
                let left_eb = match left_child {
                    ChildSpec::Leaf(sym) => Some(EBITS_TABLE[usize::from(*sym)]),
                    ChildSpec::Node(child) => uniform[usize::from(*child)],
                };
                let right_eb = match right_child {
                    ChildSpec::Leaf(sym) => Some(EBITS_TABLE[usize::from(*sym)]),
                    ChildSpec::Node(child) => uniform[usize::from(*child)],
                };
                match (left_eb, right_eb) {
                    (Some(le), Some(re)) if le == re => Some(le),
                    _ => None,
                }
            }
        };
    }
    uniform
}

fn compile_subtree(
    tree: &RawTree,
    tree_depth: u8,
    next_rank: &mut u8,
    allow_flat: bool,
    nodes: &mut Vec<PivcoNode>,
) -> u16 {
    let idx = nodes.len() as u16;
    nodes.push(PivcoNode::SingleLeaf { sym: 0 }); // placeholder

    match tree {
        RawTree::Leaf(sym) => {
            *next_rank += 1;
            nodes[usize::from(idx)] = PivcoNode::SingleLeaf { sym: *sym };
        }
        RawTree::Branch(left, right) => {
            let (min_d, max_d) = tree.min_max_depth();
            if allow_flat && min_d == max_d && (MIN_FLAT_DEPTH..=MAX_FLAT_DEPTH).contains(&min_d) {
                let base_rank = *next_rank;
                let mut leaves = Vec::with_capacity(1usize << min_d);
                tree.collect_leaves(&mut leaves);
                *next_rank += leaves.len() as u8;
                let mut c2s = Box::new([0u8; 64]);
                for (i, &s) in leaves.iter().enumerate() {
                    c2s[i] = s;
                }
                nodes[usize::from(idx)] = PivcoNode::Flat {
                    depth: min_d,
                    base_rank,
                    c2s,
                };
                return idx;
            }

            let left_spec = match left.as_ref() {
                RawTree::Leaf(s) => {
                    *next_rank += 1;
                    ChildSpec::Leaf(*s)
                }
                other => ChildSpec::Node(compile_subtree(
                    other,
                    tree_depth + 1,
                    next_rank,
                    allow_flat,
                    nodes,
                )),
            };
            let pivot_rank = *next_rank;
            let right_spec = match right.as_ref() {
                RawTree::Leaf(s) => {
                    *next_rank += 1;
                    ChildSpec::Leaf(*s)
                }
                other => ChildSpec::Node(compile_subtree(
                    other,
                    tree_depth + 1,
                    next_rank,
                    allow_flat,
                    nodes,
                )),
            };

            nodes[usize::from(idx)] = PivcoNode::Internal {
                tree_depth,
                pivot_rank,
                left_child: left_spec,
                right_child: right_spec,
            };
        }
    }
    idx
}

const TAG_LEAF: u8 = 0x00;
const TAG_INTERNAL: u8 = 0x01;
const MIN_FLAT_DEPTH: u8 = 0x02;
const MAX_FLAT_DEPTH: u8 = 0x06;

fn serialize_node(tree: &PivcoTree, idx: u16, out: &mut Vec<u8>) {
    match &tree.nodes[usize::from(idx)] {
        PivcoNode::SingleLeaf { sym } => {
            out.push(TAG_LEAF);
            out.push(*sym);
        }
        PivcoNode::Flat { depth, c2s, .. } => {
            out.push(*depth);
            out.extend_from_slice(&c2s[..1usize << *depth]);
        }
        PivcoNode::Internal {
            left_child,
            right_child,
            ..
        } => {
            out.push(TAG_INTERNAL);
            match left_child {
                ChildSpec::Leaf(s) => {
                    out.push(TAG_LEAF);
                    out.push(*s);
                }
                ChildSpec::Node(n) => serialize_node(tree, *n, out),
            }
            match right_child {
                ChildSpec::Leaf(s) => {
                    out.push(TAG_LEAF);
                    out.push(*s);
                }
                ChildSpec::Node(n) => serialize_node(tree, *n, out),
            }
        }
    }
}

fn read_validated_sym(
    bytes: &[u8],
    pos: &mut usize,
    seen: &mut [bool; MAX_SYMS],
    sym_to_rank: &mut [u8; MAX_SYMS],
    rank_to_sym: &mut Vec<u8>,
) -> VortexResult<u8> {
    vortex_ensure!(*pos < bytes.len(), "Unexpected EOF reading PivcoTree leaf");
    let sym = bytes[*pos];
    *pos += 1;
    vortex_ensure!(
        usize::from(sym) < MAX_SYMS,
        "PivcoTree leaf symbol {sym} >= {MAX_SYMS}"
    );
    vortex_ensure!(!seen[usize::from(sym)], "Duplicate leaf symbol {sym}");
    seen[usize::from(sym)] = true;
    let rank = rank_to_sym.len() as u8;
    sym_to_rank[usize::from(sym)] = rank;
    rank_to_sym.push(sym);
    Ok(sym)
}

fn deserialize_root(
    bytes: &[u8],
    pos: &mut usize,
    seen: &mut [bool; MAX_SYMS],
    sym_to_rank: &mut [u8; MAX_SYMS],
    rank_to_sym: &mut Vec<u8>,
    nodes: &mut Vec<PivcoNode>,
) -> VortexResult<()> {
    vortex_ensure!(*pos < bytes.len(), "Unexpected EOF in PivcoTree");
    match bytes[*pos] {
        TAG_LEAF => {
            *pos += 1;
            let sym = read_validated_sym(bytes, pos, seen, sym_to_rank, rank_to_sym)?;
            nodes.push(PivcoNode::SingleLeaf { sym });
            Ok(())
        }
        _ => {
            let _ = deserialize_non_leaf(bytes, pos, 0, seen, sym_to_rank, rank_to_sym, nodes)?;
            Ok(())
        }
    }
}

fn deserialize_child(
    bytes: &[u8],
    pos: &mut usize,
    tree_depth: u8,
    seen: &mut [bool; MAX_SYMS],
    sym_to_rank: &mut [u8; MAX_SYMS],
    rank_to_sym: &mut Vec<u8>,
    nodes: &mut Vec<PivcoNode>,
) -> VortexResult<ChildSpec> {
    vortex_ensure!(
        tree_depth <= MAX_TREE_DEPTH,
        "PivcoTree exceeds max depth {MAX_TREE_DEPTH}"
    );
    vortex_ensure!(*pos < bytes.len(), "Unexpected EOF in PivcoTree");
    if bytes[*pos] == TAG_LEAF {
        *pos += 1;
        let sym = read_validated_sym(bytes, pos, seen, sym_to_rank, rank_to_sym)?;
        Ok(ChildSpec::Leaf(sym))
    } else {
        let idx = deserialize_non_leaf(
            bytes,
            pos,
            tree_depth,
            seen,
            sym_to_rank,
            rank_to_sym,
            nodes,
        )?;
        Ok(ChildSpec::Node(idx))
    }
}

fn deserialize_non_leaf(
    bytes: &[u8],
    pos: &mut usize,
    tree_depth: u8,
    seen: &mut [bool; MAX_SYMS],
    sym_to_rank: &mut [u8; MAX_SYMS],
    rank_to_sym: &mut Vec<u8>,
    nodes: &mut Vec<PivcoNode>,
) -> VortexResult<u16> {
    vortex_ensure!(*pos < bytes.len(), "Unexpected EOF in PivcoTree");
    let tag = bytes[*pos];
    *pos += 1;
    match tag {
        TAG_INTERNAL => {
            vortex_ensure!(
                tree_depth < MAX_TREE_DEPTH,
                "PivcoTree internal node at depth {tree_depth} exceeds max depth {MAX_TREE_DEPTH}"
            );
            let idx = nodes.len() as u16;
            nodes.push(PivcoNode::SingleLeaf { sym: 0 }); // placeholder
            let left_child = deserialize_child(
                bytes,
                pos,
                tree_depth + 1,
                seen,
                sym_to_rank,
                rank_to_sym,
                nodes,
            )?;
            let pivot_rank = rank_to_sym.len() as u8;
            let right_child = deserialize_child(
                bytes,
                pos,
                tree_depth + 1,
                seen,
                sym_to_rank,
                rank_to_sym,
                nodes,
            )?;
            nodes[usize::from(idx)] = PivcoNode::Internal {
                tree_depth,
                pivot_rank,
                left_child,
                right_child,
            };
            Ok(idx)
        }
        d @ MIN_FLAT_DEPTH..=MAX_FLAT_DEPTH => {
            vortex_ensure!(
                tree_depth.saturating_add(d) <= MAX_TREE_DEPTH,
                "PivcoTree flat subtree at depth {tree_depth} + {d} exceeds max depth {MAX_TREE_DEPTH}"
            );
            let count = 1usize << d;
            vortex_ensure!(
                bytes.len().saturating_sub(*pos) >= count,
                "Unexpected EOF reading PivcoTree flat subtree of depth {d}"
            );
            let base_rank = rank_to_sym.len() as u8;
            let mut c2s = Box::new([0u8; 64]);
            for slot in c2s.iter_mut().take(count) {
                *slot = read_validated_sym(bytes, pos, seen, sym_to_rank, rank_to_sym)?;
            }
            let idx = nodes.len() as u16;
            nodes.push(PivcoNode::Flat {
                depth: d,
                base_rank,
                c2s,
            });
            Ok(idx)
        }
        other => vortex_bail!("Invalid PivcoTree opcode {other:#x}"),
    }
}

/// Builds a length-limited (`<= 11`) flat-aware Huffman tree using van Leeuwen's `O(n)` two-queue
/// length builder and Marcin Żukowski's set-bit chunk decomposition (`PIVCO_TREE_MODE_OPTIMIZED`).
fn build_huffman_raw(active: &[(u8, u64)]) -> RawTree {
    let mut sym_lens = build_lengths_two_queue(active);
    let max_len = sym_lens.iter().map(|&(_, len, _)| len).max().unwrap_or(1);
    if max_len > MAX_TREE_DEPTH {
        clamp_kraft_lengths(&mut sym_lens, MAX_TREE_DEPTH);
    }
    build_flat_aware_tree(&sym_lens)
}

/// Computes optimal unconstrained Huffman code lengths via van Leeuwen's `O(n)` two-queue merge.
///
/// Leaves arrive in ascending symbol order; stably sorting by frequency reproduces binary-heap
/// tie-breaking (`<=` prefers a leaf over an internal node on equal weight).
fn build_lengths_two_queue(active: &[(u8, u64)]) -> Vec<(u8, u8, u64)> {
    let n_used = active.len();
    debug_assert!((2..=MAX_SYMS).contains(&n_used));

    let mut leaves: [(u8, u64); MAX_SYMS] = [(0, 0); MAX_SYMS];
    for (slot, &(sym, w)) in leaves[..n_used].iter_mut().zip(active.iter()) {
        *slot = (sym, w.max(1));
    }
    // Stable sort by weight ascending; leaves were collected in ascending symbol order.
    leaves[..n_used].sort_by_key(|&(_, w)| w);

    // Nodes `0..n_used` are sorted leaves; `n_used..2*n_used-1` are internal nodes in creation
    // order (monotonically non-decreasing frequency).
    let mut nfreq = [0u64; MAX_SYMS * 2];
    let mut parent = [0usize; MAX_SYMS * 2];
    for i in 0..n_used {
        nfreq[i] = leaves[i].1;
    }

    let mut leaf_head = 0usize;
    let mut int_head = n_used;
    let mut next_int = n_used;
    for _ in 0..(n_used - 1) {
        let first = if leaf_head < n_used
            && (int_head == next_int || nfreq[leaf_head] <= nfreq[int_head])
        {
            let idx = leaf_head;
            leaf_head += 1;
            idx
        } else {
            let idx = int_head;
            int_head += 1;
            idx
        };
        let second = if leaf_head < n_used
            && (int_head == next_int || nfreq[leaf_head] <= nfreq[int_head])
        {
            let idx = leaf_head;
            leaf_head += 1;
            idx
        } else {
            let idx = int_head;
            int_head += 1;
            idx
        };
        nfreq[next_int] = nfreq[first].saturating_add(nfreq[second]);
        parent[first] = next_int;
        parent[second] = next_int;
        next_int += 1;
    }

    let root = next_int - 1;
    let mut depth = [0u8; MAX_SYMS * 2];
    depth[root] = 0;
    for i in (0..root).rev() {
        depth[i] = depth[parent[i]].saturating_add(1);
    }

    let mut sym_lens = Vec::with_capacity(n_used);
    for i in 0..n_used {
        let (sym, weight) = leaves[i];
        sym_lens.push((sym, depth[i].max(1), weight));
    }
    sym_lens
}

fn clamp_kraft_lengths(sym_lens: &mut [(u8, u8, u64)], max_len: u8) {
    let scale = 1i64 << max_len;
    let mut kraft = 0i64;
    for entry in sym_lens.iter_mut() {
        if entry.1 > max_len {
            entry.1 = max_len;
        }
        kraft += 1i64 << (max_len - entry.1);
    }
    // If clamping caused Kraft sum > 2^max_len, lengthen symbols with smallest weight (`< max_len`).
    if kraft > scale {
        // Order: least frequent symbols with `len < max_len` get lengthened first.
        sym_lens.sort_by_key(|&(sym, _, w)| (w, Reverse(sym)));
        while kraft > scale {
            for entry in sym_lens.iter_mut() {
                if kraft <= scale {
                    break;
                }
                if entry.1 < max_len {
                    let old_w = 1i64 << (max_len - entry.1);
                    entry.1 += 1;
                    let new_w = 1i64 << (max_len - entry.1);
                    kraft -= old_w - new_w;
                }
            }
        }
    }
    // If Kraft sum < 2^max_len, shorten most frequent symbols so tree is strictly full binary.
    if kraft < scale {
        sym_lens.sort_by_key(|&(sym, _, w)| (Reverse(w), sym));
        let mut changed = true;
        while kraft < scale && changed {
            changed = false;
            for entry in sym_lens.iter_mut() {
                if entry.1 > 1 {
                    let delta = 1i64 << (max_len - entry.1);
                    if kraft + delta <= scale {
                        entry.1 -= 1;
                        kraft += delta;
                        changed = true;
                    }
                }
            }
        }
    }
    debug_assert_eq!(kraft, scale);
}

/// Restructures Huffman code lengths into a tree maximizing `2^D` flat subtrees (`D in 2..=6`)
/// without changing any symbol's code length (`pivco-huffman`'s `PIVCO_TREE_MODE_OPTIMIZED`).
///
/// For each length `L`, the symbol count `c_L` is decomposed by its binary set bits high-to-low:
/// - bits `>= 7` are split into `2^(bit - 6)` chunks of `64` (`D = 6`) rooted at `L - 6`,
/// - bits `2..=6` form `D = bit` complete subtrees rooted at `L - bit`,
/// - bit `1` forms a `D = 1` sibling pair rooted at `L - 1`,
/// - bit `0` forms a singleton leaf at depth `L`.
///
/// Chunks are stably sorted by root depth ascending (keeping `L` asc and larger-bit-first within
/// ties so lone leaves always land on left children) and paired bottom-up into a full binary tree.
fn build_flat_aware_tree(sym_lens: &[(u8, u8, u64)]) -> RawTree {
    // Counting sort symbols by `(len, sym)` in O(MAX_SYMS).
    let mut lengths = [0u8; MAX_SYMS];
    let mut sym_count = [0usize; MAX_TREE_DEPTH as usize + 1];
    let mut max_len = 0usize;
    for &(sym, len, _) in sym_lens {
        let len_usize = usize::from(len);
        lengths[usize::from(sym)] = len;
        sym_count[len_usize] += 1;
        if len_usize > max_len {
            max_len = len_usize;
        }
    }

    let mut flat_items = [0u8; MAX_SYMS];
    let mut per_len_start = [0usize; MAX_TREE_DEPTH as usize + 2];
    let mut cursor = [0usize; MAX_TREE_DEPTH as usize + 2];
    let mut acc = 0usize;
    for len in 1..=max_len {
        per_len_start[len] = acc;
        cursor[len] = acc;
        acc += sym_count[len];
    }
    per_len_start[max_len + 1] = acc;
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 {
            let len_usize = usize::from(len);
            flat_items[cursor[len_usize]] = sym as u8;
            cursor[len_usize] += 1;
        }
    }

    let mut chunks: Vec<(u8, RawTree)> = Vec::with_capacity(sym_lens.len());
    for len in 1..=max_len {
        let count = sym_count[len];
        if count == 0 {
            continue;
        }
        let mut cur = per_len_start[len];
        for bit in (0..=usize::from(MAX_TREE_DEPTH)).rev() {
            if (count & (1usize << bit)) == 0 {
                continue;
            }
            if bit > usize::from(MAX_FLAT_DEPTH) {
                // Split `2^bit` (`bit >= 7`) into `2^(bit - 6)` chunks of `64` (`D = 6`) so every
                // flat subtree stays within the `0x02..=0x06` wire opcode budget.
                let chunk_bits = usize::from(MAX_FLAT_DEPTH);
                let chunk_syms = 1usize << chunk_bits;
                let num_subchunks = 1usize << (bit - chunk_bits);
                let root_depth = (len - chunk_bits) as u8;
                for _ in 0..num_subchunks {
                    chunks.push((
                        root_depth,
                        build_complete_raw_tree(&flat_items[cur..cur + chunk_syms]),
                    ));
                    cur += chunk_syms;
                }
            } else {
                let n_syms = 1usize << bit;
                let root_depth = (len - bit) as u8;
                chunks.push((
                    root_depth,
                    build_complete_raw_tree(&flat_items[cur..cur + n_syms]),
                ));
                cur += n_syms;
            }
        }
    }

    // Stable sort by root depth ascending (preserves `L` asc, larger-bit-first within each `L`).
    chunks.sort_by_key(|&(root_depth, _)| root_depth);
    pair_chunks_bottom_up(chunks)
}

fn build_complete_raw_tree(syms: &[u8]) -> RawTree {
    debug_assert!(syms.len().is_power_of_two());
    if syms.len() == 1 {
        return RawTree::Leaf(syms[0]);
    }
    let mid = syms.len() / 2;
    RawTree::Branch(
        Box::new(build_complete_raw_tree(&syms[..mid])),
        Box::new(build_complete_raw_tree(&syms[mid..])),
    )
}

fn pair_chunks_bottom_up(mut items: Vec<(u8, RawTree)>) -> RawTree {
    // Repeatedly pair the leftmost two chunks at the deepest root depth `d` into `d - 1`.
    // Because `items` stays sorted by `root_depth` ascending and the chunk-level Kraft sum is 1,
    // the count at `max_d` is always even and the leftmost two chunks at `max_d` are adjacent.
    while items.len() > 1 {
        let max_d = items
            .iter()
            .map(|(d, _)| *d)
            .max()
            .vortex_expect("items is non-empty");
        let pos = items
            .iter()
            .position(|(d, _)| *d == max_d)
            .vortex_expect("max_d must exist in items");
        let (_, left) = items.remove(pos);
        let (d2, right) = items.remove(pos);
        debug_assert_eq!(max_d, d2);
        items.insert(
            pos,
            (max_d - 1, RawTree::Branch(Box::new(left), Box::new(right))),
        );
    }
    items.pop().vortex_expect("items has root tree").1
}

/// Builds an alphabet-ordered (`sym` strictly increasing left-to-right) probability-balanced binary
/// tree (Shannon-Fano bisection) strictly bounded to `MAX_TREE_DEPTH` (`11`) so `< const` / `== const`
/// predicates evaluate directly on the tree.
fn build_order_preserving_raw(active_sorted: &[(u8, u64)], depth: u8) -> RawTree {
    let n = active_sorted.len();
    if n == 1 {
        return RawTree::Leaf(active_sorted[0].0);
    }
    if n == 2 {
        return RawTree::Branch(
            Box::new(RawTree::Leaf(active_sorted[0].0)),
            Box::new(RawTree::Leaf(active_sorted[1].0)),
        );
    }

    // Bound each child's leaf count so the remaining depth budget `MAX_TREE_DEPTH - depth - 1`
    // always suffices (`n <= 150 <= 2^11`, so `min_split <= max_split` holds inductively).
    let remaining_child_depth = MAX_TREE_DEPTH.saturating_sub(depth + 1);
    let max_child_leaves = 1usize << remaining_child_depth;
    let min_split = n.saturating_sub(max_child_leaves).max(1);
    let max_split = max_child_leaves.min(n - 1);

    let total: u64 = active_sorted.iter().map(|&(_, w)| w).sum();
    let half = total / 2;
    let mut cum: u64 = active_sorted[..min_split - 1].iter().map(|&(_, w)| w).sum();
    let mut best_split = min_split;
    let mut best_diff = u64::MAX;
    for split in min_split..=max_split {
        cum += active_sorted[split - 1].1;
        let diff = cum.abs_diff(half);
        if diff < best_diff {
            best_diff = diff;
            best_split = split;
        }
    }

    RawTree::Branch(
        Box::new(build_order_preserving_raw(
            &active_sorted[..best_split],
            depth + 1,
        )),
        Box::new(build_order_preserving_raw(
            &active_sorted[best_split..],
            depth + 1,
        )),
    )
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;

    fn collect_compiled_depths(
        tree: &PivcoTree,
        node_idx: u16,
        cur_depth: u8,
        out: &mut [u8; MAX_SYMS],
    ) {
        match &tree.nodes[usize::from(node_idx)] {
            PivcoNode::SingleLeaf { sym } => {
                out[usize::from(*sym)] = cur_depth;
            }
            PivcoNode::Flat { depth, c2s, .. } => {
                for &sym in &c2s[..1usize << *depth] {
                    out[usize::from(sym)] = cur_depth + *depth;
                }
            }
            PivcoNode::Internal {
                left_child,
                right_child,
                ..
            } => {
                match left_child {
                    ChildSpec::Leaf(sym) => out[usize::from(*sym)] = cur_depth + 1,
                    ChildSpec::Node(idx) => collect_compiled_depths(tree, *idx, cur_depth + 1, out),
                }
                match right_child {
                    ChildSpec::Leaf(sym) => out[usize::from(*sym)] = cur_depth + 1,
                    ChildSpec::Node(idx) => collect_compiled_depths(tree, *idx, cur_depth + 1, out),
                }
            }
        }
    }

    #[test]
    fn shifted_equal_length_run_forms_maximal_flat_chunk() -> VortexResult<()> {
        // 1 symbol at depth 2 + 5 symbols at depth 3 + 2 symbols at depth 4
        // (Kraft sum: 1/4 + 5/8 + 2/16 = 1). Pure canonical code assignment fragments the 5
        // depth-3 symbols across the 01* / 10* / 11* boundaries with zero Flat(d=2) chunks;
        // set-bit decomposition must group 4 of them into one Flat(d=2) chunk plus 1 singleton.
        let mut counts = [0u64; MAX_SYMS];
        counts[0] = 250; // len 2
        counts[1] = 130; // len 3
        counts[2] = 128; // len 3
        counts[3] = 126; // len 3
        counts[4] = 124; // len 3
        counts[5] = 122; // len 3
        counts[6] = 60; // len 4
        counts[7] = 60; // len 4

        let tree = PivcoTree::build(&counts, PivcoTreeMode::Huffman, true);
        let flat_depths: Vec<u8> = tree
            .nodes
            .iter()
            .filter_map(|n| match n {
                PivcoNode::Flat { depth, .. } => Some(*depth),
                _ => None,
            })
            .collect();
        assert_eq!(flat_depths, vec![2]);

        // Verify exact Huffman code lengths are preserved.
        let mut compiled_depths = [0u8; MAX_SYMS];
        collect_compiled_depths(&tree, 0, 0, &mut compiled_depths);
        assert_eq!(compiled_depths[0], 2);
        for &d in &compiled_depths[1..=5] {
            assert_eq!(d, 3);
        }
        assert_eq!(compiled_depths[6], 4);
        assert_eq!(compiled_depths[7], 4);

        // Verify lone leaves always sit on the left child in Huffman mode.
        for node in &tree.nodes {
            if let PivcoNode::Internal {
                left_child,
                right_child,
                ..
            } = node
            {
                assert!(
                    !(matches!(left_child, ChildSpec::Node(_))
                        && matches!(right_child, ChildSpec::Leaf(_))),
                    "unexpected LEAF_RIGHT internal node in Huffman mode: {node:?}"
                );
            }
        }

        let roundtrip = PivcoTree::from_bytes(&tree.to_bytes())?;
        assert_eq!(roundtrip, tree);
        Ok(())
    }

    #[test]
    fn multi_tier_shifted_runs_form_all_maximal_flat_chunks() -> VortexResult<()> {
        // 1 sym at depth 2 (1/4) + 10 syms at depth 4 (10/16) + 4 syms at depth 5 (4/32) = 1.
        // c_4 = 10 = 8 + 2 -> Flat(d=3); c_5 = 4 -> Flat(d=2).
        let mut counts = [0u64; MAX_SYMS];
        counts[0] = 2500;
        for slot in &mut counts[1..=10] {
            *slot = 625;
        }
        for slot in &mut counts[11..=14] {
            *slot = 312;
        }

        let tree = PivcoTree::build(&counts, PivcoTreeMode::Huffman, true);
        let mut flat_depths: Vec<u8> = tree
            .nodes
            .iter()
            .filter_map(|n| match n {
                PivcoNode::Flat { depth, .. } => Some(*depth),
                _ => None,
            })
            .collect();
        flat_depths.sort_unstable();
        assert_eq!(flat_depths, vec![2, 3]);

        let roundtrip = PivcoTree::from_bytes(&tree.to_bytes())?;
        assert_eq!(roundtrip, tree);
        Ok(())
    }

    #[test]
    fn large_uniform_alphabets_split_bit7_into_flat6_and_roundtrip() -> VortexResult<()> {
        // 128 uniform symbols -> c_7 = 128 -> splits bit 7 into two Flat(d=6) subtrees.
        let mut counts128 = [0u64; MAX_SYMS];
        counts128[..128].fill(10);
        let tree128 = PivcoTree::build(&counts128, PivcoTreeMode::Huffman, true);
        let flat128: Vec<u8> = tree128
            .nodes
            .iter()
            .filter_map(|n| match n {
                PivcoNode::Flat { depth, .. } => Some(*depth),
                _ => None,
            })
            .collect();
        assert_eq!(flat128, vec![6, 6]);
        assert_eq!(PivcoTree::from_bytes(&tree128.to_bytes())?, tree128);

        // Full 150 uniform symbols -> c_7 = 106 (64+32+8+2), c_8 = 44 (32+8+4).
        let counts150 = [10u64; MAX_SYMS];
        let tree150 = PivcoTree::build(&counts150, PivcoTreeMode::Huffman, true);
        let mut flat150: Vec<u8> = tree150
            .nodes
            .iter()
            .filter_map(|n| match n {
                PivcoNode::Flat { depth, .. } => Some(*depth),
                _ => None,
            })
            .collect();
        flat150.sort_unstable();
        assert_eq!(flat150, vec![2, 3, 3, 5, 5, 6]);
        assert_eq!(PivcoTree::from_bytes(&tree150.to_bytes())?, tree150);

        // Exponentially skewed 40 symbols forcing length clamping to MAX_TREE_DEPTH = 11.
        let mut fib = [0u64; MAX_SYMS];
        let mut a = 1u64;
        let mut b = 1u64;
        for slot in &mut fib[..40] {
            *slot = a;
            let next = a.saturating_add(b);
            a = b;
            b = next;
        }
        let tree_fib = PivcoTree::build(&fib, PivcoTreeMode::Huffman, true);
        let mut depths = [0u8; MAX_SYMS];
        collect_compiled_depths(&tree_fib, 0, 0, &mut depths);
        assert!(
            depths[..40]
                .iter()
                .all(|&d| (1..=MAX_TREE_DEPTH).contains(&d))
        );
        assert_eq!(PivcoTree::from_bytes(&tree_fib.to_bytes())?, tree_fib);
        Ok(())
    }
}
