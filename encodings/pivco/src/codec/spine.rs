// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! Direct spine-walking bitmap predicate pushdown (`Eq` / `NotEq` and `Lt`..`Gte`).

#![expect(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::BlockReader;
use super::DecodeWorkspace;
use super::MAX_BM_BYTES;
use super::read_left_subtree_bytes;
use crate::bitmap_rans::DenseBitmap;
use crate::bitmap_rans::NodeBitmap;
use crate::simd::MERGE_OVERREAD;
use crate::simd::merge_cst_cst;
use crate::simd::merge_cst_vec_in_place;
use crate::simd::merge_vec_cst_in_place;
use crate::simd::pack_64_bytes01;
use crate::simd::popcount_bits;
use crate::simd::unpack_flat_codes;
use crate::tree::ChildSpec;
use crate::tree::PivcoNode;
use crate::tree::PivcoTree;

/// Spine predicate target in DFS rank space (`0..num_leaves`).
#[derive(Clone, Copy)]
pub(crate) enum RankPredicate {
    /// `rank == target_rank` (exact equality when `ebits(sym) == 0`, valid in ALL tree modes).
    Eq(u8),
    /// `rank < split_rank` (exact range `< threshold` in `OrderPreserving` mode).
    Lt(u8),
}

impl DecodeWorkspace {
    pub(super) fn eval_spine_block(
        &mut self,
        tree: &PivcoTree,
        block_len: usize,
        pred: RankPredicate,
        block_tree_bytes: &[u8],
    ) -> VortexResult<&[u8]> {
        self.ensure_for_tree(tree, block_len);
        let mut cursor = 0usize;
        eval_spine_node(
            tree,
            0,
            0,
            tree.rank_to_sym.len() as u8,
            block_len,
            pred,
            block_tree_bytes,
            &mut cursor,
            &mut self.root_syms[..block_len + MERGE_OVERREAD],
            &mut self.bm_stack,
            self.bm_stride,
        )?;
        Ok(&self.root_syms[..block_len])
    }
}

impl BlockReader<'_> {
    /// Evaluates `pred` across this sliced [`crate::PivcoArray`] by walking ONLY the `O(depth)`
    /// spine of each block's wavelet tree (never decoding sibling subtrees or `raw_bits`),
    /// optionally negating (`NotEq`, `Gte`, `Gt`).
    pub(crate) fn evaluate_spine_predicate(
        &self,
        pred: RankPredicate,
        negate: bool,
    ) -> VortexResult<BitBuffer> {
        if self.len == 0 {
            return Ok(BitBuffer::empty());
        }
        let block_values = self.block_values();
        let offset = usize::from(self.offset);
        let total_end = offset + self.len;
        let num_blocks = self.num_blocks();
        let xor_u64 = if negate { !0u64 } else { 0u64 };

        // Fast root-level special case: when `offset == 0` and the predicate is decided purely by
        // the root node's bitmap, copy/invert the root bitmap bytes straight into packed bits
        // without byte-expansion or SIMD merge.
        if offset == 0
            && let Some(root_invert) = root_only_decision(self.tree, pred)
        {
            let invert_bits = root_invert ^ negate;
            let out_bytes_len = self.len.div_ceil(8);
            let mut packed_out = ByteBufferMut::with_capacity(out_bytes_len);
            let mut bm_scratch = [MaybeUninit::<u8>::uninit(); MAX_BM_BYTES];
            let mut written_bits = 0usize;

            for block_idx in 0..num_blocks {
                let block_len = self.block_len(block_idx);
                let copy_bits = (self.len - written_bits).min(block_len);
                let copy_bytes = copy_bits.div_ceil(8);

                let block_tree_bytes = self.block_tree_slice(block_idx);
                let mut cursor = 0usize;
                let bm =
                    NodeBitmap::read(block_tree_bytes, &mut cursor, block_len, &mut bm_scratch)?;

                // SAFETY: `bm_scratch` was just initialized by `NodeBitmap::read` for `bm`.
                match unsafe { bm.into_dense(&mut bm_scratch, block_len) } {
                    DenseBitmap::AllZero => {
                        packed_out.push_n(if invert_bits { 0xFF } else { 0x00 }, copy_bytes);
                    }
                    DenseBitmap::AllOne => {
                        packed_out.push_n(if invert_bits { 0x00 } else { 0xFF }, copy_bytes);
                    }
                    DenseBitmap::Bytes(s) => {
                        if invert_bits {
                            for &b in &s[..copy_bytes] {
                                packed_out.push(!b);
                            }
                        } else {
                            packed_out.extend_from_slice(&s[..copy_bytes]);
                        }
                    }
                }
                written_bits += copy_bits;
            }
            let rem_bits = self.len & 7;
            if rem_bits != 0
                && let Some(last) = packed_out.last_mut()
            {
                *last &= (1u8 << rem_bits) - 1;
            }
            return Ok(BitBuffer::new(packed_out.freeze(), self.len));
        }

        // General spine-walking path (`O(depth)` merges along a single path), writing packed `u64`
        // words directly into an aligned `BufferMut<u64>` with zero extra copy.
        let mut ws = DecodeWorkspace::for_tree(self.tree, block_values);
        let mut u64_words = BufferMut::<u64>::with_capacity(self.len.div_ceil(64));
        let mut bit_acc = 0u64;
        let mut acc_bits = 0usize;

        for block_idx in 0..num_blocks {
            let block_abs_start = block_idx * block_values;
            let in_block_start = if block_idx == 0 { offset } else { 0 };
            let in_block_end = (total_end - block_abs_start).min(block_values);

            let block_len = self.block_len(block_idx);
            let block_tree_bytes = self.block_tree_slice(block_idx);
            let block01 = ws.eval_spine_block(self.tree, block_len, pred, block_tree_bytes)?;
            let slice01 = &block01[in_block_start..in_block_end];

            let mut i = 0usize;
            if acc_bits != 0 {
                let need = (64 - acc_bits).min(slice01.len());
                for &b in &slice01[..need] {
                    if (b != 0) ^ negate {
                        bit_acc |= 1u64 << acc_bits;
                    }
                    acc_bits += 1;
                }
                i = need;
                if acc_bits == 64 {
                    u64_words.push(bit_acc.to_le());
                    bit_acc = 0;
                    acc_bits = 0;
                }
            }
            if acc_bits == 0 {
                while i + 64 <= slice01.len() {
                    let w = pack_64_bytes01(&slice01[i..i + 64]) ^ xor_u64;
                    u64_words.push(w.to_le());
                    i += 64;
                }
            }
            while i < slice01.len() {
                if (slice01[i] != 0) ^ negate {
                    bit_acc |= 1u64 << acc_bits;
                }
                acc_bits += 1;
                i += 1;
            }
        }
        if acc_bits > 0 {
            u64_words.push(bit_acc.to_le());
        }

        Ok(BitBuffer::new(
            u64_words.freeze().into_byte_buffer(),
            self.len,
        ))
    }
}

/// Returns `Some(invert)` if `pred` is 100% answered by the root node's bitmap (`false` => `bit == 1`
/// is true, `true` => `bit == 0` is true), or `None` if deeper nodes are required.
fn root_only_decision(tree: &PivcoTree, pred: RankPredicate) -> Option<bool> {
    let PivcoNode::Internal {
        pivot_rank,
        left_child,
        right_child,
        ..
    } = &tree.nodes[0]
    else {
        return None;
    };

    match pred {
        RankPredicate::Eq(target_rank) => {
            if target_rank == 0 && *pivot_rank == 1 && matches!(left_child, ChildSpec::Leaf(_)) {
                Some(true) // bit == 0 => true!
            } else if target_rank == *pivot_rank
                && usize::from(*pivot_rank) + 1 == tree.rank_to_sym.len()
                && matches!(right_child, ChildSpec::Leaf(_))
            {
                Some(false) // bit == 1 => true!
            } else {
                None
            }
        }
        RankPredicate::Lt(split_rank) => (split_rank == *pivot_rank).then_some(true),
    }
}

#[inline]
fn rank_outcome(lo_rank: u8, hi_rank: u8, pred: RankPredicate) -> Option<u8> {
    match pred {
        RankPredicate::Eq(t) => {
            if t < lo_rank || t >= hi_rank {
                Some(0)
            } else if lo_rank + 1 == hi_rank && lo_rank == t {
                Some(1)
            } else {
                None
            }
        }
        RankPredicate::Lt(s) => {
            if hi_rank <= s {
                Some(1)
            } else if lo_rank >= s {
                Some(0)
            } else {
                None
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn eval_spine_node(
    tree: &PivcoTree,
    node_idx: u16,
    lo_rank: u8,
    hi_rank: u8,
    n: usize,
    pred: RankPredicate,
    wire: &[u8],
    cursor: &mut usize,
    out01: &mut [u8],
    bm_stack: &mut [MaybeUninit<u8>],
    bm_stride: usize,
) -> VortexResult<()> {
    if n == 0 {
        return Ok(());
    }
    if let Some(cst) = rank_outcome(lo_rank, hi_rank, pred) {
        out01[..n].fill(cst);
        return Ok(());
    }

    match &tree.nodes[usize::from(node_idx)] {
        PivcoNode::SingleLeaf { .. } => {
            let v = rank_outcome(lo_rank, hi_rank, pred).unwrap_or(0);
            out01[..n].fill(v);
            Ok(())
        }
        PivcoNode::Flat {
            depth, base_rank, ..
        } => {
            let nbytes = (n * usize::from(*depth)).div_ceil(8);
            vortex_ensure!(*cursor + nbytes <= wire.len(), "Truncated Pivco flat eval");
            let packed = &wire[*cursor..*cursor + nbytes];
            *cursor += nbytes;
            let mut c2s_mask = [0u8; 64];
            for (code, slot) in c2s_mask.iter_mut().enumerate().take(1usize << *depth) {
                let rank = *base_rank + (code as u8);
                *slot = match pred {
                    RankPredicate::Eq(target) => u8::from(rank == target),
                    RankPredicate::Lt(split) => u8::from(rank < split),
                };
            }
            unpack_flat_codes(*depth, n, packed, &c2s_mask, &mut out01[..n]);
            Ok(())
        }
        PivcoNode::Internal {
            pivot_rank,
            left_child,
            right_child,
            ..
        } => {
            vortex_ensure!(bm_stack.len() >= bm_stride, "Pivco spine stack overflow");
            let (bm_head, bm_tail) = bm_stack.split_at_mut(bm_stride);
            let bm = NodeBitmap::read(wire, cursor, n, bm_head)?;
            let left_bytes = if matches!(left_child, ChildSpec::Node(_)) {
                Some(read_left_subtree_bytes(wire, cursor)?)
            } else {
                None
            };
            let left_const = rank_outcome(lo_rank, *pivot_rank, pred);
            let right_const = rank_outcome(*pivot_rank, hi_rank, pred);

            // SAFETY: `bm_head` was just initialized by `NodeBitmap::read` for `bm`.
            match unsafe { bm.into_dense(bm_head, n) } {
                DenseBitmap::AllZero => {
                    if let Some(lc) = left_const {
                        out01[..n].fill(lc);
                        return Ok(());
                    }
                    match *left_child {
                        ChildSpec::Leaf(_) => unreachable!("single-rank leaf has Some(left_const)"),
                        ChildSpec::Node(l_idx) => eval_spine_node(
                            tree,
                            l_idx,
                            lo_rank,
                            *pivot_rank,
                            n,
                            pred,
                            wire,
                            cursor,
                            out01,
                            bm_tail,
                            bm_stride,
                        ),
                    }
                }
                DenseBitmap::AllOne => {
                    if let Some(lb) = left_bytes {
                        vortex_ensure_eq!(
                            lb,
                            0,
                            "AllOne Pivco node must have 0 left_subtree_bytes"
                        );
                    }
                    if let Some(rc) = right_const {
                        out01[..n].fill(rc);
                        return Ok(());
                    }
                    match *right_child {
                        ChildSpec::Leaf(_) => {
                            unreachable!("single-rank leaf has Some(right_const)")
                        }
                        ChildSpec::Node(r_idx) => eval_spine_node(
                            tree,
                            r_idx,
                            *pivot_rank,
                            hi_rank,
                            n,
                            pred,
                            wire,
                            cursor,
                            out01,
                            bm_tail,
                            bm_stride,
                        ),
                    }
                }
                DenseBitmap::Bytes(bm_slice) => {
                    let k_right = popcount_bits(bm_slice, n);
                    let k_left = n - k_right;
                    if k_left == 0
                        && let Some(lb) = left_bytes
                    {
                        vortex_ensure_eq!(
                            lb,
                            0,
                            "Empty left subtree must have 0 left_subtree_bytes"
                        );
                    }
                    match (left_const, right_const) {
                        (Some(lc), Some(rc)) => {
                            merge_cst_cst(bm_slice, n, lc, rc, &mut out01[..n]);
                            Ok(())
                        }
                        (None, Some(rc)) => {
                            match *left_child {
                                ChildSpec::Leaf(_) => {
                                    unreachable!("single-rank leaf has Some(left_const)")
                                }
                                ChildSpec::Node(l_idx) => eval_spine_node(
                                    tree,
                                    l_idx,
                                    lo_rank,
                                    *pivot_rank,
                                    k_left,
                                    pred,
                                    wire,
                                    cursor,
                                    &mut out01[k_right..],
                                    bm_tail,
                                    bm_stride,
                                )?,
                            }
                            // SAFETY: `k_right == popcount_bits(bm_slice, n)`.
                            unsafe { merge_vec_cst_in_place(bm_slice, n, k_right, rc, out01) };
                            Ok(())
                        }
                        (Some(lc), None) => {
                            *cursor += left_bytes.unwrap_or(0);
                            match *right_child {
                                ChildSpec::Leaf(_) => {
                                    unreachable!("single-rank leaf has Some(right_const)")
                                }
                                ChildSpec::Node(r_idx) => eval_spine_node(
                                    tree,
                                    r_idx,
                                    *pivot_rank,
                                    hi_rank,
                                    k_right,
                                    pred,
                                    wire,
                                    cursor,
                                    &mut out01[k_left..],
                                    bm_tail,
                                    bm_stride,
                                )?,
                            }
                            // SAFETY: `k_left == n - popcount_bits(bm_slice, n)`.
                            unsafe { merge_cst_vec_in_place(bm_slice, n, lc, k_left, out01) };
                            Ok(())
                        }
                        (None, None) => {
                            unreachable!("Eq(t) and Lt(s) have at most one non-constant child")
                        }
                    }
                }
            }
        }
    }
}
