// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! Bottom-up SIMD block decoder (in-place L1 ping-pong) and zero-heap-allocation top-down
//! point-access decoder.

#![expect(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::BlockReader;
use super::DecodeWorkspace;
use super::MAX_BM_BYTES;
use super::read_left_subtree_bytes;
use crate::bitmap_rans::DenseBitmap;
use crate::bitmap_rans::NodeBitmap;
use crate::bitmap_rans::scratch_as_init;
use crate::pow2_bin::BASE_TABLE;
use crate::pow2_bin::EBITS_TABLE;
use crate::simd::MERGE_OVERREAD;
use crate::simd::merge_cst_cst;
use crate::simd::merge_cst_vec_in_place;
use crate::simd::merge_vec_cst_in_place;
use crate::simd::merge_vec_vec_in_place_left;
use crate::simd::merge_vec_vec_in_place_right;
use crate::simd::popcount_bits;
use crate::simd::unpack_flat_codes;
use crate::tree::ChildSpec;
use crate::tree::PivcoNode;
use crate::tree::PivcoTree;

/// Maximum `ebits` where `(bit_pos & 7) + ebits <= 64` holds for every byte-unaligned `u64` load.
const FAST_EXTRA_MAX_EBITS: u8 = 57;

#[inline]
fn read_raw_bits_tail(raw_bytes: &[u8], byte_idx: usize) -> u64 {
    let mut word = 0u64;
    if byte_idx < raw_bytes.len() {
        for (i, &byte) in raw_bytes[byte_idx..].iter().take(8).enumerate() {
            word |= u64::from(byte) << (i * 8);
        }
    }
    word
}

#[inline]
fn read_raw_bits_fast(raw_bytes: &[u8], bit_pos: usize, eb: u8) -> u64 {
    let byte_idx = bit_pos >> 3;
    let word = if byte_idx + 8 <= raw_bytes.len() {
        // SAFETY: `byte_idx + 8 <= raw_bytes.len()` holds by the branch condition.
        unsafe {
            raw_bytes
                .as_ptr()
                .add(byte_idx)
                .cast::<u64>()
                .read_unaligned()
        }
        .to_le()
    } else {
        read_raw_bits_tail(raw_bytes, byte_idx)
    };
    let mask = (1u64 << eb) - 1;
    (word >> (bit_pos & 7)) & mask
}

struct WideBitReader<'a> {
    chunks: &'a [[u8; 8]],
    pos: usize,
    acc: u128,
    bits: u32,
}

impl<'a> WideBitReader<'a> {
    #[inline]
    fn new_at_bit(raw_bytes: &'a [u8], bit_offset: usize) -> Self {
        let chunks = raw_bytes.as_chunks::<8>().0;
        let word_idx = bit_offset >> 6;
        let rem_bits = (bit_offset & 63) as u32;
        let mut acc = 0u128;
        let mut bits = 0u32;
        let mut pos = word_idx;
        if pos < chunks.len() {
            acc = u128::from(u64::from_le_bytes(chunks[pos]));
            bits = 64;
            pos += 1;
        }
        if pos < chunks.len() {
            acc |= u128::from(u64::from_le_bytes(chunks[pos])) << 64;
            bits = 128;
            pos += 1;
        }
        acc >>= rem_bits;
        bits = bits.saturating_sub(rem_bits);
        if bits <= 64 && pos < chunks.len() {
            acc |= u128::from(u64::from_le_bytes(chunks[pos])) << bits;
            pos += 1;
            bits += 64;
        }
        Self {
            chunks,
            pos,
            acc,
            bits,
        }
    }

    #[inline]
    fn read(&mut self, ebits: u8) -> u64 {
        let n = u32::from(ebits);
        let val = (self.acc as u64) & ((1u64 << n) - 1);
        self.acc >>= n;
        self.bits = self.bits.saturating_sub(n);
        if self.bits <= 64 && self.pos < self.chunks.len() {
            self.acc |= u128::from(u64::from_le_bytes(self.chunks[self.pos])) << self.bits;
            self.pos += 1;
            self.bits += 64;
        }
        val
    }
}

impl DecodeWorkspace {
    pub(super) fn decode_block_symbols(
        &mut self,
        tree: &PivcoTree,
        block_len: usize,
        block_tree_bytes: &[u8],
    ) -> VortexResult<&[u8]> {
        self.ensure_for_tree(tree, block_len);
        let mut cursor = 0usize;
        decode_subtree_ping_pong(
            tree,
            0,
            block_len,
            block_tree_bytes,
            &mut cursor,
            &mut self.root_syms[..block_len + MERGE_OVERREAD],
            &mut self.scratch[..block_len + MERGE_OVERREAD],
            &mut self.bm_stack,
            self.bm_stride,
        )?;
        let trailing = block_tree_bytes.len().saturating_sub(cursor);
        vortex_ensure!(
            cursor <= block_tree_bytes.len() && trailing < 4,
            "Pivco block tree_payload has unexpected trailing bytes ({trailing})"
        );
        Ok(&self.root_syms[..block_len])
    }
}

impl BlockReader<'_> {
    /// Decodes elements `[in_block_start..in_block_end)` of `block_idx` directly into `dst_slice`,
    /// applying `map_out` to each reconstructed `u64` value in register.
    pub(crate) fn decode_range_into<Out, F>(
        &self,
        block_idx: usize,
        in_block_start: usize,
        in_block_end: usize,
        ws: &mut DecodeWorkspace,
        dst_slice: &mut [MaybeUninit<Out>],
        map_out: F,
    ) -> VortexResult<()>
    where
        Out: Copy,
        F: Copy + Fn(u64) -> Out,
    {
        let copy_len = in_block_end - in_block_start;
        debug_assert_eq!(dst_slice.len(), copy_len);
        if copy_len == 0 {
            return Ok(());
        }

        let block_len = self.block_len(block_idx);
        vortex_ensure!(
            in_block_end <= block_len,
            "Pivco decode range {in_block_end} exceeds block_len {block_len}"
        );

        let block_tree_bytes = self.block_tree_slice(block_idx);
        let syms = ws.decode_block_symbols(self.tree, block_len, block_tree_bytes)?;
        let target_syms = &syms[in_block_start..in_block_end];

        if !self.has_extra_bits {
            for (dst, &s) in dst_slice.iter_mut().zip(target_syms) {
                dst.write(map_out(u64::from(s)));
            }
            return Ok(());
        }

        let raw_bytes = self.block_raw_slice(block_idx);
        let mut bit_pos = 0usize;
        for &s in &syms[..in_block_start] {
            bit_pos += usize::from(EBITS_TABLE[usize::from(s)]);
        }
        let mut total_block_bits = bit_pos;
        for &s in &syms[in_block_start..block_len] {
            total_block_bits += usize::from(EBITS_TABLE[usize::from(s)]);
        }
        vortex_ensure!(
            total_block_bits.div_ceil(8) <= raw_bytes.len(),
            "Truncated Pivco raw_bits for block {block_idx}: need {} bytes, have {}",
            total_block_bits.div_ceil(8),
            raw_bytes.len()
        );

        if self.max_ebits <= FAST_EXTRA_MAX_EBITS {
            for (dst, &s) in dst_slice.iter_mut().zip(target_syms) {
                let sym_idx = usize::from(s);
                let eb = EBITS_TABLE[sym_idx];
                let raw = read_raw_bits_fast(raw_bytes, bit_pos, eb);
                bit_pos += usize::from(eb);
                dst.write(map_out(BASE_TABLE[sym_idx] | raw));
            }
        } else {
            let mut reader = WideBitReader::new_at_bit(raw_bytes, bit_pos);
            for (dst, &s) in dst_slice.iter_mut().zip(target_syms) {
                let sym_idx = usize::from(s);
                let eb = EBITS_TABLE[sym_idx];
                let raw = if eb > 0 { reader.read(eb) } else { 0 };
                dst.write(map_out(BASE_TABLE[sym_idx] | raw));
            }
        }

        Ok(())
    }

    /// Decodes a single unsigned integer value at `(block_idx, pos_in_block)` with zero heap
    /// allocation and zero upfront stack `memset`.
    pub(crate) fn decode_value_at(
        &self,
        block_idx: usize,
        pos_in_block: usize,
    ) -> VortexResult<u64> {
        let block_len = self.block_len(block_idx);
        vortex_ensure!(
            pos_in_block < block_len,
            "Pivco point access pos_in_block {pos_in_block} >= block_len {block_len}"
        );

        let block_tree_bytes = self.block_tree_slice(block_idx);
        let mut bm_scratch = [MaybeUninit::<u8>::uninit(); MAX_BM_BYTES];

        if !self.has_extra_bits {
            let (sym, _) = decode_symbol_at_inner(
                self.tree,
                block_tree_bytes,
                block_len,
                pos_in_block,
                &mut bm_scratch,
                None,
            )?;
            return Ok(BASE_TABLE[usize::from(sym)]);
        }

        let mut root_cache_buf = [MaybeUninit::<u8>::uninit(); MAX_BM_BYTES];
        let mut root_after_cursor = 0usize;
        let mut root_left_bytes = 0usize;
        let (sym, cached_root) = decode_symbol_at_inner(
            self.tree,
            block_tree_bytes,
            block_len,
            pos_in_block,
            &mut bm_scratch,
            Some((
                &mut root_cache_buf,
                &mut root_after_cursor,
                &mut root_left_bytes,
            )),
        )?;
        let sym_idx = usize::from(sym);
        let eb = EBITS_TABLE[sym_idx];
        if eb == 0 {
            return Ok(BASE_TABLE[sym_idx]);
        }

        let bit_pos = if pos_in_block == 0 {
            0usize
        } else {
            let mut cursor = 0usize;
            let root_cached = cached_root.then(|| {
                let nbytes = block_len.div_ceil(8);
                (
                    scratch_as_init(&root_cache_buf, nbytes),
                    root_after_cursor,
                    root_left_bytes,
                )
            });
            prefix_ebits_sum(
                self.tree,
                0,
                block_len,
                pos_in_block,
                root_cached,
                block_tree_bytes,
                &mut cursor,
                &mut bm_scratch,
            )?
        };

        let raw_bytes = self.block_raw_slice(block_idx);
        vortex_ensure!(
            (bit_pos + usize::from(eb)).div_ceil(8) <= raw_bytes.len(),
            "Truncated Pivco raw_bits for block {block_idx}: need {} bytes, have {}",
            (bit_pos + usize::from(eb)).div_ceil(8),
            raw_bytes.len()
        );
        let raw = if self.max_ebits <= FAST_EXTRA_MAX_EBITS && eb <= FAST_EXTRA_MAX_EBITS {
            read_raw_bits_fast(raw_bytes, bit_pos, eb)
        } else {
            let mut reader = WideBitReader::new_at_bit(raw_bytes, bit_pos);
            reader.read(eb)
        };

        Ok(BASE_TABLE[sym_idx] | raw)
    }
}

// ============================================================================
// Zero-Heap-Alloc Top-Down `O(depth)` Point Access (`scalar_at`)
// ============================================================================

/// Inner top-down symbol walk sharing [`NodeBitmap`] validation and rank math.
fn decode_symbol_at_inner(
    tree: &PivcoTree,
    payload_bytes: &[u8],
    block_len: usize,
    pos_in_block: usize,
    bm_scratch: &mut [MaybeUninit<u8>; MAX_BM_BYTES],
    mut root_rans_cache: Option<(&mut [MaybeUninit<u8>; MAX_BM_BYTES], &mut usize, &mut usize)>,
) -> VortexResult<(u8, bool)> {
    vortex_ensure!(
        pos_in_block < block_len,
        "Pivco point access pos_in_block {pos_in_block} >= block_len {block_len}"
    );
    let mut node_idx = 0u16;
    let mut len = block_len;
    let mut pos = pos_in_block;
    let mut cursor = 0usize;
    let mut cached_root = false;

    loop {
        match &tree.nodes[usize::from(node_idx)] {
            PivcoNode::SingleLeaf { sym } => return Ok((*sym, cached_root)),
            PivcoNode::Flat { depth, c2s, .. } => {
                let depth_usize = usize::from(*depth);
                let nbytes = (len * depth_usize).div_ceil(8);
                vortex_ensure!(
                    cursor + nbytes <= payload_bytes.len(),
                    "Truncated Pivco flat subtree ({nbytes} bytes)"
                );
                let packed = &payload_bytes[cursor..cursor + nbytes];
                let bit_pos = pos * depth_usize;
                let byte_idx = bit_pos >> 3;
                let bit_off = bit_pos & 7;
                let mut val = u32::from(packed[byte_idx]);
                if bit_off + depth_usize > 8 && byte_idx + 1 < packed.len() {
                    val |= u32::from(packed[byte_idx + 1]) << 8;
                }
                let code = ((val >> bit_off) & ((1u32 << depth_usize) - 1)) as usize;
                return Ok((c2s[code], cached_root));
            }
            PivcoNode::Internal {
                left_child,
                right_child,
                ..
            } => {
                let is_root = node_idx == 0;
                let bm = NodeBitmap::read(payload_bytes, &mut cursor, len, bm_scratch)?;
                let left_bytes = if matches!(left_child, ChildSpec::Node(_)) {
                    Some(read_left_subtree_bytes(payload_bytes, &mut cursor)?)
                } else {
                    None
                };

                if is_root
                    && matches!(bm, NodeBitmap::Scratch)
                    && let Some((cache_buf, after_cursor, cached_left_bytes)) =
                        root_rans_cache.take()
                {
                    let nbytes = len.div_ceil(8);
                    cache_buf[..nbytes].copy_from_slice(&bm_scratch[..nbytes]);
                    *after_cursor = cursor;
                    *cached_left_bytes = left_bytes.unwrap_or(0);
                    cached_root = true;
                }

                let (go_right, ones_before, total_ones) = bm.bit_and_ranks(bm_scratch, len, pos);
                let k_left = len - total_ones;
                if k_left == 0
                    && let Some(lb) = left_bytes
                {
                    vortex_ensure_eq!(lb, 0, "Empty left subtree must have 0 left_subtree_bytes");
                }

                if !go_right {
                    match left_child {
                        ChildSpec::Leaf(sym) => return Ok((*sym, cached_root)),
                        ChildSpec::Node(l_idx) => {
                            pos -= ones_before;
                            len = k_left;
                            node_idx = *l_idx;
                        }
                    }
                } else {
                    match right_child {
                        ChildSpec::Leaf(sym) => return Ok((*sym, cached_root)),
                        ChildSpec::Node(r_idx) => {
                            cursor += left_bytes.unwrap_or(0);
                            pos = ones_before;
                            len = total_ones;
                            node_idx = *r_idx;
                        }
                    }
                }
            }
        }
    }
}

/// Computes `sum_{i = 0 .. prefix_len} EBITS_TABLE[sym_i]` inside `node_idx` with zero heap
/// allocation, pruning uniform-`ebits` subtrees and jumping over skipped left subtrees in `O(1)`.
#[expect(clippy::too_many_arguments)]
fn prefix_ebits_sum(
    tree: &PivcoTree,
    node_idx: u16,
    len: usize,
    prefix_len: usize,
    cached_bm: Option<(&[u8], usize, usize)>,
    wire: &[u8],
    cursor: &mut usize,
    bm_scratch: &mut [MaybeUninit<u8>; MAX_BM_BYTES],
) -> VortexResult<usize> {
    if len == 0 || prefix_len == 0 {
        return Ok(0);
    }
    if let Some(uniform_eb) = tree.node_uniform_ebits[usize::from(node_idx)] {
        return Ok(prefix_len * usize::from(uniform_eb));
    }

    match &tree.nodes[usize::from(node_idx)] {
        PivcoNode::SingleLeaf { sym } => {
            Ok(prefix_len * usize::from(EBITS_TABLE[usize::from(*sym)]))
        }
        PivcoNode::Flat { depth, c2s, .. } => {
            let depth_usize = usize::from(*depth);
            let nbytes = (len * depth_usize).div_ceil(8);
            vortex_ensure!(
                *cursor + nbytes <= wire.len(),
                "Truncated Pivco flat subtree ({nbytes} bytes)"
            );
            let packed = &wire[*cursor..*cursor + nbytes];
            *cursor += nbytes;
            let mask = (1u32 << depth_usize) - 1;
            let mut sum = 0usize;
            for idx in 0..prefix_len {
                let bit_pos = idx * depth_usize;
                let byte_idx = bit_pos >> 3;
                let bit_off = bit_pos & 7;
                let mut val = u32::from(packed[byte_idx]);
                if bit_off + depth_usize > 8 && byte_idx + 1 < packed.len() {
                    val |= u32::from(packed[byte_idx + 1]) << 8;
                }
                let code = ((val >> bit_off) & mask) as usize;
                sum += usize::from(EBITS_TABLE[usize::from(c2s[code])]);
            }
            Ok(sum)
        }
        PivcoNode::Internal {
            left_child,
            right_child,
            ..
        } => {
            let (prefix_right, total_right, left_bytes) =
                if let Some((bm_slice, after_cursor, cached_left_bytes)) = cached_bm {
                    *cursor = after_cursor;
                    (
                        popcount_bits(bm_slice, prefix_len),
                        popcount_bits(bm_slice, len),
                        matches!(left_child, ChildSpec::Node(_)).then_some(cached_left_bytes),
                    )
                } else {
                    let bm = NodeBitmap::read(wire, cursor, len, bm_scratch)?;
                    let left_bytes = if matches!(left_child, ChildSpec::Node(_)) {
                        Some(read_left_subtree_bytes(wire, cursor)?)
                    } else {
                        None
                    };
                    let (pr, tr) = bm.rank1_and_count1(bm_scratch, len, prefix_len);
                    (pr, tr, left_bytes)
                };

            let prefix_left = prefix_len - prefix_right;
            let k_left = len - total_right;
            let k_right = total_right;
            if k_left == 0
                && let Some(lb) = left_bytes
            {
                vortex_ensure_eq!(lb, 0, "Empty left subtree must have 0 left_subtree_bytes");
            }
            let right_start = *cursor + left_bytes.unwrap_or(0);

            let left_sum = match left_child {
                ChildSpec::Leaf(sym) => prefix_left * usize::from(EBITS_TABLE[usize::from(*sym)]),
                ChildSpec::Node(l_idx) => {
                    if k_left == 0 || prefix_left == 0 {
                        0
                    } else {
                        prefix_ebits_sum(
                            tree,
                            *l_idx,
                            k_left,
                            prefix_left,
                            None,
                            wire,
                            cursor,
                            bm_scratch,
                        )?
                    }
                }
            };

            *cursor = right_start;

            let right_sum = match right_child {
                ChildSpec::Leaf(sym) => prefix_right * usize::from(EBITS_TABLE[usize::from(*sym)]),
                ChildSpec::Node(r_idx) => {
                    if k_right == 0 || prefix_right == 0 {
                        0
                    } else {
                        prefix_ebits_sum(
                            tree,
                            *r_idx,
                            k_right,
                            prefix_right,
                            None,
                            wire,
                            cursor,
                            bm_scratch,
                        )?
                    }
                }
            };

            Ok(left_sum + right_sum)
        }
    }
}

// ============================================================================
// Bottom-Up In-Place L1 Ping-Pong SIMD Subtree Decoder
// ============================================================================

/// Decodes subtree `node_idx` (`n` symbols) into `out[..n]`, writing strictly within `out[..n]`
/// and `scratch[..n]` while requiring `out.len() >= n + MERGE_OVERREAD` and
/// `scratch.len() >= n + MERGE_OVERREAD` for readable SIMD overread slack.
#[expect(clippy::too_many_arguments)]
fn decode_subtree_ping_pong(
    tree: &PivcoTree,
    node_idx: u16,
    n: usize,
    wire: &[u8],
    cursor: &mut usize,
    out: &mut [u8],
    scratch: &mut [u8],
    bm_stack: &mut [MaybeUninit<u8>],
    bm_stride: usize,
) -> VortexResult<()> {
    if n == 0 {
        return Ok(());
    }
    debug_assert!(out.len() >= n + MERGE_OVERREAD);
    debug_assert!(scratch.len() >= n + MERGE_OVERREAD);

    match &tree.nodes[usize::from(node_idx)] {
        PivcoNode::SingleLeaf { sym } => {
            out[..n].fill(*sym);
            Ok(())
        }
        PivcoNode::Flat { depth, c2s, .. } => {
            let nbytes = (n * usize::from(*depth)).div_ceil(8);
            vortex_ensure!(
                *cursor + nbytes <= wire.len(),
                "Truncated Pivco flat subtree ({nbytes} bytes)"
            );
            let packed = &wire[*cursor..*cursor + nbytes];
            *cursor += nbytes;
            unpack_flat_codes(*depth, n, packed, c2s, &mut out[..n]);
            Ok(())
        }
        PivcoNode::Internal {
            left_child,
            right_child,
            ..
        } => {
            vortex_ensure!(
                bm_stack.len() >= bm_stride,
                "Pivco tree recursion exceeded bitmap stack limit"
            );
            let (bm_head, bm_tail) = bm_stack.split_at_mut(bm_stride);
            let bm = NodeBitmap::read(wire, cursor, n, bm_head)?;
            let left_bytes = if matches!(left_child, ChildSpec::Node(_)) {
                Some(read_left_subtree_bytes(wire, cursor)?)
            } else {
                None
            };
            let left_start = *cursor;

            match bm.into_dense(bm_head, n) {
                DenseBitmap::AllZero => {
                    match left_child {
                        ChildSpec::Leaf(l_sym) => {
                            out[..n].fill(*l_sym);
                        }
                        ChildSpec::Node(l_idx) => {
                            decode_subtree_ping_pong(
                                tree, *l_idx, n, wire, cursor, out, scratch, bm_tail, bm_stride,
                            )?;
                            if let Some(lb) = left_bytes {
                                vortex_ensure_eq!(
                                    *cursor - left_start,
                                    lb,
                                    "Pivco left_subtree_bytes framing mismatch"
                                );
                            }
                        }
                    }
                    Ok(())
                }
                DenseBitmap::AllOne => {
                    if let Some(lb) = left_bytes {
                        vortex_ensure_eq!(
                            lb,
                            0,
                            "AllOne Pivco node must have 0 left_subtree_bytes"
                        );
                    }
                    match right_child {
                        ChildSpec::Leaf(r_sym) => {
                            out[..n].fill(*r_sym);
                            Ok(())
                        }
                        ChildSpec::Node(r_idx) => decode_subtree_ping_pong(
                            tree, *r_idx, n, wire, cursor, out, scratch, bm_tail, bm_stride,
                        ),
                    }
                }
                DenseBitmap::Bytes(bm_slice) => merge_internal_children(
                    tree,
                    *left_child,
                    *right_child,
                    bm_slice,
                    left_bytes,
                    left_start,
                    n,
                    wire,
                    cursor,
                    out,
                    scratch,
                    bm_tail,
                    bm_stride,
                ),
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn merge_internal_children(
    tree: &PivcoTree,
    left_child: ChildSpec,
    right_child: ChildSpec,
    bm_bytes: &[u8],
    left_bytes: Option<usize>,
    left_start: usize,
    n: usize,
    wire: &[u8],
    cursor: &mut usize,
    out: &mut [u8],
    scratch: &mut [u8],
    bm_tail: &mut [MaybeUninit<u8>],
    bm_stride: usize,
) -> VortexResult<()> {
    let k_right = popcount_bits(bm_bytes, n);
    let k_left = n - k_right;

    match (left_child, right_child) {
        (ChildSpec::Leaf(l_sym), ChildSpec::Leaf(r_sym)) => {
            merge_cst_cst(bm_bytes, n, l_sym, r_sym, &mut out[..n]);
            Ok(())
        }
        (ChildSpec::Leaf(l_sym), ChildSpec::Node(r_idx)) => {
            decode_subtree_ping_pong(
                tree,
                r_idx,
                k_right,
                wire,
                cursor,
                &mut out[k_left..],
                scratch,
                bm_tail,
                bm_stride,
            )?;
            merge_cst_vec_in_place(bm_bytes, n, l_sym, k_left, out);
            Ok(())
        }
        (ChildSpec::Node(l_idx), ChildSpec::Leaf(r_sym)) => {
            decode_subtree_ping_pong(
                tree,
                l_idx,
                k_left,
                wire,
                cursor,
                &mut out[k_right..],
                scratch,
                bm_tail,
                bm_stride,
            )?;
            if let Some(lb) = left_bytes {
                vortex_ensure_eq!(
                    *cursor - left_start,
                    lb,
                    "Pivco left_subtree_bytes framing mismatch"
                );
            }
            merge_vec_cst_in_place(bm_bytes, n, k_right, r_sym, out);
            Ok(())
        }
        (ChildSpec::Node(l_idx), ChildSpec::Node(r_idx)) => {
            if k_left >= k_right {
                // Larger child (`left`) decodes in-place into `out[k_right..]`.
                decode_subtree_ping_pong(
                    tree,
                    l_idx,
                    k_left,
                    wire,
                    cursor,
                    &mut out[k_right..],
                    scratch,
                    bm_tail,
                    bm_stride,
                )?;
                if let Some(lb) = left_bytes {
                    vortex_ensure_eq!(
                        *cursor - left_start,
                        lb,
                        "Pivco left_subtree_bytes framing mismatch"
                    );
                }
                // Smaller child (`right`) decodes into `scratch[..k_right]`, using `out[..k_right]`
                // (strictly disjoint from `left` in `out[k_right..]`) as its writable partner.
                decode_subtree_ping_pong(
                    tree,
                    r_idx,
                    k_right,
                    wire,
                    cursor,
                    &mut scratch[..k_right + MERGE_OVERREAD],
                    &mut out[..k_right + MERGE_OVERREAD],
                    bm_tail,
                    bm_stride,
                )?;
                merge_vec_vec_in_place_left(
                    bm_bytes,
                    n,
                    k_right,
                    &scratch[..k_right + MERGE_OVERREAD],
                    out,
                );
            } else {
                // Wire carries `left` first: smaller child (`left`) decodes into `scratch[..k_left]`
                // using `out[..k_left]` as its writable partner.
                decode_subtree_ping_pong(
                    tree,
                    l_idx,
                    k_left,
                    wire,
                    cursor,
                    &mut scratch[..k_left + MERGE_OVERREAD],
                    &mut out[..k_left + MERGE_OVERREAD],
                    bm_tail,
                    bm_stride,
                )?;
                if let Some(lb) = left_bytes {
                    vortex_ensure_eq!(
                        *cursor - left_start,
                        lb,
                        "Pivco left_subtree_bytes framing mismatch"
                    );
                }
                // Larger child (`right`) decodes in-place into `out[k_left..]` using the disjoint
                // tail `scratch[k_left..]` (`>= k_right + MERGE_OVERREAD` bytes) as its partner.
                decode_subtree_ping_pong(
                    tree,
                    r_idx,
                    k_right,
                    wire,
                    cursor,
                    &mut out[k_left..],
                    &mut scratch[k_left..],
                    bm_tail,
                    bm_stride,
                )?;
                merge_vec_vec_in_place_right(
                    bm_bytes,
                    n,
                    &scratch[..k_left + MERGE_OVERREAD],
                    k_left,
                    out,
                );
            }
            Ok(())
        }
    }
}
