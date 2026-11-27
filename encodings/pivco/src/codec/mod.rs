// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! Core `Pow2-Bin + PIVCO` block wire layout, encoder, bottom-up SIMD wavelet-tree decoder,
//! direct spine-walking bitmap predicate pushdown, and shared compression-ratio estimator.

#![expect(clippy::cast_possible_truncation)]

mod decode;
mod encode;
mod spine;

use std::mem::MaybeUninit;

pub(crate) use encode::encode_primitive_slice;
pub(crate) use spine::RankPredicate;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::bitmap_rans::estimate_node_bitmap_bits;
use crate::pow2_bin::EBITS_TABLE;
use crate::pow2_bin::MAX_SYMS;
use crate::simd::MERGE_OVERREAD;
use crate::tree::ChildSpec;
use crate::tree::PivcoNode;
use crate::tree::PivcoTree;
use crate::tree::PivcoTreeMode;

/// Minimum supported `log2(block_values)` (`2^10 = 1,024`).
pub const MIN_BLOCK_SHIFT: u8 = 10;

/// Default `log2(block_values)` (`2^12 = 4,096`).
pub const DEFAULT_BLOCK_SHIFT: u8 = 12;

/// Maximum supported `log2(block_values)` (`2^13 = 8,192`).
pub const MAX_BLOCK_SHIFT: u8 = 13;

/// Maximum number of elements per block (`8,192`).
pub const MAX_BLOCK_VALUES: usize = 1 << MAX_BLOCK_SHIFT;

/// Maximum bytes in one block's bitmap (`8,192 / 8 = 1,024`).
pub(super) const MAX_BM_BYTES: usize = MAX_BLOCK_VALUES / 8;

/// Byte size of one `block_headers` sentinel entry when `!has_extra_bits` (`[tree_byte_offset: u32]`).
pub const BLOCK_HEADER_BYTES_NO_EXTRA: usize = 4;

/// Byte size of one `block_headers` sentinel entry when `has_extra_bits`
/// (`[tree_byte_offset: u32, raw_word_offset: u32]`).
pub const BLOCK_HEADER_BYTES_WITH_EXTRA: usize = 8;

/// Returns the byte stride of each entry in `block_headers` (`4` without extra bits, `8` with extra bits).
#[inline]
pub const fn block_header_stride(has_extra_bits: bool) -> usize {
    if has_extra_bits {
        BLOCK_HEADER_BYTES_WITH_EXTRA
    } else {
        BLOCK_HEADER_BYTES_NO_EXTRA
    }
}

/// Returns the encoded element count of block `block_idx` in a chunk with `num_blocks` blocks.
#[inline]
pub(crate) const fn block_len_for(
    block_idx: usize,
    num_blocks: usize,
    block_values: usize,
    last_block_len: u16,
) -> usize {
    if block_idx + 1 < num_blocks {
        block_values
    } else {
        last_block_len as usize
    }
}

/// Encoder configuration for `vortex-pivco`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PivcoOptions {
    /// Tree construction mode (`Huffman` or `OrderPreserving`).
    pub tree_mode: PivcoTreeMode,
    /// Enable skewed per-node bitmap compression (`PHA`: sparse indices + 32-lane static byte-rANS).
    pub pha_enabled: bool,
    /// Enable equal-depth flat-subtree (`D in 2..=6`) SIMD packing.
    pub allow_flat: bool,
    /// `log2(block_values)` in `10..=13` (`1,024..=8,192`), or `0` for `DEFAULT_BLOCK_SHIFT` (12, 4,096 rows).
    pub block_shift: u8,
}

impl Default for PivcoOptions {
    fn default() -> Self {
        Self::pha()
    }
}

impl PivcoOptions {
    /// Raw wavelet-tree bitmaps + flat subtrees (`PH` mode: maximum decode speed, zero rANS).
    pub const fn ph() -> Self {
        Self {
            tree_mode: PivcoTreeMode::Huffman,
            pha_enabled: false,
            allow_flat: true,
            block_shift: DEFAULT_BLOCK_SHIFT,
        }
    }

    /// Adaptive skewed-bitmap wavelet tree (`PHA` mode: breaks Huffman's `1.0 bpv` floor).
    pub const fn pha() -> Self {
        Self {
            tree_mode: PivcoTreeMode::Huffman,
            pha_enabled: true,
            allow_flat: true,
            block_shift: DEFAULT_BLOCK_SHIFT,
        }
    }

    /// Order-preserving alphabet tree + adaptive skewed-bitmap coding (`OP-PHA` mode: enables
    /// direct `<` / `<=` / `>` / `>=` predicate pushdown on node bitmaps).
    pub const fn op_pha() -> Self {
        Self {
            tree_mode: PivcoTreeMode::OrderPreserving,
            pha_enabled: true,
            allow_flat: true,
            block_shift: DEFAULT_BLOCK_SHIFT,
        }
    }

    /// Overrides `block_shift` (`10..=13`, or `0` for `DEFAULT_BLOCK_SHIFT`).
    pub const fn with_block_shift(mut self, block_shift: u8) -> Self {
        self.block_shift = block_shift;
        self
    }
}

/// Estimates total compressed payload bits (wavelet-tree bitmaps + flat codes + raw extra bits)
/// for `total_valid` elements with `Pow2-Bin` symbol histogram `counts` encoded at `block_len`
/// elements per block using `options`.
///
/// If `counts.iter().sum::<u64>() < total_valid as u64`, the remaining positions are null-filled
/// into the most frequent exact symbol in `0..32`, matching the encoder's null-fill policy.
///
/// Returns `(estimated_bits, is_order_preserving)`, where `is_order_preserving` is `true` iff the
/// compiled tree's leaf order is strictly sorted (enabling range predicate pushdown).
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn estimate_compressed_bits(
    counts: &[u64; MAX_SYMS],
    total_valid: usize,
    block_len: usize,
    options: PivcoOptions,
) -> VortexResult<(usize, bool)> {
    vortex_ensure!(
        (1..=MAX_BLOCK_VALUES).contains(&block_len),
        "Pivco block_len {block_len} must be in 1..={MAX_BLOCK_VALUES}"
    );
    let hist_sum = counts
        .iter()
        .copied()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(|| vortex_err!("Pivco histogram sum overflow"))?;
    let total_u64 = total_valid as u64;
    if total_valid == 0 {
        vortex_ensure_eq!(
            hist_sum,
            0,
            "Non-zero histogram counts for total_valid == 0"
        );
        return Ok((0, true));
    }
    vortex_ensure!(
        hist_sum <= total_u64,
        "Pivco histogram sum ({hist_sum}) exceeds total_valid ({total_valid})"
    );

    let mut sym_counts = *counts;
    if hist_sum < total_u64 {
        let mut fill_sym = 0usize;
        let mut best_exact = 0u64;
        for (sym, &cnt) in sym_counts[..32].iter().enumerate() {
            if cnt > best_exact {
                best_exact = cnt;
                fill_sym = sym;
            }
        }
        sym_counts[fill_sym] += total_u64 - hist_sum;
    }

    let tree = PivcoTree::build(&sym_counts, options.tree_mode, options.allow_flat);
    let mut rank_counts = vec![0u64; tree.rank_to_sym.len()];
    for (rank, &sym) in tree.rank_to_sym.iter().enumerate() {
        rank_counts[rank] = sym_counts[usize::from(sym)];
    }

    let mut raw_extra_bits = 0u128;
    for (sym, &cnt) in sym_counts.iter().enumerate() {
        let extra = EBITS_TABLE[sym];
        if extra > 0 && cnt > 0 {
            raw_extra_bits += u128::from(cnt) * u128::from(extra);
        }
    }

    let tree_bits = estimate_subtree_bits(
        &tree,
        0,
        0,
        tree.rank_to_sym.len(),
        &rank_counts,
        total_valid,
        block_len,
        options.pha_enabled,
    );
    let total_bits = (tree_bits + (raw_extra_bits as f64)).ceil() as usize;
    Ok((total_bits, tree.is_order_preserving))
}

#[expect(clippy::too_many_arguments)]
#[cfg_attr(not(test), expect(dead_code))]
fn estimate_subtree_bits(
    tree: &PivcoTree,
    node_idx: u16,
    lo_rank: usize,
    hi_rank: usize,
    rank_counts: &[u64],
    total_rows: usize,
    block_len: usize,
    pha_enabled: bool,
) -> f64 {
    let n_sample: u64 = rank_counts[lo_rank..hi_rank].iter().sum();
    if n_sample == 0 {
        return 0.0;
    }
    match &tree.nodes[usize::from(node_idx)] {
        PivcoNode::SingleLeaf { .. } => 0.0,
        PivcoNode::Flat { depth, .. } => (n_sample as f64) * f64::from(*depth),
        PivcoNode::Internal {
            tree_depth,
            pivot_rank,
            left_child,
            right_child,
        } => {
            let piv = usize::from(*pivot_rank);
            let k_left_sample: u64 = rank_counts[lo_rank..piv].iter().sum();
            let k_right_sample = n_sample - k_left_sample;

            let bm_bits = if total_rows > block_len {
                let num_blocks = (total_rows as f64) / (block_len as f64);
                let n_block = ((n_sample as f64) / num_blocks)
                    .round()
                    .clamp(1.0, block_len as f64) as usize;
                let k_right_block = if k_right_sample == 0 {
                    0
                } else if k_right_sample == n_sample {
                    n_block
                } else {
                    (((k_right_sample as f64) * (n_block as f64)) / (n_sample as f64))
                        .round()
                        .clamp(1.0, n_block.saturating_sub(1).max(1) as f64)
                        as usize
                };
                (estimate_node_bitmap_bits(
                    n_block,
                    k_right_block,
                    usize::from(*tree_depth),
                    pha_enabled,
                ) as f64)
                    * num_blocks
            } else {
                estimate_node_bitmap_bits(
                    n_sample as usize,
                    k_right_sample as usize,
                    usize::from(*tree_depth),
                    pha_enabled,
                ) as f64
            };

            let l_bits = match left_child {
                ChildSpec::Leaf(_) => 0.0,
                ChildSpec::Node(l_idx) => estimate_subtree_bits(
                    tree,
                    *l_idx,
                    lo_rank,
                    piv,
                    rank_counts,
                    total_rows,
                    block_len,
                    pha_enabled,
                ),
            };
            let r_bits = match right_child {
                ChildSpec::Leaf(_) => 0.0,
                ChildSpec::Node(r_idx) => estimate_subtree_bits(
                    tree,
                    *r_idx,
                    piv,
                    hi_rank,
                    rank_counts,
                    total_rows,
                    block_len,
                    pha_enabled,
                ),
            };
            bm_bits + l_bits + r_bits
        }
    }
}

/// Sentinel offset entry stored in `block_headers` (`(num_blocks + 1)` entries total).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PivcoBlockOffset {
    /// Byte offset (before subtracting `tree_base_offset`) into `tree_payload`.
    pub tree_byte_offset: u32,
    /// `u64`-word offset (before subtracting `raw_base_word_offset`) into `raw_bits` (`0` when `!has_extra_bits`).
    pub raw_word_offset: u32,
}

impl PivcoBlockOffset {
    /// Reads sentinel entry `entry_idx` (`0..=num_blocks`) from host `block_headers`.
    #[inline]
    pub fn read_at(headers_bytes: &[u8], entry_idx: usize, has_extra_bits: bool) -> Self {
        if has_extra_bits {
            let base = entry_idx * BLOCK_HEADER_BYTES_WITH_EXTRA;
            let hdr = &headers_bytes[base..base + BLOCK_HEADER_BYTES_WITH_EXTRA];
            Self {
                tree_byte_offset: u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]),
                raw_word_offset: u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]),
            }
        } else {
            let base = entry_idx * BLOCK_HEADER_BYTES_NO_EXTRA;
            let hdr = &headers_bytes[base..base + BLOCK_HEADER_BYTES_NO_EXTRA];
            Self {
                tree_byte_offset: u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]),
                raw_word_offset: 0,
            }
        }
    }

    #[inline]
    pub(super) fn write_into(self, has_extra_bits: bool, dst: &mut ByteBufferMut) {
        dst.extend_from_slice(&self.tree_byte_offset.to_le_bytes());
        if has_extra_bits {
            dst.extend_from_slice(&self.raw_word_offset.to_le_bytes());
        }
    }
}

/// Host-pinned view over a `PivcoArray`'s buffers and metadata used by block decode, point access,
/// and spine predicate pushdown.
#[derive(Clone, Copy)]
pub(crate) struct BlockReader<'a> {
    pub(crate) tree: &'a PivcoTree,
    pub(crate) headers_bytes: &'a [u8],
    pub(crate) tree_payload: &'a [u8],
    pub(crate) raw_bits: &'a [u8],
    pub(crate) tree_base_offset: u32,
    pub(crate) raw_base_word_offset: u32,
    pub(crate) block_shift: u8,
    pub(crate) offset: u16,
    pub(crate) len: usize,
    pub(crate) last_block_len: u16,
    pub(crate) has_extra_bits: bool,
    pub(crate) max_ebits: u8,
}

impl<'a> BlockReader<'a> {
    #[inline]
    pub(crate) const fn block_values(&self) -> usize {
        1usize << self.block_shift
    }

    #[inline]
    pub(crate) const fn num_blocks(&self) -> usize {
        if self.len == 0 {
            0
        } else {
            (self.offset as usize + self.len).div_ceil(1usize << self.block_shift)
        }
    }

    #[inline]
    pub(crate) const fn block_len(&self, block_idx: usize) -> usize {
        block_len_for(
            block_idx,
            self.num_blocks(),
            self.block_values(),
            self.last_block_len,
        )
    }

    #[inline]
    pub(super) fn block_tree_slice(&self, block_idx: usize) -> &'a [u8] {
        let start_hdr =
            PivcoBlockOffset::read_at(self.headers_bytes, block_idx, self.has_extra_bits);
        let end_hdr =
            PivcoBlockOffset::read_at(self.headers_bytes, block_idx + 1, self.has_extra_bits);
        let base = self.tree_base_offset as usize;
        let rel_start = (start_hdr.tree_byte_offset as usize) - base;
        let rel_end = (end_hdr.tree_byte_offset as usize) - base;
        &self.tree_payload[rel_start..rel_end]
    }

    #[inline]
    pub(super) fn block_raw_slice(&self, block_idx: usize) -> &'a [u8] {
        let start_hdr = PivcoBlockOffset::read_at(self.headers_bytes, block_idx, true);
        let end_hdr = PivcoBlockOffset::read_at(self.headers_bytes, block_idx + 1, true);
        let base = self.raw_base_word_offset as usize;
        let rel_start = ((start_hdr.raw_word_offset as usize) - base) * 8;
        let rel_end = ((end_hdr.raw_word_offset as usize) - base) * 8;
        &self.raw_bits[rel_start..rel_end]
    }
}

/// Validates all `(expected_blocks + 1)` sentinel entries in `block_headers` against
/// `tree_payload_len`, `raw_bits_len`, `tree_base_offset`, `raw_base_word_offset`, and
/// `has_extra_bits`.
pub(crate) fn validate_block_headers(
    headers_bytes: &[u8],
    tree_payload_len: usize,
    raw_bits_len: usize,
    tree_base_offset: u32,
    raw_base_word_offset: u32,
    expected_blocks: usize,
    has_extra_bits: bool,
) -> VortexResult<()> {
    if expected_blocks == 0 {
        vortex_ensure!(
            headers_bytes.is_empty(),
            "Empty PivcoArray must have empty block_headers"
        );
        vortex_ensure_eq!(
            tree_payload_len,
            0,
            "Empty PivcoArray must have empty tree_payload"
        );
        vortex_ensure_eq!(raw_bits_len, 0, "Empty PivcoArray must have empty raw_bits");
        vortex_ensure_eq!(
            tree_base_offset,
            0,
            "Empty PivcoArray must have tree_base_offset == 0"
        );
        vortex_ensure_eq!(
            raw_base_word_offset,
            0,
            "Empty PivcoArray must have raw_base_word_offset == 0"
        );
        return Ok(());
    }

    vortex_ensure!(
        tree_base_offset.is_multiple_of(4),
        "Pivco tree_base_offset ({tree_base_offset}) must be 4-byte aligned"
    );

    let stride = block_header_stride(has_extra_bits);
    vortex_ensure_eq!(
        headers_bytes.len(),
        (expected_blocks + 1) * stride,
        "Pivco block_headers byte length mismatch"
    );

    let first = PivcoBlockOffset::read_at(headers_bytes, 0, has_extra_bits);
    vortex_ensure!(
        first.tree_byte_offset.is_multiple_of(4),
        "Pivco hdr[0].tree_byte_offset ({}) must be 4-byte aligned",
        first.tree_byte_offset
    );
    vortex_ensure!(
        first.tree_byte_offset >= tree_base_offset,
        "Pivco hdr[0].tree_byte_offset ({}) precedes tree_base_offset ({tree_base_offset})",
        first.tree_byte_offset
    );
    if has_extra_bits {
        vortex_ensure!(
            first.raw_word_offset >= raw_base_word_offset,
            "Pivco hdr[0].raw_word_offset ({}) precedes raw_base_word_offset ({raw_base_word_offset})",
            first.raw_word_offset
        );
    } else {
        vortex_ensure_eq!(
            raw_base_word_offset,
            0,
            "Pivco raw_base_word_offset must be 0 when !has_extra_bits"
        );
        vortex_ensure_eq!(
            raw_bits_len,
            0,
            "Pivco raw_bits must be empty when !has_extra_bits"
        );
    }

    let mut prev = first;
    for b in 0..expected_blocks {
        let next = PivcoBlockOffset::read_at(headers_bytes, b + 1, has_extra_bits);
        vortex_ensure!(
            next.tree_byte_offset.is_multiple_of(4),
            "Pivco hdr[{}].tree_byte_offset ({}) must be 4-byte aligned",
            b + 1,
            next.tree_byte_offset
        );
        vortex_ensure!(
            prev.tree_byte_offset <= next.tree_byte_offset,
            "Pivco block {b} tree_byte_offset not monotonic ({} > {})",
            prev.tree_byte_offset,
            next.tree_byte_offset
        );
        if has_extra_bits {
            vortex_ensure!(
                prev.raw_word_offset <= next.raw_word_offset,
                "Pivco block {b} raw_word_offset not monotonic ({} > {})",
                prev.raw_word_offset,
                next.raw_word_offset
            );
        }
        prev = next;
    }

    let needed_tree_len = (prev.tree_byte_offset - tree_base_offset) as usize;
    vortex_ensure!(
        needed_tree_len <= tree_payload_len,
        "Pivco hdr[B].tree_byte_offset - tree_base_offset ({needed_tree_len}) exceeds tree_payload length ({tree_payload_len})"
    );

    if has_extra_bits {
        let needed_raw_words = (prev.raw_word_offset - raw_base_word_offset) as usize;
        let needed_raw_bytes = needed_raw_words
            .checked_mul(8)
            .ok_or_else(|| vortex_err!("Pivco raw_bits byte length overflow"))?;
        vortex_ensure!(
            needed_raw_bytes <= raw_bits_len,
            "Pivco (hdr[B].raw_word_offset - raw_base_word_offset) * 8 ({needed_raw_bytes}) exceeds raw_bits length ({raw_bits_len})"
        );
    }

    Ok(())
}

/// Reads the `u16` byte length of an internal node's serialized left subtree (`O(1)` skip header)
/// and validates that the referenced byte range stays within `wire`.
#[inline]
pub(super) fn read_left_subtree_bytes(wire: &[u8], cursor: &mut usize) -> VortexResult<usize> {
    vortex_ensure!(
        *cursor + 2 <= wire.len(),
        "Truncated Pivco left_subtree_bytes header"
    );
    let len = usize::from(u16::from_le_bytes([wire[*cursor], wire[*cursor + 1]]));
    *cursor += 2;
    vortex_ensure!(
        *cursor + len <= wire.len(),
        "Pivco left_subtree_bytes ({len}) exceeds payload bounds"
    );
    Ok(len)
}

/// L1-resident reusable in-place ping-pong scratch buffers for multi-block `vortex-pivco` decoding
/// and spine predicate pushdown.
///
/// Larger children decode in-place into `out`'s tail while smaller children ping-pong through
/// `scratch`, bounding both symbol buffers to `block_values + MERGE_OVERREAD` regardless of tree
/// alphabet width (`~13.5 KiB` worst-case at `4,096` rows and `150` leaves).
pub(crate) struct DecodeWorkspace {
    /// Decoded root symbol buffer (`block_values + MERGE_OVERREAD`).
    pub(super) root_syms: Vec<u8>,
    /// Ping-pong partner symbol scratch buffer (`block_values + MERGE_OVERREAD`).
    pub(super) scratch: Vec<u8>,
    /// Flat per-internal-level bitmap scratch (`max_internal_depth * bm_stride`), lazily written.
    pub(super) bm_stack: Vec<MaybeUninit<u8>>,
    /// Byte stride of one bitmap level (`block_values.div_ceil(8)`).
    pub(super) bm_stride: usize,
}

impl DecodeWorkspace {
    /// Allocates an L1-resident workspace sized tightly to `tree` and `block_values`.
    pub(crate) fn for_tree(tree: &PivcoTree, block_values: usize) -> Self {
        let sym_cap = block_values + MERGE_OVERREAD;
        let bm_stride = block_values.div_ceil(8);
        let bm_levels = usize::from(tree.max_internal_depth);
        Self {
            root_syms: vec![0u8; sym_cap],
            scratch: vec![0u8; sym_cap],
            bm_stack: vec![MaybeUninit::uninit(); bm_levels * bm_stride],
            bm_stride,
        }
    }

    /// Total heap bytes reserved across the ping-pong symbol and bitmap scratch buffers.
    #[cfg(test)]
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.root_syms.len() + self.scratch.len() + self.bm_stack.len()
    }

    #[inline]
    pub(super) fn ensure_for_tree(&mut self, tree: &PivcoTree, block_len: usize) {
        let sym_cap = block_len + MERGE_OVERREAD;
        if self.root_syms.len() < sym_cap {
            self.root_syms.resize(sym_cap, 0);
        }
        if self.scratch.len() < sym_cap {
            self.scratch.resize(sym_cap, 0);
        }
        let needed_stride = block_len.div_ceil(8);
        if self.bm_stride < needed_stride {
            self.bm_stride = needed_stride;
        }
        let needed_bm = usize::from(tree.max_internal_depth) * self.bm_stride;
        if self.bm_stack.len() < needed_bm {
            self.bm_stack.resize(needed_bm, MaybeUninit::uninit());
        }
    }
}
