// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! `Pow2-Bin + PIVCO` block encoder.

#![expect(clippy::cast_possible_truncation)]

use num_traits::AsPrimitive;
use vortex_array::dtype::NativePType;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::DEFAULT_BLOCK_SHIFT;
use super::MAX_BLOCK_SHIFT;
use super::MAX_BM_BYTES;
use super::MIN_BLOCK_SHIFT;
use super::PivcoBlockOffset;
use super::PivcoOptions;
use super::block_header_stride;
use crate::bitmap_rans::write_node_bitmap;
use crate::pow2_bin::MAX_SYMS;
use crate::pow2_bin::encode_pow2_bin;
use crate::pow2_bin::pow2_sym_and_ebits;
use crate::simd::pack_flat_codes;
use crate::simd::partition_ranks_by_pivot;
use crate::tree::ChildSpec;
use crate::tree::MAX_TREE_DEPTH;
use crate::tree::PivcoNode;
use crate::tree::PivcoTree;
use crate::tree::compute_tree_max_ebits;

/// Maximum internal-node recursion stack depth (`MAX_TREE_DEPTH = 11`).
const MAX_STACK_DEPTH: usize = MAX_TREE_DEPTH as usize;

struct BitWriter {
    words: Vec<u64>,
    acc: u128,
    bits: u32,
}

impl BitWriter {
    fn with_capacity(cap_words: usize) -> Self {
        Self {
            words: Vec::with_capacity(cap_words),
            acc: 0,
            bits: 0,
        }
    }

    #[inline]
    fn write(&mut self, val: u64, nbits: u8) {
        self.acc |= u128::from(val) << self.bits;
        self.bits += u32::from(nbits);
        if self.bits >= 64 {
            self.words.push((self.acc as u64).to_le());
            self.acc >>= 64;
            self.bits -= 64;
        }
    }

    #[inline]
    fn finish_word_aligned(&mut self) {
        if self.bits > 0 {
            self.words.push((self.acc as u64).to_le());
            self.acc = 0;
            self.bits = 0;
        }
    }

    fn clear(&mut self) {
        self.words.clear();
        self.acc = 0;
        self.bits = 0;
    }

    #[inline]
    fn flush_into(&self, dst: &mut BufferMut<u64>) -> usize {
        dst.extend_from_slice(&self.words);
        self.words.len()
    }
}

/// Output of encoding a slice of non-negative primitive integers with `vortex-pivco`.
pub(crate) struct EncodedPivcoBuffers {
    /// Selected `log2(block_values)` in `10..=13`.
    pub(crate) block_shift: u8,
    /// Encoded element count of the final block (`1..=block_values`, or `0` when empty).
    pub(crate) last_block_len: u16,
    /// Compiled chunk-level wavelet tree.
    pub(crate) tree: PivcoTree,
    /// 4-byte-aligned sentinel offset table (`(num_blocks + 1) * stride` bytes, empty when `len == 0`).
    pub(crate) block_headers: ByteBuffer,
    /// Byte-packed contiguous pre-order wavelet-tree node streams across all blocks.
    pub(crate) tree_payload: ByteBuffer,
    /// 8-byte-aligned contiguous `u64` words of verbatim extra bits (`0` bytes when `!has_extra_bits`).
    pub(crate) raw_bits: ByteBuffer,
}

/// Encodes non-negative `values` (`T` integer slice) with validity `mask` into `vortex-pivco` buffers.
pub(crate) fn encode_primitive_slice<T>(
    values: &[T],
    is_signed: bool,
    mask: &Mask,
    opts: PivcoOptions,
) -> VortexResult<EncodedPivcoBuffers>
where
    T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let block_shift = if opts.block_shift == 0 {
        DEFAULT_BLOCK_SHIFT
    } else {
        opts.block_shift
    };
    vortex_ensure!(
        (MIN_BLOCK_SHIFT..=MAX_BLOCK_SHIFT).contains(&block_shift),
        "Pivco block_shift {block_shift} must be in {MIN_BLOCK_SHIFT}..={MAX_BLOCK_SHIFT}"
    );
    let block_values = 1usize << block_shift;

    let len = values.len();
    if len == 0 {
        let counts = [0u64; MAX_SYMS];
        let tree = PivcoTree::build(&counts, opts.tree_mode, opts.allow_flat);
        return Ok(EncodedPivcoBuffers {
            block_shift,
            last_block_len: 0,
            tree,
            block_headers: ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u32>()).freeze(),
            tree_payload: ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u32>()).freeze(),
            raw_bits: BufferMut::<u64>::with_capacity(0)
                .freeze()
                .into_byte_buffer(),
        });
    }

    let (counts, fill_sym) = count_chunk_symbols(values, is_signed, mask)?;
    let tree = PivcoTree::build(&counts, opts.tree_mode, opts.allow_flat);
    let has_extra_bits = compute_tree_max_ebits(&tree) > 0;

    let num_blocks = len.div_ceil(block_values);
    let rem = len & (block_values - 1);
    let last_block_len = if rem == 0 {
        block_values as u16
    } else {
        rem as u16
    };
    let stride = block_header_stride(has_extra_bits);
    let mut headers_buf =
        ByteBufferMut::with_capacity_aligned((num_blocks + 1) * stride, Alignment::of::<u32>());
    let mut tree_payload_buf =
        ByteBufferMut::with_capacity_aligned(num_blocks * 128, Alignment::of::<u32>());
    let mut raw_bits_words =
        BufferMut::<u64>::with_capacity(if has_extra_bits { num_blocks * 16 } else { 0 });

    let mut scratch = EncodeScratch::new(block_values);
    let fill_rank = tree.sym_to_rank[usize::from(fill_sym)];
    let mut raw_word_cursor = 0usize;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * block_values;
        let block_end = (block_start + block_values).min(len);
        scratch.encode_one_block(
            values,
            block_start,
            block_end,
            is_signed,
            has_extra_bits,
            fill_rank,
            mask,
            &tree,
            opts.pha_enabled,
            &mut raw_word_cursor,
            &mut headers_buf,
            &mut tree_payload_buf,
            &mut raw_bits_words,
        )?;
    }

    // Write final sentinel entry `hdr[num_blocks]`.
    let end_tree_offset = u32::try_from(tree_payload_buf.len())
        .map_err(|_| vortex_err!("Pivco tree_payload exceeds 4 GiB u32 offset limit"))?;
    let end_raw_word_offset = if has_extra_bits {
        u32::try_from(raw_word_cursor)
            .map_err(|_| vortex_err!("Pivco raw_bits exceeds 32 GiB u32 word-offset limit"))?
    } else {
        0u32
    };
    PivcoBlockOffset {
        tree_byte_offset: end_tree_offset,
        raw_word_offset: end_raw_word_offset,
    }
    .write_into(has_extra_bits, &mut headers_buf);

    Ok(EncodedPivcoBuffers {
        block_shift,
        last_block_len,
        tree,
        block_headers: headers_buf.freeze(),
        tree_payload: tree_payload_buf.freeze(),
        raw_bits: raw_bits_words.freeze().into_byte_buffer(),
    })
}

fn count_chunk_symbols<T>(
    values: &[T],
    is_signed: bool,
    mask: &Mask,
) -> VortexResult<([u64; MAX_SYMS], u8)>
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let len = values.len();
    let mut counts = [0u64; MAX_SYMS];
    let mut valid_count = 0u64;

    if matches!(mask, Mask::AllTrue(_)) {
        if is_signed {
            for &v in values {
                let signed = AsPrimitive::<i64>::as_(v);
                vortex_ensure!(
                    signed >= 0,
                    "Pivco::encode requires non-negative integer values (got {signed}); wrap negative arrays with ZigZag"
                );
                counts[pow2_sym_and_ebits(signed as u64).0] += 1;
            }
        } else {
            for &v in values {
                counts[pow2_sym_and_ebits(AsPrimitive::<u64>::as_(v)).0] += 1;
            }
        }
        valid_count = len as u64;
    } else if let vortex_mask::AllOr::Some(buf) = mask.bit_buffer() {
        for (start, end) in buf.set_slices() {
            if is_signed {
                for &v in &values[start..end] {
                    let signed = AsPrimitive::<i64>::as_(v);
                    if signed < 0 {
                        vortex_bail!(
                            "Pivco::encode requires non-negative integer values (got {signed}); wrap negative arrays with ZigZag"
                        );
                    }
                    counts[pow2_sym_and_ebits(signed as u64).0] += 1;
                }
            } else {
                for &v in &values[start..end] {
                    counts[pow2_sym_and_ebits(AsPrimitive::<u64>::as_(v)).0] += 1;
                }
            }
            valid_count += (end - start) as u64;
        }
    }

    let mut fill_sym = 0u8;
    let mut best_exact_count = 0u64;
    for (sym, &cnt) in counts[..32].iter().enumerate() {
        if cnt > best_exact_count {
            best_exact_count = cnt;
            fill_sym = sym as u8;
        }
    }

    let fill_positions = (len as u64).saturating_sub(valid_count);
    counts[usize::from(fill_sym)] += fill_positions;
    Ok((counts, fill_sym))
}

struct EncodeScratch {
    ranks_root: Vec<u8>,
    part_arena: Vec<u8>,
    bm_scratch: [u8; MAX_BM_BYTES],
    rans_words: Vec<u16>,
    tree_bytes: Vec<u8>,
    bit_writer: BitWriter,
}

impl EncodeScratch {
    fn new(block_values: usize) -> Self {
        Self {
            ranks_root: vec![0u8; block_values + 16],
            part_arena: vec![0u8; 2 * MAX_STACK_DEPTH * (block_values + 16)],
            bm_scratch: [0u8; MAX_BM_BYTES],
            rans_words: Vec::with_capacity(block_values / 16),
            tree_bytes: Vec::with_capacity(block_values),
            bit_writer: BitWriter::with_capacity(block_values),
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn encode_one_block<T>(
        &mut self,
        values: &[T],
        block_start: usize,
        block_end: usize,
        is_signed: bool,
        has_extra_bits: bool,
        fill_rank: u8,
        mask: &Mask,
        tree: &PivcoTree,
        pha_enabled: bool,
        raw_word_cursor: &mut usize,
        headers_buf: &mut ByteBufferMut,
        tree_payload_buf: &mut ByteBufferMut,
        raw_bits_words: &mut BufferMut<u64>,
    ) -> VortexResult<()>
    where
        T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
    {
        let block_len = block_end - block_start;
        self.bit_writer.clear();
        self.tree_bytes.clear();

        populate_block_ranks(
            &values[block_start..block_end],
            block_start,
            is_signed,
            has_extra_bits,
            fill_rank,
            mask,
            &tree.sym_to_rank,
            &mut self.ranks_root[..block_len],
            &mut self.bit_writer,
        );

        encode_subtree(
            tree,
            0,
            &self.ranks_root[..block_len],
            pha_enabled,
            &mut self.bm_scratch,
            &mut self.rans_words,
            &mut self.part_arena,
            &mut self.tree_bytes,
        )?;

        let tree_byte_offset = u32::try_from(tree_payload_buf.len())
            .map_err(|_| vortex_err!("Pivco tree_payload exceeds 4 GiB u32 offset limit"))?;
        let raw_word_offset = if has_extra_bits {
            u32::try_from(*raw_word_cursor)
                .map_err(|_| vortex_err!("Pivco raw_bits exceeds 32 GiB u32 word-offset limit"))?
        } else {
            0u32
        };

        PivcoBlockOffset {
            tree_byte_offset,
            raw_word_offset,
        }
        .write_into(has_extra_bits, headers_buf);

        debug_assert!(tree_payload_buf.len().is_multiple_of(4));
        tree_payload_buf.extend_from_slice(&self.tree_bytes);
        let pad = (4 - (tree_payload_buf.len() & 3)) & 3;
        if pad > 0 {
            tree_payload_buf.extend_from_slice(&[0u8; 4][..pad]);
        }

        if has_extra_bits {
            *raw_word_cursor += self.bit_writer.flush_into(raw_bits_words);
        }

        Ok(())
    }
}

#[expect(clippy::too_many_arguments)]
fn populate_block_ranks<T>(
    slice: &[T],
    block_start: usize,
    is_signed: bool,
    has_extra_bits: bool,
    fill_rank: u8,
    mask: &Mask,
    sym_to_rank: &[u8; MAX_SYMS],
    ranks_out: &mut [u8],
    bit_writer: &mut BitWriter,
) where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    match mask {
        Mask::AllTrue(_) => {
            if has_extra_bits {
                for (dst, &v) in ranks_out.iter_mut().zip(slice) {
                    let u = if is_signed {
                        AsPrimitive::<i64>::as_(v) as u64
                    } else {
                        AsPrimitive::<u64>::as_(v)
                    };
                    let (sym, ebits, raw) = encode_pow2_bin(u);
                    *dst = sym_to_rank[usize::from(sym)];
                    if ebits > 0 {
                        bit_writer.write(raw, ebits);
                    }
                }
                bit_writer.finish_word_aligned();
            } else {
                for (dst, &v) in ranks_out.iter_mut().zip(slice) {
                    let u = if is_signed {
                        AsPrimitive::<i64>::as_(v) as u64
                    } else {
                        AsPrimitive::<u64>::as_(v)
                    };
                    *dst = sym_to_rank[usize::from(u as u8)];
                }
            }
        }
        Mask::Values(mv) => {
            // `fill_rank` always corresponds to an exact symbol in `0..32` (`ebits == 0`), so
            // pre-filling `ranks_out` and visiting valid runs in ascending order emits the exact
            // `raw_bits` stream with zero per-element `BitBuffer::value(i)` calls.
            ranks_out.fill(fill_rank);
            let block_bits = mv
                .bit_buffer()
                .slice(block_start..block_start + slice.len());
            for (run_start, run_end) in block_bits.set_slices() {
                for (dst, &v) in ranks_out[run_start..run_end]
                    .iter_mut()
                    .zip(&slice[run_start..run_end])
                {
                    let u = if is_signed {
                        AsPrimitive::<i64>::as_(v) as u64
                    } else {
                        AsPrimitive::<u64>::as_(v)
                    };
                    let (sym, ebits, raw) = encode_pow2_bin(u);
                    *dst = sym_to_rank[usize::from(sym)];
                    if has_extra_bits && ebits > 0 {
                        bit_writer.write(raw, ebits);
                    }
                }
            }
            if has_extra_bits {
                bit_writer.finish_word_aligned();
            }
        }
        Mask::AllFalse(_) => {
            ranks_out.fill(fill_rank);
            if has_extra_bits {
                bit_writer.finish_word_aligned();
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn encode_subtree(
    tree: &PivcoTree,
    node_idx: u16,
    ranks: &[u8],
    pha_enabled: bool,
    bm_scratch: &mut [u8; MAX_BM_BYTES],
    rans_words: &mut Vec<u16>,
    arena: &mut [u8],
    dst: &mut Vec<u8>,
) -> VortexResult<()> {
    let n = ranks.len();
    if n == 0 {
        return Ok(());
    }

    match &tree.nodes[usize::from(node_idx)] {
        PivcoNode::SingleLeaf { .. } => Ok(()),
        PivcoNode::Flat {
            depth, base_rank, ..
        } => {
            pack_flat_codes(*depth, ranks, *base_rank, dst);
            Ok(())
        }
        PivcoNode::Internal {
            tree_depth,
            pivot_rank,
            left_child,
            right_child,
        } => {
            let stride = n + 16;
            let (left_buf, rest1) = arena.split_at_mut(stride);
            let (right_buf, rest2) = rest1.split_at_mut(stride);
            let k_right = partition_ranks_by_pivot(
                ranks,
                *pivot_rank,
                &mut bm_scratch[..n.div_ceil(8)],
                left_buf,
                right_buf,
            );
            let k_left = n - k_right;
            write_node_bitmap(
                &bm_scratch[..n.div_ceil(8)],
                n,
                k_right,
                usize::from(*tree_depth),
                pha_enabled,
                rans_words,
                dst,
            );

            if let ChildSpec::Node(l_idx) = left_child {
                let len_pos = dst.len();
                dst.extend_from_slice(&0u16.to_le_bytes());
                let left_start = dst.len();
                if k_left > 0 {
                    encode_subtree(
                        tree,
                        *l_idx,
                        &left_buf[..k_left],
                        pha_enabled,
                        bm_scratch,
                        rans_words,
                        rest2,
                        dst,
                    )?;
                }
                let left_bytes = dst.len() - left_start;
                let left_bytes_u16 = u16::try_from(left_bytes).map_err(|_| {
                    vortex_err!("Pivco left_subtree_bytes {left_bytes} exceeds u16::MAX")
                })?;
                dst[len_pos..len_pos + 2].copy_from_slice(&left_bytes_u16.to_le_bytes());
            }
            if let ChildSpec::Node(r_idx) = right_child
                && k_right > 0
            {
                encode_subtree(
                    tree,
                    *r_idx,
                    &right_buf[..k_right],
                    pha_enabled,
                    bm_scratch,
                    rans_words,
                    rest2,
                    dst,
                )?;
            }
            Ok(())
        }
    }
}
