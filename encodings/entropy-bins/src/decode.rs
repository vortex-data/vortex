// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block parsing and the portable decoder. The x86 kernels must produce the same output.

// Single-letter names follow the ANS literature: x is a coder state, s the state log, l = 2^s.
#![allow(clippy::many_single_char_names)]

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::EntropyBinsChunk;
use crate::coder::FLAG_STOP;
use crate::coder::FLAG_UNIFORM;
use crate::coder::IdTable;
use crate::coder::LANES;
use crate::coder::MAX_BLOCK_VALUES;
use crate::coder::MAX_LAG;

/// One block's id stream and offsets, borrowed from the array's data buffer. `words` and
/// `offsets` run to the end of the buffer (including its tail padding) so vector loads may read
/// past the block.
pub(crate) struct BlockView<'a> {
    pub(crate) n: usize,
    pub(crate) uniform: Option<u8>,
    pub(crate) states: [u8; LANES],
    /// Absolute round from which each lane stops refilling.
    pub(crate) stop: [u16; LANES],
    pub(crate) words: &'a [u8],
    pub(crate) offsets: &'a [u8],
}

/// Read `w <= 64` bits at bit position `pos`; bytes past the end read as zero.
pub(crate) fn read_bits(bytes: &[u8], pos: usize, w: u32) -> u64 {
    if w == 0 {
        return 0;
    }
    let start = pos >> 3;
    let mut buf = [0u8; 16];
    let avail = bytes.len().saturating_sub(start).min(16);
    buf[..avail].copy_from_slice(&bytes[start..start + avail]);
    let v = u128::from_le_bytes(buf) >> (pos & 7);
    // Truncation keeps the low `w` bits, which is the value being read.
    #[allow(clippy::cast_possible_truncation)]
    let v = v as u64;
    if w == 64 { v } else { v & ((1u64 << w) - 1) }
}

/// Parse the block starting at `start` in `data`.
pub(crate) fn parse_block<'a>(
    data: &'a [u8],
    start: usize,
    n: usize,
    table: Option<&IdTable>,
) -> VortexResult<BlockView<'a>> {
    if start >= data.len() {
        vortex_bail!("block start {start} out of bounds");
    }
    let flags = data[start];
    if flags & FLAG_UNIFORM != 0 {
        let Some(&sym) = data.get(start + 1) else {
            vortex_bail!("truncated uniform block");
        };
        return Ok(BlockView {
            n,
            uniform: Some(sym),
            states: [0; LANES],
            stop: [0; LANES],
            words: &[],
            offsets: &data[start + 2..],
        });
    }
    let Some(t) = table else {
        vortex_bail!("coded block in a single-bin chunk");
    };
    let mut p = start + 1;
    let Some(nw) = data.get(p..p + 2) else {
        vortex_bail!("truncated block header");
    };
    let n_words = usize::from(u16::from_le_bytes([nw[0], nw[1]]));
    p += 2;
    let state_bytes = (LANES * t.s as usize).div_ceil(8);
    let mut states = [0u8; LANES];
    for (lane, st) in states.iter_mut().enumerate() {
        *st = u8::try_from(read_bits(&data[p..], lane * t.s as usize, t.s))?;
    }
    p += state_bytes;
    let n_pad = n.div_ceil(LANES) * LANES;
    let rounds = (n_pad / LANES).div_ceil(t.r);
    let rounds = u16::try_from(rounds)?;
    let mut stop = [rounds; LANES];
    if flags & FLAG_STOP != 0 {
        for (lane, st) in stop.iter_mut().enumerate() {
            let rel = u16::try_from(read_bits(&data[p..], lane * 3, 3))?;
            *st = rounds.saturating_sub(rel);
        }
        p += 6;
    }
    let word_bytes = t.word_bits as usize / 8;
    if p + word_bytes * n_words > data.len() {
        vortex_bail!("block words out of bounds");
    }
    Ok(BlockView {
        n,
        uniform: None,
        states,
        stop,
        words: &data[p..],
        offsets: &data[p + word_bytes * n_words..],
    })
}

/// Decode the bin ids of positions `[0, min(n, limit))`, rounded up to a multiple of 16, into
/// `out` (which must hold at least `b.n` bytes rounded up to a multiple of 16).
pub(crate) fn ids_scalar(t: &IdTable, b: &BlockView<'_>, out: &mut [u8], limit: usize) {
    let steps = b.n.min(limit).div_ceil(LANES);
    let l = 1u32 << t.s;
    let mut x = [0u32; LANES];
    for lane in 0..LANES {
        x[lane] = u32::from(b.states[lane]) + l;
    }
    let mut buf = [0u32; LANES];
    let mut avail = [0u32; LANES];
    let mut p = 0usize;
    let wb = t.word_bits;
    let word = |p: usize| -> u32 {
        if wb == 8 {
            return b.words.get(p).map_or(0, |&w| u32::from(w));
        }
        match b.words.get(2 * p..2 * p + 2) {
            Some(w) => u32::from(u16::from_le_bytes([w[0], w[1]])),
            None => 0,
        }
    };
    for step in 0..steps {
        if step % t.r == 0 {
            let round = step / t.r;
            for lane in 0..LANES {
                if round < usize::from(b.stop[lane]) && avail[lane] < wb {
                    buf[lane] |= word(p) << avail[lane];
                    avail[lane] += wb;
                    p += 1;
                }
            }
        }
        for lane in 0..LANES {
            let xi = (x[lane] & 127) as usize;
            let xs = u32::from(t.xs_tab[xi]);
            let nb = t.s - (31 - xs.leading_zeros());
            let bits = buf[lane] & ((1u32 << nb) - 1);
            buf[lane] >>= nb;
            // A lane's last step may consume bits that were never written (see `encode_ids`).
            avail[lane] = avail[lane].wrapping_sub(nb);
            x[lane] = (xs << nb) | bits;
            out[step * LANES + lane] = t.sym_tab[xi];
        }
    }
}

/// Per-chunk decode state: bins, id tables and the transformed lower bounds.
#[derive(Debug)]
pub(crate) struct ChunkDecoder {
    pub(crate) table: Option<IdTable>,
    pub(crate) widths: Vec<u32>,
    /// `lower + base` per bin (wrapping), so a value is `tl[id] + offset`.
    pub(crate) tl: Vec<u64>,
    /// Register tables for the vector merges, zero-padded.
    pub(crate) wtab: [u8; 64],
    pub(crate) tl64: [u64; 32],
    pub(crate) mask64: [u64; 16],
    pub(crate) tl32: [u32; 32],
    pub(crate) w32: [u32; 32],
    pub(crate) max_width: u32,
}

impl ChunkDecoder {
    pub(crate) fn new(chunk: &EntropyBinsChunk, base: u64, word_bits: u32) -> VortexResult<Self> {
        let tl: Vec<u64> = chunk.lowers.iter().map(|&l| l.wrapping_add(base)).collect();
        let mut wtab = [0u8; 64];
        let mut tl64 = [0u64; 32];
        let mut mask64 = [0u64; 16];
        let mut tl32 = [0u32; 32];
        let mut w32 = [0u32; 32];
        for (i, (&w, &t)) in chunk.widths.iter().zip(&tl).enumerate() {
            if i < 64 {
                wtab[i] = u8::try_from(w.min(255))?;
            }
            if i < 32 {
                tl64[i] = t;
                // Low 32 bits: the 16-wide merge works mod 2^32.
                #[allow(clippy::cast_possible_truncation)]
                let t32 = t as u32;
                tl32[i] = t32;
                w32[i] = w;
            }
            if i < 16 {
                mask64[i] = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
            }
        }
        Ok(Self {
            table: IdTable::new(chunk, word_bits)?,
            widths: chunk.widths.clone(),
            max_width: chunk.widths.iter().copied().max().unwrap_or(0),
            tl,
            wtab,
            tl64,
            mask64,
            tl32,
            w32,
        })
    }

    /// Fill `ids[..]` for the block (any `limit`) using the fastest available path.
    pub(crate) fn ids(&self, b: &BlockView<'_>, ids: &mut [u8], limit: usize) {
        match (b.uniform, &self.table) {
            (Some(u), _) => {
                let m = b.n.min(limit).div_ceil(LANES) * LANES;
                ids[..m].fill(u);
            }
            // A single-bin chunk only has uniform blocks; `parse_block` rejects anything else.
            (None, None) => {}
            (None, Some(t)) => {
                #[cfg(target_arch = "x86_64")]
                if crate::x86::has_avx512() {
                    // SAFETY: the CPU supports the required AVX-512 features, `ids` holds a full
                    // block and the block's words are followed by the buffer's tail padding.
                    unsafe { crate::x86::ids16::<1>(t, [b], [ids.as_mut_ptr()], limit) };
                    return;
                }
                ids_scalar(t, b, ids, limit);
            }
        }
    }

    /// Bit position of value `l` in the block's offsets.
    pub(crate) fn offset_pos(&self, ids: &[u8], l: usize) -> usize {
        ids[..l]
            .iter()
            .map(|&id| self.widths[usize::from(id)] as usize)
            .sum()
    }

    /// The coded value (before truncation to the output width) at position `l`: the row's value,
    /// or its difference from the row `lag` back.
    pub(crate) fn value_at(&self, b: &BlockView<'_>, ids: &[u8], l: usize) -> u64 {
        let pos = self.offset_pos(ids, l);
        let id = usize::from(ids[l]);
        self.tl[id].wrapping_add(read_bits(b.offsets, pos, self.widths[id]))
    }
}

/// Values the decoder can write: integers truncated from the 64-bit value.
pub(crate) trait OutInt: Copy + Default {
    const BYTES: usize;
    fn truncate_from(v: u64) -> Self;
}

macro_rules! out_int {
    ($($t:ty),*) => {$(
        impl OutInt for $t {
            const BYTES: usize = size_of::<$t>();
            // Truncation to the output width is the decoding step.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            fn truncate_from(v: u64) -> Self {
                v as $t
            }
        }
    )*};
}
out_int!(u8, u16, u32, u64, i8, i16, i32, i64);

/// Portable merge: `out[i] = tl[id] + offset` for the block. With `k = seeds.len() > 0`
/// those are differences: the first `k` values are the seeds and every later value adds its
/// difference to the value `k` rows back.
pub(crate) fn merge_scalar<T: OutInt>(
    d: &ChunkDecoder,
    b: &BlockView<'_>,
    ids: &[u8],
    out: &mut [T],
    seeds: &[u64],
) {
    let lag = seeds.len();
    let mut ring = [0u64; MAX_LAG];
    ring[..lag].copy_from_slice(seeds);
    let mut pos = 0usize;
    for i in 0..b.n {
        let id = usize::from(ids[i]);
        let w = d.widths[id];
        let v = d.tl[id].wrapping_add(read_bits(b.offsets, pos, w));
        pos += w as usize;
        out[i] = T::truncate_from(match lag {
            0 => v,
            _ if i < lag => seeds[i],
            _ => {
                let slot = &mut ring[i % lag];
                *slot = slot.wrapping_add(v);
                *slot
            }
        });
    }
}

/// Decode the ids of `views` (one to four blocks of one chunk) into consecutive
/// [`IDS_SCRATCH`]-byte slots of `ids`. Four coded blocks of equal length decode in lockstep.
pub(crate) fn decode_ids(d: &ChunkDecoder, views: &[BlockView<'_>], ids: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if let (Some(t), [b0, b1, b2, b3]) = (&d.table, views)
        && crate::x86::has_avx512()
        && views.iter().all(|v| v.uniform.is_none() && v.n == b0.n)
    {
        let p = ids.as_mut_ptr();
        // SAFETY: AVX-512 is available, `ids` holds four block-sized slots, the blocks are coded
        // and full, and their words come from a padded buffer.
        unsafe {
            crate::x86::ids16::<4>(
                t,
                [b0, b1, b2, b3],
                [
                    p,
                    p.add(IDS_SCRATCH),
                    p.add(2 * IDS_SCRATCH),
                    p.add(3 * IDS_SCRATCH),
                ],
                usize::MAX,
            )
        };
        return;
    }
    for (k, v) in views.iter().enumerate() {
        d.ids(
            v,
            &mut ids[k * IDS_SCRATCH..(k + 1) * IDS_SCRATCH],
            usize::MAX,
        );
    }
}

/// Merge one block's decoded ids with its offsets into `out[..b.n]` (see [`merge_scalar`]).
pub(crate) fn merge_block<T: OutInt>(
    d: &ChunkDecoder,
    b: &BlockView<'_>,
    ids: &[u8],
    out: &mut [T],
    seeds: &[u64],
) {
    #[cfg(target_arch = "x86_64")]
    if crate::x86::has_avx512() && crate::x86::merge(d, b, ids, out, seeds) {
        return;
    }
    merge_scalar(d, b, ids, out, seeds);
}

/// Block capacity of the id scratch buffer.
pub(crate) const IDS_SCRATCH: usize = MAX_BLOCK_VALUES + 64;
