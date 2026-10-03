// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A block-structured binned tANS stream of latents: the codec's leaf.
//!
//! Latents are split into blocks of [`BLOCK_SIZE`]. One set of bins and one tANS table is shared
//! by the whole stream, but each block is independently decodable through a small index: its
//! starting bit, the length of its tANS stream, its final tANS states, and its delta initial
//! values. The index is stored as [`Packed`] residuals from a linear trend, so it costs close to
//! nothing on regular data. Reading one value decodes at most one block, and with no delta
//! encoding only up to that value.
//!
//! A block's bits are its tANS stream (bin indices) followed by its offset stream (the position
//! of each value within its bin, as raw bits). With a single bin there is no tANS stream and
//! decoding is plain bit unpacking.

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ans;
use crate::ans::N_LANES;
use crate::bins;
use crate::bits::BitWriter;
use crate::bits::MAX_FAST_READ_BITS;
use crate::bits::PADDING;
use crate::bits::load_u64;
use crate::bits::read_bits;
use crate::bits::read_bits_wide;
use crate::latent::Latent;
use crate::latent::offset_bits;
use crate::packed::Packed;

pub const BLOCK_SIZE: usize = 1024;

#[derive(Clone, Debug)]
pub struct Config {
    /// Log2 of the number of candidate bins before merging.
    pub n_bins_log: u8,
    pub max_ans_size_log: u8,
    /// Highest consecutive delta order to try (Pco also allows up to 7).
    pub max_delta_order: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_bins_log: 8,
            max_ans_size_log: 10,
            max_delta_order: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bin<L> {
    pub lower: L,
    pub offset_bits: u8,
    pub weight: u16,
}

#[derive(Clone, Debug)]
pub struct Stream<L> {
    pub n: usize,
    pub delta_order: u8,
    pub ans_size_log: u8,
    pub bins: Vec<Bin<L>>,
    /// Bit position where each block starts.
    pub starts: Packed,
    /// Length in bits of each block's tANS stream.
    pub ans_bits: Packed,
    /// Final tANS states, [`N_LANES`] per coded block; empty with a single bin.
    pub states: Packed,
    /// The stream's most common symbol. A block whose symbols all equal it is "uniform": it has
    /// no bits and no states.
    pub dominant: L,
    /// For each block, the number of non-uniform blocks before it, plus a final total. Indexes
    /// `states` and marks uniform blocks (those whose count does not increase).
    pub coded_before: Packed,
    /// Delta initial values: one [`Packed`] per order, with one value per block.
    pub inits: Vec<Packed>,
    /// Block bits. Unpadded, so it can be a zero-copy view of a file buffer.
    pub data: ByteBuffer,
}

impl<L: Latent> Stream<L> {
    /// Serialized size: everything a reader needs, excluding the zero padding.
    pub fn nbytes(&self) -> usize {
        let latent_bytes = (L::BITS / 8) as usize;
        8 + self.bins.len() * (latent_bytes + 3)
            + self.starts.nbytes()
            + self.ans_bits.nbytes()
            + self.states.nbytes()
            + self.coded_before.nbytes()
            + (L::BITS / 8) as usize
            + self.inits.iter().map(Packed::nbytes).sum::<usize>()
            + self.data.len()
    }

    pub fn n_blocks(&self) -> usize {
        self.n.div_ceil(BLOCK_SIZE)
    }

    fn block_len(&self, block: usize) -> usize {
        (self.n - block * BLOCK_SIZE).min(BLOCK_SIZE)
    }
}

/// Block-local consecutive deltas. Pushes the block's initial value for each order (zero past
/// the block's length) onto `inits` and appends its remaining symbols to `symbols`.
fn block_symbols<L: Latent>(
    block: &[L],
    order: u8,
    inits: &mut [Vec<u64>],
    symbols: &mut Vec<L>,
) {
    let order = usize::from(order);
    let k = order.min(block.len());
    let start = symbols.len();
    symbols.extend_from_slice(block);
    for level in 0..order {
        if level >= k {
            inits[level].push(0);
            continue;
        }
        let seq = &mut symbols[start + level..];
        inits[level].push(seq[0].to_u64());
        for i in (1..seq.len()).rev() {
            seq[i] = seq[i].wrapping_sub(seq[i - 1]).wrapping_add(L::MID);
        }
    }
    // Drop the initial values, which sit at the front of each level's sequence.
    symbols.drain(start..start + k);
}

struct Candidate<L> {
    order: u8,
    dominant: L,
    symbols: Vec<L>,
    inits: Vec<Packed>,
    bins: Vec<bins::RawBin<L>>,
    ans_size_log: u8,
    bits: f64,
}

fn candidate<L: Latent>(latents: &[L], order: u8, config: &Config) -> Candidate<L> {
    let mut inits = vec![Vec::new(); usize::from(order)];
    let mut symbols = Vec::with_capacity(latents.len());
    for block in latents.chunks(BLOCK_SIZE) {
        block_symbols(block, order, &mut inits, &mut symbols);
    }
    let inits: Vec<Packed> = inits.iter().map(|v| Packed::new(v)).collect();
    let mut sorted = symbols.clone();
    sorted.sort_unstable();
    let dominant = sorted
        .chunk_by(|a, b| a == b)
        .max_by_key(|run| run.len())
        .map_or(L::ZERO, |run| run[0]);
    let ans_size_log =
        bins::ans_size_log(symbols.len(), config.n_bins_log, config.max_ans_size_log);
    let raw = bins::optimize(&bins::histogram(&sorted, config.n_bins_log), ans_size_log);
    let mut coded_blocks = 0usize;
    let mut pos = 0;
    for block in latents.chunks(BLOCK_SIZE) {
        let n_syms = block.len() - usize::from(order).min(block.len());
        if symbols[pos..pos + n_syms].iter().any(|&s| s != dominant) {
            coded_blocks += 1;
        }
        pos += n_syms;
    }
    let index_bits = if raw.len() > 1 {
        // Rough cost of the start, tANS length and state columns.
        coded_blocks as f64 * (8.0 + 8.0 + (N_LANES as f64) * f64::from(ans_size_log))
    } else {
        coded_blocks as f64 * 8.0
    };
    let bits = bins::estimated_bits(&raw, ans_size_log) * symbols.len() as f64
        + inits.iter().map(|p| p.nbytes() as f64 * 8.0).sum::<f64>()
        + index_bits;
    Candidate {
        order,
        dominant,
        symbols,
        inits,
        bins: raw,
        ans_size_log,
        bits,
    }
}

/// Estimated encoded bits for these latents at their best delta order.
pub fn estimate_bits<L: Latent>(latents: &[L], config: &Config) -> f64 {
    (0..=config.max_delta_order)
        .map(|order| candidate(latents, order, config).bits)
        .fold(f64::INFINITY, f64::min)
}

const SAMPLE_RUNS: usize = 8;

/// Contiguous block-sized runs spread across the data, so delta estimates still see neighbors.
pub(crate) fn sample<T: Copy>(values: &[T]) -> Vec<T> {
    if values.len() <= SAMPLE_RUNS * BLOCK_SIZE {
        return values.to_vec();
    }
    let stride = values.len() / SAMPLE_RUNS;
    (0..SAMPLE_RUNS)
        .flat_map(|r| values[r * stride..r * stride + BLOCK_SIZE].iter().copied())
        .collect()
}

pub fn compress_latents<L: Latent>(latents: &[L], config: &Config) -> VortexResult<Stream<L>> {
    // Choose the delta order on a sample, then build only that candidate from all the data.
    let sampled = sample(latents);
    let order = (0..=config.max_delta_order)
        .map(|order| (order, candidate(&sampled, order, config).bits))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(order, _)| order);
    let best = candidate(latents, order, config);

    let ans_size_log = if best.bins.len() <= 1 {
        0
    } else {
        best.ans_size_log
    };
    let counts: Vec<u64> = best.bins.iter().map(|b| b.count).collect();
    let weights = if best.bins.len() <= 1 {
        vec![1; best.bins.len()]
    } else {
        ans::quantize_weights(&counts, ans_size_log)?
    };
    let bins: Vec<Bin<L>> = best
        .bins
        .iter()
        .zip(&weights)
        .map(|(b, &weight)| Bin {
            lower: b.lower,
            offset_bits: offset_bits(b.upper.to_u64() - b.lower.to_u64()),
            weight,
        })
        .collect();
    let lowers: Vec<L> = bins.iter().map(|b| b.lower).collect();
    let encoder = (bins.len() > 1)
        .then(|| ans::Encoder::new(ans_size_log, &weights))
        .transpose()?;

    let n_blocks = latents.len().div_ceil(BLOCK_SIZE);
    let mut starts = Vec::with_capacity(n_blocks);
    let mut ans_lens = Vec::with_capacity(n_blocks);
    let mut states_col = Vec::with_capacity(if encoder.is_some() { n_blocks * N_LANES } else { 0 });
    let mut coded_before = Vec::with_capacity(n_blocks + 1);
    let mut n_coded = 0u64;
    let mut writer = BitWriter::default();
    let mut sym_start = 0;
    let mut bin_idxs = Vec::with_capacity(BLOCK_SIZE);
    let mut chunks = Vec::with_capacity(BLOCK_SIZE);
    for block in 0..n_blocks {
        let block_len = (latents.len() - block * BLOCK_SIZE).min(BLOCK_SIZE);
        let n_syms = block_len - usize::from(best.order).min(block_len);
        let syms = &best.symbols[sym_start..sym_start + n_syms];
        sym_start += n_syms;

        starts.push(writer.bit_len() as u64);
        coded_before.push(n_coded);
        if syms.iter().all(|&s| s == best.dominant) {
            ans_lens.push(0);
            continue;
        }
        n_coded += 1;
        bin_idxs.clear();
        bin_idxs.extend(syms.iter().map(|&s| lowers.partition_point(|&l| l <= s) - 1));

        if let Some(encoder) = &encoder {
            let mut states = [0u16; N_LANES];
            chunks.clear();
            chunks.resize(n_syms, (0u64, 0u32));
            for i in (0..n_syms).rev() {
                let lane = i % N_LANES;
                #[allow(clippy::cast_possible_truncation)]
                let (state, bits, n) = encoder.encode(states[lane], bin_idxs[i] as u16);
                states[lane] = state;
                chunks[i] = (bits, n);
            }
            let ans_start = writer.bit_len();
            for &(bits, n) in &chunks {
                writer.write(bits, n);
            }
            ans_lens.push((writer.bit_len() - ans_start) as u64);
            states_col.extend(states.iter().map(|&s| u64::from(s)));
        } else {
            ans_lens.push(0);
        }

        for (&s, &b) in syms.iter().zip(&bin_idxs) {
            let bin = &bins[b];
            writer.write(
                s.to_u64() - bin.lower.to_u64(),
                u32::from(bin.offset_bits),
            );
        }
    }
    coded_before.push(n_coded);
    let data = ByteBuffer::from(writer.finish());

    Ok(Stream {
        n: latents.len(),
        delta_order: best.order,
        ans_size_log,
        bins,
        starts: Packed::new(&starts),
        ans_bits: Packed::new(&ans_lens),
        states: Packed::new(&states_col),
        dominant: best.dominant,
        coded_before: Packed::new(&coded_before),
        inits: best.inits,
        data,
    })
}

/// One fused decode-table position: the bin it decodes to and the next tANS transition. Masks are
/// precomputed because variable-width masks cost several instructions without BMI2.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Entry<L> {
    lower: L,
    off_mask: L,
    ans_mask: u16,
    next_base: u16,
    offset_bits: u8,
    ans_bits: u8,
}

/// Decoding state derived once from a [`Stream`]: the fused decode table and stream bounds.
pub struct StreamDecoder<L> {
    n: usize,
    table: Vec<Entry<L>>,
    /// The only bin, with its lower bound shifted like the table's, when there is one bin.
    single_bin: Option<Bin<L>>,
    /// The dominant symbol, shifted like the table's lower bounds.
    dominant: L,
    state_mask: u16,
    max_offset_bits: u8,
    bmi2: bool,
}

fn mask<L: Latent>(bits: u8) -> L {
    if u32::from(bits) >= u64::BITS {
        L::from_u64(u64::MAX)
    } else {
        L::from_u64((1u64 << bits) - 1)
    }
}

impl<L: Latent> StreamDecoder<L> {
    /// Validates `stream` and builds its decode tables. Decoding must then be given the same
    /// stream.
    pub fn new(stream: &Stream<L>) -> VortexResult<Self> {
        let n_blocks = stream.n_blocks();
        let order = usize::from(stream.delta_order);
        vortex_ensure!(
            stream.starts.len() == n_blocks && stream.ans_bits.len() == n_blocks,
            "expected {n_blocks} block index entries"
        );
        vortex_ensure!(
            stream.inits.len() == order && stream.inits.iter().all(|p| p.len() == n_blocks),
            "wrong number of delta initial values"
        );
        let n_symbols: usize = (0..n_blocks)
            .map(|b| stream.block_len(b) - order.min(stream.block_len(b)))
            .sum();
        vortex_ensure!(
            n_symbols == 0 || !stream.bins.is_empty(),
            "{n_symbols} symbols need at least one bin"
        );
        vortex_ensure!(
            stream.ans_size_log <= ans::MAX_SIZE_LOG,
            "tANS size log {} too large",
            stream.ans_size_log
        );
        vortex_ensure!(
            stream.coded_before.len() == n_blocks + 1,
            "expected {} coded-block counts",
            n_blocks + 1
        );
        let mut prev = 0;
        for b in 0..=n_blocks {
            let c = stream.coded_before.get(b);
            vortex_ensure!(
                c == prev || (b > 0 && c == prev + 1),
                "coded-block counts must start at 0 and step by 0 or 1"
            );
            prev = c;
        }
        if stream.bins.len() > 1 {
            vortex_ensure!(
                stream.states.len() as u64 == prev * N_LANES as u64,
                "expected {} tANS states",
                prev * N_LANES as u64
            );
        }
        let data_bits = stream.data.len() as u64 * 8;
        for b in 0..n_blocks {
            vortex_ensure!(
                stream
                    .starts
                    .get(b)
                    .checked_add(stream.ans_bits.get(b))
                    .is_some_and(|end| end <= data_bits),
                "block {b} extends past the data"
            );
        }
        let mut max_offset_bits = 0;
        for bin in &stream.bins {
            vortex_ensure!(
                u32::from(bin.offset_bits) <= L::BITS,
                "bin offset bits {} exceed the latent width",
                bin.offset_bits
            );
            max_offset_bits = max_offset_bits.max(bin.offset_bits);
        }

        // Delta symbols are centered on `MID`; folding the un-centering into each bin's lower
        // bound takes it off the prefix sum's dependency chain.
        let lower_shift = if stream.delta_order > 0 {
            L::MID
        } else {
            L::ZERO
        };
        let dominant = stream.dominant.wrapping_sub(lower_shift);
        let single_bin = stream.bins.first().map(|b| Bin {
            lower: b.lower.wrapping_sub(lower_shift),
            ..*b
        });
        let table = if stream.bins.len() > 1 {
            let weights: Vec<u16> = stream.bins.iter().map(|b| b.weight).collect();
            ans::decode_steps(stream.ans_size_log, &weights)?
                .into_iter()
                .map(|step| {
                    let bin = &stream.bins[usize::from(step.symbol)];
                    #[allow(clippy::cast_possible_truncation)]
                    Entry {
                        lower: bin.lower.wrapping_sub(lower_shift),
                        off_mask: mask(bin.offset_bits),
                        ans_mask: ((1u32 << step.n_bits) - 1) as u16,
                        next_base: step.next_base,
                        offset_bits: bin.offset_bits,
                        ans_bits: step.n_bits,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        #[cfg(target_arch = "x86_64")]
        let bmi2 = std::arch::is_x86_feature_detected!("bmi2");
        #[cfg(not(target_arch = "x86_64"))]
        let bmi2 = false;

        #[allow(clippy::cast_possible_truncation)]
        Ok(Self {
            n: stream.n,
            table,
            single_bin,
            dominant,
            state_mask: ((1u32 << stream.ans_size_log) - 1) as u16,
            max_offset_bits,
            bmi2,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Decodes the first `n_syms` symbols of `block` into `out`.
    fn decode_symbols(&self, stream: &Stream<L>, block: usize, n_syms: usize, out: &mut [L]) {
        if n_syms == 0 {
            return;
        }
        let coded = stream.coded_before.get(block);
        if stream.coded_before.get(block + 1) == coded {
            out[..n_syms].fill(self.dominant);
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        let (start, ans_len) = (
            stream.starts.get(block) as usize,
            stream.ans_bits.get(block) as usize,
        );
        // Worst-case bits either stream can consume; corrupt data can claim fewer than it reads.
        let ans_end = start + n_syms * usize::from(stream.ans_size_log);
        let off_end = start + ans_len + n_syms * usize::from(self.max_offset_bits);
        let need = ans_end.max(off_end) / 8 + 1 + PADDING;

        let data = stream.data.as_slice();
        let scratch;
        let (buf, shift) = if need <= data.len() {
            (data, 0)
        } else {
            let first = start / 8;
            let available = &data[first..];
            let mut padded = vec![0u8; need - first];
            padded[..available.len()].copy_from_slice(available);
            scratch = padded;
            (scratch.as_slice(), first * 8)
        };

        let mut states = [0usize; N_LANES];
        if !self.table.is_empty() {
            for (lane, s) in states.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let state = stream.states.get(coded as usize * N_LANES + lane) as usize;
                *s = state & usize::from(self.state_mask);
            }
        }
        let args = KernelArgs {
            buf,
            ans_pos: start - shift,
            off_pos: start + ans_len - shift,
            states,
            out: &mut out[..n_syms],
        };
        let wide = self.max_offset_bits > MAX_FAST_READ_BITS;
        let offsets = self.max_offset_bits > 0;
        // SAFETY: `buf` holds at least `PADDING` bytes past the furthest either stream can read,
        // states are masked to the table size, and BMI2 kernels only run on CPUs with BMI2.
        unsafe {
            match (&self.single_bin, self.table.is_empty()) {
                (Some(bin), true) => match (wide, self.bmi2) {
                    (true, _) => single_bin::<L, true>(bin, args),
                    #[cfg(target_arch = "x86_64")]
                    (false, true) => single_bin_bmi2(bin, args),
                    (false, _) => single_bin::<L, false>(bin, args),
                },
                (_, false) => {
                    let t = self.table.as_slice();
                    match (offsets, wide, self.bmi2) {
                        (_, true, _) => ans_kernel::<L, true, true>(t, args),
                        #[cfg(target_arch = "x86_64")]
                        (true, false, true) => ans_bmi2::<L, true>(t, args),
                        #[cfg(target_arch = "x86_64")]
                        (false, false, true) => ans_bmi2::<L, false>(t, args),
                        (true, false, _) => ans_kernel::<L, true, false>(t, args),
                        (false, false, _) => ans_kernel::<L, false, false>(t, args),
                    }
                }
                // Validation rules out symbols without bins.
                (None, true) => {}
            }
        }
    }

    /// Decodes the first `n_values` latents of `block` into `out[..n_values]` (and possibly a
    /// few more, up to the delta order). `out` must hold the whole block.
    pub fn decode_block_prefix(
        &self,
        stream: &Stream<L>,
        block: usize,
        n_values: usize,
        out: &mut [L],
    ) {
        let block_len = stream.block_len(block);
        let order = usize::from(stream.delta_order);
        let k = order.min(block_len);
        let n_values = n_values.min(block_len).max(k);
        let n_syms = n_values - k;
        self.decode_symbols(stream, block, n_syms, &mut out[k..]);
        // The innermost level's symbols come out of the decode table already un-centered, so its
        // prefix sum is a single dependent add per value.
        for level in (0..k).rev() {
            out[level] = L::from_u64(stream.inits[level].get(block));
            let shift = if level + 1 == k { L::ZERO } else { L::MID };
            for i in level + 1..n_values {
                let delta = out[i].wrapping_sub(shift);
                out[i] = out[i - 1].wrapping_add(delta);
            }
        }
    }

    pub fn decode_all(&self, stream: &Stream<L>) -> Vec<L> {
        let mut out = vec![L::ZERO; self.n];
        for (block, chunk) in out.chunks_mut(BLOCK_SIZE).enumerate() {
            self.decode_block_prefix(stream, block, chunk.len(), chunk);
        }
        out
    }
}

struct KernelArgs<'a, L> {
    buf: &'a [u8],
    /// Bit position of the tANS stream.
    ans_pos: usize,
    /// Bit position of the offset stream.
    off_pos: usize,
    states: [usize; N_LANES],
    out: &'a mut [L],
}

/// Reads the offset at bit `pos`.
///
/// # Safety
///
/// `buf` has [`PADDING`] readable bytes past `pos / 8`.
#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn read_offset<L: Latent, const WIDE: bool>(buf: &[u8], pos: usize, bits: u8, mask: L) -> L {
    // SAFETY: forwarded from the caller.
    unsafe {
        if WIDE {
            L::from_u64(read_bits_wide(buf, pos, bits))
        } else {
            L::from_u64(load_u64(buf, pos >> 3) >> (pos & 7)) & mask
        }
    }
}

/// Fixed-width bit unpacking for a single bin.
///
/// # Safety
///
/// `args.buf` has [`PADDING`] readable bytes past the end of the offset stream.
#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn single_bin<L: Latent, const WIDE: bool>(bin: &Bin<L>, args: KernelArgs<'_, L>) {
    let bits = bin.offset_bits;
    if bits == 0 {
        args.out.fill(bin.lower);
        return;
    }
    let off_mask = mask::<L>(bits);
    let mut pos = args.off_pos;
    for o in args.out.iter_mut() {
        // SAFETY: forwarded from the caller.
        let offset = unsafe { read_offset::<L, WIDE>(args.buf, pos, bits, off_mask) };
        *o = bin.lower.wrapping_add(offset);
        pos += usize::from(bits);
    }
}

/// Fused tANS + offset decoding with `N_LANES` interleaved states.
///
/// # Safety
///
/// `args.buf` has [`PADDING`] readable bytes past the furthest point either stream can reach,
/// and every state in `args.states` indexes `table`.
#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn ans_kernel<L: Latent, const OFFSETS: bool, const WIDE: bool>(
    table: &[Entry<L>],
    args: KernelArgs<'_, L>,
) {
    let KernelArgs {
        buf,
        mut ans_pos,
        mut off_pos,
        mut states,
        out,
    } = args;
    let n_syms = out.len();

    let mut groups = out.chunks_exact_mut(N_LANES);
    for group in &mut groups {
        // Four states read at most 4 * MAX_SIZE_LOG = 48 bits, so one load covers the group.
        // SAFETY: forwarded from the caller.
        let mut word = unsafe { load_u64(buf, ans_pos >> 3) } >> (ans_pos & 7);
        for lane in 0..N_LANES {
            // SAFETY: states index the table: every `next_base + bits` is below its size.
            let entry = unsafe { *table.get_unchecked(states[lane]) };
            group[lane] = if OFFSETS {
                // SAFETY: forwarded from the caller.
                let offset = unsafe {
                    read_offset::<L, WIDE>(buf, off_pos, entry.offset_bits, entry.off_mask)
                };
                off_pos += usize::from(entry.offset_bits);
                entry.lower.wrapping_add(offset)
            } else {
                entry.lower
            };
            #[allow(clippy::cast_possible_truncation)]
            let bits = (word as usize) & usize::from(entry.ans_mask);
            word >>= entry.ans_bits;
            ans_pos += usize::from(entry.ans_bits);
            states[lane] = usize::from(entry.next_base) + bits;
        }
    }
    let tail_start = n_syms - groups.into_remainder().len();
    for i in tail_start..n_syms {
        let lane = i % N_LANES;
        // SAFETY: as above.
        let entry = unsafe { *table.get_unchecked(states[lane]) };
        out[i] = if OFFSETS {
            // SAFETY: forwarded from the caller.
            let offset =
                unsafe { read_offset::<L, WIDE>(buf, off_pos, entry.offset_bits, entry.off_mask) };
            off_pos += usize::from(entry.offset_bits);
            entry.lower.wrapping_add(offset)
        } else {
            entry.lower
        };
        // SAFETY: forwarded from the caller.
        let bits = unsafe { read_bits(buf, ans_pos, entry.ans_bits) };
        ans_pos += usize::from(entry.ans_bits);
        #[allow(clippy::cast_possible_truncation)]
        {
            states[lane] = usize::from(entry.next_base) + bits as usize;
        }
    }
}

/// [`ans_kernel`] compiled with BMI2, whose variable shifts and masks are single instructions.
///
/// # Safety
///
/// As [`ans_kernel`], and the CPU supports BMI2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,bmi2")]
unsafe fn ans_bmi2<L: Latent, const OFFSETS: bool>(table: &[Entry<L>], args: KernelArgs<'_, L>) {
    // SAFETY: forwarded from the caller.
    unsafe { ans_kernel::<L, OFFSETS, false>(table, args) }
}

/// [`single_bin`] compiled with BMI2.
///
/// # Safety
///
/// As [`single_bin`], and the CPU supports BMI2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,bmi2")]
unsafe fn single_bin_bmi2<L: Latent>(bin: &Bin<L>, args: KernelArgs<'_, L>) {
    // SAFETY: forwarded from the caller.
    unsafe { single_bin::<L, false>(bin, args) }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn lcg(seed: u64) -> impl Iterator<Item = u64> {
        let mut x = seed;
        std::iter::repeat_with(move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x >> 11
        })
    }

    fn roundtrip<L: Latent>(latents: &[L]) -> VortexResult<Stream<L>> {
        let stream = compress_latents(latents, &Config::default())?;
        let dec = StreamDecoder::new(&stream)?;
        assert_eq!(dec.decode_all(&stream), latents);
        let mut buf = [L::ZERO; BLOCK_SIZE];
        for i in (0..latents.len()).step_by(97) {
            dec.decode_block_prefix(&stream, i / BLOCK_SIZE, i % BLOCK_SIZE + 1, &mut buf);
            assert_eq!(buf[i % BLOCK_SIZE], latents[i], "index {i}");
        }
        Ok(stream)
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(5)]
    #[case(1024)]
    #[case(1025)]
    #[case(10_000)]
    fn various_lengths(#[case] n: usize) -> VortexResult<()> {
        let latents: Vec<u64> = lcg(1).take(n).map(|x| x % 1000 + (1 << 63)).collect();
        roundtrip(&latents)?;
        Ok(())
    }

    #[test]
    fn skewed_distribution_compresses() -> VortexResult<()> {
        let latents: Vec<u32> = lcg(2)
            .take(50_000)
            .map(|x| if x % 10 < 8 { (x % 4) as u32 } else { (x % 100_000) as u32 })
            .collect();
        let stream = roundtrip(&latents)?;
        assert!(stream.nbytes() < latents.len() * 4 / 3, "{}", stream.nbytes());
        Ok(())
    }

    #[test]
    fn regular_timestamps_are_nearly_free() -> VortexResult<()> {
        let latents: Vec<u64> = (0..100_000u64).map(|i| 1_700_000_000_000 + i * 60_000).collect();
        let stream = roundtrip(&latents)?;
        assert_eq!(stream.delta_order, 1);
        assert!(stream.nbytes() < 200, "{}", stream.nbytes());
        Ok(())
    }

    #[test]
    fn wide_offsets() -> VortexResult<()> {
        let latents: Vec<u64> = lcg(3).take(20_000).map(|x| x.rotate_left(13)).collect();
        roundtrip(&latents)?;
        Ok(())
    }

    #[test]
    fn constant_is_tiny() -> VortexResult<()> {
        let stream = roundtrip(&vec![42u16; 5000])?;
        assert!(stream.nbytes() < 120, "{}", stream.nbytes());
        Ok(())
    }

    #[test]
    fn small_latents() -> VortexResult<()> {
        let latents: Vec<u8> = lcg(5).take(3000).map(|x| (x % 7) as u8).collect();
        roundtrip(&latents)?;
        Ok(())
    }
}
