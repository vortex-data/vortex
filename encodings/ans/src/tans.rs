// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Single-state table-based asymmetric numeral systems (tANS).
//!
//! This is a from-scratch implementation that follows pco's `ans` module
//! algorithmically (table layout, weight quantization, renormalization
//! cutoff) so that the produced byte stream is structurally compatible
//! with pco's tANS even though we do not re-emit pco's framing. We do
//! not depend on pco's `ans` module directly because it is `mod ans;`
//! (private) inside the pco crate and therefore not callable from
//! downstream code.
//!
//! Single-state means we do not interleave four streams as pco does for
//! SIMD throughput; that is a P6 concern. Compression ratio is the same.
//!
//! # Bit stream layout
//!
//! Encoding processes symbols in reverse and emits `renorm_bits` of the
//! current state into a backwards bit stream. The final state is stored
//! in the prost metadata. Decoding initializes the state from metadata,
//! reads bits from the **end** of the bit stream forward (LIFO), and
//! emits symbols in the original order.
//!
//! Bits are packed LSB-first within each byte; symbols are processed
//! such that the bit reader for decoding consumes from the high end of
//! the buffer.

#![expect(clippy::cast_possible_truncation)]

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// Minimum allowed `ans_size_log` (matches pco's range).
pub const MIN_ANS_SIZE_LOG: u8 = 4;
/// Maximum allowed `ans_size_log` (matches pco's `MAX_ANS_BITS`).
pub const MAX_ANS_SIZE_LOG: u8 = 14;

/// State width chosen so that `state ∈ [table_size, 2 * table_size)`.
/// We use `u32` to match pco's `AnsState`.
pub type AnsState = u32;

/// A symbol index into the per-symbol weights table.
pub type Symbol = u32;

/// Per-symbol weight (frequency mass in the tANS table).
pub type Weight = u32;

/// Build the `state_symbols` table by spreading each symbol across
/// `table_size` slots in a deterministic way matching pco's stride
/// scheme. The result is `Vec<Symbol>` of length `1 << size_log` where
/// `state_symbols[i]` is the symbol decoded from state index `i`.
pub(crate) fn spread_state_symbols(size_log: u8, weights: &[Weight]) -> VortexResult<Vec<Symbol>> {
    let table_size: Weight = weights.iter().copied().sum();
    if table_size != 1 << size_log {
        vortex_bail!(
            "tANS weight sum {} does not match table size {}",
            table_size,
            1u32 << size_log,
        );
    }

    // Stride: roughly 3/5 of the table size, always odd.
    let mut stride = (3 * table_size) / 5;
    if stride.is_multiple_of(2) {
        stride += 1;
    }
    // Mask equivalent to `% table_size`.
    let mask = (1u32 << size_log).wrapping_sub(1);

    let mut state_symbols = vec![0 as Symbol; table_size as usize];
    let mut step: u32 = 0;
    for (symbol, &weight) in weights.iter().enumerate() {
        for _ in 0..weight {
            let state_idx = (stride.wrapping_mul(step)) & mask;
            state_symbols[state_idx as usize] = symbol as Symbol;
            step = step.wrapping_add(1);
        }
    }
    Ok(state_symbols)
}

/// Compact precomputed encoder state for a single symbol.
struct SymbolInfo {
    renorm_bit_cutoff: AnsState,
    min_renorm_bits: u8,
    /// `next_states[x_s - weight]` for `x_s ∈ [weight, 2 * weight)`.
    next_states: Vec<AnsState>,
}

/// The tANS encoder. Construct via [`Encoder::new`] from a precomputed
/// weights vector whose sum equals `1 << size_log`.
pub struct Encoder {
    symbol_infos: Vec<SymbolInfo>,
    size_log: u8,
}

impl Encoder {
    /// Build the encoder from `(size_log, weights)`.
    pub fn new(size_log: u8, weights: &[Weight]) -> VortexResult<Self> {
        let state_symbols = spread_state_symbols(size_log, weights)?;
        let mut symbol_infos: Vec<SymbolInfo> = weights
            .iter()
            .map(|&weight| {
                let max_x_s = 2 * weight - 1;
                let min_renorm_bits = (size_log as u32 - max_x_s.ilog2()) as u8;
                let renorm_bit_cutoff: AnsState = 2 * weight * (1u32 << min_renorm_bits);
                SymbolInfo {
                    renorm_bit_cutoff,
                    min_renorm_bits,
                    next_states: Vec::with_capacity(weight as usize),
                }
            })
            .collect();
        let table_size = 1u32 << size_log;
        for (state_idx, &symbol) in state_symbols.iter().enumerate() {
            symbol_infos[symbol as usize]
                .next_states
                .push(table_size + state_idx as AnsState);
        }
        Ok(Self {
            symbol_infos,
            size_log,
        })
    }

    /// Returns the initial state used at the start of encoding. We pick
    /// the minimum of the valid range so the first symbol typically
    /// needs the fewest renormalization bits.
    pub fn default_state(&self) -> AnsState {
        1u32 << self.size_log
    }

    /// `ans_size_log`.
    pub fn size_log(&self) -> u8 {
        self.size_log
    }

    /// Encode one symbol given the current state. Returns
    /// `(new_state, bits_to_write)`: the caller must write the low
    /// `bits_to_write` bits of `state` to the (reverse) bit stream.
    #[inline]
    pub fn encode(&self, state: AnsState, symbol: Symbol) -> (AnsState, u8) {
        let info = &self.symbol_infos[symbol as usize];
        let renorm_bits = if state >= info.renorm_bit_cutoff {
            info.min_renorm_bits + 1
        } else {
            info.min_renorm_bits
        };
        let x_s = state >> renorm_bits;
        let weight = info.next_states.len() as AnsState;
        let new_state = info.next_states[(x_s - weight) as usize];
        (new_state, renorm_bits)
    }
}

/// Decoder node: matches pco's layout.
#[derive(Clone)]
struct Node {
    next_state_idx_base: u32,
    bits_to_read: u8,
}

/// The tANS decoder.
pub struct Decoder {
    nodes: Vec<Node>,
}

impl Decoder {
    /// Build the decoder from `(size_log, weights)`.
    pub fn new(size_log: u8, weights: &[Weight]) -> VortexResult<Self> {
        let state_symbols = spread_state_symbols(size_log, weights)?;
        let table_size = 1u32 << size_log;
        let mut symbol_x_s = weights.to_vec();
        let mut nodes = Vec::with_capacity(state_symbols.len());
        // `nodes[state_idx]` knows where state `state_idx + table_size`
        // came from; the consumed-symbol lookup uses `state_symbols`.
        for &symbol in &state_symbols {
            let next_state_base = symbol_x_s[symbol as usize];
            let bits_to_read = next_state_base.leading_zeros() - table_size.leading_zeros();
            let next_state_base_shifted = next_state_base << bits_to_read;
            nodes.push(Node {
                next_state_idx_base: next_state_base_shifted - table_size,
                bits_to_read: bits_to_read as u8,
            });
            symbol_x_s[symbol as usize] += 1;
        }
        Ok(Self { nodes })
    }

    /// Look up the per-state symbol table size for the given decoder.
    pub fn state_symbols(size_log: u8, weights: &[Weight]) -> VortexResult<Vec<Symbol>> {
        spread_state_symbols(size_log, weights)
    }

    /// Number of state nodes (= table size).
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn bits_to_read_at(&self, state_idx: u32) -> u8 {
        self.nodes[state_idx as usize].bits_to_read
    }

    pub(crate) fn next_state_idx_base(&self, state_idx: u32) -> u32 {
        self.nodes[state_idx as usize].next_state_idx_base
    }
}

/// Compute `(size_log, weights)` from a per-symbol count vector,
/// targeting `max_size_log` and adjusting downward if all weights share
/// a common power-of-2 factor. Mirrors pco's `quantize_weights`.
pub fn quantize_weights(counts: &[u32], total_count: usize, max_size_log: u8) -> (u8, Vec<Weight>) {
    if counts.len() <= 1 {
        return (0, vec![1]);
    }
    let min_size_log = (u32::BITS - (counts.len() as u32 - 1).leading_zeros()) as u8;
    let mut size_log = max_size_log.max(min_size_log);

    let mut weights = quantize_weights_to(counts, total_count, size_log);

    let power_of_2 = weights
        .iter()
        .map(|w| w.trailing_zeros() as u8)
        .min()
        .unwrap_or(0);
    size_log -= power_of_2;
    for w in &mut weights {
        *w >>= power_of_2;
    }
    (size_log, weights)
}

#[expect(clippy::cast_precision_loss)]
fn quantize_weights_to(counts: &[u32], total_count: usize, size_log: u8) -> Vec<Weight> {
    if size_log == 0 {
        return vec![1];
    }
    let required_weight_sum: Weight = 1 << size_log;
    let total_count = total_count as f32;
    let multiplier = required_weight_sum as f32 / total_count;
    let desired_surplus_per_bin: Vec<f32> = counts
        .iter()
        .map(|&c| (c as f32 * multiplier - 1.0).max(0.0))
        .collect();
    let desired_surplus: f32 = desired_surplus_per_bin.iter().sum();
    let required_surplus = required_weight_sum - counts.len() as Weight;
    let surplus_mult = if desired_surplus == 0.0 {
        0.0
    } else {
        required_surplus as f32 / desired_surplus
    };
    let float_weights: Vec<f32> = desired_surplus_per_bin
        .iter()
        .map(|&s| 1.0 + s * surplus_mult)
        .collect();
    let mut weights: Vec<Weight> = float_weights.iter().map(|&w| w.round() as Weight).collect();
    let mut weight_sum: Weight = weights.iter().sum();
    let mut i = 0usize;
    while weight_sum > required_weight_sum {
        if weights[i] > 1 && (weights[i] as f32) > float_weights[i] {
            weights[i] -= 1;
            weight_sum -= 1;
        }
        i = (i + 1) % weights.len();
    }
    let mut i = 0usize;
    while weight_sum < required_weight_sum {
        if (weights[i] as f32) < float_weights[i] {
            weights[i] += 1;
            weight_sum += 1;
        }
        i = (i + 1) % weights.len();
    }
    weights
}

/// Reverse bit writer: appends bits into a `Vec<u8>` in LSB-first
/// order. The encoder emits symbols in reverse, so the resulting
/// buffer is read **forward** by the decoder using a forward-LSB
/// reader, but the symbols are emitted in their original order.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    bit_cursor: u64,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            bit_cursor: 0,
        }
    }

    /// Append `n` low bits of `value`. `n` must be `<= 32`.
    #[inline]
    pub fn write(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let mut value = if n == 32 {
            value
        } else {
            value & ((1u32 << n) - 1)
        };
        let mut remaining = n as u64;
        while remaining > 0 {
            let byte_idx = (self.bit_cursor >> 3) as usize;
            let bit_in_byte = (self.bit_cursor & 7) as u8;
            if byte_idx >= self.out.len() {
                self.out.push(0);
            }
            let space = 8 - bit_in_byte;
            let take = remaining.min(space as u64) as u8;
            let chunk = (value & ((1u32 << take) - 1)) as u8;
            self.out[byte_idx] |= chunk << bit_in_byte;
            value >>= take;
            self.bit_cursor += take as u64;
            remaining -= take as u64;
        }
    }

    pub fn finish(self) -> (Vec<u8>, u64) {
        (self.out, self.bit_cursor)
    }
}

/// Forward bit reader for the encoded stream. Reads `n` bits at a time
/// in LSB-first order.
pub(crate) struct BitReader<'a> {
    src: &'a [u8],
    bit_cursor: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self { src, bit_cursor: 0 }
    }

    #[inline]
    pub fn read(&mut self, n: u8) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        let mut out: u32 = 0;
        let mut shift: u32 = 0;
        let mut remaining = n as u64;
        while remaining > 0 {
            let byte_idx = (self.bit_cursor >> 3) as usize;
            let bit_in_byte = (self.bit_cursor & 7) as u8;
            let space = 8 - bit_in_byte;
            let take = remaining.min(space as u64) as u8;
            let mask = if take == 8 {
                0xFFu32
            } else {
                (1u32 << take) - 1
            };
            let chunk = ((self.src[byte_idx] >> bit_in_byte) as u32) & mask;
            out |= chunk << shift;
            shift += take as u32;
            self.bit_cursor += take as u64;
            remaining -= take as u64;
        }
        out
    }
}

/// Encode a `symbols` stream with tANS using the precomputed weights.
/// Returns `(compressed_bytes, total_bit_count, final_state)`.
///
/// The encoder iterates symbols **in reverse** (tANS is LIFO). On the
/// way out, the (state, bits) pairs are queued and then written into a
/// forward LSB-first bit stream in reverse order: that is, the **first
/// thing written** is the bits emitted by the **last symbol consumed**
/// in the reverse traversal, which corresponds to the **first symbol**
/// of the input. The decoder reads forward and emits symbols in input
/// order.
///
/// We validate that every symbol fits in the weights table.
pub fn encode_symbols(
    symbols: &[Symbol],
    encoder: &Encoder,
) -> VortexResult<(Vec<u8>, u64, AnsState)> {
    let n_symbols = encoder.symbol_infos.len() as Symbol;
    for &s in symbols {
        vortex_ensure!(
            s < n_symbols,
            "tANS symbol {s} out of range; alphabet size is {n_symbols}",
        );
    }
    let mut state = encoder.default_state();
    let mut emissions: Vec<(AnsState, u8)> = Vec::with_capacity(symbols.len());
    for &symbol in symbols.iter().rev() {
        let (new_state, bits) = encoder.encode(state, symbol);
        emissions.push((state, bits));
        state = new_state;
    }
    let final_state = state;
    let mut writer = BitWriter::new();
    for (val, bits) in emissions.into_iter().rev() {
        writer.write(val, bits);
    }
    let (bytes, total_bits) = writer.finish();
    Ok((bytes, total_bits, final_state))
}

/// Decode a tANS-compressed stream back to its symbol vector.
pub fn decode_symbols(
    encoded: &[u8],
    n_symbols: u64,
    final_state: AnsState,
    size_log: u8,
    weights: &[Weight],
) -> VortexResult<Vec<Symbol>> {
    let decoder = Decoder::new(size_log, weights)?;
    let state_symbols = spread_state_symbols(size_log, weights)?;
    let table_size = 1u32 << size_log;
    let mut reader = BitReader::new(encoded);
    let mut state_idx = final_state.checked_sub(table_size).ok_or_else(|| {
        vortex_error::vortex_err!(
            "tANS final_state {} below table_size {}",
            final_state,
            table_size,
        )
    })?;
    if n_symbols == 0 {
        return Ok(Vec::new());
    }
    let mut out: Vec<Symbol> = Vec::with_capacity(n_symbols as usize);
    for _ in 0..n_symbols {
        let symbol = state_symbols[state_idx as usize];
        out.push(symbol);
        let bits = decoder.bits_to_read_at(state_idx);
        let base = decoder.next_state_idx_base(state_idx);
        let extra = reader.read(bits);
        state_idx = base + extra;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts_from(symbols: &[Symbol], alphabet: u32) -> Vec<u32> {
        let mut out = vec![0u32; alphabet as usize];
        for &s in symbols {
            out[s as usize] += 1;
        }
        out
    }

    #[test]
    fn round_trip_small_uniform() -> VortexResult<()> {
        let symbols: Vec<Symbol> = (0..32).map(|i| (i % 4) as Symbol).collect();
        let counts = counts_from(&symbols, 4);
        let (size_log, weights) = quantize_weights(&counts, symbols.len(), 8);
        let encoder = Encoder::new(size_log, &weights)?;
        let (bytes, _bits, final_state) = encode_symbols(&symbols, &encoder)?;
        let decoded = decode_symbols(
            &bytes,
            symbols.len() as u64,
            final_state,
            size_log,
            &weights,
        )?;
        assert_eq!(decoded, symbols);
        Ok(())
    }

    #[test]
    fn round_trip_skewed() -> VortexResult<()> {
        // Mostly symbol 0, occasional 1, 2.
        let mut symbols: Vec<Symbol> = Vec::new();
        for _ in 0..100 {
            symbols.extend([0, 0, 0, 0, 0, 0, 0, 1]);
        }
        let counts = counts_from(&symbols, 2);
        let (size_log, weights) = quantize_weights(&counts, symbols.len(), 4);
        let encoder = Encoder::new(size_log, &weights)?;
        let (bytes, _bits, final_state) = encode_symbols(&symbols, &encoder)?;
        let decoded = decode_symbols(
            &bytes,
            symbols.len() as u64,
            final_state,
            size_log,
            &weights,
        )?;
        assert_eq!(decoded, symbols);
        // Compression should beat 1 byte per symbol.
        assert!(
            bytes.len() < symbols.len(),
            "no compression: {} bytes for {} symbols",
            bytes.len(),
            symbols.len()
        );
        Ok(())
    }

    #[test]
    fn quantize_keeps_sum_invariant() {
        let counts = vec![777u32, 1u32];
        let (size_log, weights) = quantize_weights(&counts, 778, 4);
        assert_eq!(weights.iter().sum::<u32>(), 1u32 << size_log);
    }
}
