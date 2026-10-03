// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Table-based asymmetric numeral systems (tANS) over bin indices.
//!
//! States live in `[T, 2T)` for a table size `T = 2^size_log`; we store them as indices in
//! `[0, T)`. Symbols are encoded in reverse so the decoder reads forward, with `N_LANES`
//! interleaved states: value `i` uses lane `i % N_LANES`.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

pub(crate) const N_LANES: usize = 4;
pub(crate) const MAX_SIZE_LOG: u8 = 12;

/// Spreads each symbol's weight across the table with a fixed odd stride, as Pco does, so the
/// states of one symbol are scattered rather than clustered.
pub(crate) fn spread(size_log: u8, weights: &[u16]) -> VortexResult<Vec<u16>> {
    let table_size = 1usize << size_log;
    let total: usize = weights.iter().map(|&w| usize::from(w)).sum();
    vortex_ensure!(
        total == table_size,
        "tANS weights sum to {total}, expected {table_size}"
    );
    vortex_ensure!(
        weights.iter().all(|&w| w > 0),
        "tANS weights must be positive"
    );
    let mut stride = (3 * table_size) / 5;
    if stride % 2 == 0 {
        stride += 1;
    }
    let mask = table_size - 1;
    let mut symbols = vec![0u16; table_size];
    let mut step = 0usize;
    for (symbol, &weight) in weights.iter().enumerate() {
        for _ in 0..weight {
            #[allow(clippy::cast_possible_truncation)]
            {
                symbols[(stride * step) & mask] = symbol as u16;
            }
            step += 1;
        }
    }
    Ok(symbols)
}

/// One decode-table position: the symbol it decodes to, how many bits to read next, and the base
/// state index those bits are added to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodeStep {
    pub symbol: u16,
    pub n_bits: u8,
    pub next_base: u16,
}

pub(crate) fn decode_steps(size_log: u8, weights: &[u16]) -> VortexResult<Vec<DecodeStep>> {
    let symbols = spread(size_log, weights)?;
    let table_size = 1u32 << size_log;
    let mut seen = vec![0u32; weights.len()];
    let mut steps = Vec::with_capacity(symbols.len());
    for &symbol in &symbols {
        let s = usize::from(symbol);
        let y = u32::from(weights[s]) + seen[s];
        seen[s] += 1;
        let n_bits = u32::from(size_log) - y.ilog2();
        #[allow(clippy::cast_possible_truncation)]
        steps.push(DecodeStep {
            symbol,
            n_bits: n_bits as u8,
            next_base: ((y << n_bits) - table_size) as u16,
        });
    }
    Ok(steps)
}

pub(crate) struct Encoder {
    size_log: u8,
    /// Per symbol: the state indices holding that symbol, in table order.
    states: Vec<Vec<u16>>,
    /// Per symbol: `(weight, floor(log2(weight)))`.
    weights: Vec<(u32, u32)>,
}

impl Encoder {
    pub(crate) fn new(size_log: u8, weights: &[u16]) -> VortexResult<Self> {
        let symbols = spread(size_log, weights)?;
        let mut states = vec![Vec::new(); weights.len()];
        for (idx, &symbol) in symbols.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            states[usize::from(symbol)].push(idx as u16);
        }
        let weights = weights
            .iter()
            .map(|&w| (u32::from(w), u32::from(w).ilog2()))
            .collect();
        Ok(Self {
            size_log,
            states,
            weights,
        })
    }

    /// Encodes `symbol` from state index `state`, returning the new state index and the
    /// `(bits, n_bits)` chunk the decoder will read when it decodes this symbol.
    #[inline]
    pub(crate) fn encode(&self, state: u16, symbol: u16) -> (u16, u64, u32) {
        let x = u32::from(state) + (1 << self.size_log);
        let (w, w_log) = self.weights[usize::from(symbol)];
        let base_bits = u32::from(self.size_log) - w_log;
        let n_bits = base_bits - u32::from(x < (w << base_bits));
        let y = x >> n_bits;
        let bits = u64::from(x & ((1 << n_bits) - 1));
        let new_state = self.states[usize::from(symbol)][(y - w) as usize];
        (new_state, bits, n_bits)
    }
}

/// Scales `counts` to positive integer weights summing to `2^size_log`.
pub(crate) fn quantize_weights(counts: &[u64], size_log: u8) -> VortexResult<Vec<u16>> {
    let table_size = 1u64 << size_log;
    let n_symbols = counts.len() as u64;
    if n_symbols > table_size {
        vortex_bail!("{n_symbols} symbols do not fit a tANS table of size {table_size}");
    }
    let total: u64 = counts.iter().sum();
    let scale = table_size as f64 / total as f64;
    let mut weights: Vec<u64> = counts
        .iter()
        .map(|&c| ((c as f64 * scale).floor() as u64).max(1))
        .collect();
    let mut sum: u64 = weights.iter().sum();

    // Hand out the remainder by largest fractional shortfall, or take back the excess (caused by
    // the minimum weight of 1) from the symbols that lose the least per unit taken.
    if sum < table_size {
        let mut order: Vec<usize> = (0..counts.len()).collect();
        order.sort_by(|&a, &b| {
            let fa = counts[a] as f64 * scale - weights[a] as f64;
            let fb = counts[b] as f64 * scale - weights[b] as f64;
            fb.total_cmp(&fa)
        });
        let mut i = 0;
        while sum < table_size {
            weights[order[i % order.len()]] += 1;
            sum += 1;
            i += 1;
        }
    }
    while sum > table_size {
        let (idx, _) = weights
            .iter()
            .enumerate()
            .filter(|&(_, &w)| w > 1)
            .map(|(i, &w)| (i, counts[i] as f64 * ((w as f64).ln() - ((w - 1) as f64).ln())))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| vortex_error::vortex_err!("cannot shrink tANS weights"))?;
        weights[idx] -= 1;
        sum -= 1;
    }
    weights
        .into_iter()
        .map(|w| u16::try_from(w).map_err(Into::into))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() -> VortexResult<()> {
        let weights = quantize_weights(&[500, 300, 150, 40, 9, 1], 8)?;
        assert_eq!(weights.iter().map(|&w| u32::from(w)).sum::<u32>(), 256);
        let enc = Encoder::new(8, &weights)?;
        let steps = decode_steps(8, &weights)?;
        let symbols: Vec<u16> = (0..997u32).map(|i| ((i * 7 + i / 3) % 6) as u16).collect();

        let mut states = [0u16; N_LANES];
        let mut chunks = vec![(0u64, 0u32); symbols.len()];
        for i in (0..symbols.len()).rev() {
            let lane = i % N_LANES;
            let (s, bits, n) = enc.encode(states[lane], symbols[i]);
            states[lane] = s;
            chunks[i] = (bits, n);
        }
        for (i, &sym) in symbols.iter().enumerate() {
            let lane = i % N_LANES;
            let step = steps[usize::from(states[lane])];
            assert_eq!(step.symbol, sym);
            assert_eq!(u32::from(step.n_bits), chunks[i].1);
            states[lane] = step.next_base + u16::try_from(chunks[i].0)?;
        }
        assert_eq!(states, [0; N_LANES]);
        Ok(())
    }
}
