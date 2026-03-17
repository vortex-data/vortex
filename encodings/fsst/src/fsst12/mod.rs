// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST-12: An extended FSST variant using 12-bit codes (up to 4095 symbols).
//!
//! Standard FSST uses 8-bit codes (255 symbols + 1 escape code = 256 codes total),
//! where symbols are up to 8 bytes. FSST-12 uses 12-bit codes stored as u16 values,
//! with codes 0-255 reserved for single-byte literals and codes 256-4094 available
//! for multi-byte symbols (up to 3839 symbols).
//!
//! Key design decisions:
//! - **No escapes needed**: Codes 0-255 directly represent each possible byte value,
//!   so every input byte always matches at least a 1-byte code.
//! - **More multi-byte symbols**: 3839 slots for multi-byte patterns vs FSST's ~255 total.
//! - **Trade-off**: Each code costs 2 bytes of output vs 1 byte for FSST, so symbols
//!   must average >2 bytes to achieve better compression than FSST. FSST-12 works best
//!   on data with many distinct long patterns that exhaust FSST's 255-symbol limit.

// FSST-12 is an experimental compression algorithm that intentionally uses low-level
// bit operations and pointer casts where truncation is controlled by design.
#![allow(
    clippy::cast_possible_truncation,
    clippy::collapsible_if,
    clippy::option_map_or_none
)]

#[cfg(test)]
mod tests;

/// Multi-byte symbol codes start at 256.
const SYMBOL_CODE_BASE: u16 = 256;

/// Maximum code value (exclusive). We use 12 bits = 4096 values.
/// Codes 0-255: raw bytes, codes 256-4094: multi-byte symbols, 4095: unused.
const MAX_CODE: u16 = 4095;

/// Maximum number of multi-byte symbols.
pub const MAX_MULTI_SYMBOLS: usize = (MAX_CODE - SYMBOL_CODE_BASE) as usize;

/// Maximum symbol length in bytes.
const MAX_SYMBOL_LEN: usize = 8;

/// A symbol in the FSST-12 table: up to 8 bytes stored as a u64.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol12 {
    value: u64,
    len: u8,
}

#[allow(clippy::len_without_is_empty)]
impl Symbol12 {
    /// Create a symbol from a byte slice (up to 8 bytes).
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert!(!bytes.is_empty() && bytes.len() <= MAX_SYMBOL_LEN);
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        Self {
            value: u64::from_le_bytes(buf),
            len: bytes.len() as u8,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Concatenate two symbols if the result fits within 8 bytes.
    pub fn concat(self, other: Self) -> Option<Self> {
        let new_len = self.len() + other.len();
        if new_len > MAX_SYMBOL_LEN {
            return None;
        }
        let value = self.value | (other.value << (8 * self.len()));
        Some(Self {
            value,
            len: new_len as u8,
        })
    }

    #[inline]
    fn matches_at(&self, word: u64, available: usize) -> bool {
        if self.len() > available {
            return false;
        }
        let mask = if self.len() == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * self.len())) - 1
        };
        (word & mask) == self.value
    }
}

/// FSST-12 Compressor.
///
/// Uses codes 0-255 for single bytes and codes 256+ for multi-byte symbols.
/// This means no escape codes are ever needed - every byte has a direct encoding.
#[derive(Clone)]
pub struct Compressor12 {
    /// Multi-byte symbol table: code N maps to symbols\[N - SYMBOL_CODE_BASE\].
    symbols: Vec<Symbol12>,

    /// Symbol lengths for fast lookup during decompression.
    /// Index is (code - SYMBOL_CODE_BASE).
    symbol_lengths: Vec<u8>,

    /// Symbol values for fast lookup during decompression.
    /// Index is (code - SYMBOL_CODE_BASE).
    symbol_values: Vec<u64>,

    /// Inverted index for 2-byte lookups: first_two_bytes -> code.
    /// Returns HASH_EMPTY if no 2-byte symbol exists.
    codes_two_byte: Vec<u16>,

    /// Hash table for 3+ byte symbols.
    hash_table: Vec<HashEntry>,
}

const HASH_TABLE_SIZE: usize = 16384;
const HASH_EMPTY: u16 = 0;

#[derive(Copy, Clone, Default)]
struct HashEntry {
    value: u64,
    len_and_code: u32, // low 16 bits = code, high 16 bits = len
}

impl HashEntry {
    #[inline]
    fn new(value: u64, len: u8, code: u16) -> Self {
        Self {
            value,
            len_and_code: (code as u32) | ((len as u32) << 16),
        }
    }

    #[inline]
    fn code(&self) -> u16 {
        #[allow(clippy::cast_possible_truncation)]
        let result = self.len_and_code as u16;
        result
    }

    #[inline]
    fn len(&self) -> usize {
        (self.len_and_code >> 16) as usize
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len_and_code == 0
    }
}

impl Compressor12 {
    /// Train a compressor from a corpus of byte strings.
    pub fn train(samples: &[&[u8]]) -> Self {
        if samples.is_empty() {
            return Self::empty();
        }

        let sample = Self::make_sample(samples);
        let sample_refs: Vec<&[u8]> = sample.iter().map(|v| v.as_slice()).collect();

        let mut builder = TableBuilder::new();

        let generations = [8usize, 38, 68, 98, 128];
        for sample_frac in generations {
            let counts = builder.count_frequencies(&sample_refs, sample_frac);
            builder.optimize(&counts, sample_frac);
        }

        builder.build()
    }

    fn empty() -> Self {
        Self {
            symbols: Vec::new(),
            symbol_lengths: Vec::new(),
            symbol_values: Vec::new(),
            codes_two_byte: vec![HASH_EMPTY; 65536],
            hash_table: vec![HashEntry::default(); HASH_TABLE_SIZE],
        }
    }

    /// Rebuild a compressor from an existing multi-byte symbol table.
    pub fn rebuild(symbols: &[Symbol12]) -> Self {
        assert!(symbols.len() <= MAX_MULTI_SYMBOLS);

        let mut codes_two_byte = vec![HASH_EMPTY; 65536];
        let mut hash_table = vec![HashEntry::default(); HASH_TABLE_SIZE];
        let mut symbol_lengths = Vec::with_capacity(symbols.len());
        let mut symbol_values = Vec::with_capacity(symbols.len());

        #[allow(clippy::cast_possible_truncation)]
        for (idx, sym) in symbols.iter().enumerate() {
            let code = SYMBOL_CODE_BASE + idx as u16;
            symbol_lengths.push(sym.len);
            symbol_values.push(sym.value);

            match sym.len() {
                2 => {
                    codes_two_byte[sym.value as u16 as usize] = code;
                }
                3..=8 => {
                    let h = hash_symbol(sym.value) as usize & (HASH_TABLE_SIZE - 1);
                    for i in 0..HASH_TABLE_SIZE {
                        let slot = (h + i) & (HASH_TABLE_SIZE - 1);
                        if hash_table[slot].is_empty() {
                            hash_table[slot] = HashEntry::new(sym.value, sym.len, code);
                            break;
                        }
                    }
                }
                _ => {} // 1-byte symbols are handled by byte codes directly
            }
        }

        Self {
            symbols: symbols.to_vec(),
            symbol_lengths,
            symbol_values,
            codes_two_byte,
            hash_table,
        }
    }

    /// Compress a single byte string.
    ///
    /// Output format: sequence of u16 codes in little-endian.
    /// Codes 0-255 represent literal bytes, codes 256+ represent multi-byte symbols.
    pub fn compress(&self, input: &[u8]) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(input.len());
        let mut pos = 0;

        while pos < input.len() {
            let remaining = input.len() - pos;
            let word = load_word(&input[pos..]);

            // Try hash table (3+ bytes) for longest match
            if remaining >= 3 {
                if let Some((code, len)) = self.lookup_hash(word, remaining) {
                    output.extend_from_slice(&code.to_le_bytes());
                    pos += len;
                    continue;
                }
            }

            // Try 2-byte match
            #[allow(clippy::cast_possible_truncation)]
            if remaining >= 2 {
                let code = self.codes_two_byte[word as u16 as usize];
                if code != HASH_EMPTY {
                    output.extend_from_slice(&code.to_le_bytes());
                    pos += 2;
                    continue;
                }
            }

            // Fallback: emit raw byte as code 0-255
            let byte_code = input[pos] as u16;
            output.extend_from_slice(&byte_code.to_le_bytes());
            pos += 1;
        }

        output
    }

    /// Get the multi-byte symbol table.
    pub fn symbols(&self) -> &[Symbol12] {
        &self.symbols
    }

    /// Create a decompressor from this compressor's symbol table.
    pub fn decompressor(&self) -> Decompressor12 {
        Decompressor12 {
            symbol_lengths: self.symbol_lengths.clone(),
            symbol_values: self.symbol_values.clone(),
        }
    }

    #[inline]
    fn lookup_hash(&self, word: u64, available: usize) -> Option<(u16, usize)> {
        let h = hash_symbol(word) as usize & (HASH_TABLE_SIZE - 1);
        let mut best: Option<(u16, usize)> = None;

        for i in 0..8 {
            let idx = (h + i) & (HASH_TABLE_SIZE - 1);
            let entry = &self.hash_table[idx];
            if entry.is_empty() {
                break;
            }
            let entry_len = entry.len();
            if entry_len <= available {
                let mask = if entry_len == 8 {
                    u64::MAX
                } else {
                    (1u64 << (8 * entry_len)) - 1
                };
                if (word & mask) == entry.value
                    && best.is_none_or(|(_, best_len)| entry_len > best_len)
                {
                    best = Some((entry.code(), entry_len));
                }
            }
        }

        best
    }

    fn make_sample(inputs: &[&[u8]]) -> Vec<Vec<u8>> {
        let total_size: usize = inputs.iter().map(|s| s.len()).sum();
        if total_size <= 16384 {
            return inputs.iter().map(|s| s.to_vec()).collect();
        }

        let mut rng = hash_rng(4637947);
        let mut sample = Vec::new();
        let mut sample_size = 0;

        while sample_size < 16384 {
            let idx = (rng as usize) % inputs.len();
            rng = hash_rng(rng);
            let line = inputs[idx];
            if line.is_empty() {
                continue;
            }
            let chunk_start = (rng as usize) % line.len();
            rng = hash_rng(rng);
            let chunk_len = 512.min(line.len() - chunk_start);
            sample.push(line[chunk_start..chunk_start + chunk_len].to_vec());
            sample_size += chunk_len;
        }

        sample
    }
}

/// FSST-12 Decompressor.
#[derive(Clone)]
pub struct Decompressor12 {
    symbol_lengths: Vec<u8>,
    symbol_values: Vec<u64>,
}

impl Decompressor12 {
    /// Decompress a byte stream produced by [`Compressor12::compress`].
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(compressed.len() * 4);
        let mut pos = 0;

        while pos + 1 < compressed.len() {
            let code = u16::from_le_bytes([compressed[pos], compressed[pos + 1]]);
            pos += 2;

            if code < SYMBOL_CODE_BASE {
                // Raw byte
                #[allow(clippy::cast_possible_truncation)]
                output.push(code as u8);
            } else {
                // Multi-byte symbol
                let idx = (code - SYMBOL_CODE_BASE) as usize;
                let len = self.symbol_lengths[idx] as usize;
                let val = self.symbol_values[idx];
                let bytes = val.to_le_bytes();
                output.extend_from_slice(&bytes[..len]);
            }
        }

        output
    }

    /// Decompress into a pre-allocated buffer, returning bytes written.
    pub fn decompress_into(&self, compressed: &[u8], output: &mut [u8]) -> usize {
        let mut out_pos = 0;
        let mut pos = 0;

        while pos + 1 < compressed.len() {
            let code = u16::from_le_bytes([compressed[pos], compressed[pos + 1]]);
            pos += 2;

            if code < SYMBOL_CODE_BASE {
                #[allow(clippy::cast_possible_truncation)]
                {
                    output[out_pos] = code as u8;
                }
                out_pos += 1;
            } else {
                let idx = (code - SYMBOL_CODE_BASE) as usize;
                let len = self.symbol_lengths[idx] as usize;
                let val = self.symbol_values[idx];

                // Write up to 8 bytes at once
                if out_pos + 8 <= output.len() {
                    // SAFETY: we check bounds above. Write 8 bytes and only advance by `len`.
                    unsafe {
                        let ptr = output.as_mut_ptr().add(out_pos);
                        (ptr as *mut u64).write_unaligned(val);
                    }
                    out_pos += len;
                } else {
                    let bytes = val.to_le_bytes();
                    output[out_pos..out_pos + len].copy_from_slice(&bytes[..len]);
                    out_pos += len;
                }
            }
        }

        out_pos
    }
}

// --- Training infrastructure ---

struct TableBuilder {
    symbols: Vec<Symbol12>,
    // Fast lookup during training
    codes_two_byte: Vec<u16>,
    hash_table: Vec<HashEntry>,
}

/// For training, we track extended codes: 0-255 = raw byte, 256+ = symbol index.
const TRAIN_CODE_BASE: usize = 256;
/// Limit the number of distinct codes we track for pairs (keep memory manageable).
const TRAIN_CODE_RANGE: usize = 512; // 256 bytes + up to 256 top symbols

struct FrequencyCounts {
    counts1: Vec<usize>,
    counts2: Vec<usize>,
    observed1: Vec<bool>,
    observed2: Vec<bool>,
}

impl FrequencyCounts {
    fn new() -> Self {
        Self {
            counts1: vec![0; TRAIN_CODE_RANGE],
            counts2: vec![0; TRAIN_CODE_RANGE * TRAIN_CODE_RANGE],
            observed1: vec![false; TRAIN_CODE_RANGE],
            observed2: vec![false; TRAIN_CODE_RANGE * TRAIN_CODE_RANGE],
        }
    }

    #[inline]
    fn record1(&mut self, code: usize) {
        if code < TRAIN_CODE_RANGE {
            self.counts1[code] += 1;
            self.observed1[code] = true;
        }
    }

    #[inline]
    fn record2(&mut self, code1: usize, code2: usize) {
        if code1 < TRAIN_CODE_RANGE && code2 < TRAIN_CODE_RANGE {
            let idx = code1 * TRAIN_CODE_RANGE + code2;
            self.counts2[idx] += 1;
            self.observed2[idx] = true;
        }
    }
}

impl TableBuilder {
    fn new() -> Self {
        Self {
            symbols: Vec::new(),
            codes_two_byte: vec![HASH_EMPTY; 65536],
            hash_table: vec![HashEntry::default(); HASH_TABLE_SIZE],
        }
    }

    fn rebuild_lookup(&mut self) {
        self.codes_two_byte.fill(HASH_EMPTY);
        self.hash_table.fill(HashEntry::default());

        #[allow(clippy::cast_possible_truncation)]
        for (idx, sym) in self.symbols.iter().enumerate() {
            let code = (TRAIN_CODE_BASE + idx) as u16;
            match sym.len() {
                2 => {
                    self.codes_two_byte[sym.value as u16 as usize] = code;
                }
                3..=8 => {
                    let h = hash_symbol(sym.value) as usize & (HASH_TABLE_SIZE - 1);
                    for i in 0..HASH_TABLE_SIZE {
                        let slot = (h + i) & (HASH_TABLE_SIZE - 1);
                        if self.hash_table[slot].is_empty() {
                            self.hash_table[slot] = HashEntry::new(sym.value, sym.len, code);
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    #[inline]
    fn find_longest_match(&self, word: u64, available: usize) -> (usize, usize) {
        // Returns (train_code, consumed_bytes)
        #[allow(clippy::cast_possible_truncation)]
        let mut best_code = (word as u8) as usize; // raw byte fallback
        let mut best_len = 1usize;

        // Check 2-byte table
        #[allow(clippy::cast_possible_truncation)]
        if available >= 2 {
            let code = self.codes_two_byte[word as u16 as usize];
            if code != HASH_EMPTY {
                best_code = code as usize;
                best_len = 2;
            }
        }

        // Check hash table for 3+ byte matches
        if available >= 3 {
            let h = hash_symbol(word) as usize & (HASH_TABLE_SIZE - 1);
            for i in 0..8 {
                let idx = (h + i) & (HASH_TABLE_SIZE - 1);
                let entry = &self.hash_table[idx];
                if entry.is_empty() {
                    break;
                }
                let entry_len = entry.len();
                if entry_len <= available && entry_len > best_len {
                    let mask = if entry_len == 8 {
                        u64::MAX
                    } else {
                        (1u64 << (8 * entry_len)) - 1
                    };
                    if (word & mask) == entry.value {
                        best_code = entry.code() as usize;
                        best_len = entry_len;
                    }
                }
            }
        }

        (best_code, best_len)
    }

    fn count_frequencies(&self, samples: &[&[u8]], sample_frac: usize) -> FrequencyCounts {
        let mut counts = FrequencyCounts::new();

        for (i, sample) in samples.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            if sample_frac < 128 && (hash_rng(i as u64) & 127) as usize > sample_frac {
                continue;
            }

            let mut pos = 0;
            let mut prev_code = TRAIN_CODE_RANGE; // sentinel

            while pos < sample.len() {
                let remaining = sample.len() - pos;
                let word = load_word(&sample[pos..]);
                let (code, len) = self.find_longest_match(word, remaining);

                counts.record1(code);
                if prev_code < TRAIN_CODE_RANGE {
                    counts.record2(prev_code, code);
                }

                #[allow(clippy::cast_possible_truncation)]
                if len > 1 {
                    let first_byte = (word as u8) as usize;
                    counts.record1(first_byte);
                    if prev_code < TRAIN_CODE_RANGE {
                        counts.record2(prev_code, first_byte);
                    }
                }

                prev_code = code;
                pos += len;
            }
        }

        counts
    }

    fn optimize(&mut self, counts: &FrequencyCounts, sample_frac: usize) {
        let mut candidates: Vec<(usize, Symbol12)> = Vec::new();

        for code in 0..TRAIN_CODE_RANGE {
            if !counts.observed1[code] {
                continue;
            }

            let count = counts.counts1[code];
            let threshold = 5 * sample_frac / 128;
            if count < threshold {
                continue;
            }

            let (symbol, sym_len) = if code < TRAIN_CODE_BASE {
                #[allow(clippy::cast_possible_truncation)]
                let byte = code as u8;
                (Symbol12::from_bytes(&[byte]), 1)
            } else {
                let idx = code - TRAIN_CODE_BASE;
                if idx >= self.symbols.len() {
                    continue;
                }
                (self.symbols[idx], self.symbols[idx].len())
            };

            // Gain calculation: multi-byte symbols replace multiple 2-byte codes with one 2-byte code
            let mut gain = count * sym_len;

            // Boost single-byte symbols to reduce escape counts
            if code < TRAIN_CODE_BASE {
                gain *= 4;
            }

            candidates.push((gain, symbol));

            // Try merging with following symbols (skip on last round or if symbol is max length)
            if sample_frac >= 128 || sym_len >= MAX_SYMBOL_LEN {
                continue;
            }

            for code2 in 0..TRAIN_CODE_RANGE {
                let idx2 = code * TRAIN_CODE_RANGE + code2;
                if !counts.observed2[idx2] {
                    continue;
                }

                let symbol2 = if code2 < TRAIN_CODE_BASE {
                    #[allow(clippy::cast_possible_truncation)]
                    let byte2 = code2 as u8;
                    Symbol12::from_bytes(&[byte2])
                } else {
                    let idx = code2 - TRAIN_CODE_BASE;
                    if idx >= self.symbols.len() {
                        continue;
                    }
                    self.symbols[idx]
                };

                if let Some(merged) = symbol.concat(symbol2) {
                    let pair_count = counts.counts2[idx2];
                    let pair_gain = pair_count * merged.len();
                    candidates.push((pair_gain, merged));
                }
            }
        }

        // Sort by gain descending
        candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        // Pick top multi-byte symbols (skip 1-byte since those are handled by byte codes)
        self.symbols.clear();
        let mut seen_values: vortex_utils::aliases::hash_set::HashSet<(u64, u8)> =
            vortex_utils::aliases::hash_set::HashSet::default();

        for (_, sym) in candidates {
            if self.symbols.len() >= MAX_MULTI_SYMBOLS.min(TRAIN_CODE_RANGE - TRAIN_CODE_BASE) {
                break;
            }
            if sym.len() < 2 {
                continue; // Skip 1-byte symbols - they're handled by byte codes 0-255
            }
            let key = (sym.value, sym.len);
            if seen_values.insert(key) {
                self.symbols.push(sym);
            }
        }

        self.rebuild_lookup();
    }

    fn build(self) -> Compressor12 {
        Compressor12::rebuild(&self.symbols)
    }
}

#[inline]
fn load_word(data: &[u8]) -> u64 {
    if data.len() >= 8 {
        // SAFETY: we've checked data.len() >= 8 above
        let bytes: [u8; 8] = unsafe { *(data.as_ptr() as *const [u8; 8]) };
        u64::from_le_bytes(bytes)
    } else {
        let mut buf = [0u8; 8];
        buf[..data.len()].copy_from_slice(data);
        u64::from_le_bytes(buf)
    }
}

#[inline]
fn hash_symbol(value: u64) -> u64 {
    value.wrapping_mul(2971215073) ^ value.wrapping_shr(15)
}

#[inline]
fn hash_rng(value: u64) -> u64 {
    value.wrapping_mul(2971215073) ^ value.wrapping_shr(15)
}
