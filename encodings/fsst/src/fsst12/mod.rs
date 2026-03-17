// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST-12: An FSST variant using 12-bit codes (up to 4096 symbols).
//!
//! Standard FSST uses 8-bit codes (255 symbols + 1 escape code = 256 codes total),
//! where symbols are up to 8 bytes. FSST-12 uses 12-bit codes packed at 1.5 bytes
//! per code, with codes 0-255 reserved for single-byte literals and codes 256-4095
//! available for multi-byte symbols (up to 3840 symbols).
//!
//! Key design decisions (per the reference cwida/fsst implementation):
//! - **No escapes needed**: Codes 0-255 directly represent each possible byte value,
//!   so every input byte always matches at least a 1-byte code.
//! - **12-bit packing**: Two codes are packed into 3 bytes (24 bits), not 2 bytes each.
//!   A trailing odd code uses 2 bytes. This gives 1.5 bytes/code average.
//! - **More multi-byte symbols**: 3840 slots for multi-byte patterns vs FSST's ~255.
//! - **Trade-off**: Each code costs 1.5 bytes vs 1 byte for FSST-8 (but FSST-8 escapes
//!   cost 2 bytes). FSST-12 wins when data has many distinct patterns that exhaust
//!   FSST-8's 255-symbol limit, or when escape rates are high.

// FSST-12 intentionally uses low-level bit operations and pointer casts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::collapsible_if,
    clippy::option_map_or_none
)]

#[cfg(test)]
mod tests;

/// Multi-byte symbol codes start at 256.
const SYMBOL_CODE_BASE: u16 = 256;

/// Maximum code value (12 bits = 4096 values).
/// Codes 0-255: raw bytes, codes 256-4095: multi-byte symbols.
const MAX_CODE: u16 = 4096;

/// Maximum number of multi-byte symbols.
pub const MAX_MULTI_SYMBOLS: usize = (MAX_CODE - SYMBOL_CODE_BASE) as usize;

/// Maximum symbol length in bytes.
const MAX_SYMBOL_LEN: usize = 8;

/// Hash prime from reference cwida/fsst implementation.
const FSST_HASH_PRIME: u64 = 2971215073;

/// Hash shift from reference cwida/fsst implementation.
const FSST_HASH_SHIFT: u32 = 15;

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

    /// Returns the byte-length of the symbol (1-8).
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns the symbol value as a little-endian u64.
    #[inline]
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns the symbol bytes as a slice.
    pub fn as_bytes(&self) -> Vec<u8> {
        let bytes = self.value.to_le_bytes();
        bytes[..self.len as usize].to_vec()
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
}

/// FSST-12 Compressor.
///
/// Uses codes 0-255 for single bytes and codes 256+ for multi-byte symbols.
/// Codes are packed in 12-bit format: 2 codes per 3 bytes, trailing odd code in 2 bytes.
#[derive(Clone)]
pub struct Compressor12 {
    /// Multi-byte symbol table: code N maps to `symbols[N - SYMBOL_CODE_BASE]`.
    symbols: Vec<Symbol12>,

    /// Symbol lengths for fast lookup during decompression.
    symbol_lengths: Vec<u8>,

    /// Symbol values for fast lookup during decompression.
    symbol_values: Vec<u64>,

    /// Direct lookup for 2-byte patterns: `first_two_bytes -> code`.
    /// Returns `HASH_EMPTY` if no 2-byte symbol exists.
    codes_two_byte: Vec<u16>,

    /// Hash table for 3+ byte symbols (open addressing, linear probing).
    hash_table: Vec<HashEntry>,
}

const HASH_TABLE_SIZE: usize = 1 << 14; // 16384, must be power of 2
const HASH_EMPTY: u16 = 0;

/// Maximum number of linear probe steps during hash lookup.
const MAX_PROBE_STEPS: usize = 16;

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
        self.len_and_code as u16
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
    ///
    /// Uses 5 training generations with progressively larger sample fractions
    /// (`sampleFrac = 8, 38, 68, 98, 128`) matching the reference cwida/fsst
    /// implementation. Each generation compresses the sample, counts symbol and
    /// bigram frequencies, then rebuilds the symbol table by concatenating
    /// high-gain adjacent symbols.
    pub fn train(samples: &[&[u8]]) -> Self {
        if samples.is_empty() {
            return Self::empty();
        }

        let sample = Self::make_sample(samples);
        let sample_refs: Vec<&[u8]> = sample.iter().map(|v| v.as_slice()).collect();

        let mut builder = TableBuilder::new();

        // 5 generations matching reference: sampleFrac starts at 8 and increments by 30.
        // The last generation (128) is the final round where no new concatenations are made.
        let mut sample_frac = 8usize;
        loop {
            let counts = builder.count_frequencies(&sample_refs, sample_frac);
            builder.optimize(&counts, sample_frac);
            if sample_frac >= 128 {
                break;
            }
            sample_frac += 30;
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
    ///
    /// This is used for deserialization: given a previously-trained set of symbols,
    /// reconstruct the lookup structures needed for compression.
    pub fn rebuild(symbols: &[Symbol12]) -> Self {
        assert!(symbols.len() <= MAX_MULTI_SYMBOLS);

        let mut codes_two_byte = vec![HASH_EMPTY; 65536];
        let mut hash_table = vec![HashEntry::default(); HASH_TABLE_SIZE];
        let mut symbol_lengths = Vec::with_capacity(symbols.len());
        let mut symbol_values = Vec::with_capacity(symbols.len());

        for (idx, sym) in symbols.iter().enumerate() {
            let code = SYMBOL_CODE_BASE + idx as u16;
            symbol_lengths.push(sym.len);
            symbol_values.push(sym.value);

            match sym.len() {
                2 => {
                    codes_two_byte[sym.value as u16 as usize] = code;
                }
                3..=8 => {
                    insert_hash(&mut hash_table, sym.value, sym.len, code);
                }
                _ => {} // 1-byte symbols handled by byte codes directly
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
    /// Output format: 12-bit packed codes. Two 12-bit codes are packed into 3 bytes
    /// (little-endian: `code1 | (code2 << 12)` written as 3 bytes). A trailing odd
    /// code is written as 2 bytes.
    pub fn compress(&self, input: &[u8]) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }

        // Phase 1: generate code sequence via greedy longest-match
        let mut codes = Vec::with_capacity(input.len());
        let mut pos = 0;

        while pos < input.len() {
            let remaining = input.len() - pos;
            let word = load_word(&input[pos..]);

            // Try hash table (3+ bytes) for longest match first
            if remaining >= 3 {
                if let Some((code, len)) = self.lookup_hash(word, remaining) {
                    codes.push(code);
                    pos += len;
                    continue;
                }
            }

            // Try 2-byte match
            if remaining >= 2 {
                let code = self.codes_two_byte[word as u16 as usize];
                if code != HASH_EMPTY {
                    codes.push(code);
                    pos += 2;
                    continue;
                }
            }

            // Fallback: emit raw byte as code 0-255
            codes.push(input[pos] as u16);
            pos += 1;
        }

        // Phase 2: pack codes into 12-bit format
        pack_12bit(&codes)
    }

    /// Get the multi-byte symbol table.
    pub fn symbols(&self) -> &[Symbol12] {
        &self.symbols
    }

    /// Returns the number of multi-byte symbols in the table.
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Create a decompressor from this compressor's symbol table.
    pub fn decompressor(&self) -> Decompressor12 {
        Decompressor12 {
            symbol_lengths: self.symbol_lengths.clone(),
            symbol_values: self.symbol_values.clone(),
        }
    }

    /// Serialize the symbol table to bytes for storage.
    ///
    /// Format: `[num_symbols: u16][for each symbol: value: u64, len: u8]`
    pub fn serialize_table(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.symbols.len() * 9);
        out.extend_from_slice(&(self.symbols.len() as u16).to_le_bytes());
        for sym in &self.symbols {
            out.extend_from_slice(&sym.value.to_le_bytes());
            out.push(sym.len);
        }
        out
    }

    /// Deserialize a symbol table from bytes and rebuild the compressor.
    ///
    /// Returns `None` if the data is malformed.
    pub fn deserialize_table(data: &[u8]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let num_symbols = u16::from_le_bytes([data[0], data[1]]) as usize;
        if data.len() < 2 + num_symbols * 9 {
            return None;
        }
        let mut symbols = Vec::with_capacity(num_symbols);
        let mut offset = 2;
        for _ in 0..num_symbols {
            let value = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            let len = data[offset + 8];
            if len == 0 || len > MAX_SYMBOL_LEN as u8 {
                return None;
            }
            symbols.push(Symbol12 { value, len });
            offset += 9;
        }
        Some(Self::rebuild(&symbols))
    }

    #[inline]
    fn lookup_hash(&self, word: u64, available: usize) -> Option<(u16, usize)> {
        let h = hash_symbol(word) as usize & (HASH_TABLE_SIZE - 1);
        let mut best: Option<(u16, usize)> = None;

        for i in 0..MAX_PROBE_STEPS {
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
        // FSST-12 has 16x more symbol slots than FSST-8, so we use a larger
        // sample to give the training algorithm enough data.
        let target_size = 1 << 16; // 64KB
        let total_size: usize = inputs.iter().map(|s| s.len()).sum();
        if total_size <= target_size {
            return inputs.iter().map(|s| s.to_vec()).collect();
        }

        let mut rng = hash_rng(4637947);
        let mut sample = Vec::new();
        let mut sample_size = 0;

        while sample_size < target_size {
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

/// Pack a sequence of 12-bit codes into bytes.
///
/// Two codes are packed into 3 bytes: `code1 | (code2 << 12)` as LE.
/// Trailing odd code uses 2 bytes.
fn pack_12bit(codes: &[u16]) -> Vec<u8> {
    let mut output = Vec::with_capacity(codes.len() * 3 / 2 + 2);
    let mut idx = 0;

    while idx + 1 < codes.len() {
        let lo = codes[idx] as u32;
        let hi = codes[idx + 1] as u32;
        let packed = lo | (hi << 12);
        output.push(packed as u8);
        output.push((packed >> 8) as u8);
        output.push((packed >> 16) as u8);
        idx += 2;
    }

    // Trailing odd code
    if idx < codes.len() {
        let code = codes[idx];
        output.push(code as u8);
        output.push((code >> 8) as u8);
    }

    output
}

/// Unpack 12-bit codes from a byte stream.
#[cfg(test)]
fn unpack_12bit(data: &[u8]) -> Vec<u16> {
    let mut codes = Vec::with_capacity(data.len() * 2 / 3 + 1);
    let mut pos = 0;

    // Process pairs (3 bytes -> 2 codes)
    while pos + 2 < data.len() {
        let b0 = data[pos] as u32;
        let b1 = data[pos + 1] as u32;
        let b2 = data[pos + 2] as u32;
        let packed = b0 | (b1 << 8) | (b2 << 16);

        codes.push((packed & 0xFFF) as u16);
        codes.push(((packed >> 12) & 0xFFF) as u16);
        pos += 3;
    }

    // Trailing odd code (2 bytes)
    if pos + 1 < data.len() {
        let code = u16::from_le_bytes([data[pos], data[pos + 1]]) & 0xFFF;
        codes.push(code);
    }

    codes
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
        let mut output = Vec::with_capacity(compressed.len() * 3);
        let mut pos = 0;

        // Process pairs (3 bytes -> 2 codes)
        while pos + 2 < compressed.len() {
            let b0 = compressed[pos] as u32;
            let b1 = compressed[pos + 1] as u32;
            let b2 = compressed[pos + 2] as u32;
            let packed = b0 | (b1 << 8) | (b2 << 16);

            let code1 = (packed & 0xFFF) as u16;
            let code2 = ((packed >> 12) & 0xFFF) as u16;

            self.emit_code(code1, &mut output);
            self.emit_code(code2, &mut output);
            pos += 3;
        }

        // Trailing odd code (2 bytes)
        if pos + 1 < compressed.len() {
            let code = u16::from_le_bytes([compressed[pos], compressed[pos + 1]]) & 0xFFF;
            self.emit_code(code, &mut output);
        }

        output
    }

    /// Decompress into a pre-allocated buffer, returning bytes written.
    pub fn decompress_into(&self, compressed: &[u8], output: &mut [u8]) -> usize {
        let mut out_pos = 0;
        let mut pos = 0;

        while pos + 2 < compressed.len() {
            let b0 = compressed[pos] as u32;
            let b1 = compressed[pos + 1] as u32;
            let b2 = compressed[pos + 2] as u32;
            let packed = b0 | (b1 << 8) | (b2 << 16);

            let code1 = (packed & 0xFFF) as u16;
            let code2 = ((packed >> 12) & 0xFFF) as u16;

            out_pos += self.emit_code_into(code1, output, out_pos);
            out_pos += self.emit_code_into(code2, output, out_pos);
            pos += 3;
        }

        if pos + 1 < compressed.len() {
            let code = u16::from_le_bytes([compressed[pos], compressed[pos + 1]]) & 0xFFF;
            out_pos += self.emit_code_into(code, output, out_pos);
        }

        out_pos
    }

    #[inline]
    fn emit_code(&self, code: u16, output: &mut Vec<u8>) {
        if code < SYMBOL_CODE_BASE {
            output.push(code as u8);
        } else {
            let idx = (code - SYMBOL_CODE_BASE) as usize;
            let len = self.symbol_lengths[idx] as usize;
            let val = self.symbol_values[idx];
            let bytes = val.to_le_bytes();
            output.extend_from_slice(&bytes[..len]);
        }
    }

    #[inline]
    fn emit_code_into(&self, code: u16, output: &mut [u8], out_pos: usize) -> usize {
        if code < SYMBOL_CODE_BASE {
            output[out_pos] = code as u8;
            1
        } else {
            let idx = (code - SYMBOL_CODE_BASE) as usize;
            let len = self.symbol_lengths[idx] as usize;
            let val = self.symbol_values[idx];
            let bytes = val.to_le_bytes();
            output[out_pos..out_pos + len].copy_from_slice(&bytes[..len]);
            len
        }
    }
}

// --- Training infrastructure ---

struct TableBuilder {
    symbols: Vec<Symbol12>,
    codes_two_byte: Vec<u16>,
    hash_table: Vec<HashEntry>,
}

/// Track up to 1024 codes during training (256 byte codes + 768 top symbols).
const TRAIN_CODE_BASE: usize = 256;
const TRAIN_CODE_RANGE: usize = 1024;

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

        for (idx, sym) in self.symbols.iter().enumerate() {
            let code = (TRAIN_CODE_BASE + idx) as u16;
            match sym.len() {
                2 => {
                    self.codes_two_byte[sym.value as u16 as usize] = code;
                }
                3..=8 => {
                    insert_hash(&mut self.hash_table, sym.value, sym.len, code);
                }
                _ => {}
            }
        }
    }

    #[inline]
    fn find_longest_match(&self, word: u64, available: usize) -> (usize, usize) {
        // Returns (train_code, consumed_bytes).
        // Lookup order: hash table (3+ bytes) -> shortCodes (2 bytes) -> byte fallback.
        let mut best_code = (word as u8) as usize; // raw byte fallback
        let mut best_len = 1usize;

        if available >= 2 {
            let code = self.codes_two_byte[word as u16 as usize];
            if code != HASH_EMPTY {
                best_code = code as usize;
                best_len = 2;
            }
        }

        if available >= 3 {
            let h = hash_symbol(word) as usize & (HASH_TABLE_SIZE - 1);
            for i in 0..MAX_PROBE_STEPS {
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
            // Sub-sample: skip some lines in early generations for speed.
            if sample_frac < 128 && (hash_rng(i as u64) & 127) as usize > sample_frac {
                continue;
            }

            let mut pos = 0;
            let mut prev_code = TRAIN_CODE_RANGE; // sentinel (no previous)

            while pos < sample.len() {
                let remaining = sample.len() - pos;
                let word = load_word(&sample[pos..]);
                let (code, len) = self.find_longest_match(word, remaining);

                counts.record1(code);
                if prev_code < TRAIN_CODE_RANGE {
                    counts.record2(prev_code, code);
                }

                // Also record the first-byte sub-code when a multi-byte symbol
                // matched. This helps bootstrap byte-level patterns in early
                // generations when the table is still sparse.
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
            // Minimum frequency threshold per the reference implementation.
            let threshold = 5 * sample_frac / 128;
            if count < threshold {
                continue;
            }

            let (symbol, sym_len) = if code < TRAIN_CODE_BASE {
                let byte = code as u8;
                (Symbol12::from_bytes(&[byte]), 1)
            } else {
                let idx = code - TRAIN_CODE_BASE;
                if idx >= self.symbols.len() {
                    continue;
                }
                (self.symbols[idx], self.symbols[idx].len())
            };

            // Gain = count * symbol_length (per reference FSST implementation).
            let gain = count * sym_len;
            if gain > 0 {
                candidates.push((gain, symbol));
            }

            // Concatenate with following symbols to form longer candidates.
            // Skip on the final generation (sample_frac >= 128) since those
            // candidates can't be evaluated in another round.
            if sample_frac >= 128 || sym_len >= MAX_SYMBOL_LEN {
                continue;
            }

            for code2 in 0..TRAIN_CODE_RANGE {
                let idx2 = code * TRAIN_CODE_RANGE + code2;
                if !counts.observed2[idx2] {
                    continue;
                }

                let symbol2 = if code2 < TRAIN_CODE_BASE {
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
                    let merged_gain = pair_count * merged.len();
                    candidates.push((merged_gain, merged));
                }
            }
        }

        // Sort by gain descending
        candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        // Pick top multi-byte symbols, deduplicating by (value, len).
        self.symbols.clear();
        let max_symbols = MAX_MULTI_SYMBOLS.min(TRAIN_CODE_RANGE - TRAIN_CODE_BASE);
        let mut seen_values: vortex_utils::aliases::hash_set::HashSet<(u64, u8)> =
            vortex_utils::aliases::hash_set::HashSet::default();

        for (_, sym) in candidates {
            if self.symbols.len() >= max_symbols {
                break;
            }
            if sym.len() < 2 {
                continue; // 1-byte symbols handled by byte codes 0-255
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

/// Insert a symbol into the hash table with linear probing.
fn insert_hash(hash_table: &mut [HashEntry], value: u64, len: u8, code: u16) {
    let h = hash_symbol(value) as usize & (HASH_TABLE_SIZE - 1);
    for i in 0..HASH_TABLE_SIZE {
        let slot = (h + i) & (HASH_TABLE_SIZE - 1);
        if hash_table[slot].is_empty() {
            hash_table[slot] = HashEntry::new(value, len, code);
            return;
        }
    }
}

/// Load up to 8 bytes from `data` as a little-endian u64, zero-padding if shorter.
#[inline]
fn load_word(data: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = data.len().min(8);
    buf[..n].copy_from_slice(&data[..n]);
    u64::from_le_bytes(buf)
}

/// Hash function matching the reference cwida/fsst implementation.
///
/// Uses the first 3 bytes (24 bits) of the symbol value, consistent with
/// `FSST_HASH(w) = (w * FSST_HASH_PRIME) ^ ((w * FSST_HASH_PRIME) >> FSST_SHIFT)`.
#[inline]
fn hash_symbol(value: u64) -> u64 {
    let w = value & 0xFFFFFF; // first 3 bytes
    let h = w.wrapping_mul(FSST_HASH_PRIME);
    h ^ h.wrapping_shr(FSST_HASH_SHIFT)
}

/// Simple deterministic PRNG for sampling.
#[inline]
fn hash_rng(value: u64) -> u64 {
    value
        .wrapping_mul(FSST_HASH_PRIME)
        .wrapping_add(1)
        .rotate_right(FSST_HASH_SHIFT)
}

#[cfg(test)]
mod packing_tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let codes: Vec<u16> = vec![0, 255, 256, 4095, 1000, 2000];
        let packed = pack_12bit(&codes);
        let unpacked = unpack_12bit(&packed);
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_odd() {
        let codes: Vec<u16> = vec![100, 200, 300];
        let packed = pack_12bit(&codes);
        // 2 codes = 3 bytes, 1 trailing = 2 bytes, total 5
        assert_eq!(packed.len(), 5);
        let unpacked = unpack_12bit(&packed);
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_size() {
        // Even number of codes: N*3/2 bytes
        let codes: Vec<u16> = vec![0; 10];
        let packed = pack_12bit(&codes);
        assert_eq!(packed.len(), 15); // 5 pairs * 3 bytes

        // Odd number: (N-1)*3/2 + 2
        let codes: Vec<u16> = vec![0; 11];
        let packed = pack_12bit(&codes);
        assert_eq!(packed.len(), 17); // 5 pairs * 3 + 2 trailing
    }
}
