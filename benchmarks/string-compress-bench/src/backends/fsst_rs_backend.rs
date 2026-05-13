// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Backend wrapping the pure-Rust [`fsst-rs`](https://crates.io/crates/fsst-rs)
//! port. Equality pushdown is implemented by FSST-compressing the needle once
//! and doing a byte-equality test against the per-row compressed codes — this
//! works because FSST is a position-stable code (equal plaintexts → equal
//! codes).

use fsst::Compressor;

use super::{Backend, Pushdown};

pub struct FsstRsBackend {
    compressor: Compressor,
    /// Codes per row (no escapes table, just the raw output).
    codes: Vec<Vec<u8>>,
}

impl FsstRsBackend {
    pub fn train_and_compress(strings: &[Vec<u8>]) -> Self {
        let refs: Vec<&[u8]> = strings.iter().map(|s| s.as_slice()).collect();
        let compressor = Compressor::train(&refs);
        let codes: Vec<Vec<u8>> = refs.iter().map(|s| compressor.compress(s)).collect();
        Self { compressor, codes }
    }

    pub fn symbol_table_bytes(&self) -> usize {
        // 256 entries × (Symbol(u64) + length(u8))
        self.compressor.symbol_table().len() * (size_of::<u64>() + 1)
    }
}

impl Backend for FsstRsBackend {
    fn name(&self) -> &'static str {
        "fsst-rs"
    }

    fn compressed_payload_bytes(&self) -> usize {
        self.codes.iter().map(|c| c.len()).sum()
    }

    fn total_compressed_bytes(&self) -> usize {
        // Codes + symbol table + per-row offsets (i32).
        self.compressed_payload_bytes()
            + self.symbol_table_bytes()
            + self.codes.len() * size_of::<u32>()
    }

    fn decompress_all(&self) -> Vec<Vec<u8>> {
        let dec = self.compressor.decompressor();
        self.codes.iter().map(|c| dec.decompress(c)).collect()
    }
}

impl Pushdown for FsstRsBackend {
    fn equals(&self, needle: &[u8]) -> Vec<usize> {
        // FSST equality pushdown: compress the needle, compare against the
        // already-compressed per-row codes byte for byte. Two compressed
        // strings are equal iff their plaintexts are equal because FSST is a
        // deterministic prefix-free code.
        let target = self.compressor.compress(needle);
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (c.as_slice() == target.as_slice()).then_some(i))
            .collect()
    }

    fn contains(&self, needle: &[u8]) -> Vec<usize> {
        // FSST does not let you substring-search the compressed bytes
        // directly: a needle might straddle a code boundary. Decompress and
        // run a plain `memmem`. This matches the upstream `vortex-fsst` LIKE
        // fallback path for `%needle%` patterns that exceed the DFA limits.
        let dec = self.compressor.decompressor();
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let plain = dec.decompress(c);
                memmem(&plain, needle).is_some().then_some(i)
            })
            .collect()
    }

    fn starts_with(&self, prefix: &[u8]) -> Vec<usize> {
        let dec = self.compressor.decompressor();
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let plain = dec.decompress(c);
                (plain.len() >= prefix.len() && &plain[..prefix.len()] == prefix).then_some(i)
            })
            .collect()
    }
}

#[inline]
fn memmem(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
