// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String-compression backends compared by the benchmark suite.

pub mod fsst_rs_backend;
#[cfg(feature = "fsst-cpp")]
pub mod fsst_cpp_8;
#[cfg(feature = "fsst-cpp")]
pub mod fsst_cpp_12;
#[cfg(feature = "fsst-cpp")]
pub mod fsst_cpp_ffi;
#[cfg(feature = "onpair")]
pub mod onpair_backend;
#[cfg(feature = "onpair")]
pub mod onpair16_backend;
#[cfg(feature = "onpair-cpp")]
pub mod onpair_cpp_backend;
#[cfg(feature = "onpair-cpp")]
pub mod onpair_cpp_ffi;

/// One trained compressor over a slice of input strings. Owning. Each `Backend`
/// keeps the data needed to decompress (symbol table or dictionary) plus the
/// compressed byte streams. Pushdown-capable backends additionally implement
/// `Pushdown`.
pub trait Backend {
    /// Human-readable identifier, e.g. `"fsst-rs"`.
    fn name(&self) -> &'static str;

    /// Compressed bytes (codes only, excluding the symbol table). Used as the
    /// numerator of the compression ratio.
    fn compressed_payload_bytes(&self) -> usize;

    /// Total bytes the backend would have to spill if it serialised itself
    /// (codes + symbol/dictionary metadata + offsets). Lower bound on the
    /// "fair" compressed size when comparing across backends.
    fn total_compressed_bytes(&self) -> usize;

    /// Decompress every string back into a fresh `Vec<Vec<u8>>`. The harness
    /// uses this to verify round-trip correctness.
    fn decompress_all(&self) -> Vec<Vec<u8>>;
}

/// Optional pushdown capability: equality / substring / prefix predicates
/// evaluated directly on the compressed representation when the backend
/// supports it.
pub trait Pushdown: Backend {
    /// SQL `col = ?` — returns row indices matching `needle` exactly.
    fn equals(&self, needle: &[u8]) -> Vec<usize>;

    /// SQL `col LIKE '%needle%'` — returns row indices containing `needle`.
    fn contains(&self, needle: &[u8]) -> Vec<usize>;

    /// SQL `col LIKE 'prefix%'` — returns row indices starting with `prefix`.
    fn starts_with(&self, prefix: &[u8]) -> Vec<usize>;
}

/// Knobs forwarded into each backend's `train_and_compress`. Each backend
/// reads only the fields relevant to it.
#[derive(Copy, Clone, Debug)]
pub struct BackendConfig {
    /// `onpair` / `onpair16` merging threshold (`OnPair::new(threshold)`).
    pub onpair_threshold: u16,
    /// `onpair-cpp` code width: 2^bits dictionary slots, codes pack at
    /// `bits/8` bytes per token in the stream. 14 is the upstream default.
    pub onpair_cpp_bits: u8,
    /// `onpair-cpp` training RNG seed (controls dictionary shuffle order).
    pub onpair_cpp_seed: u32,
    /// `onpair-cpp` fixed merge threshold when > 0; 0 selects the upstream
    /// dynamic threshold that fills the dictionary to capacity.
    pub onpair_cpp_fixed_threshold: u32,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            onpair_threshold: 4,
            onpair_cpp_bits: 14,
            onpair_cpp_seed: 42,
            onpair_cpp_fixed_threshold: 0,
        }
    }
}
