// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex tANS (table-based Asymmetric Numeral Systems) entropy-coded
//! array — phase P5 of the layered pco stack.
//!
//! [`AnsArray`] compresses a `Primitive<u8>` stream of small-alphabet
//! symbols (typically the `bin_idx` child of a `BinPartitionArray`
//! from the `vortex-bin-partition` crate) into a single bit-packed
//! buffer plus a compact metadata record describing the tANS table.
//!
//! # Strategy
//!
//! `pco`'s `ans` module is private (`mod ans;`), so its `Encoder` and
//! `Decoder` types are not reachable from downstream crates. This
//! crate implements the same single-state algorithm directly. We do
//! not interleave four streams the way pco does for SIMD throughput —
//! that perf optimisation is deferred to P6. The compression ratio is
//! the same.
//!
//! See [`AnsArray`] for the public type and the design doc at
//! `encodings/pco/DESIGN.md` for the role of P5 in the layered stack.

pub use array::*;

mod array;
pub mod tans;

/// Prost-encoded metadata for an [`AnsArray`].
///
/// `alphabet[s]` gives the original `u8` value for dense symbol id `s`;
/// `weights[s]` is its weight in the tANS table; `final_state` and
/// `bit_len` reconstruct the decoder's initial state and the bit length
/// of the encoded stream.
#[derive(Clone, prost::Message)]
pub struct AnsMetadata {
    /// `2^ans_size_log` is the tANS table size. Range: `[4, 14]`.
    #[prost(uint32, tag = "1")]
    pub ans_size_log: u32,
    /// Number of symbols this array represents.
    #[prost(uint64, tag = "2")]
    pub n_symbols: u64,
    /// Distinct `u8` values seen at encode time, in first-occurrence
    /// order. The decoded symbol id `s` maps to `alphabet[s as usize]`.
    #[prost(uint32, repeated, tag = "3")]
    pub alphabet: Vec<u32>,
    /// Per-symbol weight (frequency-derived). Sum equals
    /// `1 << ans_size_log` when `alphabet.len() > 1`.
    #[prost(uint32, repeated, tag = "4")]
    pub weights: Vec<u32>,
    /// Final tANS state after encoding, in `[1 << ans_size_log,
    /// 1 << (ans_size_log + 1))`.
    #[prost(uint32, tag = "5")]
    pub final_state: u32,
    /// Bit length used by the encoded stream (`<= 8 * encoded.len()`).
    #[prost(uint64, tag = "6")]
    pub bit_len: u64,
}
