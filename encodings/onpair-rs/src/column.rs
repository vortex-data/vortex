// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Owning compressed column. API-compatible with the subset of
// `vortex-onpair-sys::Column` that `vortex-onpair` actually consumes:
// `compress`, `len`, `bits`, `dict_size`, `parts`. The shim accepts
// `&[u64]` row offsets so callers don't need to truncate to u32; internally
// we sanity-check and downcast.

use crate::config::{Error, OnPairTrainingConfig};
use crate::dict::Dictionary;
use crate::parser::parse;
use crate::store::Store;
use crate::trainer::{TrainResult, train};
use crate::types::is_valid_bits;

/// Owning compressed column. Built by [`Column::compress`].
#[derive(Debug, Clone)]
pub struct Column {
    dict: Dictionary,
    store: Store,
    num_rows: usize,
}

/// Borrowed raw arrays of a column. Mirrors `vortex-onpair-sys::Parts`.
#[derive(Copy, Clone)]
pub struct Parts<'a> {
    /// Concatenated dictionary entry bytes. The C++ shim's caller pads this
    /// with `MAX_TOKEN_SIZE` zeros before handing to the decoder; we expose
    /// the unpadded logical slice (length `dict_offsets.last()`).
    pub dict_bytes: &'a [u8],
    /// Length `dict_size + 1`.
    pub dict_offsets: &'a [u32],
    /// LSB-first bit-packed token stream.
    pub codes_packed: &'a [u64],
    /// Length `num_rows + 1`.
    pub codes_boundaries: &'a [u32],
    /// Bits per token (9..=16).
    pub bits: u32,
    pub num_rows: usize,
}

impl Column {
    /// Compress `n` byte strings described by a flat `bytes` blob and an
    /// `offsets` array of length `n + 1`. Matches
    /// `vortex-onpair-sys::Column::compress`.
    pub fn compress(
        bytes: &[u8],
        offsets: &[u64],
        config: OnPairTrainingConfig,
    ) -> Result<Self, Error> {
        if offsets.is_empty() {
            return Err(Error::InvalidArg);
        }
        if !is_valid_bits(config.bits as u8) {
            return Err(Error::InvalidArg);
        }
        let n = offsets.len() - 1;

        // Downcast u64 offsets to u32. Bail on overflow rather than wrap.
        let mut off32 = Vec::with_capacity(offsets.len());
        for &o in offsets {
            if o > u32::MAX as u64 {
                return Err(Error::InvalidArg);
            }
            off32.push(o as u32);
        }
        if (off32[n] as usize) > bytes.len() {
            return Err(Error::InvalidArg);
        }

        let cfg = config.into();
        let TrainResult { dict, lpm } = train(bytes, &off32, n, &cfg);
        let mut store = Store::default();
        parse(bytes, &off32, n, &lpm, config.bits as u8, &mut store);

        Ok(Self { dict, store, num_rows: n })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_rows
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    #[inline]
    pub fn bits(&self) -> u32 {
        self.store.bit_width as u32
    }

    #[inline]
    pub fn dict_size(&self) -> usize {
        self.dict.num_tokens()
    }

    /// Borrow the column's raw arrays for downstream consumers (decode loop,
    /// predicate kernels). Mirrors `vortex-onpair-sys::Column::parts`.
    pub fn parts(&self) -> Result<Parts<'_>, Error> {
        // dict_bytes: logical-size slice, not including decoder padding. The
        // C++ shim returns the same thing — `dict_bytes_len` is the byte
        // count from offsets.back(), and `vortex-onpair`'s compress.rs adds
        // MAX_TOKEN_SIZE of trailing zero padding itself.
        let true_dict_bytes = *self.dict.offsets.last().unwrap_or(&0) as usize;
        // Skip the trailing zero sentinel `BitWriter::flush` appended.
        let codes_packed = if self.store.packed.is_empty() {
            &self.store.packed[..]
        } else {
            &self.store.packed[..self.store.packed.len() - 1]
        };
        Ok(Parts {
            dict_bytes: &self.dict.bytes[..true_dict_bytes],
            dict_offsets: &self.dict.offsets,
            codes_packed,
            codes_boundaries: &self.store.boundaries,
            bits: self.store.bit_width as u32,
            num_rows: self.num_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_unpack::unpack_codes_to_u16;
    use crate::config::DEFAULT_DICT12_CONFIG;
    use crate::types::Token;

    fn pack_inputs(strings: &[&str]) -> (Vec<u8>, Vec<u64>) {
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(strings.len() + 1);
        offsets.push(0u64);
        for s in strings {
            bytes.extend_from_slice(s.as_bytes());
            offsets.push(bytes.len() as u64);
        }
        (bytes, offsets)
    }

    fn decode_row(parts: &Parts<'_>, dict_bytes: &[u8], row: usize) -> Vec<u8> {
        let begin = parts.codes_boundaries[row] as usize;
        let end = parts.codes_boundaries[row + 1] as usize;
        let codes = unpack_codes_to_u16(parts.codes_packed, end, parts.bits);
        let mut out = Vec::new();
        for &c in &codes[begin..end] {
            let s = parts.dict_offsets[c as usize] as usize;
            let e = parts.dict_offsets[c as usize + 1] as usize;
            out.extend_from_slice(&dict_bytes[s..e]);
        }
        out
    }

    #[test]
    fn empty_offsets_returns_invalid_arg() {
        let r = Column::compress(&[], &[], DEFAULT_DICT12_CONFIG);
        assert_eq!(r.err(), Some(Error::InvalidArg));
    }

    #[test]
    fn invalid_bits_returns_invalid_arg() {
        let cfg = OnPairTrainingConfig { bits: 8, threshold: 0.5, seed: 0 };
        let r = Column::compress(&[], &[0], cfg);
        assert_eq!(r.err(), Some(Error::InvalidArg));
    }

    #[test]
    fn zero_rows_compress_succeeds() {
        let col = Column::compress(&[], &[0], DEFAULT_DICT12_CONFIG).unwrap();
        assert_eq!(col.len(), 0);
        let parts = col.parts().unwrap();
        assert_eq!(parts.num_rows, 0);
        assert_eq!(parts.codes_boundaries, &[0u32]);
        assert!(parts.codes_packed.is_empty());
    }

    #[test]
    fn roundtrip_simple_strings() {
        let strings = ["user_000001", "user_000002", "admin_001", "user_000003", "guest_001"];
        let (bytes, offsets) = pack_inputs(&strings);
        let cfg = OnPairTrainingConfig { bits: 12, threshold: 0.5, seed: 7 };
        let col = Column::compress(&bytes, &offsets, cfg).unwrap();
        assert_eq!(col.len(), 5);
        assert_eq!(col.bits(), 12);
        let parts = col.parts().unwrap();
        let dict_bytes_padded = {
            let mut v = parts.dict_bytes.to_vec();
            v.extend(std::iter::repeat_n(0u8, crate::MAX_TOKEN_SIZE));
            v
        };
        for (i, &s) in strings.iter().enumerate() {
            let decoded = decode_row(&parts, &dict_bytes_padded, i);
            assert_eq!(decoded, s.as_bytes(), "row {i}");
        }
    }

    #[test]
    fn roundtrip_with_binary_data_and_all_bit_widths() {
        let strings: Vec<Vec<u8>> = (0..30u8)
            .map(|i| {
                let mut v = Vec::with_capacity(20);
                for j in 0..20u8 {
                    v.push(i.wrapping_add(j));
                }
                v
            })
            .collect();
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(strings.len() + 1);
        offsets.push(0u64);
        for s in &strings {
            bytes.extend_from_slice(s);
            offsets.push(bytes.len() as u64);
        }
        for bw in 9u32..=16 {
            let cfg = OnPairTrainingConfig { bits: bw, threshold: 0.5, seed: 99 };
            let col = Column::compress(&bytes, &offsets, cfg).unwrap();
            assert_eq!(col.bits(), bw);
            let parts = col.parts().unwrap();
            let dict_bytes_padded = {
                let mut v = parts.dict_bytes.to_vec();
                v.extend(std::iter::repeat_n(0u8, crate::MAX_TOKEN_SIZE));
                v
            };
            for (i, s) in strings.iter().enumerate() {
                let decoded = decode_row(&parts, &dict_bytes_padded, i);
                assert_eq!(decoded, *s, "bits={bw} row={i}");
            }
        }
    }

    #[test]
    fn dict_first_256_tokens_cover_all_bytes_after_sort() {
        let strings = ["hello world", "another row"];
        let (bytes, offsets) = pack_inputs(&strings);
        let col = Column::compress(&bytes, &offsets, DEFAULT_DICT12_CONFIG).unwrap();
        let parts = col.parts().unwrap();
        // Every byte value 0..=255 must appear as a single-byte token somewhere
        // in the dictionary.
        let mut found = [false; 256];
        for i in 0..parts.dict_offsets.len() - 1 {
            let s = parts.dict_offsets[i] as usize;
            let e = parts.dict_offsets[i + 1] as usize;
            if e - s == 1 {
                found[parts.dict_bytes[s] as usize] = true;
            }
        }
        for (i, &f) in found.iter().enumerate() {
            assert!(f, "byte {i} missing");
        }
    }

    #[test]
    fn parts_codes_packed_excludes_sentinel() {
        let strings = ["x", "y", "z"];
        let (bytes, offsets) = pack_inputs(&strings);
        let col = Column::compress(&bytes, &offsets, DEFAULT_DICT12_CONFIG).unwrap();
        let parts = col.parts().unwrap();
        // Number of token bits actually used.
        let total_tokens = *parts.codes_boundaries.last().unwrap() as usize;
        let needed_words = (total_tokens * parts.bits as usize).div_ceil(64);
        assert_eq!(parts.codes_packed.len(), needed_words);
        // unpack should still yield exactly total_tokens valid codes.
        let codes = unpack_codes_to_u16(parts.codes_packed, total_tokens, parts.bits);
        assert_eq!(codes.len(), total_tokens);
        // All codes must be valid dict indices.
        let dict_size = col.dict_size() as Token;
        for &c in &codes {
            assert!(c < dict_size);
        }
    }
}
