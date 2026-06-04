// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Greedy longest-prefix-match tokeniser for OnPair predicate kernels.
//!
//! OnPair's dictionary is stored in **lexicographic order** (per
//! `onpair_cpp/include/onpair/core/dictionary.h`). For any byte `b` the
//! dict ids whose first byte equals `b` form a contiguous range we can
//! find in O(1) via a 257-entry first-byte index. The tokeniser walks
//! `needle` left-to-right and at each position picks the *longest* dict
//! entry that's a prefix of `needle[pos..]` — exactly the same strategy
//! `EQSearch` / `PrefixAutomaton` use on the C++ side.
//!
//! Returns:
//! * `Some(Vec<u16>)` — the unique LPM token sequence for `needle`. Two
//!   strings with the same byte content compress to the same token
//!   sequence under the same dict, so token-sequence equality on the
//!   `codes` child is exactly equivalent to byte equality on the
//!   decoded rows. **No decoding required** in the predicate hot loop.
//! * `None` — `needle` contains a byte that's not the start of any dict
//!   entry (degenerate dict; OnPair training normally guarantees the
//!   256 single-byte entries exist). Callers should fall back to byte
//!   matching.

use vortex_error::vortex_panic;

use crate::decode::DecodeView;

/// Per-byte index into the dictionary: `range_for(b) = lo..hi` is the
/// half-open range of dict ids whose first byte equals `b`. Empty if
/// no such dict entry exists.
///
/// Stored as 257 `u32` so `range_for(b) = lo..hi` reads two adjacent
/// entries with no branch.
pub struct DictIndex {
    by_first_byte: [u32; 257],
}

impl DictIndex {
    pub fn build(dv: &DecodeView<'_>) -> Self {
        let mut by_first_byte = [0u32; 257];
        // OnPair training caps dict_size at 2^bits ≤ 65 536, well within u32.
        let dict_size: u32 = u32::try_from(dv.dict_table.len())
            .unwrap_or_else(|_| vortex_panic!("OnPair dict_size > u32::MAX"));
        // The dict is sorted lexicographically, so the first dict id
        // whose first byte is `b` is the lowest `i` with that property.
        // Fill `by_first_byte[0..=first]` with `i` lazily and tail-fill
        // with `dict_size`.
        let mut last_first: usize = 0;
        for (i, &entry) in dv.dict_table.iter().enumerate() {
            let off = (entry >> 16) as usize;
            let len = (entry & 0xffff) as usize;
            if len == 0 {
                continue; // defensive: OnPair dicts have len >= 1
            }
            let first = dv.dict_bytes[off] as usize;
            let i_u32 =
                u32::try_from(i).unwrap_or_else(|_| vortex_panic!("OnPair dict id > u32::MAX"));
            while last_first <= first {
                by_first_byte[last_first] = i_u32;
                last_first += 1;
            }
        }
        while last_first <= 256 {
            by_first_byte[last_first] = dict_size;
            last_first += 1;
        }
        Self { by_first_byte }
    }

    /// Range of dict ids whose first byte is `b`. Empty if none.
    #[inline]
    pub fn range_for(&self, b: u8) -> std::ops::Range<usize> {
        let lo = self.by_first_byte[b as usize] as usize;
        let hi = self.by_first_byte[b as usize + 1] as usize;
        lo..hi
    }
}

/// Tokenise `needle` via greedy longest-prefix-match against the
/// OnPair dict. Returns `None` if any byte of the needle has no
/// matching dict entry.
pub fn tokenize_needle(dv: &DecodeView<'_>, index: &DictIndex, needle: &[u8]) -> Option<Vec<u16>> {
    let mut tokens = Vec::with_capacity(needle.len());
    let mut pos = 0usize;
    while pos < needle.len() {
        let candidates = index.range_for(needle[pos]);
        if candidates.is_empty() {
            return None;
        }
        let remaining = &needle[pos..];
        let mut best_len: usize = 0;
        let mut best_id: u16 = 0;
        for id in candidates {
            // SAFETY: `id < dict_table.len()` (range from index).
            let entry = unsafe { *dv.dict_table.get_unchecked(id) };
            let off = (entry >> 16) as usize;
            let len = (entry & 0xffff) as usize;
            if len <= best_len || len > remaining.len() {
                continue;
            }
            // SAFETY: dict_bytes was validated; off + len ≤ dict_bytes.len().
            let entry_bytes = unsafe { dv.dict_bytes.get_unchecked(off..off + len) };
            if remaining.starts_with(entry_bytes) {
                best_len = len;
                // OnPair caps `bits ≤ 16`, so dict ids fit in u16.
                best_id = u16::try_from(id)
                    .unwrap_or_else(|_| vortex_panic!("OnPair dict id > u16::MAX"));
            }
        }
        if best_len == 0 {
            return None;
        }
        tokens.push(best_id);
        pos += best_len;
    }
    Some(tokens)
}

// `LIKE 'prefix%'` could *not* use a token-prefix shortcut: the LPM of
// the row's leading bytes may merge what would otherwise be two prefix
// tokens into a single longer token whose end extends past the literal
// prefix. The byte-streaming check in `compute/like.rs::row_starts_with`
// is the correct minimum-work option.

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;
    use crate::DEFAULT_DICT12_CONFIG;
    use crate::decode::OwnedDecodeInputs;
    use crate::onpair_compress;

    fn build_array(strings: &[&str]) -> OwnedDecodeInputs {
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_bytes())),
            DType::Utf8(Nullability::NonNullable),
        );
        let arr =
            onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        OwnedDecodeInputs::collect(arr.as_view(), &mut ctx).unwrap()
    }

    #[test]
    fn tokenise_round_trip() {
        let strings: Vec<String> = (0..200).map(|i| format!("row-{i:04}-tail")).collect();
        let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let inputs = build_array(&str_refs);
        let dv = inputs.view();
        let index = DictIndex::build(&dv);

        for s in &strings {
            let needle = s.as_bytes();
            let toks = tokenize_needle(&dv, &index, needle).expect("LPM must tokenise");
            // Round-trip: decode the token sequence back to bytes.
            let mut decoded = Vec::with_capacity(needle.len());
            for &t in &toks {
                let entry = dv.dict_table[t as usize];
                let off = (entry >> 16) as usize;
                let len = (entry & 0xffff) as usize;
                decoded.extend_from_slice(&dv.dict_bytes[off..off + len]);
            }
            assert_eq!(decoded, needle, "LPM didn't reconstruct {s:?}");
        }
    }

    #[test]
    fn tokenise_prefix_matches_row_prefix() {
        let strings: &[&str] = &[
            "https://example.com/items/0001",
            "https://example.com/items/0002",
            "https://example.com/users/abc",
            "ftp://other.example.com/x",
        ];
        let inputs = build_array(strings);
        let dv = inputs.view();
        let index = DictIndex::build(&dv);

        // Prefixes that should tokenise and match the right rows.
        let pfx = b"https://example.com/items/";
        let pfx_toks = tokenize_needle(&dv, &index, pfx).expect("prefix must tokenise");
        // For each row, check whether its codes start with pfx_toks.
        let codes_offsets = dv.codes_offsets;
        let codes = dv.codes;
        for (r, s) in strings.iter().enumerate() {
            let lo = codes_offsets[r] as usize;
            let hi = codes_offsets[r + 1] as usize;
            let row_toks = &codes[lo..hi];
            let token_match =
                row_toks.len() >= pfx_toks.len() && row_toks[..pfx_toks.len()] == pfx_toks[..];
            assert_eq!(
                token_match,
                s.as_bytes().starts_with(pfx),
                "row {r} ({s:?}) prefix mismatch"
            );
        }
    }
}
