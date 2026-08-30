// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Greedy tokenisation via a first-two-bytes bucket index.
//!
//! Instead of hashing (the descending short-map probe ladder) or a per-byte
//! automaton walk, index a 65,536-entry table by the first two bytes of the
//! input. Each bucket holds the dictionary tokens starting with those two
//! bytes, sorted by descending length, so an ordered scan of the few
//! candidates (typically a handful) finds the longest match with masked
//! suffix compares — no hash mixing, no per-byte table transitions, and no
//! automaton build. Single-byte tokens live in a direct 256-entry fallback.

use crate::bits::BitWriter;
use crate::dict::Dictionary;
use crate::store::Store;
use crate::types::BitWidth;
use crate::types::MAX_TOKEN_SIZE;
use crate::types::Token;

/// One bucket entry: the token bytes after the two-byte key packed
/// little-endian (up to 14 of them), the full token length, and the id.
#[derive(Copy, Clone)]
struct BucketEntry {
    rest: u128,
    len: u8,
    token: Token,
}

/// Greedy longest-prefix matcher indexed by the first two token bytes.
///
/// Dictionaries produced by OnPair contain all 256 single-byte tokens, so
/// matching is total at every byte position.
pub struct TwoByteLpm {
    /// `(start, count)` into `entries` per two-byte key.
    heads: Vec<(u32, u32)>,
    /// Bucket entries grouped by key, each group sorted by descending length.
    entries: Vec<BucketEntry>,
    /// Token id for each single byte.
    one_byte: [Token; 256],
    num_tokens: usize,
}

/// Pack `len` bytes at `data[pos..]` little-endian, zero-padded to 16.
#[inline]
fn load_window(data: &[u8], pos: usize, len: usize) -> u128 {
    debug_assert!(pos + len <= data.len());
    if pos + 16 <= data.len() {
        let full = u128::from_le_bytes(data[pos..pos + 16].try_into().expect("sixteen bytes"));
        if len >= 16 {
            full
        } else {
            full & ((1u128 << (len * 8)) - 1)
        }
    } else {
        let mut buf = [0u8; 16];
        buf[..len].copy_from_slice(&data[pos..pos + len]);
        u128::from_le_bytes(buf)
    }
}

impl TwoByteLpm {
    /// Build the bucket index for `dict`.
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        assert!(
            dict.num_tokens() <= usize::from(Token::MAX) + 1,
            "OnPair dictionary has more than 65,536 tokens"
        );

        let mut one_byte = [0 as Token; 256];
        let mut buckets: Vec<Vec<BucketEntry>> = vec![Vec::new(); 1 << 16];
        for id in 0..dict.num_tokens() {
            let token = dict.data(id as Token);
            assert!(
                !token.is_empty() && token.len() <= MAX_TOKEN_SIZE,
                "OnPair token length must be in 1..={MAX_TOKEN_SIZE}"
            );
            if token.len() == 1 {
                // Duplicate token bytes keep the newest id, matching the
                // other matchers' tie-break.
                one_byte[token[0] as usize] = id as Token;
                continue;
            }
            let key = u16::from_le_bytes([token[0], token[1]]) as usize;
            let rest = load_window(token, 2, token.len() - 2);
            let entries = &mut buckets[key];
            if let Some(existing) = entries
                .iter_mut()
                .find(|e| e.len as usize == token.len() && e.rest == rest)
            {
                existing.token = id as Token;
                continue;
            }
            entries.push(BucketEntry {
                rest,
                len: token.len() as u8,
                token: id as Token,
            });
            // Descending length keeps the first hit the longest.
            entries.sort_by(|a, b| b.len.cmp(&a.len));
        }

        let total: usize = buckets.iter().map(Vec::len).sum();
        let mut heads = Vec::with_capacity(1 << 16);
        let mut entries = Vec::with_capacity(total);
        for bucket in &buckets {
            heads.push((entries.len() as u32, bucket.len() as u32));
            entries.extend_from_slice(bucket);
        }

        Self {
            heads,
            entries,
            one_byte,
            num_tokens: dict.num_tokens(),
        }
    }

    /// Number of dictionary patterns in the matcher.
    pub fn size(&self) -> usize {
        self.num_tokens
    }

    /// Heap bytes used by the bucket index.
    pub fn table_bytes(&self) -> usize {
        self.heads.len() * size_of::<(u32, u32)>() + self.entries.len() * size_of::<BucketEntry>()
    }

    /// Longest token that prefixes `data[pos..end]`, as `(token, length)`.
    #[inline]
    fn longest_match(&self, data: &[u8], pos: usize, end: usize) -> (Token, usize) {
        let remaining = end - pos;
        if remaining >= 2 {
            // SAFETY: `pos + 1 < end <= data.len()`.
            let key = unsafe {
                u16::from_le_bytes([*data.get_unchecked(pos), *data.get_unchecked(pos + 1)])
            } as usize;
            // SAFETY: `heads` has an entry for every u16 key.
            let (start, count) = unsafe { *self.heads.get_unchecked(key) };
            if count != 0 {
                let max_len = remaining.min(MAX_TOKEN_SIZE);
                let window = load_window(data, pos, max_len).wrapping_shr(16);
                let entries = &self.entries[start as usize..start as usize + count as usize];
                for e in entries {
                    let elen = e.len as usize;
                    if elen <= max_len {
                        let mask = if elen >= 16 {
                            u128::MAX
                        } else {
                            (1u128 << ((elen - 2) * 8)) - 1
                        };
                        if window & mask == e.rest {
                            return (e.token, elen);
                        }
                    }
                }
            }
        }
        // SAFETY: `pos < end <= data.len()`.
        let byte = unsafe { *data.get_unchecked(pos) };
        (self.one_byte[byte as usize], 1)
    }
}

/// Encode all rows greedily with the two-byte bucket matcher.
pub fn parse_twobyte(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &TwoByteLpm,
    bits: BitWidth,
    store: &mut Store,
) {
    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();
    let mut writer = BitWriter::new(store);
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0);
    for row in 0..n {
        let mut pos = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        while pos < end {
            let (token, len) = matcher.longest_match(data, pos, end);
            debug_assert!(len != 0);
            writer.write(token);
            pos += len;
        }
        boundaries.push(writer.tokens_written() as u32);
    }
    drop(writer);
    store.boundaries = boundaries;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FixedThreshold;
    use crate::config::ThresholdSpec;
    use crate::config::TrainingConfig;
    use crate::parser::parse;
    use crate::test_corpus::binary_strings;
    use crate::test_corpus::make_raw;
    use crate::test_corpus::random_ascii_strings;
    use crate::test_corpus::user_strings;
    use crate::trainer::train;

    fn assert_matches_greedy(rows: &[Vec<u8>], bits: BitWidth) {
        let raw = make_raw(rows);
        let config = TrainingConfig {
            bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(7),
        };
        let trained = train(&raw.data, &raw.offsets, raw.n, &config);
        let mut greedy = Store::default();
        parse(
            &raw.data,
            &raw.offsets,
            raw.n,
            &trained.lpm,
            bits,
            &mut greedy,
        );

        let matcher = TwoByteLpm::from_dictionary(&trained.dict);
        let mut twobyte = Store::default();
        parse_twobyte(&raw.data, &raw.offsets, raw.n, &matcher, bits, &mut twobyte);
        assert_eq!(twobyte.packed, greedy.packed);
        assert_eq!(twobyte.boundaries, greedy.boundaries);
    }

    #[test]
    fn twobyte_matches_greedy_tokens() {
        for bits in [12u8, 16u8] {
            let user: Vec<Vec<u8>> = user_strings(200)
                .into_iter()
                .map(String::into_bytes)
                .collect();
            assert_matches_greedy(&user, bits);
            assert_matches_greedy(&random_ascii_strings(512, 80, 97), bits);
            assert_matches_greedy(&binary_strings(256, 80, 31), bits);
        }
    }

    #[test]
    fn empty_rows_and_boundaries_are_preserved() {
        let rows = vec![
            b"".to_vec(),
            b"abcabcabc".to_vec(),
            b"".to_vec(),
            b"abc".to_vec(),
        ];
        assert_matches_greedy(&rows, 12);
    }
}
