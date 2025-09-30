// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module implements Bloom Filters, which are treated as another type of "zone map".
//!
//! Bloom filters may optionally be populated at write time, and consulted at read time.

pub mod sbbf;
mod serde;

use std::ops::Index;
use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_expr::{BinaryVTable, LikeVTable, LiteralVTable, Operator, VortexExpr};
use vortex_mask::Mask;

use crate::layouts::dict::bloom::sbbf::{Block, Sbbf};

/// A Bloom Filter that allows probabilistic set membership queries for the values in an array.
///
/// Bloom Filters can be constructed at file write, and probed at query time to enable pruning of
/// zones and avoid fetching and decoding entire blocks, greatly speeding up the execution time of
/// string queries.
#[derive(Clone)]
pub enum BloomFilter {
    /// A split-block Bloom Filter that uses word-level tokenization.
    ///
    ///
    /// ## Tokenization
    ///
    /// Every string in a `Utf8` array will tokenize its scalar and insert the stream of tokens into
    /// the filter.
    ///
    /// Tokens are roughly "words", specifically adjacent runs of alphanumeric Unicode scalar values.
    ///
    /// For example, the string `https://google.com` will be tokenized into
    /// `["https", ":", "/", "/", "google", ".", "com"]`.
    ///
    /// The string `I ❤️ databases" will be tokenized into `["I", " ", "❤️", " ", "databases"]`.
    ///
    /// ## Hashing
    ///
    /// We use the xxHash algorithm, and specifically XXH64. This is identical to the implementation
    /// that is used by Parquet.
    ///
    /// ## "Split-block"
    ///
    /// The Split-block structure is a method of construction L1-cache efficient filters. The crux
    /// of it is that you have a collection of 32 byte "blocks", broken up into 8 32-bit "words".
    /// The block is sized to fit into a single L1 cache line on all modern processors.
    ///
    /// Both insertion and membership queries take as input a 64-bit hash, where the top 32 bits
    /// are used to index a single block in the filter, and the bottom 32 bits are used to update
    /// the block.
    ///
    /// The split-block design is lifted straight from the Parquet specification.
    SplitBlockWord(Sbbf),
}

/// The type of membership operation we're trying to execute
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CheckOp {
    Eq,
    NotEq,
    Contains,
    StartsWith,
    EndsWith,
}

#[allow(clippy::cast_possible_truncation)]
fn num_of_bits_from_ndv_fpp(ndv: u64, fpp: f64) -> usize {
    let num_bits = -8.0 * ndv as f64 / (1.0 - fpp.powf(1.0 / 8.0)).ln();
    num_bits as usize
}

impl BloomFilter {
    const BF_MAX_BYTES: usize = 128 * 1024 * 1024;
    const BF_MIN_BYTES: usize = size_of::<Block>();

    /// Create a new split-block Bloom filter that uses XXH64 hasher for element insertions and
    /// lookups.
    ///
    /// The provided tokenizer is used to map input string data into tokens, which are then hashed
    /// and inserted.
    pub fn new_sbbf(blocks: usize) -> Self {
        Self::SplitBlockWord(Sbbf::new(BufferMut::zeroed(blocks)))
    }

    /// Create a new split-block Bloom filter with the target number of distinct values, and target
    /// false positivity rate (e.g. 0.01 = 1%).
    ///
    /// This will never yield a filter with < 1 block, or > ~4mm blocks.
    pub fn new_sbbf_ndv_fp(ndv: usize, fp: f64) -> Self {
        let bit_size = num_of_bits_from_ndv_fpp(ndv as u64, fp);
        let byte_size = bit_size / 8;
        let num_bytes = byte_size
            .clamp(Self::BF_MIN_BYTES, Self::BF_MAX_BYTES)
            .next_power_of_two();

        let num_blocks = num_bytes / size_of::<Block>();
        Self::new_sbbf(num_blocks)
    }

    /// Insert into the filter.
    pub fn insert(&mut self, input: &str) {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => {
                // Insert every word token individually into the filter
                for token in sbbf::tokenize(input) {
                    sbbf.insert_hash(token);
                }

                // Insert the full string. This allows us to bypass tokenization when evaluating
                // exact match predicates.
                sbbf.insert_hash(input);
            }
        }
    }

    /// Insert a bytestring "raw", without any tokenization.
    ///
    /// This is helpful for anything that supports
    pub fn insert_raw(&mut self, raw: impl AsRef<[u8]>) {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => sbbf.insert_hash(raw),
        }
    }

    /// Check if a bloom filter contains the provided `search` string.
    ///
    /// Returns `true` if the filter **may** contain the string, and `false if it **certainly**
    /// does not contain the given string.
    pub fn check(&self, search: &str) -> bool {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => sbbf.check(search),
        }
    }

    /// Check if the bloom filter contains a string that has the given prefix.
    ///
    /// Returns `true` if the filter **may** contain a string with the prefix, and `false` if
    /// the filter **certainly** does not contain a string with the provided prefix.
    pub fn check_prefix(&self, prefix: &str) -> bool {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => {
                for token in sbbf::tokenize_starts_with(prefix) {
                    if !sbbf.check(token) {
                        return false;
                    }
                }

                true
            }
        }
    }

    /// Check if the bloom filter contains any strings with the given suffix.
    ///
    /// Returns `true` if the filter **may** contain a string with the suffix, and `false` if the
    /// filter **certainly** does not contain a string with the suffix.
    pub fn check_suffix(&self, suffix: &str) -> bool {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => {
                for token in sbbf::tokenize_ends_with(suffix) {
                    if !sbbf.check(token) {
                        return false;
                    }
                }

                true
            }
        }
    }

    /// Check if the bloom filter contains any strings which contain the given `query` string.
    ///
    /// Returns `true` if the filter **may** contain a string that itself contains `query`, and
    /// `false` if it **certainly** does not contain one.
    pub fn check_contains(&self, query: &str) -> bool {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => {
                for token in sbbf::tokenize_contains(query) {
                    if !sbbf.check(token) {
                        return false;
                    }
                }

                true
            }
        }
    }
}

impl BloomFilter {
    /// Returns the fraction of bits in the filter that have been set.
    pub fn load(&self) -> f64 {
        match self {
            BloomFilter::SplitBlockWord(sbbf) => sbbf.load(),
        }
    }
}

/// A shareable set of Bloom filters, one for each zone.
#[derive(Clone)]
pub struct BloomFilters {
    filters: Arc<[BloomFilter]>,
}

impl BloomFilters {
    /// Create a new set of bloom filters from a collection of `BloomFilter`, assumed to be one
    /// filter per zone.
    pub fn new(filters: impl Into<Arc<[BloomFilter]>>) -> Self {
        Self {
            filters: filters.into(),
        }
    }

    /// Build a pruning mask by turning the expression into a sequence of probes against the zone
    /// Bloom filters.
    ///
    /// The pruning mask will have the length equal to the number of zones. Each bit in the mask
    /// indicates:
    ///
    /// * `true` indicating the zone **can** be pruned, as determined by the Bloom filter
    /// * `false` indicating the zone **cannot** be pruned, because the Bloom filter indicated
    ///   that the target _may_ be contained in the zone.
    pub fn prune(&self, expr: &dyn VortexExpr) -> Mask {
        // Handle pruning of `=` and `<>` against a string literal
        if let Some(binary) = expr.as_opt::<BinaryVTable>()
            && let Some(rhs) = binary.rhs().as_opt::<LiteralVTable>()
            && let Some(phrase) = rhs.value().as_utf8_opt().and_then(|s| s.value())
        {
            let op = match binary.op() {
                Operator::Eq => CheckOp::Eq,
                // Other operators not supported, no pruning executed
                _ => return Mask::new_false(self.filters.len()),
            };

            self.prune_op(phrase.as_str(), op)
        }
        // Handle `LIKE` expressions against a literal pattern
        else if let Some(like) = expr.as_opt::<LikeVTable>()
            && !like.case_insensitive()
            && !like.negated()
            && let Some(rhs) = like.pattern().as_opt::<LiteralVTable>()
            && let Some(pattern) = rhs.value().as_utf8_opt().and_then(|s| s.value())
        {
            match parse_like_pattern(pattern.as_str()) {
                LikePattern::Exact(search) => self.prune_op(search, CheckOp::Eq),
                LikePattern::Suffix(suffix) => self.prune_op(suffix, CheckOp::EndsWith),
                LikePattern::Prefix(prefix) => self.prune_op(prefix, CheckOp::StartsWith),
                LikePattern::Contains(contains) => self.prune_op(contains, CheckOp::Contains),
                LikePattern::Other(_) => Mask::new_false(self.filters.len()),
            }
        } else {
            Mask::new_false(self.filters.len())
        }
    }

    /// Generate a Mask with a bit for each zone based on the presence of tokens for the given
    /// check operation.
    fn prune_op(&self, target: &str, op: CheckOp) -> Mask {
        match op {
            CheckOp::Eq => Mask::from_iter(self.filters.iter().map(|f| !f.check(target))),
            CheckOp::NotEq => Mask::from_iter(self.filters.iter().map(|f| f.check(target))),
            CheckOp::Contains => {
                Mask::from_iter(self.filters.iter().map(|f| !f.check_contains(target)))
            }
            CheckOp::StartsWith => {
                Mask::from_iter(self.filters.iter().map(|f| !f.check_prefix(target)))
            }
            CheckOp::EndsWith => {
                Mask::from_iter(self.filters.iter().map(|f| !f.check_suffix(target)))
            }
        }
    }

    pub fn len(&self) -> usize {
        self.filters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    pub fn load(&self) -> Vec<f64> {
        self.filters.iter().map(|f| f.load()).collect()
    }
}

impl Index<usize> for BloomFilters {
    type Output = BloomFilter;

    fn index(&self, index: usize) -> &Self::Output {
        &self.filters[index]
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LikePattern<'a> {
    Exact(&'a str),
    Suffix(&'a str),
    Prefix(&'a str),
    Contains(&'a str),
    Other(&'a str),
}

fn parse_like_pattern(pattern: &str) -> LikePattern<'_> {
    // Check for empty pattern
    if pattern.is_empty() {
        return LikePattern::Other(pattern);
    }

    // Count % and _ characters
    let percent_count = pattern.chars().filter(|&c| c == '%').count();
    let underscore_count = pattern.chars().filter(|&c| c == '_').count();

    // If there are any underscores, it's Other (unsupported)
    if underscore_count > 0 {
        return LikePattern::Other(pattern);
    }

    match percent_count {
        0 => {
            // No wildcards - exact match, treat as Other
            LikePattern::Exact(pattern)
        }
        1 => {
            if let Some(suffix) = pattern.strip_prefix('%') {
                // %XYZ - Suffix pattern
                if suffix.is_empty() {
                    LikePattern::Other(pattern) // Just "%" is not a valid suffix
                } else {
                    LikePattern::Suffix(suffix)
                }
            } else if let Some(prefix) = pattern.strip_suffix('%') {
                // XYZ% - Prefix pattern
                if prefix.is_empty() {
                    LikePattern::Other(pattern) // Just "%" is not a valid prefix
                } else {
                    LikePattern::Prefix(prefix)
                }
            } else {
                // % somewhere in the middle - unsupported
                LikePattern::Other(pattern)
            }
        }
        2 => {
            if pattern.starts_with('%') && pattern.ends_with('%') {
                // %XYZ% - Contains pattern
                let contained = &pattern[1..pattern.len() - 1];
                if contained.is_empty() {
                    LikePattern::Other(pattern) // "%%" is not a valid contains
                } else {
                    LikePattern::Contains(contained)
                }
            } else {
                // Multiple % but not surrounding - unsupported
                LikePattern::Other(pattern)
            }
        }
        _ => {
            // More than 2 % characters - unsupported
            LikePattern::Other(pattern)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_patterns() {
        assert_eq!(parse_like_pattern("%xyz"), LikePattern::Suffix("xyz"));
        assert_eq!(parse_like_pattern("%hello"), LikePattern::Suffix("hello"));
        assert_eq!(parse_like_pattern("%"), LikePattern::Other("%"));
    }

    #[test]
    fn test_prefix_patterns() {
        assert_eq!(parse_like_pattern("xyz%"), LikePattern::Prefix("xyz"));
        assert_eq!(parse_like_pattern("hello%"), LikePattern::Prefix("hello"));
        assert_eq!(parse_like_pattern("%"), LikePattern::Other("%"));
    }

    #[test]
    fn test_contains_patterns() {
        assert_eq!(parse_like_pattern("%xyz%"), LikePattern::Contains("xyz"));
        assert_eq!(
            parse_like_pattern("%hello%"),
            LikePattern::Contains("hello")
        );
        assert_eq!(parse_like_pattern("%%"), LikePattern::Other("%%"));
    }

    #[test]
    fn test_other_patterns() {
        // Patterns with underscores
        assert_eq!(parse_like_pattern("xy_z"), LikePattern::Other("xy_z"));
        assert_eq!(parse_like_pattern("%xy_z%"), LikePattern::Other("%xy_z%"));

        // Multiple % characters
        assert_eq!(parse_like_pattern("%%xyz"), LikePattern::Other("%%xyz"));
        assert_eq!(parse_like_pattern("xyz%%"), LikePattern::Other("xyz%%"));
        assert_eq!(parse_like_pattern("%xy%z%"), LikePattern::Other("%xy%z%"));

        // % in middle
        assert_eq!(parse_like_pattern("xy%z"), LikePattern::Other("xy%z"));

        // Exact match (no wildcards)
        assert_eq!(parse_like_pattern("xyz"), LikePattern::Exact("xyz"));

        // Empty pattern
        assert_eq!(parse_like_pattern(""), LikePattern::Other(""));
    }

    #[test]
    fn test_edge_cases() {
        // Single characters
        assert_eq!(parse_like_pattern("%a"), LikePattern::Suffix("a"));
        assert_eq!(parse_like_pattern("a%"), LikePattern::Prefix("a"));
        assert_eq!(parse_like_pattern("%a%"), LikePattern::Contains("a"));

        // Special characters in content
        assert_eq!(
            parse_like_pattern("%hello world%"),
            LikePattern::Contains("hello world")
        );
        assert_eq!(
            parse_like_pattern("hello@world%"),
            LikePattern::Prefix("hello@world")
        );
    }
}
