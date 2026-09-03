// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split block Bloom filters (SBBF) implementation for Vortex.
//!
//! This implementation follows the original paper but
//! with the following noticeable changes:
//! - Renaming `bucket` to `block`,
//! - Small changes that help the Rust compiler generate optimized, vectorized
//!   code for `make_mask`, `add_hash`, and `find_hash`
//! - A different salt order.
//!
//! [Split block Bloom filters]: https://arxiv.org/pdf/2101.01719

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use twox_hash::XxHash3_64;
use vortex_error::VortexError;
use vortex_error::vortex_err;

use super::BloomOptions;

mod aggregate;
mod scalar;
mod serde;

pub(super) const SPLITS_PER_BLOCK: usize = 8;
pub(super) const BYTES_PER_SPLIT: usize = size_of::<u32>(); // 4 bytes

/// Block size (32 bytes [256 bits])
pub(super) const BLOCK_SIZE: usize = SPLITS_PER_BLOCK * BYTES_PER_SPLIT;

/// Eight odd constants for multiply-shift hashing.
///
/// They fit in one 256-bit SIMD vector, and the order matches the one
/// used by the Apache Parquet specification. The paper's example uses
/// the same values but in a different order. This was not
/// intentional for having compatibility with Apache Parquet, but remains
/// as a common-order in implementations.
///
/// It is important to notice that while order doesn't affect validity,
/// it changes the final bits set in each split/lane.
const SALT: [u32; 8] = [
    0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
];

/// Hash function to use in a bloom filter.
///
/// The only supported hash function is XXH3 64-bit.
///
/// The serialized options retain a hash-function identifier
/// so future variants can be added without changing the metadata layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum HashFn {
    XxHash3_64 = 0, // Default
}

impl Display for HashFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::XxHash3_64 => "xxhash3_64",
        })
    }
}

impl TryFrom<u32> for HashFn {
    type Error = VortexError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::XxHash3_64),
            _ => Err(vortex_err!("unknown bloom hash function ID: {value}")),
        }
    }
}

/// Represents a Split block Bloom Filter for a single layout zone.
///
/// The filter stores hashes of byte representations of values. `Scalar`
/// values are converted to their underlying bytes before insertion. Other
/// types whose underlying values can be represented as bytes can also be
/// stored in the filter.
///
/// The current implementation defaults to `XxHash3_64`,
/// which is currently the only supported hash function and the fastest
/// variant from the xxHash family.
///
/// Usage example:
///
/// ```rust
/// use vortex_array::dtype::{DType, Nullability};
/// use vortex_layout::layouts::zoned::aggregates::bloom_filter::{BloomFilter, BloomOptions};
/// use vortex_array::aggregate_fn::AggregateFnVTable;
///
/// let filter = BloomFilter {};
/// let mut zone = filter
///     .empty_partial(
///         &BloomOptions::default(),
///         &DType::Binary(Nullability::NonNullable),
///     )
///     .expect("valid partial");
///
/// zone.insert(b"Denmark");
///
/// assert_eq!(zone.contains(b"Denmark"), true);
/// assert_eq!(zone.contains(b"Japan"), false);
/// assert_eq!(zone.contains(b"Brazil"), false);
/// ```
pub struct BloomPartial {
    blocks: Vec<[u32; 8]>,
    hash_fn: HashFn,
}

impl BloomPartial {
    /// Returns the blocks len.
    ///
    /// Matches [BloomOptions::blocks_count]
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Inserts a value expressed in bytes into
    /// the filter.
    #[inline]
    pub fn insert<T>(&mut self, value: T)
    where
        T: AsRef<[u8]>,
    {
        let hash = self.hash(value);
        self.add_hash(hash);
    }

    /// Returns `true` if `value` might be present in the filter.
    ///
    /// A `false` result guarantees that the value is absent. A `true` result may
    /// be a false positive.
    ///
    /// Use `BloomPartial::contains_scalar` for scalar values.
    #[inline]
    pub fn contains<T>(&self, value: T) -> bool
    where
        T: AsRef<[u8]>,
    {
        let hash = self.hash(value);
        self.find_hash(hash)
    }

    /// Produces a 64-bit hash.
    ///
    /// This follows the reference implementation, where
    /// the upper 32 bits select the block and the lower 32 bits determine the bit
    /// positions within that block.
    #[inline]
    fn hash<T>(&self, value: T) -> u64
    where
        T: AsRef<[u8]>,
    {
        XxHash3_64::oneshot(value.as_ref())
    }

    /// Returns the lower 32 bits of the hash used to construct the block mask.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the mask uses the low 32 bits of the 64-bit hash"
    )]
    fn lower_hash_bits(&self, hash: u64) -> u32 {
        hash as u32
    }

    /// Adds a hash into a single block of the bloom filter.
    fn add_hash(&mut self, hash: u64) {
        // 1. Use the upper 32 bits from the hash value to select a block.
        let block_idx = self.block_index(hash, self.blocks.len());

        // 2. Use the lower 32 bits to construct a mask.
        let mask = self.make_mask(self.lower_hash_bits(hash));

        // 3. Apply the mask to the block selected in step 1.
        for i in 0..8 {
            self.blocks[block_idx][i] |= mask[i];
        }
    }

    /// Checks whether a hash is (probably) present in the filter.
    fn find_hash(&self, hash: u64) -> bool {
        let idx = self.block_index(hash, self.blocks.len());
        let mask = self.make_mask(self.lower_hash_bits(hash));

        let mut missing = 0u32;
        let block = &self.blocks[idx];

        // The original solution uses _mm256_testc_si256
        // checks if all the bits in mask are also set in *block. Scalar
        // equivalent: (~block & mask) == 0
        for i in 0..8 {
            missing |= !block[i] & mask[i];
        }

        missing == 0
    }

    /// Takes a hash value and creates a mask with one bit set in each 32-bit lane.
    /// These are the bits to set or check when accessing the block.
    fn make_mask(&self, hash: u32) -> [u32; 8] {
        let mut out = [0u32; 8];

        for i in 0..8 {
            // Shift all data right, reducing the hash values from 32 bits to five bits.
            // Those five bits represent an index in [0, 31)
            let y = hash.wrapping_mul(SALT[i]) >> 27;

            // Set a bit in each lane based on using the [0, 32) data as shift values.
            out[i] = 1u32 << y;
        }

        out
    }

    /// Returns the index of the block to which a hash belongs.
    ///
    /// For details about the algorithm, see
    /// [Lemire's FastRange](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/).
    ///
    /// Although `blocks_count` is a `usize`, its value is limited to `u32::MAX`
    /// by [`BloomOptions`] and the serialization format.
    #[inline]
    fn block_index(&self, hash: u64, blocks_count: usize) -> usize {
        (((hash >> 32) * blocks_count as u64) >> 32) as usize
    }
}

/// Useful conversion used mostly for tests, and to
/// start an empty partial from [`super::BloomFilter`].
impl From<&BloomOptions> for BloomPartial {
    fn from(options: &BloomOptions) -> Self {
        Self {
            blocks: vec![[0u32; 8]; options.blocks_count.get() as usize],
            hash_fn: options.hash_fn,
        }
    }
}

impl PartialEq for BloomPartial {
    fn eq(&self, other: &Self) -> bool {
        // Currently, the Bloom filter only supports one hash function,
        // so two partials with the same blocks are equal.
        // If the filter supports more hash functions in the future,
        // this would no longer be true, because the same blocks could represent
        // different values.
        self.blocks == other.blocks && self.hash_fn == other.hash_fn
    }
}

#[cfg(test)]
impl From<Vec<[u32; 8]>> for BloomPartial {
    fn from(value: Vec<[u32; 8]>) -> Self {
        BloomPartial {
            blocks: value,
            hash_fn: HashFn::XxHash3_64, // Default. Only used for tests.
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use rstest::rstest;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;
    use crate::layouts::zoned::aggregates::bloom_filter::DEFAULT_BLOCKS_COUNT;
    use crate::layouts::zoned::aggregates::bloom_filter::HashFn;

    #[test]
    fn bigger_filter_size() {
        // The idea is to create a bigger Bloom filter than the default one (1000x approx. ~8MiB).
        let options = BloomOptions::new(
            NonZeroU32::new(DEFAULT_BLOCKS_COUNT * 1000).expect("valid nonzero u32"),
            HashFn::XxHash3_64,
        );
        let mut bloom_filter = BloomPartial::from(&options);

        for i in 1..=10u64 {
            bloom_filter.insert(i.to_le_bytes());
        }

        assert!(
            bloom_filter.contains(10u64.to_le_bytes()),
            "expected to contain value"
        );

        assert!(
            !bloom_filter.contains(11u64.to_le_bytes()),
            "expected to not contain value"
        );
    }

    /// Another regression test for bloom serialization,
    /// but in this case to detect mask salt changes.
    /// It just verifies that a filter's serialized representation remains stable.
    #[test]
    fn serialized_bits_are_stable() {
        let options = BloomOptions::new(NonZeroU32::MIN, HashFn::XxHash3_64);
        let mut bloom_filter = BloomPartial::from(&options);

        bloom_filter.insert(b"vortex");

        // Because we have only one block and this is the only value inserted,
        // these splits are equal to its mask: `empty | mask == mask`.
        let expected_splits: [u32; 8] = [
            0x0000_1000,
            0x0200_0000,
            0x0000_2000,
            0x0800_0000,
            0x0200_0000,
            0x0000_0040,
            0x0000_4000,
            0x0000_1000,
        ];

        let expected_bytes: Vec<u8> = expected_splits
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();

        let bytes: Vec<u8> = bloom_filter.serialize();
        assert_eq!(bytes, expected_bytes);
    }

    // Similar to the goldenfile tests, but for hash functions.
    //
    // Useful compatibility test to catch an accidental hash-algorithm or seed change.
    #[rstest]
    #[case(HashFn::XxHash3_64, 16649171463689419262)]
    fn hash_output_is_stable(#[case] hash_fn: HashFn, #[case] expected: u64) {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::new(
            NonZeroU32::new(256).expect("valid non-zero"),
            hash_fn,
        ));
        assert_eq!(bloom_filter.hash(b"vortex"), expected);

        // Additional check
        bloom_filter.insert(b"vortex");
        assert!(bloom_filter.contains(b"vortex"));
    }
}
