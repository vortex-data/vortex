// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bloom-filter aggregate implementation for zoned layouts.
//!
//! For more documentation about the bloom filter, see [`BloomFilter`]. For
//! information about the metadata/options, see [`BloomOptions`], and for the
//! actual implementation, see [`BloomPartial`].

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::num::NonZeroU32;

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

mod canonical;
mod partial;
mod scalar_fn;

pub(in crate::layouts::zoned) mod constant;
pub use partial::BloomPartial;
pub use partial::HashFn;

/// The default value is derived from the default `WriteStrategyBuilder::row_block_size`
const DEFAULT_BLOCKS_COUNT: u32 = 256;
/// Serialized options size expressed in bytes
const OPTIONS_BYTES_LEN: usize = size_of::<u32>() * 2;

/// Bloom-filter tuning options
///
/// **Serialization**
///
/// Bloom options are serialized in little-endian format:
///
/// ```text
/// ┌───────────────────────┬───────────────────────┐
/// │ bytes 0..4            │ bytes 4..8            │
/// ├───────────────────────┼───────────────────────┤
/// │ blocks_count (u32 LE) │ hash_fn (u32 LE)      │
/// └───────────────────────┴───────────────────────┘
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BloomOptions {
    /// Number of blocks in the split block Bloom filter (SBBF).
    ///
    /// Defaults to: [DEFAULT_BLOCKS_COUNT].
    ///
    /// The filter is partitioned into 256-bit blocks. More blocks reduce the
    /// false-positive rate at the cost of a larger filter.
    ///
    /// ### Block size and memory usage
    ///
    /// Approximate memory used by the filter for one zone:
    ///
    /// | `blocks_count`  |      Memory | Notes   |
    /// | --------------: | ----------: | ------- |
    /// |               8 |   **256 B** |         |
    /// |             256 |   **8 KiB** | Default |
    /// |           8,192 | **256 KiB** |         |
    /// |          65,536 |   **2 MiB** |         |
    /// |       1,048,576 |  **32 MiB** |         |
    blocks_count: NonZeroU32,
    /// Hashing function to use.
    ///
    /// Defaults to: [`HashFn::XxHash3_64`].
    hash_fn: HashFn,
}

impl BloomOptions {
    pub fn new(blocks_count: NonZeroU32, hash_fn: HashFn) -> Self {
        Self {
            blocks_count,
            hash_fn,
        }
    }

    pub fn blocks_count(&self) -> NonZeroU32 {
        self.blocks_count
    }

    /// Deserialize options from their layout metadata representation.
    pub(in crate::layouts::zoned) fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure_eq!(
            bytes.len(),
            OPTIONS_BYTES_LEN,
            "invalid bloom metadata length"
        );

        // Both options are u32
        let (chunks, remainder) = bytes.as_chunks::<4>();
        vortex_ensure_eq!(remainder.len(), 0, "expected no trailing metadata bytes");

        let blocks_count = u32::from_le_bytes(chunks[0]);
        let hash_fn = HashFn::try_from(u32::from_le_bytes(chunks[1]))?;

        Ok(Self {
            blocks_count: NonZeroU32::new(blocks_count)
                .ok_or_else(|| vortex_err!("bloom blocks length must be non-zero"))?,
            hash_fn,
        })
    }

    /// Serialize options into their layout metadata representation.
    pub(in crate::layouts::zoned) fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(OPTIONS_BYTES_LEN);
        bytes.extend(self.blocks_count.get().to_le_bytes());
        bytes.extend((self.hash_fn as u32).to_le_bytes());
        bytes
    }
}

impl Default for BloomOptions {
    fn default() -> Self {
        Self {
            blocks_count: NonZeroU32::new(DEFAULT_BLOCKS_COUNT)
                .vortex_expect("valid nonzero u32 value"),
            hash_fn: HashFn::XxHash3_64,
        }
    }
}

impl Display for BloomOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "blocks={}, hash_fn={}", self.blocks_count, self.hash_fn)
    }
}

/// A Bloom filter is an approximate membership query structure.
/// In Vortex layouts, it helps determine if a value is probably
/// present in a single-column zone.
///
/// Because membership is approximate, the filter can produce false positives
/// but never false negatives. In other words, it can report that
/// an absent value is present in a zone, but it never excludes a zone containing the value.
/// The false-positive probability depends on the number of distinct values
/// and the filter configuration.
///
/// ### Implementation
///
/// Implementation is based on the Split block Bloom Filter (SBBF),
/// a Bloom filter variant that is cache-friendly and takes advantage of SIMD.
/// As a tradeoff, it is less space-efficient than a traditional Bloom filter.
///
/// The filter is made up of 256-bit blocks, where each block "splits" into eight sections.
/// When a zone writer inserts a value, the value gets hashed and assigned to a block.
/// And then a mask derived from the hash and applied to the assigned block.
/// For more details about the process, see [Insertion](#insertion) for how a block
/// is selected and updated. For the actual implementation code, refer to [BloomPartial].
///
/// ### Representation
///
/// The internal state is represented as `blocks: Vec<[u32; 8]>`, with each block
/// containing its eight splits/sections.
///
/// An empty filter looks as follows:
///
/// ```text
/// ┌──────────────────┬─────┬──────────────────────┐
/// │ block 0 [u32; 8] │ ... │ block N -1 [u32; 8]  │
/// ├──────────────────┼─────┼──────────────────────┤
/// │ 00000...00000    │ ... │ 00000...00000        │
/// └──────────────────┴─────┴──────────────────────┘
/// ```
///
/// If we zoom in on a particular block it would look like this:
///
/// ```text
/// ┌────────────────┬─────┬────────────────┐
/// │ split 0 [u32]  │ ... │ split 7 [u32]  │
/// ├────────────────┼─────┼────────────────┤
/// │ 00000...00000  │ ... │ 00000...00000  │
/// └────────────────┴─────┴────────────────┘
/// ```
///
/// ### Insertion
///
/// During insertion, the value to insert gets hashed into a 64-bit value.
/// From the resulting hash, the upper 32 bits are used to select a block,
/// while the lower 32 bits are used to create the mask.
///
/// ```text
///                    hash [u64]
///            10100...11000_00011...11011
///                         │
///                         ▼
///          ┌────────────────┬────────────────┐
///          │ upper [u32]    │ lower [u32]    │
///          ├────────────────┼────────────────┤
///          │ 10100...11000  │ 00011...11011  │
///          └───────┬────────┴───────┬────────┘
///                  │                │
///                  ▼                ▼
///          block_index(upper)  make_mask(lower)
///                  │                │
///                  ▼                ▼
///          block_idx [usize]   mask [u32; 8]
/// ```
///
/// The mask has the same structure as a block, but with one bit set for
/// each of its eight splits.
///
/// The following action is to OR the mask and block together, turning those bits on
/// without changing any bits that were already set:
///
/// `block = block OR mask`
///
/// And this is how the updated block fits back into the filter:
///
/// ```text
/// ┌────────────────┬─────┬────────────────┐
/// │ split 0 [u32]  │ ... │ split 7 [u32]  │
/// ├────────────────┼─────┼────────────────┤
/// │ 10000...00000  │ ... │ 00000...00100  │
/// └────────────────┴─────┴────────────────┘
/// ```
///
/// Bloom filter block visualised (splits flattened):
///
/// ```text
/// ┌─────┬────────────────────────────────┬─────┐
/// │ ... │ block 4                        │ ... │
/// ├─────┼────────────────────────────────┼─────┤
/// │ ... │ 10000...00100                  │ ... │
/// └─────┴────────────────────────────────┴─────┘
/// ```
///
/// ### Serialization
///
/// Serialization is simple, it is just flattening the blocks `Vec<[u32; 8]>`
/// into `Vec<u8>`. So the only difference is that the blocks boundaries
/// are now implicit, while the bits remain the same.
///
/// To deserialize, it is just enough to split the byte sequence into 32-byte blocks.
///
/// ```text
/// ┌────────────────────────┬─────┬────────────────────────────────┐
/// │ bytes 0..32            │ ... │ bytes (N - 1) * 32..N * 32     │
/// ├────────────────────────┼─────┼────────────────────────────────┤
/// │ block 0: 8 x u32 (LE)  │ ... │ block N - 1                    │
/// └────────────────────────┴─────┴────────────────────────────────┘
/// ```
///
/// ### Notice
///
/// Only valid (non-null) scalar values are stored in the filter.
#[derive(Clone, Debug)]
pub struct BloomFilter;

impl AggregateFnVTable for BloomFilter {
    type Options = BloomOptions;
    type Partial = BloomPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.bloom_filter.sbbf");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        BloomOptions::deserialize(metadata)
    }

    /// Returns [Binary(Nullability::NonNullable)] when input [DType] is valid.
    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_bloom_valid_dtype(input_dtype).then_some(DType::Binary(Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    /// Returns an empty Bloom filter with all blocks zero-initialized.
    fn empty_partial(&self, options: &Self::Options, _: &DType) -> VortexResult<Self::Partial> {
        Ok(BloomPartial::from(options))
    }

    // Combination happens by doing an OR between both filters bits
    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }

        let other_as_bytes = other
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("non-null bloom partial has no bytes"))?;

        // This assumes that `other` was created using the same hash function as
        // `partial`. Ideally, an assertion here about which `hash_fn` was used to create `other`
        // would catch this invariant.
        partial.merge(other_as_bytes)
    }

    /// Returns the non-nullable binary representation of a bloom filter
    ///
    /// Basically turns each block into a single byte sequence.
    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let bytes: Vec<u8> = partial.serialize();
        Ok(Scalar::binary(bytes, Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.reset();
    }

    /// Returns true if all the blocks are full.
    ///
    /// When a bloom filter is saturated, it cannot rule out any values.
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.is_saturated()
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(constant) => constant::accumulate_constant(constant, partial)?,
            Columnar::Canonical(canonical) => {
                canonical::accumulate_canonical(canonical, partial, ctx)?
            }
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

/// Returns true if the type is valid for the bloom index to acc/contain.
///
/// This is defined by the available implementations in
/// [constant::accumulate_constant] and [canonical::accumulate_canonical]
pub(super) fn is_bloom_valid_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => is_bloom_valid_dtype(ext.storage_dtype()),
        DType::Bool(_) | DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_) => true,
        _ => false,
    }
}

// The following functions are utils/useful for tests in canonical and constants.
#[cfg(test)]
pub(in crate::layouts::zoned::aggregates::bloom_filter) mod test_utils {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::test_harness::check_metadata;

    use super::*;

    pub fn setup() -> VortexResult<ExecutionCtx> {
        let session = vortex_array::array_session();
        let options = BloomOptions::default();
        let metadata = BloomFilter
            .serialize(&options)?
            .expect("bloom is serializable");
        assert_eq!(BloomFilter.deserialize(&metadata, &session)?, options);

        let ctx = session.create_execution_ctx();

        Ok(ctx)
    }

    pub fn build_filter(
        batch: ArrayRef,
        dtype: DType,
        mut ctx: ExecutionCtx,
    ) -> VortexResult<BloomPartial> {
        let mut accumulator = Accumulator::try_new(BloomFilter, BloomOptions::default(), dtype)?;
        accumulator.accumulate(&batch.into_array(), &mut ctx)?;
        let state = accumulator.finish()?;
        let bytes = state
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("bloom state must be non-null"))?;

        let bloom_filter = BloomPartial::deserialize(bytes.as_slice())?;

        Ok(bloom_filter)
    }

    #[test]
    fn saturation_false_when_empty() -> VortexResult<()> {
        let options = BloomOptions::default();
        let partial =
            BloomFilter.empty_partial(&options, &DType::Binary(Nullability::NonNullable))?;
        assert!(!BloomFilter.is_saturated(&partial));
        Ok(())
    }

    #[test]
    fn saturation_true_when_every_block_is_full() {
        let blocks = vec![[u32::MAX; 8]; 4];
        let partial = BloomPartial::from(blocks);

        assert!(BloomFilter.is_saturated(&partial));
    }

    #[test]
    fn combine_partials_rejects_mismatched_block_counts() -> VortexResult<()> {
        let mut smaller = BloomFilter.empty_partial(
            &BloomOptions::new(NonZeroU32::new(4).unwrap(), HashFn::XxHash3_64),
            &DType::Binary(Nullability::NonNullable),
        )?;
        let bigger = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;

        let bigger_scalar = BloomFilter.to_scalar(&bigger)?;
        let result = BloomFilter.combine_partials(&mut smaller, bigger_scalar);

        assert!(
            result.is_err(),
            "combining partials built with different blocks_count must fail loudly, not corrupt state"
        );
        Ok(())
    }

    #[test]
    fn combine_partials_unions_two_disjoint_partials() -> VortexResult<()> {
        let mut partial = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;
        for i in 0..50i64 {
            partial.insert(i.to_le_bytes());
        }

        let mut secondary_partial = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;
        for i in 50..100i64 {
            secondary_partial.insert(i.to_le_bytes());
        }

        // The following expected works because seed is equal for all.
        // If the seed is different for both partials, then this will fail.
        let mut expected = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;
        for i in 0..100i64 {
            expected.insert(i.to_le_bytes());
        }

        let secondary_partial_as_scalar = BloomFilter.to_scalar(&secondary_partial)?;
        BloomFilter.combine_partials(&mut partial, secondary_partial_as_scalar)?;

        assert!(
            partial == expected,
            "merging via combine_partials should equal a single filter built from the union of inputs"
        );

        for i in 0..100i64 {
            assert!(
                partial.contains(i.to_le_bytes()),
                "value {i} missing after merge"
            );
        }

        for i in 101..200i64 {
            assert!(
                !partial.contains(i.to_le_bytes()),
                "value {i} shouldn't be present after"
            );
        }

        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_bloom_metadata() {
        // Using a fixed value rather than the defaults.
        //
        // Defaults may change, but that shouldn't affect
        // how files are ser/de.
        let options = &BloomOptions::new(
            NonZeroU32::new(256).vortex_expect("valid nonzero"),
            HashFn::XxHash3_64,
        );

        check_metadata(
            "bloom_filter_sbbf_xxhash3_64.metadata",
            &options.serialize(),
        );
    }

    #[test]
    fn bloom_options_equality_compares_all_fields() {
        let default = BloomOptions::default();
        let same_as_default = BloomOptions::new(
            NonZeroU32::new(DEFAULT_BLOCKS_COUNT).vortex_expect("valid nonzero"),
            HashFn::XxHash3_64,
        );
        let different_block_count = BloomOptions::new(
            NonZeroU32::new(4).vortex_expect("valid nonzero"),
            HashFn::XxHash3_64,
        );

        assert_eq!(default, same_as_default);
        assert_ne!(default, different_block_count);
    }

    #[test]
    fn options_roundtrip() -> VortexResult<()> {
        let options = BloomOptions::new(
            NonZeroU32::new(256).expect("valid non-zero block count"),
            HashFn::XxHash3_64,
        );

        assert_eq!(BloomOptions::deserialize(&options.serialize())?, options);
        Ok(())
    }

    #[rstest]
    #[case::empty(&[])]
    #[case::invalid_len_too_short(&[0; OPTIONS_BYTES_LEN - 1])]
    #[case::invalid_zero_blocks(&[0_u8; OPTIONS_BYTES_LEN])]
    #[case::unknown_hash_fn(&[
        1_u32.to_le_bytes(), // `blocks_count` = 1
        1_u32.to_le_bytes() // `HashFn` = 1 (doesn't exist)
    ].concat())]
    fn invalid_options_error(#[case] bytes: &[u8]) {
        assert!(
            BloomOptions::deserialize(bytes).is_err(),
            "expected invalid metadata to return an error: {bytes:?}"
        );
    }
}
