// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ser/de for Split Block Bloom Filters (SBBF) in Vortex layouts.
//!
//! The idea is to follow a similar approach to `AggregateFnVTable`
//! and have one method for serialization and another for deserialization.
//! This is preferred over traits like `TryFrom<&[u8]>` on purpose,
//! to avoid footguns.

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::BLOCK_SIZE;
use super::Block;
use super::Blocks;
use super::BloomPartial;

impl BloomPartial {
    /// Deserialize a partial from its byte representation.
    ///
    /// The bytes are the filter's splits as little-endian `u32`s, which is also how Vortex lays
    /// out every primitive buffer, so nothing is decoded: the partial shares `bytes` when they are
    /// aligned for `u32` and copies them once otherwise. Both `partial_from_scalar` and
    /// `BloomContains` parse stored filters through here.
    #[inline]
    pub(in crate::layouts::zoned) fn deserialize(bytes: ByteBuffer) -> VortexResult<Self> {
        vortex_ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(BLOCK_SIZE),
            "invalid bloom filter byte length: {}",
            bytes.len()
        );
        vortex_ensure!(
            u32::try_from(bytes.len() / BLOCK_SIZE).is_ok(),
            "bloom blocks length must be non-zero and lower than u32::MAX",
        );

        let blocks = Buffer::<Block>::from_byte_buffer(bytes.aligned(Alignment::of::<Block>()));
        Ok(BloomPartial {
            blocks: Blocks::Frozen(blocks),
        })
    }

    /// Serialize the filter into the bytes a layout zone stores: its splits as little-endian
    /// `u32`s.
    ///
    /// A frozen partial hands back the buffer it was parsed from; a thawed one is copied.
    pub(in crate::layouts::zoned) fn serialize(&self) -> ByteBuffer {
        match &self.blocks {
            Blocks::Frozen(blocks) => blocks.clone().into_byte_buffer(),
            Blocks::Thawed(blocks) => Buffer::copy_from(blocks.as_slice()).into_byte_buffer(),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::ToBytes;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;

    #[test]
    fn valid_serde() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let bytes = bloom_filter.serialize();
        let valid_filter = BloomPartial::deserialize(bytes).unwrap();

        assert!(
            valid_filter.contains(32.to_le_bytes()),
            "expect filter to have value"
        );

        assert!(
            !valid_filter.contains(14.to_le_bytes()),
            "expect filter to not have value"
        );
    }

    #[test]
    fn invalid_serde() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let mut bytes = bloom_filter.serialize().as_slice().to_vec();
        bytes.pop();
        let invalid_filter = BloomPartial::deserialize(ByteBuffer::from(bytes));

        assert!(invalid_filter.is_err(), "expect filter to be invalid");

        let mut bytes = bloom_filter.serialize().as_slice().to_vec();
        bytes.push(0u8);
        let invalid_filter = BloomPartial::deserialize(ByteBuffer::from(bytes));

        assert!(invalid_filter.is_err(), "expect filter to be invalid");
    }

    /// Merging a stored filter must not decode it: the partial reads the zone's own bytes.
    #[test]
    fn deserialize_shares_aligned_bytes() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let bytes = bloom_filter.serialize().aligned(Alignment::of::<u32>());
        let parsed = BloomPartial::deserialize(bytes.clone()).unwrap();

        assert_eq!(parsed.blocks().as_ptr().cast::<u8>(), bytes.as_ptr());
        assert!(parsed.contains(32.to_le_bytes()));
    }

    /// Bytes that are not aligned for `u32` are copied into alignment rather than rejected.
    #[test]
    fn deserialize_copies_unaligned_bytes() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let mut padded = vec![0u8];
        padded.extend_from_slice(bloom_filter.serialize().as_slice());
        let unaligned = ByteBuffer::from(padded).slice(1..);
        let parsed = BloomPartial::deserialize(unaligned).unwrap();

        assert!(
            parsed == bloom_filter,
            "a copied filter must equal its source"
        );
        assert!(parsed.contains(32.to_le_bytes()));
    }
}
