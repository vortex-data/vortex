// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ser/de for Split Block Bloom Filters (SBBF) in Vortex layouts.
//!
//! The idea is to follow a similar approach to `AggregateFnVTable`
//! and have one method for serialization and another for deserialization.
//! This is preferred over traits like `TryFrom<&[u8]>` on purpose,
//! to avoid footguns.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::BLOCK_SIZE;
use super::BYTES_PER_SPLIT;
use super::BloomPartial;
use crate::layouts::zoned::aggregates::bloom_filter::HashFn;

impl BloomPartial {
    /// Deserialize a partial from its byte representation.
    pub(in crate::layouts::zoned) fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(BLOCK_SIZE),
            "invalid bloom filter byte length: {}",
            bytes.len()
        );

        let blocks = bytes
            .as_chunks::<BLOCK_SIZE>()
            .0
            .iter()
            .map(|chunk| {
                let (split_bytes, remainder) = chunk.as_chunks::<BYTES_PER_SPLIT>();
                let mut block = [0u32; 8];
                vortex_ensure!(
                    remainder.is_empty(),
                    "invalid bloom filter, unexpected remainder bytes"
                );

                for (split, split_bytes) in block.iter_mut().zip(split_bytes) {
                    *split = u32::from_le_bytes(*split_bytes);
                }

                Ok(block)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        vortex_ensure!(
            !blocks.is_empty() && u32::try_from(blocks.len()).is_ok(),
            "bloom blocks length must be non-zero and lower than u32::MAX",
        );

        Ok(BloomPartial {
            blocks,
            hash_fn: HashFn::XxHash3_64, // Default option
        })
    }

    /// Serialize partial filter into its bytes format (little endian)
    /// to store into a layout zone. Basically it flattens out the blocks
    /// structure from `Vec<[u32; 8]> -> Vec<[u8]>`
    pub(in crate::layouts::zoned) fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.len() * BLOCK_SIZE);
        bytes.extend(
            self.blocks
                .iter()
                .flatten()
                .flat_map(|block| block.to_le_bytes()),
        );

        bytes
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::ToBytes;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;

    #[test]
    fn valid_serde() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let bytes = bloom_filter.serialize();
        let valid_filter = BloomPartial::deserialize(bytes.as_slice()).unwrap();

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

        let mut bytes: Vec<u8> = bloom_filter.serialize();
        bytes.pop();
        let invalid_filter = BloomPartial::deserialize(bytes.as_slice());

        assert!(invalid_filter.is_err(), "expect filter to be invalid");

        let mut bytes: Vec<u8> = bloom_filter.serialize();
        bytes.push(0u8);
        let invalid_filter = BloomPartial::deserialize(bytes.as_slice());

        assert!(invalid_filter.is_err(), "expect filter to be invalid");
    }
}
