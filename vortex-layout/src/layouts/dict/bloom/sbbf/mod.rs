// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod prune;
mod tokenizer;

pub use prune::*;
pub use tokenizer::*;
use twox_hash::XxHash64;
use vortex_buffer::{Buffer, BufferMut, ByteBuffer};
use vortex_error::{VortexResult, vortex_ensure};
// TODO(aduffy): performance test the twox_hash impl

const XX_SEED: u64 = 0;

// Constants from the Parquet format specification.
const C0: u32 = 0x47b6137b;
const C1: u32 = 0x44974d91;
const C2: u32 = 0x8824ad5b;
const C3: u32 = 0xa2b7289d;
const C4: u32 = 0x705495c7;
const C5: u32 = 0x2df1424b;
const C6: u32 = 0x9efc4947;
const C7: u32 = 0x5c6bfb31;

/// The SBBF filter, using the XXH64 hash algorithm with a seed value of 0.
///
/// This configuration is identical to what Parquet uses, and the rationale is laid out in
/// "Split block Bloom filters" <https://arxiv.org/pdf/2101.01719>.
#[derive(Clone)]
pub struct Sbbf {
    blocks: BufferMut<Block>,
}

impl Sbbf {
    pub fn new(blocks: BufferMut<Block>) -> Self {
        Self { blocks }
    }

    /// Serialize the filter to a new buffer.
    pub fn serialize(&self) -> ByteBuffer {
        BufferMut::copy_from(&self.blocks)
            .freeze()
            .into_byte_buffer()
    }

    /// Deserialize the split-block filter.
    pub fn try_deserialize(bytes: impl AsRef<[u8]>) -> VortexResult<Self> {
        let bytes = bytes.as_ref();

        vortex_ensure!(
            bytes.len() % size_of::<Block>() == 0,
            "Provided missized buffer for blocks: {}",
            bytes.len()
        );

        let blocks = Buffer::<Block>::from_byte_buffer(ByteBuffer::copy_from(bytes)).into_mut();

        Ok(Self { blocks })
    }
}

impl Sbbf {
    /// Insert a bytestring value into the filter using XXH64 hasher.
    #[allow(clippy::cast_possible_truncation)]
    pub fn insert_hash(&mut self, value: impl AsRef<[u8]>) {
        let hash = XxHash64::oneshot(XX_SEED, value.as_ref());
        let block_index = self.hash_to_block_index(hash);
        self.blocks[block_index].insert(hash as u32);
    }

    #[inline]
    fn hash_to_block_index(&self, hash: u64) -> usize {
        (((hash >> 32).saturating_mul(self.blocks.len() as u64)) >> 32) as usize
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn check(&self, value: impl AsRef<[u8]>) -> bool {
        let hash = XxHash64::oneshot(XX_SEED, value.as_ref());
        let block_index = self.hash_to_block_index(hash);
        self.blocks[block_index].check(hash as u32)
    }

    /// Get the total **load** of the filter, i.e. the fraction of bits that are set.
    ///
    /// This is an indication of how "full" the filter is.
    pub fn load(&self) -> f64 {
        self.blocks
            .iter()
            .flat_map(|b| b.0.iter())
            .map(|&word| word.count_ones() as f64)
            .sum::<f64>()
            / 256.0
            / (self.blocks.len() as f64)
    }
}

/// A single 32-byte filter block.
#[derive(Debug, Copy, Clone)]
pub struct Block([u32; 8]);

impl Block {
    pub fn insert(&mut self, value: u32) {
        let block = mask(value);
        self.0[0] |= block.0[0];
        self.0[1] |= block.0[1];
        self.0[2] |= block.0[2];
        self.0[3] |= block.0[3];
        self.0[4] |= block.0[4];
        self.0[5] |= block.0[5];
        self.0[6] |= block.0[6];
        self.0[7] |= block.0[7];
    }

    pub fn check(&self, value: u32) -> bool {
        let block = mask(value);

        for i in 0..8 {
            if self.0[i] & block.0[i] == 0 {
                return false;
            }
        }

        true
    }
}

fn mask(value: u32) -> Block {
    let mut block = [0u32; 8];

    block[0] |= 1 << (value.wrapping_mul(C0) >> 27);
    block[1] |= 1 << (value.wrapping_mul(C1) >> 27);
    block[2] |= 1 << (value.wrapping_mul(C2) >> 27);
    block[3] |= 1 << (value.wrapping_mul(C3) >> 27);
    block[4] |= 1 << (value.wrapping_mul(C4) >> 27);
    block[5] |= 1 << (value.wrapping_mul(C5) >> 27);
    block[6] |= 1 << (value.wrapping_mul(C6) >> 27);
    block[7] |= 1 << (value.wrapping_mul(C7) >> 27);

    Block(block)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferMut;

    use crate::layouts::dict::bloom::sbbf::Sbbf;

    #[test]
    fn test_sbbf() {
        let blocks = BufferMut::zeroed(4);
        let mut filter = Sbbf::new(blocks);

        filter.insert_hash("Google");
        filter.insert_hash("google");
        filter.insert_hash("boogle");

        // round trip serde
        let bytes = filter.serialize();
        let filter = Sbbf::try_deserialize(&bytes).expect("deserialize sbbf from bytes");

        // No false negatives
        assert!(filter.check("Google"));
        assert!(filter.check("google"));
        assert!(filter.check("boogle"));

        // There may be false negatives. We won't exhaustively check the 64-bit hash space, but here
        // are some examples of filter misses, just to prove to you that the filter is working.
        assert!(!filter.check("koogle"));
        assert!(!filter.check("stroodle"));
        assert!(!filter.check("poodle"));
    }
}
