// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of lane-wise patches instead of linear patches. This layout for exception
//! patching enables fully parallel GPU execution, as outlined by Hepkema et al. in
//! "G-ALP: Rethinking Light-weight Encodings for GPUs" <https://doi.org/10.1145/3736227.3736242>

use fastlanes::BitPacking;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;

#[derive(Debug, Default, Clone)]
pub(crate) struct Transposed<V> {
    pub(crate) chunks: Vec<Chunk<V>>,
}

#[derive(Debug, Clone)]
pub(crate) struct Chunk<V> {
    pub(crate) lanes: Vec<Lane<V>>,
}

impl<V: BitPacking + Default> Default for Chunk<V> {
    fn default() -> Self {
        Self {
            lanes: vec![Lane::<V>::default(); V::LANES],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Lane<V> {
    pub(crate) indices: Vec<u16>,
    pub(crate) values: Vec<V>,
}

impl<V: Copy> Lane<V> {
    pub fn push(&mut self, index: u16, value: V) {
        self.indices.push(index);
        self.values.push(value);
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub(crate) fn transpose<I: IntegerPType, V: NativePType + BitPacking>(
    indices: &[I],
    values: &[V],
    array_len: usize,
) -> Transposed<V> {
    // Total number of slots is number of chunks times number of lanes.
    let n_chunks = array_len.div_ceil(1024);
    let mut chunks: Vec<Chunk<V>> = vec![Chunk::default(); n_chunks];

    // For each chunk, for each lane, push new values
    for (index, &value) in std::iter::zip(indices, values) {
        let index = index.as_();

        let chunk = index / 1024;
        let lane = index % V::LANES;

        chunks[chunk].lanes[lane].push((index % 1024) as u16, value);
    }

    Transposed { chunks }
}

#[cfg(test)]
mod tests {
    use vortex::buffer::BufferMut;
    use vortex::buffer::buffer;

    use crate::kernel::patches::types::transpose;

    #[test]
    fn test_transpose_patches() {
        let patch_values = buffer![0u32, 10, 20, 30, 40, 50, 60, 70, 80];

        let mut patch_indices = BufferMut::empty();
        // CHUNK 0. patch_values have value type i32, which means there will be 32 lanes.
        patch_indices.extend_from_slice(&[0, 31, 63, 64]);

        // CHUNK 1.
        patch_indices.extend_from_slice(&[1024, 1056, 1058]);

        // CHUNK 2: empty
        patch_indices.extend_from_slice(&[]);

        // CHUNK 3
        patch_indices.extend_from_slice(&[3073, 3076]);

        let patch_indices = patch_indices.freeze();

        let transposed = transpose(patch_indices.as_slice(), patch_values.as_slice(), 1024 * 5);

        // Chunk 0 should have patches in lanes 0, 31
        assert_eq!(transposed.chunks[0].lanes[0].values, vec![0, 30]);
        assert_eq!(transposed.chunks[0].lanes[0].indices, vec![0, 64]);
        assert_eq!(transposed.chunks[0].lanes[31].values, vec![10, 20]);
        assert_eq!(transposed.chunks[0].lanes[31].indices, vec![31, 63]);

        // Chunk 1 should have patches in lanes 0, 2
        assert_eq!(transposed.chunks[1].lanes[0].values, vec![40, 50]);
        assert_eq!(transposed.chunks[1].lanes[0].indices, vec![0, 32]);
        assert_eq!(transposed.chunks[1].lanes[2].values, vec![60]);
        assert_eq!(transposed.chunks[1].lanes[2].indices, vec![34]);

        // Chunk 2 should be empty
        for lane in 0..31 {
            assert!(transposed.chunks[2].lanes[lane].is_empty());
        }

        // Chunk 3 contains patches at lanes 1, 4
        assert_eq!(transposed.chunks[3].lanes[1].values, vec![70]);
        assert_eq!(transposed.chunks[3].lanes[1].indices, vec![1]);
        assert_eq!(transposed.chunks[3].lanes[4].values, vec![80]);
        assert_eq!(transposed.chunks[3].lanes[4].indices, vec![4]);
    }
}
