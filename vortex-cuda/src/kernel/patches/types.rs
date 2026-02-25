// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of lane-wise patches instead of linear patches. This layout for exception
//! patching enables fully parallel GPU execution, as outlined by Hepkema et al. in
//! "G-ALP: Rethinking Light-weight Encodings for GPUs" <https://doi.org/10.1145/3736227.3736242>

use std::marker::PhantomData;
use std::ops::Range;

use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::buffer_mut;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_error::VortexResult;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

#[derive(Debug, Default, Clone)]
pub(crate) struct Transposed<V> {
    pub(crate) chunks: Vec<Chunk<V>>,
}

impl<V: Copy> Transposed<V> {
    // Slice patches to only contain the patches that are in the range given instead.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let start_chunk = range.start / 1024;
        let stop_chunk = range.end.div_ceil(1024);

        Self {
            chunks: self.chunks[start_chunk..stop_chunk].to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Chunk<V> {
    pub(crate) lanes: Vec<Lane<V>>,
}

impl<V: Copy + Default> Default for Chunk<V> {
    fn default() -> Self {
        Self {
            lanes: vec![Lane::<V>::default(); 128 / size_of::<V>()],
        }
    }
}

// indices
// offsets

pub struct NewPatches<V> {
    n_chunks: usize,
    n_lanes: usize,
    lane_offsets: Buffer<u32>,
    indices: Buffer<u16>,
    values: Buffer<V>,
}

pub struct LanePatches<'a, V> {
    indices: &'a [u16],
    values: &'a [V],
}

impl<V: Copy> NewPatches<V> {
    /// Get number of patches for a specific lane.
    pub fn patch_count(&self, chunk: usize, lane: usize) -> usize {
        let start = chunk * self.n_lanes + lane;
        let end = start + 1;
        let count = self.lane_offsets[end] - self.lane_offsets[start];

        count as usize
    }

    pub fn patches(&self, chunk: usize, lane: usize) -> LanePatches<'_, V> {
        let start = chunk * self.n_lanes + lane;
        let end = start + 1;

        let lane_start = self.lane_offsets[start] as usize;
        let lane_stop = self.lane_offsets[end] as usize;

        LanePatches {
            indices: &self.indices[lane_start..lane_stop],
            values: &self.values[lane_start..lane_stop],
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

pub(crate) fn transpose<I: IntegerPType, V: NativePType>(
    indices: &[I],
    values: &[V],
    array_len: usize,
) -> NewPatches<V> {
    // Total number of slots is number of chunks times number of lanes.
    let n_chunks = array_len.div_ceil(1024);
    let n_lanes = 128 / size_of::<V>();
    let mut chunks: Vec<Chunk<V>> = vec![Chunk::default(); n_chunks];

    // For each chunk, for each lane, push new values
    for (index, &value) in std::iter::zip(indices, values) {
        let index = index.as_();

        let chunk = index / 1024;
        let lane = index % n_lanes;

        chunks[chunk].lanes[lane].push((index % 1024) as u16, value);
    }

    let mut offset = 0;
    let mut lane_offsets = buffer_mut![0u32];
    let mut indices_buffer = BufferMut::empty();
    let mut values_buffer = BufferMut::empty();
    for chunk in chunks {
        for lane in chunk.lanes {
            indices_buffer.extend_from_slice(&lane.indices);
            values_buffer.extend_from_slice(&lane.values);
            offset += lane.len() as u32;
            lane_offsets.push(offset);
        }
    }

    NewPatches {
        n_chunks,
        n_lanes,
        lane_offsets: lane_offsets.freeze(),
        indices: indices_buffer.freeze(),
        values: values_buffer.freeze(),
    }
}

/// Set of patches that can be copied over to the GPU with ease.
#[repr(C)]
pub struct GPUNewPatches<V> {
    pub(crate) n_chunks: u32,
    pub(crate) n_lanes: u32,
    pub(crate) lane_offsets: BufferHandle,
    pub(crate) indices: BufferHandle,
    pub(crate) values: BufferHandle,
    _marker: PhantomData<V>,
}

/// Export the transposed patches back out to the GPU so they can be read in the necessary format.
pub async fn export_gpu<V: NativePType>(
    mut transposed: NewPatches<V>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<GPUNewPatches<V>> {
    let lane_offsets = std::mem::take(&mut transposed.lane_offsets);
    let indices = std::mem::take(&mut transposed.indices);
    let values = std::mem::take(&mut transposed.values);

    // Convert each into a handle that can be passed around.
    let lane_offsets_handle = BufferHandle::new_host(lane_offsets.into_byte_buffer());
    let indices_handle = BufferHandle::new_host(indices.into_byte_buffer());
    let values_handle = BufferHandle::new_host(values.into_byte_buffer());

    let lane_offsets_handle = ctx.ensure_on_device(lane_offsets_handle).await?;
    let indices_handle = ctx.ensure_on_device(indices_handle).await?;
    let values_handle = ctx.ensure_on_device(values_handle).await?;

    Ok(GPUNewPatches {
        n_chunks: transposed.n_chunks as u32,
        n_lanes: transposed.n_lanes as u32,
        lane_offsets: lane_offsets_handle,
        indices: indices_handle,
        values: values_handle,
        _marker: PhantomData,
    })
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
        assert_eq!(transposed.patches(0, 0).values, &[0, 30]);
        assert_eq!(transposed.patches(0, 0).indices, &[0, 64]);

        assert_eq!(transposed.patches(0, 31).values, &[10, 20]);
        assert_eq!(transposed.patches(0, 31).indices, &[31, 63]);

        // Chunk 1 should have patches in lanes 0, 2
        assert_eq!(transposed.patches(1, 0).values, &[40, 50]);
        assert_eq!(transposed.patches(1, 0).indices, &[0, 32]);
        assert_eq!(transposed.patches(1, 2).values, &[60]);
        assert_eq!(transposed.patches(1, 2).indices, &[34]);

        // Chunk 2 should be empty
        for lane in 0..transposed.n_lanes {
            assert_eq!(transposed.patch_count(2, lane), 0);
        }

        // Chunk 3 contains patches at lanes 1, 4
        assert_eq!(transposed.patches(3, 1).values, &[70]);
        assert_eq!(transposed.patches(3, 1).indices, &[1]);
        assert_eq!(transposed.patches(3, 4).values, &[80]);
        assert_eq!(transposed.patches(3, 4).indices, &[4]);
    }
}
