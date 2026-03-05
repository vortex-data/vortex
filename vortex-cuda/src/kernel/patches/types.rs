// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of lane-wise patches instead of linear patches. This layout for exception
//! patching enables fully parallel GPU execution, as outlined by Hepkema et al. in
//! "G-ALP: Rethinking Light-weight Encodings for GPUs" <https://doi.org/10.1145/3736227.3736242>

use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::buffer_mut;
use vortex_array::Canonical;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::CudaExecutionCtx;

/// A set of device-resident patches that live in the GPU.
///
/// These are dynamically typed.
#[repr(C)]
pub struct DevicePatches {
    pub(crate) lane_offsets: BufferHandle,
    pub(crate) indices: BufferHandle,
    pub(crate) values: BufferHandle,
}

/// Number of lanes used at patch time for a value of type `V`.
///
/// This is *NOT* equal to the number of FastLanes lanes for the type `V`, rather this will
/// correspond with the number of CUDA threads dedicated to executing each 1024-element vector.
const fn patch_lanes<V: Sized>() -> usize {
    // For types 32-bits or smaller, we use a 32 lane configuration, and for 64-bit we use 16 lanes.
    // This matches up with the number of lanes we use to execute copying results from bit-unpacking
    // from shared to global memory.
    if size_of::<V>() < 8 { 32 } else { 16 }
}

#[derive(Clone)]
struct Chunk<V> {
    lanes: Vec<Lane<V>>,
}

impl<V: Copy + Default> Default for Chunk<V> {
    fn default() -> Self {
        Self {
            lanes: vec![Lane::<V>::default(); patch_lanes::<V>()],
        }
    }
}

/// A set of patches of values `V` existing in host buffers.
#[allow(dead_code)]
pub struct HostPatches<V> {
    n_chunks: usize,
    n_lanes: usize,
    lane_offsets: Buffer<u32>,
    indices: Buffer<u16>,
    /// Values. This is a buffer handle which might live on the new buffer type here
    values: Buffer<V>,
}

#[cfg(test)]
struct LanePatches<'a, V> {
    indices: &'a [u16],
    values: &'a [V],
}

impl<V: Copy> HostPatches<V> {
    /// Get number of patches for a specific lane.
    #[cfg(test)]
    fn patch_count(&self, chunk: usize, lane: usize) -> usize {
        let start = chunk * self.n_lanes + lane;
        let end = start + 1;
        let count = self.lane_offsets[end] - self.lane_offsets[start];

        count as usize
    }

    /// Get an ordered list of patches for the given chunk/lane.
    #[cfg(test)]
    fn patches(&self, chunk: usize, lane: usize) -> LanePatches<'_, V> {
        let start = chunk * self.n_lanes + lane;
        let end = start + 1;

        let lane_start = self.lane_offsets[start] as usize;
        let lane_stop = self.lane_offsets[end] as usize;

        LanePatches {
            indices: &self.indices[lane_start..lane_stop],
            values: &self.values[lane_start..lane_stop],
        }
    }

    /// Export the patches for use on the device associated with the provided execution context.
    pub async fn export_to_device(
        mut self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<DevicePatches> {
        let lane_offsets = std::mem::take(&mut self.lane_offsets);
        let indices = std::mem::take(&mut self.indices);
        let values = std::mem::take(&mut self.values);

        // Convert each into a handle that can be passed around.
        let lane_offsets_handle = BufferHandle::new_host(lane_offsets.into_byte_buffer());
        let indices_handle = BufferHandle::new_host(indices.into_byte_buffer());
        let values_handle = BufferHandle::new_host(values.into_byte_buffer());

        let lane_offsets_handle = ctx.ensure_on_device(lane_offsets_handle).await?;
        let indices_handle = ctx.ensure_on_device(indices_handle).await?;
        let values_handle = ctx.ensure_on_device(values_handle).await?;

        Ok(DevicePatches {
            lane_offsets: lane_offsets_handle,
            indices: indices_handle,
            values: values_handle,
        })
    }
}

#[derive(Debug, Default, Clone)]
struct Lane<V> {
    indices: Vec<u16>,
    values: Vec<V>,
}

impl<V: Copy> Lane<V> {
    fn push(&mut self, index: u16, value: V) {
        self.indices.push(index);
        self.values.push(value);
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Transpose a set of patches from the default sorted layout into the data parallel layout.
#[allow(clippy::cognitive_complexity)]
pub async fn transpose_patches(
    patches: &Patches,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DevicePatches> {
    let array_len = patches.array_len();
    let offset = patches.offset();

    let indices = patches
        .indices()
        .clone()
        .execute::<Canonical>(ctx.execution_ctx())?
        .into_primitive();

    let values = patches
        .values()
        .clone()
        .execute::<Canonical>(ctx.execution_ctx())?
        .into_primitive();

    let indices_ptype = indices.ptype();
    let values_ptype = values.ptype();

    let indices = indices.buffer_handle().to_host().await;
    let values = values.buffer_handle().to_host().await;

    match_each_unsigned_integer_ptype!(indices_ptype, |I| {
        match_each_native_ptype!(values_ptype, |V| {
            let indices: Buffer<I> = Buffer::from_byte_buffer(indices);
            let values: Buffer<V> = Buffer::from_byte_buffer(values);

            let host_patches = transpose(indices.as_slice(), values.as_slice(), offset, array_len);

            host_patches.export_to_device(ctx).await
        })
    })
}

#[allow(clippy::cast_possible_truncation)]
fn transpose<I: IntegerPType, V: NativePType>(
    indices: &[I],
    values: &[V],
    offset: usize,
    array_len: usize,
) -> HostPatches<V> {
    // Total number of slots is number of chunks times number of lanes.
    let n_chunks = array_len.div_ceil(1024);
    assert!(
        n_chunks <= u32::MAX as usize,
        "Cannot transpose patches for array with >= 4 trillion elements"
    );

    let n_lanes = patch_lanes::<V>();
    let mut chunks: Vec<Chunk<V>> = vec![Chunk::default(); n_chunks];

    // For each chunk, for each lane, push new values
    for (index, &value) in std::iter::zip(indices, values) {
        let index = index.as_() - offset;

        let chunk = index / 1024;
        let lane = index % n_lanes;

        chunks[chunk].lanes[lane].push((index % 1024) as u16, value);
    }

    // Reshuffle the different containers into a single contiguous buffer each for indices/values
    let mut lane_offset = 0;
    let mut lane_offsets = buffer_mut![0u32];
    let mut indices_buffer = BufferMut::empty();
    let mut values_buffer = BufferMut::empty();
    for chunk in chunks {
        for lane in chunk.lanes {
            indices_buffer.extend_from_slice(&lane.indices);
            values_buffer.extend_from_slice(&lane.values);
            lane_offset += lane.len() as u32;
            lane_offsets.push(lane_offset);
        }
    }

    HostPatches {
        n_chunks,
        n_lanes,
        lane_offsets: lane_offsets.freeze(),
        indices: indices_buffer.freeze(),
        values: values_buffer.freeze(),
    }
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

        let transposed = transpose(
            patch_indices.as_slice(),
            patch_values.as_slice(),
            0,
            1024 * 5,
        );

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
