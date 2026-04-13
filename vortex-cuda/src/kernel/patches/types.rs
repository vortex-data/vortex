// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of lane-wise patches instead of linear patches. This layout for exception
//! patching enables fully parallel GPU execution, as outlined by Hepkema et al. in
//! "G-ALP: Rethinking Light-weight Encodings for GPUs" <https://doi.org/10.1145/3736227.3736242>

use vortex::array::Canonical;
use vortex::array::buffer::BufferHandle;
use vortex::array::dtype::IntegerPType;
use vortex::array::dtype::NativePType;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
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

    /// Apply the patches on top of the other buffer.
    #[cfg(test)]
    fn apply(&self, output: &mut BufferMut<V>) {
        for chunk in 0..self.n_chunks {
            for lane in 0..self.n_lanes {
                let patches = self.patches(chunk, lane);
                for (&index, &value) in std::iter::zip(patches.indices, patches.values) {
                    let full_index = chunk * 1024 + (index as usize);
                    output[full_index] = value;
                }
            }
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

/// Transpose a set of patches from the default sorted layout into the data parallel layout.
#[expect(clippy::cognitive_complexity)]
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

#[expect(clippy::cast_possible_truncation)]
fn transpose<I: IntegerPType, V: NativePType>(
    indices_in: &[I],
    values_in: &[V],
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

    // We know upfront how many indices and values we'll have.
    let mut indices_buffer = BufferMut::with_capacity(indices_in.len());
    let mut values_buffer = BufferMut::with_capacity(values_in.len());

    // number of patches in each chunk.
    let mut lane_offsets: BufferMut<u32> = BufferMut::zeroed(n_chunks * n_lanes + 1);

    // Scan the index/values once to get chunk/lane counts
    for index in indices_in {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane + 1] += 1;
    }

    // Prefix-sum sizes -> offsets
    for index in 1..lane_offsets.len() {
        lane_offsets[index] += lane_offsets[index - 1];
    }

    // Loop over patches, writing them to final positions
    let indices_out = indices_buffer.spare_capacity_mut();
    let values_out = values_buffer.spare_capacity_mut();
    for (index, &value) in std::iter::zip(indices_in, values_in) {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        let position = &mut lane_offsets[chunk * n_lanes + lane];
        indices_out[*position as usize].write((index % 1024) as u16);
        values_out[*position as usize].write(value);
        *position += 1;
    }

    // SAFETY: we know there are exactly indices_in.len() indices/values, and we just
    //  set them to the appropriate values in the loop above.
    unsafe {
        indices_buffer.set_len(indices_in.len());
        values_buffer.set_len(values_in.len());
    }

    // Now, pass over all the indices and values again and subtract out the position increments.
    for index in indices_in {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane] -= 1;
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
    use vortex::array::ExecutionCtx;
    use vortex::buffer::BufferMut;
    use vortex::buffer::buffer;
    use vortex::buffer::buffer_mut;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::patches::Patches;
    use vortex_error::VortexResult;

    use crate::kernel::patches::types::transpose;

    #[crate::test]
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

    #[test]
    #[expect(clippy::cast_possible_truncation)]
    fn test_transpose_complex() -> VortexResult<()> {
        test_case(1024, 0, &[0], &[0f32])?;
        test_case(512, 512, &[512, 513, 514], &[10i8, 20, 30])?;
        test_case(10_000, 100, &[500, 1_000, 1_001, 1_002], &[1i16, 2, 3, 4])?;

        for len in (1..4096).step_by(10) {
            let offset = len / 2;

            let indices: Vec<u32> = (offset..len).map(|x| x as u32).collect();

            test_case(len, offset, &indices, &indices)?;
        }

        Ok(())
    }

    fn test_case<V: NativePType>(
        len: usize,
        offset: usize,
        patch_indices: &[u32],
        patch_values: &[V],
    ) -> VortexResult<()> {
        let mut data = buffer_mut![V::default(); len];
        let array = PrimitiveArray::from_iter(data.iter().copied());

        let patches = Patches::new(
            len,
            offset,
            PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array(),
            PrimitiveArray::from_iter(patch_values.iter().copied()).into_array(),
            None,
        )?;

        // Verify that the outputs match between Patches and transpose_patches().
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());
        let patched = array.patch(&patches, &mut ctx)?.into_array();

        let transposed = transpose(patch_indices, patch_values, offset, len);
        transposed.apply(&mut data);

        let patched_transposed = data.freeze().into_array();

        assert_arrays_eq!(patched, patched_transposed);

        Ok(())
    }
}
