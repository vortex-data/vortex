// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rustc_hash::FxHashMap;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::arrays::dict::TakeExecute;
use crate::arrays::primitive::PrimitiveArrayParts;
use crate::arrays::{Patched, PrimitiveArray};
use crate::dtype::{IntegerPType, NativePType};
use crate::{ArrayRef, DynArray, IntoArray, match_each_native_ptype};
use crate::{ExecutionCtx, match_each_unsigned_integer_ptype};

impl TakeExecute for Patched {
    fn take(
        array: &Self::Array,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Perform take on the inner array, including the placeholders.
        let inner = array
            .inner
            .take(indices.clone())?
            .execute::<PrimitiveArray>(ctx)?;

        let PrimitiveArrayParts {
            buffer,
            validity,
            ptype,
        } = inner.into_parts();

        let indices_ptype = indices.dtype().as_ptype();

        match_each_unsigned_integer_ptype!(indices_ptype, |I| {
            match_each_native_ptype!(ptype, |V| {
                let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
                let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
                take_map(
                    output.as_mut(),
                    indices.as_slice::<I>(),
                    array.offset,
                    array.len,
                    array.n_chunks,
                    array.n_lanes,
                    array.lane_offsets.as_host().reinterpret::<u32>(),
                    array.indices.as_host().reinterpret::<u16>(),
                    array.values.as_host().reinterpret::<V>(),
                );

                // SAFETY: output and validity still have same length after take_map returns.
                unsafe {
                    return Ok(Some(
                        PrimitiveArray::new_unchecked(output.freeze(), validity).into_array(),
                    ));
                }
            })
        });
    }
}

/// Take patches for the given `indices` and apply them onto an `output` using a hash map.
///
/// First, builds a hashmap from index to patch value, then uses the hashmap in a loop to collect
/// the values.
fn take_map<I: IntegerPType, V: NativePType>(
    output: &mut [V],
    indices: &[I],
    offset: usize,
    len: usize,
    n_chunks: usize,
    n_lanes: usize,
    lane_offsets: &[u32],
    patch_index: &[u16],
    patch_value: &[V],
) {
    // Build a hashmap of patch_index -> values.
    let mut index_map = FxHashMap::with_capacity(indices.len());
    for chunk in 0..n_chunks {
        for lane in 0..n_lanes {
            let [lane_start, lane_end] = lane_offsets[chunk * n_lanes + lane..][..2];
            for i in lane_start..lane_end {
                let patch_idx = patch_index[i as usize];
                let patch_value = patch_value[i as usize];

                let index = chunk * 1024 + patch_idx as usize;
                if index >= offset && index < offset + len {
                    index_map.insert(index, patch_value);
                }
            }
        }
    }

    // Now, iterate the take indices using the prebuilt hashmap.
    // Undefined/null indices will miss the hash map, which we can ignore.
    for index in indices {
        let index = index.as_();
        if let Some(&patch_value) = index_map.get(&index) {
            output[index] = patch_value;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_take() {
        // Patch some values here instead.
    }
}
