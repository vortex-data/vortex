use std::simd;

use num_traits::AsPrimitive;
use simd::num::SimdUint;
use vortex_buffer::{Alignment, Buffer, BufferMut};
use vortex_dtype::{
    NativePType, Nullability, PType, match_each_integer_ptype, match_each_native_ptype,
    match_each_native_simd_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for PrimitiveVTable {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &PrimitiveArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;

        if array.ptype() != PType::F16
            && indices.dtype().is_unsigned_int()
            && indices.all_valid()?
            && array.all_valid()?
        {
            // TODO(alex): handle nullable codes & values
            match_each_unsigned_integer_ptype!(indices.ptype(), |C| {
                match_each_native_simd_ptype!(array.ptype(), |V| {
                    // Check Indices in-bounds
                    let max: usize = indices
                        .statistics()
                        .compute_max::<C>()
                        .vortex_expect("all_valid indices must have max")
                        .as_();
                    if max >= array.len() {
                        vortex_bail!(
                            "take indices with max={} invalid for array of length {}",
                            max,
                            array.len()
                        );
                    }

                    // SIMD types larger than the SIMD register size are beneficial for
                    // performance as this leads to better instruction level parallelism.
                    // SAFETY: we have checked all indices to be in-bounds above.
                    let decoded = unsafe {
                        take_primitive_simd::<C, V, 64>(
                            indices.as_slice(),
                            array.as_slice(),
                            array.dtype().nullability() | indices.dtype().nullability(),
                        )
                    };

                    return Ok(decoded.into_array()) as VortexResult<ArrayRef>;
                })
            });
        }

        // TODO(joe): if the true count of take indices validity is low, only take array values with
        // valid indices.
        let validity = array.validity().take(indices.as_ref())?;
        match_each_native_ptype!(array.ptype(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let values = take_primitive(array.as_slice::<T>(), indices.as_slice::<I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }
}

register_kernel!(TakeKernelAdapter(PrimitiveVTable).lift());

fn take_primitive<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Buffer<T> {
    indices.iter().map(|idx| array[idx.as_()]).collect()
}

/// Takes elements from an array using SIMD indexing.
///
/// # Type Parameters
/// * `C` - Index type
/// * `V` - Value type
/// * `LANE_COUNT` - Number of SIMD lanes to process in parallel
///
/// # Parameters
/// * `indices` - Indices to gather values from
/// * `values` - Source values to index
/// * `nullability` - Nullability of the resulting array
///
/// # Returns
/// A `PrimitiveArray` containing the gathered values where each index has been replaced with
/// the corresponding value from the source array.
///
/// # Safety
///
/// This function circumvents bounds checks when accessing the `values` slice, so all `indices`
/// must be valid or it will result in out of bounds memory access.
unsafe fn take_primitive_simd<I, V, const LANE_COUNT: usize>(
    indices: &[I],
    values: &[V],
    nullability: Nullability,
) -> PrimitiveArray
where
    I: simd::SimdElement + AsPrimitive<usize>,
    V: simd::SimdElement + NativePType,
    simd::LaneCount<LANE_COUNT>: simd::SupportedLaneCount,
    simd::Simd<I, LANE_COUNT>: SimdUint<Cast<usize> = simd::Simd<usize, LANE_COUNT>>,
{
    let indices_len = indices.len();
    // Check all indices before gathering.
    for &idx in indices.iter() {
        let idx: usize = idx.as_();
        if idx >= values.len() {
            vortex_panic!(
                "cannot take with index {} on array of length {}",
                idx,
                values.len()
            );
        }
    }

    let mut buffer = BufferMut::<V>::with_capacity_aligned(
        indices_len,
        Alignment::of::<simd::Simd<V, LANE_COUNT>>(),
    );

    let buf_slice = buffer.spare_capacity_mut();

    for chunk_idx in 0..(indices_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let mask = simd::Mask::from_bitmask(u64::MAX);
        let codes_chunk = simd::Simd::<I, LANE_COUNT>::from_slice(&indices[offset..]);

        // SAFETY: we have verified that all indices are in-bounds, so gather will not
        // read invalid memory.
        unsafe {
            let selection = simd::Simd::gather_select_unchecked(
                values,
                mask,
                codes_chunk.cast::<usize>(),
                simd::Simd::<V, LANE_COUNT>::default(),
            );

            selection.store_select_ptr(buf_slice.as_mut_ptr().add(offset) as *mut V, mask.cast());
        }
    }

    for idx in ((indices_len / LANE_COUNT) * LANE_COUNT)..indices_len {
        unsafe {
            buf_slice
                .get_unchecked_mut(idx)
                .write(values[indices[idx].as_()]);
        }
    }

    unsafe {
        buffer.set_len(indices_len);
    }

    PrimitiveArray::new(buffer.freeze(), nullability.into())
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::arrays::primitive::compute::take::take_primitive;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_take() {
        let a = vec![1i32, 2, 3, 4, 5];
        let result = take_primitive(&a, &[0, 0, 4, 2]);
        assert_eq!(result.as_slice(), &[1i32, 1, 5, 3]);
    }

    #[test]
    fn test_take_with_null_indices() {
        let values = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5],
            Validity::Array(BoolArray::from_iter([true, true, false, false, true]).into_array()),
        );
        let indices = PrimitiveArray::new(
            buffer![0, 3, 4],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = take(values.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(actual.scalar_at(0).unwrap(), Scalar::from(Some(1)));
        // position 3 is null
        assert_eq!(actual.scalar_at(1).unwrap(), Scalar::null_typed::<i32>());
        // the third index is null
        assert_eq!(actual.scalar_at(2).unwrap(), Scalar::null_typed::<i32>());
    }
}
