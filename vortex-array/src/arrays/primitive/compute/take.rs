use std::mem::{MaybeUninit, transmute};
use std::simd;

use multiversion::multiversion;
use num_traits::AsPrimitive;
use simd::num::SimdUint;
use vortex_buffer::{Alignment, Buffer, BufferMut};
use vortex_dtype::{
    DType, NativePType, PType, match_each_native_simd_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{TakeKernel, TakeKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

// SIMD types larger than the SIMD register size are beneficial for
// performance as this leads to better instruction level parallelism.
const SIMD_WIDTH: usize = 64;

impl TakeKernel for PrimitiveVTable {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &PrimitiveArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let unsigned_indices = match indices.dtype() {
            DType::Primitive(p, n) => {
                if p.is_unsigned_int() {
                    indices.to_primitive()?
                } else {
                    // This will fail if all values cannot be converted to unsigned
                    cast(indices, &DType::Primitive(p.to_unsigned(), *n))?.to_primitive()?
                }
            }
            _ => vortex_bail!("Invalid indices dtype: {}", indices.dtype()),
        };

        let validity = array.validity().take(unsigned_indices.as_ref())?;
        if array.ptype() == PType::F16 {
            // Special handling for f16 to treat as opaque u16
            let decoded = match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                take_primitive_simd::<C, u16, SIMD_WIDTH>(
                    unsigned_indices.as_slice(),
                    array.reinterpret_cast(PType::U16).as_slice(),
                )
            });
            Ok(PrimitiveArray::new(decoded, validity)
                .reinterpret_cast(PType::F16)
                .into_array())
        } else {
            match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                match_each_native_simd_ptype!(array.ptype(), |V| {
                    let decoded = take_primitive_simd::<C, V, SIMD_WIDTH>(
                        unsigned_indices.as_slice(),
                        array.as_slice(),
                    );
                    Ok(PrimitiveArray::new(decoded, validity).into_array())
                })
            })
        }
    }
}

register_kernel!(TakeKernelAdapter(PrimitiveVTable).lift());

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
///
/// # Returns
/// A `PrimitiveArray` containing the gathered values where each index has been replaced with
/// the corresponding value from the source array.
#[multiversion(targets("x86_64+avx2", "x86_64+avx", "aarch64+neon"))]
fn take_primitive_simd<I, V, const LANE_COUNT: usize>(indices: &[I], values: &[V]) -> Buffer<V>
where
    I: simd::SimdElement + AsPrimitive<usize>,
    V: simd::SimdElement + NativePType,
    simd::LaneCount<LANE_COUNT>: simd::SupportedLaneCount,
    simd::Simd<I, LANE_COUNT>: SimdUint<Cast<usize> = simd::Simd<usize, LANE_COUNT>>,
{
    let indices_len = indices.len();

    let mut buffer = BufferMut::<V>::with_capacity_aligned(
        indices_len,
        Alignment::of::<simd::Simd<V, LANE_COUNT>>(),
    );

    let buf_slice = buffer.spare_capacity_mut();

    for chunk_idx in 0..(indices_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let mask = simd::Mask::from_bitmask(u64::MAX);
        let codes_chunk = simd::Simd::<I, LANE_COUNT>::from_slice(&indices[offset..]);

        let selection = simd::Simd::gather_select(
            values,
            mask,
            codes_chunk.cast::<usize>(),
            simd::Simd::<V, LANE_COUNT>::default(),
        );

        unsafe {
            selection.store_select_unchecked(
                transmute::<&mut [MaybeUninit<V>], &mut [V]>(&mut buf_slice[offset..][..64]),
                mask.cast(),
            );
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

    buffer.freeze()
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::arrays::primitive::compute::take::take_primitive_simd;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_take() {
        let a = vec![1i32, 2, 3, 4, 5];
        let result = take_primitive_simd::<u8, i32, 64>(&[0, 0, 4, 2], &a);
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

    #[test]
    fn test_take_out_of_bounds() {
        let indices = vec![2_000_000u32; 64];
        let values = vec![1i32];

        let result = take_primitive_simd::<u32, i32, 64>(&indices, &values);
        assert_eq!(result.as_slice(), [0i32; 64]);
    }
}
