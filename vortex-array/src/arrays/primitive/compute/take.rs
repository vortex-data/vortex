use std::simd;

use num_traits::AsPrimitive;
use simd::num::SimdUint;
use vortex_buffer::{Alignment, Buffer, BufferMut};
use vortex_dtype::{
    NativePType, Nullability, PType, match_each_integer_ptype, match_each_native_ptype,
    match_each_native_simd_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexResult, vortex_err};
use vortex_mask::Mask;

use crate::arrays::PrimitiveEncoding;
use crate::arrays::primitive::PrimitiveArray;
use crate::builders::{ArrayBuilder, PrimitiveBuilder};
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ToCanonical};

impl TakeFn<&PrimitiveArray> for PrimitiveEncoding {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &PrimitiveArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;
        let validity = array.validity().take(&indices)?;

        if array.ptype() != PType::F16
            && indices.dtype().is_unsigned_int()
            && indices.all_valid()?
            && array.all_valid()?
        {
            // TODO(alex): handle nullable codes & values
            match_each_unsigned_integer_ptype!(indices.ptype(), |$C| {
                match_each_native_simd_ptype!(array.ptype(), |$V| {
                    // SIMD types larger than the SIMD register size are beneficial for
                    // performance as this leads to better instruction level parallelism.
                    let decoded = take_primitive_simd::<$C, $V, 64>(
                        indices.as_slice(),
                        array.as_slice(),
                        array.dtype().nullability() | indices.dtype().nullability(),
                    );

                    return Ok(decoded.into_array()) as VortexResult<ArrayRef>;
                })
            });
        }

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive(array.as_slice::<$T>(), indices.as_slice::<$I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }

    fn take_into(
        &self,
        array: &PrimitiveArray,
        indices: &dyn Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        let indices = indices.to_primitive()?;
        // TODO(joe): impl take over mask and use `Array::validity_mask`, instead of `validity()`.
        let validity = array.validity().take(&indices)?;
        let mask = validity.to_logical(indices.len())?;

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                take_into_impl::<$T, $I>(array, &indices, mask, builder)
            })
        })
    }
}

fn take_into_impl<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &PrimitiveArray,
    indices: &PrimitiveArray,
    mask: Mask,
    builder: &mut dyn ArrayBuilder,
) -> VortexResult<()> {
    assert_eq!(indices.len(), mask.len());

    let array = array.as_slice::<T>();
    let indices = indices.as_slice::<I>();
    let builder = builder
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .ok_or_else(|| {
            vortex_err!(
                "Failed to downcast builder to PrimitiveBuilder<{}>",
                T::PTYPE
            )
        })?;
    builder.extend_with_iterator(indices.iter().map(|idx| array[idx.as_()]), mask);
    Ok(())
}

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
fn take_primitive_simd<I, V, const LANE_COUNT: usize>(
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

    let mut buffer = BufferMut::<V>::with_capacity_aligned(
        indices_len,
        Alignment::of::<simd::Simd<V, LANE_COUNT>>(),
    );

    let buf_slice = buffer.spare_capacity_mut();

    for chunk_idx in 0..(indices_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let mask = simd::Mask::from_bitmask(u64::MAX);
        let codes_chunk = simd::Simd::<I, LANE_COUNT>::from_slice(&indices[offset..]);

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
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::primitive::compute::take::take_primitive;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::builders::{ArrayBuilder as _, PrimitiveBuilder};
    use crate::compute::{scalar_at, take, take_into};
    use crate::validity::Validity;

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
        let actual = take(&values, &indices).unwrap();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::from(Some(1)));
        // position 3 is null
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::null_typed::<i32>());
        // the third index is null
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<i32>());
    }

    #[test]
    fn test_take_into() {
        let values = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
        let all_valid_indices = PrimitiveArray::new(
            buffer![0, 3, 4],
            Validity::Array(BoolArray::from_iter([true, true, true]).into_array()),
        );
        let mut builder = PrimitiveBuilder::<i32>::new(Nullability::Nullable);
        take_into(&values, &all_valid_indices, &mut builder).unwrap();
        let actual = builder.finish();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::from(Some(1)));
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::from(Some(4)));
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::from(Some(5)));

        let mixed_valid_indices = PrimitiveArray::new(
            buffer![0, 3, 4],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let mut builder = PrimitiveBuilder::<i32>::new(Nullability::Nullable);
        take_into(&values, &mixed_valid_indices, &mut builder).unwrap();
        let actual = builder.finish();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::from(Some(1)));
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::from(Some(4)));
        // the third index is null
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<i32>());

        let all_invalid_indices = PrimitiveArray::new(
            buffer![0, 3, 4],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let mut builder = PrimitiveBuilder::<i32>::new(Nullability::Nullable);
        take_into(&values, &all_invalid_indices, &mut builder).unwrap();
        let actual = builder.finish();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::null_typed::<i32>());
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::null_typed::<i32>());
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<i32>());

        let non_null_indices = PrimitiveArray::new(buffer![0, 3, 4], Validity::NonNullable);
        let mut builder = PrimitiveBuilder::<i32>::new(Nullability::NonNullable);
        take_into(&values, &non_null_indices, &mut builder).unwrap();
        let actual = builder.finish();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::from(1));
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::from(4));
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::from(5));
    }
}
