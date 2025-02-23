use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType};
use vortex_error::{vortex_err, VortexResult};
use vortex_mask::Mask;

use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::PrimitiveEncoding;
use crate::builders::{ArrayBuilder, PrimitiveBuilder};
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ToCanonical};

impl TakeFn<&PrimitiveArray> for PrimitiveEncoding {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &PrimitiveArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;
        let validity = array.validity().take(&indices)?;

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive(array.as_slice::<$T>(), indices.as_slice::<$I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }

    unsafe fn take_unchecked(
        &self,
        array: &PrimitiveArray,
        indices: &dyn Array,
    ) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;
        let validity = unsafe { array.validity().take_unchecked(&indices)? };

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive_unchecked(array.as_slice::<$T>(), indices.as_slice::<$I>());
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

// We pass a Vec<I> in case we're T == u64.
// In which case, Rust should reuse the same Vec<u64> the result.
unsafe fn take_primitive_unchecked<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Buffer<T> {
    indices
        .iter()
        .map(|idx| unsafe { *array.get_unchecked(idx.as_()) })
        .collect()
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
