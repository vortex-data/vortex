// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use num_traits::AsPrimitive;
use vortex_buffer::{BitBuffer, get_bit};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable, ConstantArray};
use crate::compute::{TakeKernel, TakeKernelAdapter, fill_null};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for BoolVTable {
    fn take(&self, array: &BoolArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices_nulls_zeroed = match indices.validity_mask() {
            Mask::AllTrue(_) => indices.to_array(),
            Mask::AllFalse(_) => {
                return Ok(ConstantArray::new(
                    Scalar::null(array.dtype().as_nullable()),
                    indices.len(),
                )
                .into_array());
            }
            Mask::Values(_) => fill_null(indices, &Scalar::from(0).cast(indices.dtype())?)?,
        };
        let indices_nulls_zeroed = indices_nulls_zeroed.to_primitive();
        let buffer = match_each_integer_ptype!(indices_nulls_zeroed.ptype(), |I| {
            take_valid_indices(array.bit_buffer(), indices_nulls_zeroed.as_slice::<I>())
        });

        Ok(BoolArray::from_bit_buffer(buffer, array.validity().take(indices)?).to_array())
    }
}

register_kernel!(TakeKernelAdapter(BoolVTable).lift());

fn take_valid_indices<I: AsPrimitive<usize>>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
    // the overhead to convert to a Vec<bool>.
    if bools.len() <= 4096 {
        let bools = bools.iter().collect_vec();
        take_byte_bool(bools, indices)
    } else {
        take_bool(bools, indices)
    }
}

fn take_byte_bool<I: AsPrimitive<usize>>(bools: Vec<bool>, indices: &[I]) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |idx| {
        bools[unsafe { indices.get_unchecked(idx).as_() }]
    })
}

fn take_bool<I: AsPrimitive<usize>>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    // We dereference to underlying buffer to avoid access cost on every index.
    let buffer = bools.inner().as_ref();
    BitBuffer::collect_bool(indices.len(), |idx| {
        // SAFETY: we can take from the indices unchecked since collect_bool just iterates len.
        let idx = unsafe { indices.get_unchecked(idx).as_() };
        get_bit(buffer, bools.offset() + idx)
    })
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::arrays::BoolArray;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray as _, ToCanonical};

    #[test]
    fn take_nullable() {
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = take(reference.as_ref(), buffer![0, 3, 4].into_array().as_ref())
            .unwrap()
            .to_bool();
        assert_eq!(
            b.bit_buffer(),
            BoolArray::from_iter([Some(false), None, Some(false)]).bit_buffer()
        );

        let nullable_bool_dtype = DType::Bool(Nullability::Nullable);
        let all_invalid_indices = PrimitiveArray::from_option_iter([None::<u32>, None, None]);
        let b = take(reference.as_ref(), all_invalid_indices.as_ref()).unwrap();
        assert_eq!(b.dtype(), &nullable_bool_dtype);
        assert_eq!(b.scalar_at(0), Scalar::null(nullable_bool_dtype.clone()));
        assert_eq!(b.scalar_at(1), Scalar::null(nullable_bool_dtype.clone()));
        assert_eq!(b.scalar_at(2), Scalar::null(nullable_bool_dtype));
    }

    #[test]
    fn test_bool_array_take_with_null_out_of_bounds_indices() {
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).to_array()),
        );
        let actual = take(values.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(actual.scalar_at(0), Scalar::from(Some(false)));
        // position 3 is null
        assert_eq!(actual.scalar_at(1), Scalar::null_typed::<bool>());
        // the third index is null
        assert_eq!(actual.scalar_at(2), Scalar::null_typed::<bool>());
    }

    #[test]
    fn test_non_null_bool_array_take_with_null_out_of_bounds_indices() {
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).to_array()),
        );
        let actual = take(values.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(actual.scalar_at(0), Scalar::from(Some(false)));
        assert_eq!(actual.scalar_at(1), Scalar::from(Some(true)));
        // the third index is null
        assert_eq!(actual.scalar_at(2), Scalar::null_typed::<bool>());
    }

    #[test]
    fn test_bool_array_take_all_null_indices() {
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).to_array()),
        );
        let actual = take(values.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(actual.scalar_at(0), Scalar::null_typed::<bool>());
        assert_eq!(actual.scalar_at(1), Scalar::null_typed::<bool>());
        assert_eq!(actual.scalar_at(2), Scalar::null_typed::<bool>());
    }

    #[test]
    fn test_non_null_bool_array_take_all_null_indices() {
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).to_array()),
        );
        let actual = take(values.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(actual.scalar_at(0), Scalar::null_typed::<bool>());
        assert_eq!(actual.scalar_at(1), Scalar::null_typed::<bool>());
        assert_eq!(actual.scalar_at(2), Scalar::null_typed::<bool>());
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true, false]))]
    #[case(BoolArray::from_iter([true]))]
    fn test_take_bool_conformance(#[case] array: BoolArray) {
        test_take_conformance(array.as_ref());
    }
}
