// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Sequence;

impl FilterKernel for Sequence {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = Validity::from(array.dtype().nullability());
        match_each_integer_ptype!(array.calculation_ptype(), |C| {
            let mul = array.multiplier().cast::<C>()?;
            let base = array.base().cast::<C>()?;
            match_each_integer_ptype!(array.dtype().as_ptype(), |O| {
                Ok(Some(filter_impl::<C, O>(mul, base, mask, validity)))
            })
        })
    }
}

fn filter_impl<C: IntegerPType, O: IntegerPType>(
    mul: C,
    base: C,
    mask: &Mask,
    validity: Validity,
) -> ArrayRef {
    let mask_values = mask
        .values()
        .vortex_expect("FilterKernel precondition: mask is Mask::Values");
    let mut buffer = BufferMut::<O>::with_capacity(mask_values.true_count());
    buffer.extend(mask_values.indices().iter().map(|&idx| {
        let i = C::from_usize(idx).vortex_expect("all valid indices fit");
        <O as NumCast>::from(base + i * mul)
            .vortex_expect("valid sequence values must fit output ptype")
    }));
    PrimitiveArray::new(buffer.freeze(), validity).into_array()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::compute::conformance::filter::LARGE_SIZE;
    use vortex_array::compute::conformance::filter::MEDIUM_SIZE;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::Nullability;

    use crate::Sequence;
    use crate::SequenceArray;

    #[rstest]
    #[case(Sequence::try_new_typed(0i32, 1, Nullability::NonNullable, 5).unwrap())]
    #[case(Sequence::try_new_typed(10i32, 2, Nullability::NonNullable, 5).unwrap())]
    #[case(Sequence::try_new_typed(100i32, -3, Nullability::NonNullable, 5).unwrap())]
    #[case(Sequence::try_new_typed(0i32, 1, Nullability::NonNullable, 1).unwrap())]
    #[case(Sequence::try_new_typed(0i32, 1, Nullability::NonNullable, MEDIUM_SIZE).unwrap())]
    #[case(Sequence::try_new_typed(0i32, 1, Nullability::NonNullable, LARGE_SIZE).unwrap())]
    #[case(Sequence::try_new_typed(0i64, 1, Nullability::NonNullable, 5).unwrap())]
    #[case(Sequence::try_new_typed(1000i64, 50, Nullability::NonNullable, 5).unwrap())]
    #[case(Sequence::try_new_typed(-100i64, 10, Nullability::NonNullable, MEDIUM_SIZE).unwrap())]
    #[case(Sequence::try_new_typed(0u32, 1, Nullability::NonNullable, 5).unwrap())]
    #[case(Sequence::try_new_typed(0u32, 5, Nullability::NonNullable, MEDIUM_SIZE).unwrap())]
    #[case(Sequence::try_new_typed(0u64, 1, Nullability::NonNullable, LARGE_SIZE).unwrap())]
    fn test_filter_sequence_conformance(#[case] array: SequenceArray) {
        test_filter_conformance(&array.into_array());
    }
}
