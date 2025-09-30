// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::{FilterKernel, FilterKernelAdapter};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};

use crate::{SequenceArray, SequenceVTable};

impl FilterKernel for SequenceVTable {
    fn filter(&self, array: &SequenceArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = Validity::from(array.dtype().nullability());
        match_each_native_ptype!(array.ptype(), |P| {
            let mul = array.multiplier().as_primitive::<P>();
            let base = array.base().as_primitive::<P>();
            Ok(filter_impl(mul, base, selection_mask, validity))
        })
    }
}

register_kernel!(FilterKernelAdapter(SequenceVTable).lift());

fn filter_impl<T: NativePType>(mul: T, base: T, mask: &Mask, validity: Validity) -> ArrayRef {
    match mask.boolean_buffer() {
        AllOr::All | AllOr::None => unreachable!("Handled by entrypoint function"),
        AllOr::Some(mask) => {
            let mut buffer = BufferMut::<T>::with_capacity(mask.count_set_bits());
            buffer.extend(mask.set_indices().map(|idx| {
                let i = T::from_usize(idx).vortex_expect("all valid indices fit");
                base + i * mul
            }));
            PrimitiveArray::new(buffer.freeze(), validity).into_array()
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::compute::conformance::filter::{
        LARGE_SIZE, MEDIUM_SIZE, test_filter_conformance,
    };
    use vortex_dtype::Nullability;

    use crate::SequenceArray;

    #[rstest]
    #[case(SequenceArray::typed_new(0i32, 1, Nullability::NonNullable, 5).unwrap())]
    #[case(SequenceArray::typed_new(10i32, 2, Nullability::NonNullable, 5).unwrap())]
    #[case(SequenceArray::typed_new(100i32, -3, Nullability::NonNullable, 5).unwrap())]
    #[case(SequenceArray::typed_new(0i32, 1, Nullability::NonNullable, 1).unwrap())]
    #[case(SequenceArray::typed_new(0i32, 1, Nullability::NonNullable, MEDIUM_SIZE).unwrap())]
    #[case(SequenceArray::typed_new(0i32, 1, Nullability::NonNullable, LARGE_SIZE).unwrap())]
    #[case(SequenceArray::typed_new(0i64, 1, Nullability::NonNullable, 5).unwrap())]
    #[case(SequenceArray::typed_new(1000i64, 50, Nullability::NonNullable, 5).unwrap())]
    #[case(SequenceArray::typed_new(-100i64, 10, Nullability::NonNullable, MEDIUM_SIZE).unwrap())]
    #[case(SequenceArray::typed_new(0u32, 1, Nullability::NonNullable, 5).unwrap())]
    #[case(SequenceArray::typed_new(0u32, 5, Nullability::NonNullable, MEDIUM_SIZE).unwrap())]
    #[case(SequenceArray::typed_new(0u64, 1, Nullability::NonNullable, LARGE_SIZE).unwrap())]
    fn test_filter_sequence_conformance(#[case] array: SequenceArray) {
        test_filter_conformance(array.as_ref());
    }
}
