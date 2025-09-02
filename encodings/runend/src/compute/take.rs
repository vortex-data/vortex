// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::{AsPrimitive, NumCast};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{Array, ArrayRef, ToCanonical, register_kernel};
use vortex_buffer::Buffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexResult, vortex_bail};

use crate::{RunEndArray, RunEndVTable};

impl TakeKernel for RunEndVTable {
    #[allow(clippy::cast_possible_truncation)]
    fn take(&self, array: &RunEndArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let primitive_indices = indices.to_primitive();

        let checked_indices = match_each_integer_ptype!(primitive_indices.ptype(), |P| {
            primitive_indices
                .as_slice::<P>()
                .iter()
                .copied()
                .map(|idx| {
                    let usize_idx = idx as usize;
                    if usize_idx >= array.len() {
                        vortex_bail!(OutOfBounds: usize_idx, 0, array.len());
                    }
                    Ok(usize_idx)
                })
                .collect::<VortexResult<Vec<_>>>()?
        });

        take_indices_unchecked(array, &checked_indices, primitive_indices.validity())
    }
}

register_kernel!(TakeKernelAdapter(RunEndVTable).lift());

/// Perform a take operation on a RunEndArray by binary searching for each of the indices.
pub fn take_indices_unchecked<T: AsPrimitive<usize>>(
    array: &RunEndArray,
    indices: &[T],
    validity: &Validity,
) -> VortexResult<ArrayRef> {
    let ends = array.ends().to_primitive();
    let ends_len = ends.len();

    // TODO(joe): use the validity mask to skip search sorted.
    let physical_indices = match_each_integer_ptype!(ends.ptype(), |I| {
        let end_slices = ends.as_slice::<I>();
        let buffer = Buffer::from_trusted_len_iter(
            indices
                .iter()
                .map(|idx| idx.as_() + array.offset())
                .map(|idx| {
                    match <I as NumCast>::from(idx) {
                        Some(idx) => end_slices.search_sorted(&idx, SearchSortedSide::Right),
                        None => {
                            // The idx is too large for I, therefore it's out of bounds.
                            SearchResult::NotFound(ends_len)
                        }
                    }
                })
                .map(|result| result.to_ends_index(ends_len) as u64),
        );

        PrimitiveArray::new(buffer, validity.clone())
    });

    take(array.values(), physical_indices.as_ref())
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::compute::take;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::{Scalar, ScalarValue};

    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn ree_take() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from_iter([9, 8, 1, 3]).as_ref(),
        )
        .unwrap();
        assert_eq!(taken.to_primitive().as_slice::<i32>(), &[5, 5, 1, 4]);
    }

    #[test]
    fn ree_take_end() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from_iter([11]).as_ref(),
        )
        .unwrap();
        assert_eq!(taken.to_primitive().as_slice::<i32>(), &[5]);
    }

    #[test]
    #[should_panic]
    fn ree_take_out_of_bounds() {
        take(
            ree_array().as_ref(),
            PrimitiveArray::from_iter([12]).as_ref(),
        )
        .unwrap();
    }

    #[test]
    fn sliced_take() {
        let sliced = ree_array().slice(4..9);
        let taken = take(
            sliced.as_ref(),
            PrimitiveArray::from_iter([1, 3, 4]).as_ref(),
        )
        .unwrap();

        assert_eq!(taken.len(), 3);
        assert_eq!(taken.scalar_at(0), 4.into());
        assert_eq!(taken.scalar_at(1), 2.into());
        assert_eq!(taken.scalar_at(2), 5.into());
    }

    #[test]
    fn ree_take_nullable() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from_option_iter([Some(1), None]).as_ref(),
        )
        .unwrap();

        assert_eq!(
            taken.scalar_at(0),
            Scalar::new(
                DType::Primitive(PType::I32, Nullability::Nullable),
                ScalarValue::from(1i32)
            )
        );
        assert_eq!(
            taken.scalar_at(1),
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
        );
    }

    #[rstest]
    #[case(ree_array())]
    #[case(RunEndArray::encode(
        PrimitiveArray::from_iter([1u8, 1, 2, 2, 2, 3, 3, 3, 3, 4]).into_array(),
    ).unwrap())]
    #[case(RunEndArray::encode(
        PrimitiveArray::from_option_iter([
            Some(10),
            Some(10),
            None,
            None,
            Some(20),
            Some(20),
            Some(20),
        ])
        .into_array(),
    ).unwrap())]
    #[case(RunEndArray::encode(PrimitiveArray::from_iter([42i32, 42, 42, 42, 42]).into_array())
        .unwrap())]
    #[case(RunEndArray::encode(
        PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]).into_array(),
    ).unwrap())]
    #[case({
        let mut values = Vec::new();
        for i in 0..20 {
            for _ in 0..=i {
                values.push(i);
            }
        }
        RunEndArray::encode(PrimitiveArray::from_iter(values).into_array()).unwrap()
    })]
    fn test_take_runend_conformance(#[case] array: RunEndArray) {
        test_take_conformance(array.as_ref());
    }

    #[rstest]
    #[case(ree_array().slice(3..6))]
    #[case({
        let array = RunEndArray::encode(
            PrimitiveArray::from_iter([1i32, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).into_array(),
        )
        .unwrap();
        array.slice(2..8)
    })]
    fn test_take_sliced_runend_conformance(#[case] sliced: ArrayRef) {
        test_take_conformance(sliced.as_ref());
    }
}
