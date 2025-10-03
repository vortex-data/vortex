// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBufferBuilder;
use vortex_dtype::{IntegerPType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_mask::Mask;

use crate::arrays::{ListArray, ListVTable, PrimitiveArray};
use crate::builders::{ArrayBuilder, PrimitiveBuilder};
use crate::compute::{TakeKernel, TakeKernelAdapter, take};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, ToCanonical, register_kernel};

/// Take implementation for [`ListArray`].
///
/// Unlike `ListView`, `ListArray` must rebuild the elements array to maintain its invariant
/// that lists are stored contiguously and in-order (`offset[i+1] >= offset[i]`). Taking
/// non-contiguous indices would violate this requirement.
impl TakeKernel for ListVTable {
    fn take(&self, array: &ListArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive();
        let offsets = array.offsets().to_primitive();

        match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                _take::<I, O>(
                    array,
                    offsets.as_slice::<O>(),
                    &indices,
                    array.validity_mask(),
                    indices.validity_mask(),
                )
            })
        })
    }
}

register_kernel!(TakeKernelAdapter(ListVTable).lift());

fn _take<I: IntegerPType, O: IntegerPType>(
    array: &ListArray,
    offsets: &[O],
    indices_array: &PrimitiveArray,
    data_validity: Mask,
    indices_validity_mask: Mask,
) -> VortexResult<ArrayRef> {
    let indices: &[I] = indices_array.as_slice::<I>();

    if !indices_validity_mask.all_true() || !data_validity.all_true() {
        return _take_nullable::<I, O>(
            array,
            offsets,
            indices,
            data_validity,
            indices_validity_mask,
        );
    }

    let mut new_offsets = PrimitiveBuilder::with_capacity(Nullability::NonNullable, indices.len());
    let mut elements_to_take =
        PrimitiveBuilder::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = O::zero();
    new_offsets.append_zero();

    for &data_idx in indices {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let start = offsets[data_idx];
        let stop = offsets[data_idx + 1];

        // Annoyingly, we can't turn (start..end) into a range, so we're doing that manually.
        //
        // We could convert start and end to usize, but that would impose a potentially
        // harder constraint - now we don't care if they fit into usize as long as their
        // difference does.
        let additional = (stop - start).to_usize().unwrap_or_else(|| {
            vortex_panic!("Failed to convert range length to usize: {}", stop - start)
        });

        elements_to_take.ensure_capacity(elements_to_take.len() + additional);
        for i in 0..additional {
            elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < additional"));
        }
        current_offset += stop - start;
        new_offsets.append_value(current_offset);
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();

    let new_elements = take(array.elements(), elements_to_take.as_ref())?;

    Ok(ListArray::try_new(
        new_elements,
        new_offsets,
        indices_array
            .validity()
            .clone()
            .and(array.validity().clone()),
    )?
    .to_array())
}

fn _take_nullable<I: IntegerPType, O: IntegerPType>(
    array: &ListArray,
    offsets: &[O],
    indices: &[I],
    data_validity: Mask,
    indices_validity: Mask,
) -> VortexResult<ArrayRef> {
    let mut new_offsets = PrimitiveBuilder::with_capacity(Nullability::NonNullable, indices.len());

    // This will be the indices we push down to the child array to call `take` with.
    //
    // There are 2 things to note here:
    // - We do not know how many elements we need to take from our child since lists are variable
    //   size: thus we arbitrarily choose a capacity of `2 * # of indices`.
    // - The type of the primitive builder needs to fit the largest offset of the (parent)
    //   `ListArray`, so we make this `PrimitiveBuilder` generic over `O` (instead of `I`).
    let mut elements_to_take =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = O::zero();
    new_offsets.append_zero();

    let mut new_validity = BooleanBufferBuilder::new(indices.len());

    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            new_offsets.append_value(current_offset);
            new_validity.append(false);
            continue;
        }

        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        if data_validity.value(data_idx) {
            let start = offsets[data_idx];
            let stop = offsets[data_idx + 1];

            // See the note it the `take` on the reasoning
            let additional = (stop - start).to_usize().unwrap_or_else(|| {
                vortex_panic!("Failed to convert range length to usize: {}", stop - start)
            });

            elements_to_take.ensure_capacity(elements_to_take.len() + additional);
            for i in 0..additional {
                elements_to_take
                    .append_value(start + O::from_usize(i).vortex_expect("i < additional"));
            }
            current_offset += stop - start;
            new_offsets.append_value(current_offset);
            new_validity.append(true);
        } else {
            new_offsets.append_value(current_offset);
            new_validity.append(false);
        }
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();
    let new_elements = take(array.elements(), elements_to_take.as_ref())?;

    let new_validity: Validity = Validity::from(new_validity.finish());
    // data are indexes are nullable, so the final result is also nullable.

    Ok(ListArray::try_new(new_elements, new_offsets, new_validity)?.to_array())
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::arrays::list::ListArray;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray as _, ToCanonical};

    #[test]
    fn nullable_take() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3, 4, 4].into_array(),
            Validity::Array(BoolArray::from_iter(vec![true, true, false, true]).to_array()),
        )
        .unwrap()
        .to_array();

        let idx =
            PrimitiveArray::from_option_iter(vec![Some(0), None, Some(1), Some(3)]).to_array();

        let result = take(&list, &idx).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );

        let result = result.to_list();

        assert_eq!(result.len(), 4);

        let element_dtype: Arc<DType> = Arc::new(I32.into());

        assert!(result.is_valid(0));
        assert_eq!(
            result.scalar_at(0),
            Scalar::list(
                element_dtype.clone(),
                vec![0i32.into(), 5.into()],
                Nullability::Nullable
            )
        );

        assert!(result.is_invalid(1));

        assert!(result.is_valid(2));
        assert_eq!(
            result.scalar_at(2),
            Scalar::list(
                element_dtype.clone(),
                vec![3i32.into()],
                Nullability::Nullable
            )
        );

        assert!(result.is_valid(3));
        assert_eq!(
            result.scalar_at(3),
            Scalar::list(element_dtype, vec![], Nullability::Nullable)
        );
    }

    #[test]
    fn change_validity() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .to_array();

        let idx = PrimitiveArray::from_option_iter(vec![Some(0), Some(1), None]).to_array();
        // since idx is nullable, the final list will also be nullable

        let result = take(&list, &idx).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );
    }

    #[test]
    fn non_nullable_take() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3, 3, 4].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .to_array();

        let idx = buffer![1, 0, 2].into_array();

        let result = take(&list, &idx).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        let result = result.to_list();

        assert_eq!(result.len(), 3);

        let element_dtype: Arc<DType> = Arc::new(I32.into());

        assert!(result.is_valid(0));
        assert_eq!(
            result.scalar_at(0),
            Scalar::list(
                element_dtype.clone(),
                vec![3i32.into()],
                Nullability::NonNullable
            )
        );

        assert!(result.is_valid(1));
        assert_eq!(
            result.scalar_at(1),
            Scalar::list(
                element_dtype.clone(),
                vec![0i32.into(), 5.into()],
                Nullability::NonNullable
            )
        );

        assert!(result.is_valid(2));
        assert_eq!(
            result.scalar_at(2),
            Scalar::list(element_dtype, vec![], Nullability::NonNullable)
        );
    }

    #[test]
    fn test_take_empty_array() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .to_array();

        let idx = PrimitiveArray::empty::<i32>(Nullability::Nullable).to_array();

        let result = take(&list, &idx).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );
        assert_eq!(result.len(), 0,);
    }

    #[rstest]
    #[case(ListArray::try_new(
        buffer![0i32, 1, 2, 3, 4, 5].into_array(),
        buffer![0, 2, 3, 5, 5, 6].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case(ListArray::try_new(
        buffer![10i32, 20, 30, 40, 50].into_array(),
        buffer![0, 2, 3, 4, 5].into_array(),
        Validity::Array(BoolArray::from_iter(vec![true, false, true, true]).to_array()),
    ).unwrap())]
    #[case(ListArray::try_new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0, 0, 2, 2, 3].into_array(), // First and third are empty
        Validity::NonNullable,
    ).unwrap())]
    #[case(ListArray::try_new(
        buffer![42i32, 43].into_array(),
        buffer![0, 2].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case({
        let elements = buffer![0i32..200].into_array();
        let mut offsets = vec![0u64];
        for i in 1..=50 {
            offsets.push(offsets[i - 1] + (i as u64 % 5)); // Variable length lists
        }
        ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).to_array(),
            Validity::NonNullable,
        ).unwrap()
    })]
    #[case(ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]).to_array(),
        buffer![0, 2, 3, 5].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    fn test_take_list_conformance(#[case] list: ListArray) {
        test_take_conformance(list.as_ref());
    }
}
