// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::list::ListArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_smallest_offset_type;

// TODO(connor)[ListView]: Re-revert to the version where we simply convert to a `ListView` and call
// the `ListView::take` compute function once `ListView` is more stable.

impl TakeExecute for List {
    /// Take implementation for [`ListArray`].
    ///
    /// Unlike `ListView`, `ListArray` must rebuild the elements array to maintain its invariant
    /// that lists are stored contiguously and in-order (`offset[i+1] >= offset[i]`). Taking
    /// non-contiguous indices would violate this requirement.
    #[expect(clippy::cognitive_complexity)]
    fn take(
        array: ArrayView<'_, List>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        // This is an over-approximation of the total number of elements in the resulting array.
        let total_approx = array.elements().len().saturating_mul(indices.len());

        match_each_integer_ptype!(array.offsets().dtype().as_ptype(), |O| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                match_smallest_offset_type!(total_approx, |OutputOffsetType| {
                    {
                        let indices = indices.as_view();
                        _take::<I, O, OutputOffsetType>(array, indices, ctx).map(Some)
                    }
                })
            })
        })
    }
}

fn _take<I: IntegerPType, O: IntegerPType, OutputOffsetType: IntegerPType>(
    array: ArrayView<'_, List>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let data_validity = array.list_validity().to_mask(array.as_ref().len(), ctx)?;
    let indices_validity = indices_array
        .validity()
        .vortex_expect("Failed to compute validity mask")
        .to_mask(indices_array.as_ref().len(), ctx)?;

    if !indices_validity.all_true() || !data_validity.all_true() {
        return _take_nullable::<I, O, OutputOffsetType>(array, indices_array, ctx);
    }

    let offsets_array = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let offsets: &[O] = offsets_array.as_slice();
    let indices: &[I] = indices_array.as_slice();

    let mut new_offsets = PrimitiveBuilder::<OutputOffsetType>::with_capacity(
        Nullability::NonNullable,
        indices.len(),
    );
    let mut elements_to_take =
        PrimitiveBuilder::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = OutputOffsetType::zero();
    new_offsets.append_zero();

    for &data_idx in indices {
        let data_idx: usize = data_idx.as_();

        let start = offsets[data_idx];
        let stop = offsets[data_idx + 1];

        // Annoyingly, we can't turn (start..end) into a range, so we're doing that manually.
        //
        // We could convert start and end to usize, but that would impose a potentially
        // harder constraint - now we don't care if they fit into usize as long as their
        // difference does.
        let additional: usize = (stop - start).as_();

        // TODO(0ax1): optimize this
        elements_to_take.reserve_exact(additional);
        for i in 0..additional {
            elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < additional"));
        }
        current_offset +=
            OutputOffsetType::from_usize((stop - start).as_()).vortex_expect("offset conversion");
        new_offsets.append_value(current_offset);
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();

    let new_elements = array.elements().take(elements_to_take)?;

    Ok(ListArray::try_new(
        new_elements,
        new_offsets,
        array.validity()?.take(indices_array.array())?,
    )?
    .into_array())
}

fn _take_nullable<I: IntegerPType, O: IntegerPType, OutputOffsetType: IntegerPType>(
    array: ArrayView<'_, List>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offsets_array = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let offsets: &[O] = offsets_array.as_slice();
    let indices: &[I] = indices_array.as_slice();
    let data_validity = array.list_validity().to_mask(array.as_ref().len(), ctx)?;
    let indices_validity = indices_array
        .validity()
        .vortex_expect("Failed to compute validity mask")
        .to_mask(indices_array.as_ref().len(), ctx)?;

    let mut new_offsets = PrimitiveBuilder::<OutputOffsetType>::with_capacity(
        Nullability::NonNullable,
        indices.len(),
    );

    // This will be the indices we push down to the child array to call `take` with.
    //
    // There are 2 things to note here:
    // - We do not know how many elements we need to take from our child since lists are variable
    //   size: thus we arbitrarily choose a capacity of `2 * # of indices`.
    // - The type of the primitive builder needs to fit the largest offset of the (parent)
    //   `ListArray`, so we make this `PrimitiveBuilder` generic over `O` (instead of `I`).
    let mut elements_to_take =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = OutputOffsetType::zero();
    new_offsets.append_zero();

    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            new_offsets.append_value(current_offset);
            continue;
        }

        let data_idx: usize = data_idx.as_();

        if !data_validity.value(data_idx) {
            new_offsets.append_value(current_offset);
            continue;
        }

        let start = offsets[data_idx];
        let stop = offsets[data_idx + 1];

        // See the note in `_take` on the reasoning.
        let additional: usize = (stop - start).as_();

        elements_to_take.reserve_exact(additional);
        for i in 0..additional {
            elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < additional"));
        }
        current_offset +=
            OutputOffsetType::from_usize((stop - start).as_()).vortex_expect("offset conversion");
        new_offsets.append_value(current_offset);
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();
    let new_elements = array.elements().take(elements_to_take)?;

    Ok(ListArray::try_new(
        new_elements,
        new_offsets,
        array.validity()?.take(indices_array.array())?,
    )?
    .into_array())
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray as _;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn nullable_take() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3, 4, 4].into_array(),
            Validity::Array(BoolArray::from_iter(vec![true, true, false, true]).into_array()),
        )
        .unwrap()
        .into_array();

        let idx =
            PrimitiveArray::from_option_iter(vec![Some(0), None, Some(1), Some(3)]).into_array();

        let result = list.take(idx).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );

        #[expect(deprecated)]
        let result = result.to_listview();

        assert_eq!(result.len(), 4);

        let element_dtype: Arc<DType> = Arc::new(I32.into());

        assert!(
            result
                .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![0i32.into(), 5.into()],
                Nullability::Nullable
            )
        );

        assert!(
            result
                .is_invalid(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );

        assert!(
            result
                .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![3i32.into()],
                Nullability::Nullable
            )
        );

        assert!(
            result
                .is_valid(3, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
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
        .into_array();

        let idx = PrimitiveArray::from_option_iter(vec![Some(0), Some(1), None]).into_array();
        // since idx is nullable, the final list will also be nullable

        let result = list.take(idx).unwrap();
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
        .into_array();

        let idx = buffer![1, 0, 2].into_array();

        let result = list.take(idx).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        #[expect(deprecated)]
        let result = result.to_listview();

        assert_eq!(result.len(), 3);

        let element_dtype: Arc<DType> = Arc::new(I32.into());

        assert!(
            result
                .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![3i32.into()],
                Nullability::NonNullable
            )
        );

        assert!(
            result
                .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![0i32.into(), 5.into()],
                Nullability::NonNullable
            )
        );

        assert!(
            result
                .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
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
        .into_array();

        let idx = PrimitiveArray::empty::<i32>(Nullability::Nullable).into_array();

        let result = list.take(idx).unwrap();
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
        Validity::Array(BoolArray::from_iter(vec![true, false, true, true]).into_array()),
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
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        ).unwrap()
    })]
    #[case(ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]).into_array(),
        buffer![0, 2, 3, 5].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    fn test_take_list_conformance(#[case] list: ListArray) {
        test_take_conformance(&list.into_array());
    }

    #[test]
    fn test_u64_offset_accumulation_non_nullable() {
        let elements = buffer![0i32; 200].into_array();
        let offsets = buffer![0u8, 200].into_array();
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .into_array();

        // Take the same large list twice - would overflow u8 but works with u64.
        let idx = buffer![0u8, 0].into_array();
        let result = list.take(idx).unwrap();

        assert_eq!(result.len(), 2);

        #[expect(deprecated)]
        let result_view = result.to_listview();
        assert_eq!(result_view.len(), 2);
        assert!(
            result_view
                .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert!(
            result_view
                .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
    }

    #[test]
    fn test_u64_offset_accumulation_nullable() {
        let elements = buffer![0i32; 150].into_array();
        let offsets = buffer![0u8, 150, 150].into_array();
        let validity = BoolArray::from_iter(vec![true, false]).into_array();
        let list = ListArray::try_new(elements, offsets, Validity::Array(validity))
            .unwrap()
            .into_array();

        // Take the same large list twice - would overflow u8 but works with u64.
        let idx = PrimitiveArray::from_option_iter(vec![Some(0u8), None, Some(0u8)]).into_array();
        let result = list.take(idx).unwrap();

        assert_eq!(result.len(), 3);

        #[expect(deprecated)]
        let result_view = result.to_listview();
        assert_eq!(result_view.len(), 3);
        assert!(
            result_view
                .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert!(
            result_view
                .is_invalid(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert!(
            result_view
                .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
    }

    /// Regression test for validity length mismatch bug.
    ///
    /// When source array has `Validity::Array(...)` and indices are non-nullable,
    /// the result validity must have length equal to indices.len(), not source.len().
    #[test]
    fn test_take_validity_length_mismatch_regression() {
        // Source array with explicit validity array (length 2).
        let list = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0, 2, 4].into_array(),
            Validity::Array(BoolArray::from_iter(vec![true, true]).into_array()),
        )
        .unwrap()
        .into_array();

        // Take more indices than source length (4 vs 2) with non-nullable indices.
        let idx = buffer![0u32, 1, 0, 1].into_array();

        // This should not panic - result should have length 4.
        let result = list.take(idx).unwrap();
        assert_eq!(result.len(), 4);
    }
}
