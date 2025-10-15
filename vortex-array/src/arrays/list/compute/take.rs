// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ListArray, ListVTable, list_view_from_list};
use crate::compute::{self, TakeKernel, TakeKernelAdapter};
use crate::{Array, ArrayRef, IntoArray, register_kernel};

// TODO(connor): For very short arrays it is probably more efficient to build the list from scratch.
/// Take implementation for [`ListArray`].
///
/// This implementation converts the [`ListArray`] to a [`ListViewArray`] and then delegates to its
/// `take` implementation. This approach avoids the need to rebuild the `elements` array.
///
/// The resulting [`ListViewArray`] can represent non-contiguous and out-of-order lists, which would
/// violate [`ListArray`]'s invariants (but not [`ListViewArray`]'s).
///
/// [`ListViewArray`]: crate::arrays::ListViewArray
impl TakeKernel for ListVTable {
    fn take(&self, array: &ListArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let list_view = list_view_from_list(array.clone());
        compute::take(&list_view.into_array(), indices)
    }
}

register_kernel!(TakeKernelAdapter(ListVTable).lift());

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

        let result = result.to_listview();

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

        let result = result.to_listview();

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
