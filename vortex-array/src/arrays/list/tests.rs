// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_buffer::BooleanBuffer;
use vortex_dtype::Nullability;
use vortex_dtype::PType::I32;
use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use super::*;
use crate::arrays::PrimitiveArray;
use crate::compute::filter;

#[test]
fn test_empty_list_array() {
    let elements = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
    let offsets = PrimitiveArray::from_iter([0]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

    assert_eq!(0, list.len());
}

#[test]
fn test_simple_list_array() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([0, 2, 4, 5]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

    assert_eq!(
        Scalar::list(
            Arc::new(I32.into()),
            vec![1.into(), 2.into()],
            Nullability::Nullable
        ),
        list.scalar_at(0)
    );
    assert_eq!(
        Scalar::list(
            Arc::new(I32.into()),
            vec![3.into(), 4.into()],
            Nullability::Nullable
        ),
        list.scalar_at(1)
    );
    assert_eq!(
        Scalar::list(Arc::new(I32.into()), vec![5.into()], Nullability::Nullable),
        list.scalar_at(2)
    );
}

#[test]
fn test_simple_list_array_from_iter() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3]);
    let offsets = PrimitiveArray::from_iter([0, 2, 3]);
    let validity = Validity::NonNullable;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

    let list_from_iter =
        ListArray::from_iter_slow::<u32, _>(vec![vec![1i32, 2], vec![3]], Arc::new(I32.into()))
            .unwrap();

    assert_eq!(list.len(), list_from_iter.len());
    assert_eq!(list.scalar_at(0), list_from_iter.scalar_at(0));
    assert_eq!(list.scalar_at(1), list_from_iter.scalar_at(1));
}

#[test]
fn test_simple_list_filter() {
    let elements = PrimitiveArray::from_option_iter([None, Some(2), Some(3), Some(4), Some(5)]);
    let offsets = PrimitiveArray::from_iter([0, 2, 4, 5]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let filtered = filter(
        &list,
        &Mask::from(BooleanBuffer::from(vec![false, true, true])),
    );

    assert!(filtered.is_ok())
}

#[test]
fn test_offset_to_0() {
    let mut builder =
        ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), Nullability::NonNullable, 5);
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![4.into(), 5.into(), 6.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![7.into(), 8.into(), 9.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![10.into(), 11.into(), 12.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![13.into(), 14.into(), 15.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    let list = builder.finish().slice(2, 4);
    let list = list.as_::<ListVTable>().reset_offsets().unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list.offsets().len(), 3);
    assert_eq!(list.elements().len(), 6);
    assert_eq!(list.offsets().scalar_at(0), 0u32.into());
}
