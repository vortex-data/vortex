// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::arrays::{BinaryView, VarBinViewArray, VarBinViewVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeKernel for VarBinViewVTable {
    fn take(&self, array: &VarBinViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // Compute the new validity

        // This is valid since all elements (of all arrays) even null values must be inside
        // min-max valid range.
        let validity = array.validity().take(indices)?;
        let indices = indices.to_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |I| {
            // This is valid since all elements even null values are inside the min-max valid range.
            take_views(array.views(), indices.as_slice::<I>())
        });

        // SAFETY: taking all components at same indices maintains invariants
        unsafe {
            Ok(VarBinViewArray::new_unchecked(
                views_buffer,
                array.buffers().clone(),
                array
                    .dtype()
                    .union_nullability(indices.dtype().nullability()),
                validity,
            )
            .into_array())
        }
    }
}

register_kernel!(TakeKernelAdapter(VarBinViewVTable).lift());

fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<BinaryView>::from_iter(indices.iter().map(|i| views_ref[i.as_()]))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::{PrimitiveArray, VarBinViewArray};
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;

    #[test]
    fn take_nullable() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let taken = take(arr.as_ref(), &buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .to_varbinview()
                .unwrap()
                .with_iterator(|it| it
                    .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                    .collect::<Vec<_>>())
                .unwrap(),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }

    #[test]
    fn take_nullable_indices() {
        let arr = VarBinViewArray::from_iter(["one", "two"].map(Some), DType::Utf8(NonNullable));

        let taken = take(
            arr.as_ref(),
            PrimitiveArray::from_option_iter(vec![Some(1), None]).as_ref(),
        )
        .unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .to_varbinview()
                .unwrap()
                .with_iterator(|it| it
                    .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                    .collect::<Vec<_>>())
                .unwrap(),
            [Some("two".to_string()), None]
        );
    }

    #[rstest]
    #[case(VarBinViewArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(NonNullable),
    ))]
    #[case(VarBinViewArray::from_iter_nullable_str([
        Some("hello"),
        None,
        Some("test"),
        Some("data"),
        None,
    ]))]
    #[case(VarBinViewArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(NonNullable),
    ))]
    #[case(VarBinViewArray::from_iter(["single"].map(Some), DType::Utf8(NonNullable)))]
    fn test_take_varbinview_conformance(#[case] array: VarBinViewArray) {
        test_take_conformance(array.as_ref());
    }
}
