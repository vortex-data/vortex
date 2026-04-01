// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::varbinview::BinaryView;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;

impl TakeExecute for VarBinView {
    /// Take involves creating a new array that references the old array, just with the given set of views.
    fn take(
        array: ArrayView<'_, VarBinView>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity().take(indices)?;
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

        let indices_mask = indices.validity_mask()?;
        let views_buffer = match_each_integer_ptype!(indices.ptype(), |I| {
            take_views(array.views(), indices.as_slice::<I>(), &indices_mask)
        });

        // SAFETY: taking all components at same indices maintains invariants
        unsafe {
            Ok(Some(
                VarBinViewArray::new_handle_unchecked(
                    BufferHandle::new_host(views_buffer.into_byte_buffer()),
                    array.data_buffers().clone(),
                    array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                    validity,
                )
                .into_array(),
            ))
        }
    }
}

fn take_views<I: AsPrimitive<usize>>(
    views_ref: &[BinaryView],
    indices: &[I],
    mask: &Mask,
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    // We do not use iter_bools directly, since the resulting dyn iterator cannot
    // implement TrustedLen.
    match mask.bit_buffer() {
        AllOr::All => {
            Buffer::<BinaryView>::from_trusted_len_iter(indices.iter().map(|i| views_ref[i.as_()]))
        }
        AllOr::None => Buffer::<BinaryView>::from_trusted_len_iter(iter::repeat_n(
            BinaryView::default(),
            indices.len(),
        )),
        AllOr::Some(buffer) => Buffer::<BinaryView>::from_trusted_len_iter(
            buffer.iter().zip(indices.iter()).map(|(valid, idx)| {
                if valid {
                    views_ref[idx.as_()]
                } else {
                    BinaryView::default()
                }
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::compute::take::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::validity::Validity;

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

        let taken = arr.take(buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken.to_varbinview().with_iterator(|it| it
                .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                .collect::<Vec<_>>()),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }

    #[test]
    fn take_nullable_indices() {
        let arr = VarBinViewArray::from_iter(["one", "two"].map(Some), DType::Utf8(NonNullable));

        let indices = PrimitiveArray::new(
            // Verify that garbage values at NULL indices are ignored.
            buffer![1u64, 999],
            Validity::from(BitBuffer::from(vec![true, false])),
        );

        let taken = arr.take(indices.into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken.to_varbinview().with_iterator(|it| it
                .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                .collect::<Vec<_>>()),
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
        test_take_conformance(&array.into_array());
    }
}
