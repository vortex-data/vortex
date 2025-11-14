// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_dtype::{IntegerPType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::binaryview::BinaryView;

use crate::arrays::varbin::VarBinArray;
use crate::arrays::{VarBinVTable, VarBinViewArray};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for VarBinVTable {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &VarBinArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let offsets = array.offsets().to_primitive();
        let data = array.bytes().clone();

        let has_null_indices = !indices.all_valid();

        // Get the validity result
        let result_validity = array.validity().take(indices)?;
        let result_dtype = array
            .dtype()
            .with_nullability(result_validity.nullability());

        let indices = indices.to_primitive();

        match_each_integer_ptype!(offsets.ptype(), |O| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let views = if has_null_indices {
                    take_to_views::<O, I, _>(
                        data.as_slice(),
                        offsets.as_slice::<O>(),
                        indices.as_slice::<I>(),
                        |index| indices.is_valid(index),
                    )
                } else {
                    take_to_views::<O, I, _>(
                        data.as_slice(),
                        offsets.as_slice::<O>(),
                        indices.as_slice::<I>(),
                        |_| true,
                    )
                };

                // SAFETY: views are constructed against validated
                //   string array, validity will have same length as indices.
                unsafe {
                    Ok(VarBinViewArray::new_unchecked(
                        views,
                        Arc::new([data]),
                        result_dtype,
                        result_validity,
                    )
                    .into_array())
                }
            })
        })
    }
}

register_kernel!(TakeKernelAdapter(VarBinVTable).lift());

/// A take implementation which yields VarBinViewArray back.
#[inline(always)]
fn take_to_views<Offset: IntegerPType, Index: IntegerPType, F: Fn(usize) -> bool>(
    bytes: &[u8],
    offsets: &[Offset],
    indices: &[Index],
    index_is_valid: F,
) -> Buffer<BinaryView> {
    indices
        .iter()
        .copied()
        .enumerate()
        .map(|(indices_index, index)| {
            if !index_is_valid(indices_index) {
                BinaryView::empty_view()
            } else {
                let index = index.as_();
                let offset = offsets[index].to_u32().vortex_expect("offset u32");
                let end = offsets[index + 1].as_();
                let string = &bytes[offset as usize..end];

                BinaryView::make_view(string, 0u32, offset)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::Array;
    use crate::arrays::{PrimitiveArray, VarBinArray};
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;

    #[test]
    fn test_null_take() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            take(arr.as_ref(), idx1.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            take(arr.as_ref(), idx2.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("test"), Some("data"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(["single"].map(Some), DType::Utf8(Nullability::NonNullable)))]
    fn test_take_varbin_conformance(#[case] array: VarBinArray) {
        test_take_conformance(array.as_ref());
    }
}
