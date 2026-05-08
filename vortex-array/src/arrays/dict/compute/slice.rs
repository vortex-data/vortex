// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::slice::SliceReduce;
use crate::scalar::Scalar;

impl SliceReduce for Dict {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let is_full_slice = range.len() == array.len();
        let sliced_code = array.codes().slice(range)?;
        // TODO(joe): if the range is size 1 replace with a constant array
        if let Some(code) = sliced_code.as_opt::<Constant>() {
            let code = code.scalar().as_primitive().as_::<usize>();
            return if let Some(code) = code {
                let values = array.values().slice(code..code + 1)?;
                // SAFETY: the only dictionary value is referenced by every non-null code when the
                // slice is non-empty. An empty code stream cannot reference a non-empty values
                // array.
                let sliced = unsafe {
                    DictArray::new_unchecked(
                        ConstantArray::new(0u8, sliced_code.len()).into_array(),
                        values,
                    )
                };
                if sliced_code.is_empty() {
                    Ok(Some(sliced.into_array()))
                } else {
                    Ok(Some(unsafe {
                        sliced.set_all_values_referenced(true).into_array()
                    }))
                }
            } else {
                Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().clone()), sliced_code.len())
                        .into_array(),
                ))
            };
        }
        // SAFETY: slicing the codes preserves invariants.
        let sliced = unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()) };
        if is_full_slice {
            // A full-length slice preserves the exact code stream, so the referenced-values
            // metadata remains sound. Partial slices may drop the only reference to a value.
            Ok(Some(unsafe {
                sliced
                    .set_all_values_referenced(array.has_all_values_referenced())
                    .into_array()
            }))
        } else {
            Ok(Some(sliced.into_array()))
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::DictArrayExt;
    use crate::arrays::dict::compute::slice::ConstantArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

    #[test]
    fn slice_constant_valid_code() -> VortexResult<()> {
        let dict = DictArray::new(
            ConstantArray::new(1u8, 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected = PrimitiveArray::from_iter([20i32, 20, 20]).into_array();
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn slice_constant_null_code() -> VortexResult<()> {
        let dict = DictArray::new(
            ConstantArray::new(Scalar::null(DType::Primitive(PType::U8, Nullable)), 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected =
            PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]).into_array();
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn full_slice_preserves_all_values_referenced_metadata() -> VortexResult<()> {
        let dict = unsafe {
            DictArray::new_unchecked(
                buffer![0u8, 1].into_array(),
                buffer![10i32, 20].into_array(),
            )
            .set_all_values_referenced(true)
        };

        let sliced = dict.slice(0..2)?;

        assert!(sliced.as_::<Dict>().has_all_values_referenced());
        Ok(())
    }

    #[test]
    fn partial_slice_drops_all_values_referenced_metadata() -> VortexResult<()> {
        let dict = unsafe {
            DictArray::new_unchecked(
                buffer![0u8, 1].into_array(),
                buffer![10i32, 20].into_array(),
            )
            .set_all_values_referenced(true)
        };

        let sliced = dict.slice(0..1)?;

        assert!(!sliced.as_::<Dict>().has_all_values_referenced());
        Ok(())
    }
}
