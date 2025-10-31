// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{FillNullKernel, FillNullKernelAdapter, Operator, compare, fill_null};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{DictArray, DictVTable};

impl FillNullKernel for DictVTable {
    fn fill_null(&self, array: &DictArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        // If the fill value exists in the dictionary, we can simply rewrite the null codes to
        // point to the value.
        let found_fill_values = compare(
            array.values(),
            ConstantArray::new(fill_value.clone(), array.values().len()).as_ref(),
            Operator::Eq,
        )?
        .to_bool();

        let Some(first_fill_value) = found_fill_values.bit_buffer().set_indices().next() else {
            // No fill values found, so we must canonicalize and fill_null.
            // TODO(ngates): compute kernels should all return Option<ArrayRef> to support this
            //  fall back.
            return fill_null(&array.to_canonical().into_array(), fill_value);
        };

        // Now we rewrite the nullable codes to point at the fill value.
        let codes = fill_null(
            array.codes(),
            &Scalar::new(
                array
                    .codes()
                    .dtype()
                    .with_nullability(fill_value.dtype().nullability()),
                ScalarValue::from(first_fill_value),
            ),
        )?;
        // And fill nulls in the values
        let values = fill_null(array.values(), fill_value)?;

        // SAFETY: invariants are still satisfied after patching nulls
        unsafe { Ok(DictArray::new_unchecked(codes, values).into_array()) }
    }
}

register_kernel!(FillNullKernelAdapter(DictVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::fill_null;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, ToCanonical, assert_arrays_eq};
    use vortex_buffer::{BitBuffer, buffer};
    use vortex_dtype::Nullability;
    use vortex_error::VortexUnwrap;
    use vortex_scalar::Scalar;

    use crate::DictArray;

    #[test]
    fn nullable_codes_fill_in_values() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2],
                Validity::from(BitBuffer::from(vec![true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(buffer![10, 20, 20], Validity::AllValid).into_array(),
        )
        .vortex_unwrap();

        let filled = fill_null(
            dict.as_ref(),
            &Scalar::primitive(20, Nullability::NonNullable),
        )
        .vortex_unwrap();
        let filled_primitive = filled.to_primitive();
        assert_arrays_eq!(filled_primitive, PrimitiveArray::from_iter([10, 20, 20]));
        assert!(filled_primitive.all_valid());
    }
}
