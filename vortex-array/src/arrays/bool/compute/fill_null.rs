// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable, ConstantArray};
use crate::compute::{FillNullKernel, FillNullKernelAdapter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, ToCanonical, register_kernel};

impl FillNullKernel for BoolVTable {
    fn fill_null(&self, array: &BoolArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        let fill = fill_value
            .as_bool()
            .value()
            .ok_or_else(|| vortex_err!("Fill value must be non null"))?;

        Ok(match array.validity() {
            Validity::NonNullable | Validity::AllValid => BoolArray::from_bit_buffer(
                array.bit_buffer().clone(),
                fill_value.dtype().nullability().into(),
            )
            .into_array(),
            Validity::AllInvalid => {
                ConstantArray::new(fill_value.clone(), array.len()).into_array()
            }
            Validity::Array(v) => {
                let bool_buffer = if fill {
                    array.bit_buffer() | &!v.to_bool().bit_buffer()
                } else {
                    array.bit_buffer() & v.to_bool().bit_buffer()
                };
                BoolArray::from_bit_buffer(bool_buffer, fill_value.dtype().nullability().into())
                    .into_array()
            }
        })
    }
}

register_kernel!(FillNullKernelAdapter(BoolVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::{BitBuffer, bitbuffer};
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    use crate::compute::fill_null;
    use crate::validity::Validity;

    #[rstest]
    #[case(true, bitbuffer![true, true, false, true])]
    #[case(false, bitbuffer![true, false, false, false])]
    fn bool_fill_null(#[case] fill_value: bool, #[case] expected: BitBuffer) {
        let bool_array = BoolArray::from_bit_buffer(
            BitBuffer::from_iter([true, true, false, false]),
            Validity::from_iter([true, false, true, false]),
        );
        let non_null_array = fill_null(bool_array.as_ref(), &fill_value.into())
            .unwrap()
            .to_bool();
        assert_eq!(non_null_array.bit_buffer(), &expected);
        assert_eq!(
            non_null_array.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }
}
