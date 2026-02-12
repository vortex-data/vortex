// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::compute::FillNullKernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl FillNullKernel for BoolVTable {
    fn fill_null(
        array: &BoolArray,
        fill_value: &Scalar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let fill = fill_value
            .as_bool()
            .value()
            .ok_or_else(|| vortex_err!("Fill value must be non null"))?;

        Ok(Some(match array.validity() {
            Validity::Array(v) => {
                let bool_buffer = if fill {
                    array.to_bit_buffer() | &!v.to_bool().to_bit_buffer()
                } else {
                    array.to_bit_buffer() & v.to_bool().to_bit_buffer()
                };
                BoolArray::new(bool_buffer, fill_value.dtype().nullability().into()).into_array()
            }
            _ => unreachable!("checked in entry point"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::bitbuffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::arrays::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::validity::Validity;

    #[rstest]
    #[case(true, bitbuffer![true, true, false, true])]
    #[case(false, bitbuffer![true, false, false, false])]
    fn bool_fill_null(#[case] fill_value: bool, #[case] expected: BitBuffer) {
        let bool_array = BoolArray::new(
            BitBuffer::from_iter([true, true, false, false]),
            Validity::from_iter([true, false, true, false]),
        );
        let non_null_array = bool_array
            .to_array()
            .fill_null(Scalar::from(fill_value))
            .unwrap()
            .to_bool();
        assert_eq!(non_null_array.to_bit_buffer(), expected);
        assert_eq!(
            non_null_array.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }
}
