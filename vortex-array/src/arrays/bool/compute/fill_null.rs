// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::validity::Validity;

impl FillNullKernel for Bool {
    fn fill_null(
        array: ArrayView<'_, Bool>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let fill = fill_value
            .as_bool()
            .value()
            .ok_or_else(|| vortex_err!("Fill value must be non null"))?;

        Ok(Some(match array.validity()? {
            Validity::Array(v) => {
                let v_bool = v.execute::<BoolArray>(ctx)?;
                let bool_buffer = if fill {
                    array.to_bit_buffer() | &!v_bool.to_bit_buffer()
                } else {
                    array.to_bit_buffer() & v_bool.to_bit_buffer()
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

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[rstest]
    #[case(true, bitbuffer![true, true, false, true])]
    #[case(false, bitbuffer![true, false, false, false])]
    fn bool_fill_null(#[case] fill_value: bool, #[case] expected: BitBuffer) {
        let bool_array = BoolArray::new(
            BitBuffer::from_iter([true, true, false, false]),
            Validity::from_iter([true, false, true, false]),
        );
        #[expect(deprecated)]
        let non_null_array = bool_array
            .into_array()
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
