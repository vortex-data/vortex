use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::Scalar;

use crate::array::{BoolArray, BoolEncoding, ConstantArray};
use crate::compute::FillNullFn;
use crate::validity::Validity;
use crate::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};

impl FillNullFn<BoolArray> for BoolEncoding {
    fn fill_null(&self, array: &BoolArray, fill_value: Scalar) -> VortexResult<ArrayData> {
        let fill = fill_value
            .as_bool()
            .value()
            .ok_or_else(|| vortex_err!("Fill value must be non null"))?;

        Ok(match array.validity() {
            Validity::NonNullable => array.clone().into_array(),
            Validity::AllValid => BoolArray::from(array.boolean_buffer()).into_array(),
            Validity::AllInvalid => ConstantArray::new(fill, array.len()).into_array(),
            Validity::Array(v) => {
                let bool_buffer = if fill {
                    &array.boolean_buffer() | &!&v.into_bool()?.boolean_buffer()
                } else {
                    &array.boolean_buffer() & &v.into_bool()?.boolean_buffer()
                };
                BoolArray::from(bool_buffer).into_array()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::array::BoolArray;
    use crate::compute::fill_null;
    use crate::validity::Validity;
    use crate::{ArrayDType, IntoArrayVariant};

    #[rstest]
    #[case(true, vec![true, true, false, true])]
    #[case(false, vec![true, false, false, false])]
    fn bool_fill_null(#[case] fill_value: bool, #[case] expected: Vec<bool>) {
        let bool_array = BoolArray::try_new(
            BooleanBuffer::from_iter([true, true, false, false]),
            Validity::from_iter([true, false, true, false]),
        )
        .unwrap();
        let non_null_array = fill_null(bool_array, fill_value.into())
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(
            non_null_array.boolean_buffer().iter().collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            non_null_array.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }
}
