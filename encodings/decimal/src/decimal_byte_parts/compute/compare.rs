use num_traits::NumCast;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::{PType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_scalar::{DecimalValue, Scalar, ScalarValue, match_each_decimal_value};

use crate::DecimalBytePartsEncoding;

impl CompareKernel for DecimalBytePartsEncoding {
    fn compare(
        &self,
        lhs: &Self::Array,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        if let [part] = &lhs.parts.as_slice() {
            let encoded_scalar = rhs_const
                .as_decimal()
                .decimal_value()
                .and_then(|value| decimal_to_primitive(value, part.as_primitive_typed().ptype()))
                .and_then(|value| Some(Scalar::new(part.dtype().clone(), value)))
                .unwrap_or_else(|| Scalar::null(part.dtype().clone()));
            let encoded_const = ConstantArray::new(encoded_scalar, rhs.len());
            return compare(part, &encoded_const.to_array(), operator).map(Some);
        }

        Ok(None)
    }
}

fn decimal_to_primitive(decimal_value: DecimalValue, ptype: PType) -> Option<ScalarValue> {
    match_each_integer_ptype!(ptype, |$P| {
        match_each_decimal_value!(decimal_value, |$decimal_v| {
            Some(ScalarValue::from(<$P as NumCast>::from($decimal_v)?))
        })
    })
}

register_kernel!(CompareKernelAdapter(DecimalBytePartsEncoding).lift());

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{Operator, compare};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::DecimalBytePartsArray;

    #[test]
    fn compare_decimal_const() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype.clone(), Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            vec![
                PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).to_array(),
            ],
            decimal_dtype,
        )
        .unwrap()
        .to_array();
        let rhs = ConstantArray::new(Scalar::new(dtype, DecimalValue::I64(400).into()), lhs.len());

        let res = compare(&lhs, &rhs, Operator::Eq).unwrap();

        assert_eq!(
            res.to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![false, false, true]
        );
    }
}
