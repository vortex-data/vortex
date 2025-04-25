use std::mem;
use std::sync::Arc;

use arrow_array::{ArrayRef as ArrowArrayRef, Decimal128Array, Decimal256Array};
use arrow_schema::DataType;
use vortex_buffer::Buffer;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::i256;

use crate::Array;
use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::ToArrowFn;

impl ToArrowFn<&DecimalArray> for DecimalEncoding {
    fn to_arrow(
        &self,
        array: &DecimalArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        let precision = array.decimal_dtype().precision();
        let scale = array.decimal_dtype().scale();
        let nulls = array.validity_mask()?.to_null_buffer();

        match array.values_type {
            DecimalValueType::I128 => {
                let DataType::Decimal128(p, s) = data_type else {
                    vortex_bail!(
                        "Target Arrow type does not match Decimal source: {:?} ≠ {:?}",
                        data_type,
                        array.decimal_dtype()
                    );
                };
                if *p != precision || *s != scale {
                    vortex_bail!(
                        "Decimal128: precision {} and scale {} do not match expected ({}, {})",
                        precision,
                        scale,
                        p,
                        s
                    );
                }

                Ok(Some(Arc::new(
                    Decimal128Array::new(array.buffer::<i128>().into_arrow_scalar_buffer(), nulls)
                        .with_precision_and_scale(precision, scale)?,
                )))
            }
            DecimalValueType::I256 => {
                let DataType::Decimal256(p, s) = data_type else {
                    vortex_bail!(
                        "Target Arrow type does not match Decimal source: {:?} ≠ {:?}",
                        data_type,
                        array.decimal_dtype()
                    );
                };
                if *p != precision || *s != scale {
                    vortex_bail!(
                        "Decimal128: precision {} and scale {} do not match expected ({}, {})",
                        precision,
                        scale,
                        p,
                        s
                    );
                }

                let buffer_i256 = array.buffer::<i256>();
                // SAFETY: vortex_scalar::i256 and arrow_buffer::i256 have same bits
                let buffer_i256: Buffer<arrow_buffer::i256> =
                    unsafe { mem::transmute(buffer_i256) };

                Ok(Some(Arc::new(
                    Decimal256Array::new(buffer_i256.into_arrow_scalar_buffer(), nulls)
                        .with_precision_and_scale(precision, scale)?,
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Decimal128Array};
    use arrow_schema::DataType;
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::arrays::DecimalArray;
    use crate::compute::to_arrow;
    use crate::validity::Validity;

    #[test]
    fn test_to_arrow() {
        // Make a very simple i128 and i256 array.
        let decimal_vortex = DecimalArray::new(
            buffer![1i128, 2i128, 3i128, 4i128, 5i128],
            DecimalDType::new(19, 2),
            Validity::NonNullable,
        );
        let arrow = to_arrow(&decimal_vortex, &DataType::Decimal128(19, 2)).unwrap();
        assert_eq!(arrow.data_type(), &DataType::Decimal128(19, 2));
        let decimal_array = arrow.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(
            decimal_array.values().as_ref(),
            &[1i128, 2i128, 3i128, 4i128, 5i128]
        );
    }
}
