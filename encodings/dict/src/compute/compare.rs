use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareFn, Operator, compare, take, try_cast};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DictArray, DictEncoding};

impl CompareFn<&DictArray> for DictEncoding {
    fn compare(
        &self,
        lhs: &DictArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(rhs) = rhs.as_constant() {
            let compare_result = compare(
                lhs.values(),
                &ConstantArray::new(rhs, lhs.values().len()),
                operator,
            )?;

            let bool = compare_result.to_bool()?;
            let bool_buffer = bool.boolean_buffer();
            let mut indices_iter = bool_buffer.set_indices();

            let result = match (indices_iter.next(), indices_iter.next()) {
                // Couldn't find a value match, so the result is all false
                (None, _) => ConstantArray::new(
                    Scalar::bool(false, lhs.dtype().nullability()),
                    lhs.codes().len(),
                )
                .into_array(),
                // We found a single matching value so we can compare the codes directly.
                // Note: the codes include nullability so we can just compare the codes directly, to the found code.
                (Some(code), None) => try_cast(
                    &compare(
                        lhs.codes(),
                        &try_cast(&ConstantArray::new(code, lhs.len()), lhs.codes().dtype())?,
                        operator,
                    )?,
                    &DType::Bool(lhs.dtype().nullability()),
                )?,
                // more than one value matches
                _ => take(&bool, lhs.codes())?,
            };
            return Ok(Some(result));
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::ConstantArray;
    use vortex_array::compute::{Operator, compare};
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::DictArray;

    #[test]
    fn test_compare_value() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        let res = compare(
            &dict,
            &ConstantArray::new(Scalar::from(1i32), 3),
            Operator::Eq,
        )
        .unwrap();
        let res = res.to_bool().unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, false]
        );
    }
}
