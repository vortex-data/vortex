use vortex_array::arrays::ConstantArray;
use vortex_array::builders::builder_with_capacity;
use vortex_array::compute::{CompareFn, Operator, compare, try_cast};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;
use vortex_mask::{AllOr, Mask};
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
            return if operator == Operator::Eq {
                let result_nullability =
                    compare_result.dtype().nullability() | lhs.dtype().nullability();
                dict_equal_to(compare_result, lhs.codes(), result_nullability).map(Some)
            } else {
                DictArray::try_new(lhs.codes().clone(), compare_result)
                    .map(|a| a.into_array())
                    .map(Some)
            };
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}

fn dict_equal_to(
    values_compare: ArrayRef,
    codes: &ArrayRef,
    result_nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let bool_result = values_compare.to_bool()?;
    let result_validity = bool_result.validity_mask()?;
    let bool_buffer = bool_result.boolean_buffer();
    let (first_match, second_match) = match result_validity.boolean_buffer() {
        AllOr::All => {
            let mut indices_iter = bool_buffer.set_indices();
            (indices_iter.next(), indices_iter.next())
        }
        AllOr::None => (None, None),
        AllOr::Some(v) => {
            let mut indices_iter = bool_buffer.set_indices().filter(|i| v.value(*i));
            (indices_iter.next(), indices_iter.next())
        }
    };

    Ok(match (first_match, second_match) {
        // Couldn't find a value match, so the result is all false
        (None, _) => match result_validity {
            Mask::AllTrue(_) => {
                let mut result_builder =
                    builder_with_capacity(&DType::Bool(result_nullability), codes.len());
                result_builder.extend_from_array(
                    &ConstantArray::new(Scalar::bool(false, result_nullability), codes.len())
                        .into_array(),
                )?;
                result_builder.set_validity(codes.validity_mask()?);
                result_builder.finish()
            }
            Mask::AllFalse(_) => ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                codes.len(),
            )
            .into_array(),
            Mask::Values(_) => {
                let mut result_builder =
                    builder_with_capacity(&DType::Bool(result_nullability), codes.len());
                result_builder.extend_from_array(
                    &ConstantArray::new(Scalar::bool(false, result_nullability), codes.len())
                        .into_array(),
                )?;
                result_builder.set_validity(
                    Validity::from_mask(result_validity, bool_result.dtype().nullability())
                        .take(codes)?
                        .to_mask(codes.len())?,
                );
                result_builder.finish()
            }
        },
        // We found a single matching value so we can compare the codes directly.
        // Note: the codes include nullability so we can just compare the codes directly, to the found code.
        (Some(code), None) => try_cast(
            &compare(
                codes,
                &try_cast(&ConstantArray::new(code, codes.len()), codes.dtype())?,
                Operator::Eq,
            )?,
            &DType::Bool(result_nullability),
        )?,
        // more than one value matches
        _ => DictArray::try_new(codes.clone(), bool_result.into_array())?.into_array(),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{Operator, compare};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;
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
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, false]
        );
    }

    #[test]
    fn test_compare_non_eq() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        let res = compare(
            &dict,
            &ConstantArray::new(Scalar::from(1i32), 3),
            Operator::Gt,
        )
        .unwrap();
        let res = res.to_bool().unwrap();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![false, true, true]
        );
    }

    #[test]
    fn test_compare_nullable() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2],
                Validity::from_iter([false, true, false]),
            )
            .into_array(),
            PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array(),
        )
        .unwrap();

        let res = compare(
            &dict,
            &ConstantArray::new(Scalar::primitive(4i32, Nullability::Nullable), 3),
            Operator::Eq,
        )
        .unwrap();
        let res = res.to_bool().unwrap();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![false, false, false]
        );
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        assert_eq!(
            res.validity_mask().unwrap(),
            Mask::from_iter([false, true, false])
        );
    }

    #[test]
    fn test_compare_null_values() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            PrimitiveArray::new(
                buffer![1i32, 2, 0],
                Validity::from_iter([true, true, false]),
            )
            .into_array(),
        )
        .unwrap();

        let res = compare(
            &dict,
            &ConstantArray::new(Scalar::primitive(4i32, Nullability::NonNullable), 3),
            Operator::Eq,
        )
        .unwrap();
        let res = res.to_bool().unwrap();
        assert_eq!(
            res.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![false, false, false]
        );
        assert_eq!(res.dtype().nullability(), Nullability::Nullable);
        assert_eq!(
            res.validity_mask().unwrap(),
            Mask::from_iter([true, true, false])
        );
    }
}
