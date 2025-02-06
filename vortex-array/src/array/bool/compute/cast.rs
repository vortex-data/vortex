use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::CastFn;
use crate::{Array, IntoArray};

impl CastFn<BoolArray> for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<Array> {
        if !matches!(dtype, DType::Bool(_)) {
            vortex_bail!(
                "Cannot cast {} to {}",
                array.dtype().to_string(),
                dtype.to_string()
            );
        }

        // If the types are the same, return the array,
        // otherwise set the array nullability as the dtype nullability.
        if dtype.is_nullable() || array.all_valid()? {
            Ok(BoolArray::new(array.boolean_buffer(), dtype.nullability()).into_array())
        } else {
            vortex_bail!("Cannot cast null array to non-nullable type");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::bool::{DType, Nullability};
    use crate::array::BoolArray;
    use crate::compute::try_cast;

    #[test]
    fn try_cast_bool_success() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), Some(true)]);

        let res = try_cast(bool, &DType::Bool(Nullability::NonNullable));
        assert!(res.is_ok());
        assert_eq!(res.unwrap().dtype(), &DType::Bool(Nullability::NonNullable));
    }

    #[test]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);

        assert!(try_cast(bool, &DType::Bool(Nullability::NonNullable)).is_err());
    }
}
