use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{Array, ArrayRef};
use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::CastFn;
use crate::IntoArray;

impl CastFn<&BoolArray> for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<ArrayRef> {
        if !matches!(dtype, DType::Bool(_)) {
            vortex_bail!("Cannot cast {} to {}", array.dtype(), dtype);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array.validity().clone().cast_nullability(new_nullability)?;
        Ok(BoolArray::new(array.boolean_buffer().clone(), new_validity).into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::BoolArray;
    use crate::compute::try_cast;

    #[test]
    fn try_cast_bool_success() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), Some(true)]);

        let res = try_cast(&bool, &DType::Bool(Nullability::NonNullable));
        assert!(res.is_ok());
        assert_eq!(res.unwrap().dtype(), &DType::Bool(Nullability::NonNullable));
    }

    #[test]
    #[should_panic]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        try_cast(&bool, &DType::Bool(Nullability::NonNullable)).unwrap();
    }
}
