use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::CastFn;
use crate::{Array, IntoArray};

impl CastFn<BoolArray> for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<Array> {
        if !matches!(dtype, DType::Bool(_)) {
            vortex_bail!("Cannot cast {} to {}", array.dtype(), dtype);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array.validity().cast_nullability(new_nullability)?;
        BoolArray::try_new(array.boolean_buffer(), new_validity).map(IntoArray::into_array)
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
    #[should_panic]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        try_cast(bool, &DType::Bool(Nullability::NonNullable)).unwrap();
    }
}
