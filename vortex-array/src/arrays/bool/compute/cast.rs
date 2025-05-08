use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::array::{Array, ArrayRef};
use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::register_kernel;

impl CastKernel for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<ArrayRef> {
        if !matches!(dtype, DType::Bool(_)) {
            vortex_bail!("Cannot cast {} to {}", array.dtype(), dtype);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array.validity().clone().cast_nullability(new_nullability)?;
        Ok(BoolArray::new(array.boolean_buffer().clone(), new_validity).into_array())
    }
}

register_kernel!(CastKernelAdapter(BoolEncoding).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::BoolArray;
    use crate::compute::cast;

    #[test]
    fn try_cast_bool_success() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), Some(true)]);

        let res = cast(&bool, &DType::Bool(Nullability::NonNullable));
        assert!(res.is_ok());
        assert_eq!(res.unwrap().dtype(), &DType::Bool(Nullability::NonNullable));
    }

    #[test]
    #[should_panic]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        cast(&bool, &DType::Bool(Nullability::NonNullable)).unwrap();
    }
}
