use vortex_array::arrays::{ConstantArray, ConstantEncoding};
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::DictArray;

impl ArrayOperationsImpl for DictArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let sliced_code = self.codes().slice(start, stop)?;
        if sliced_code.is_encoding(ConstantEncoding.id()) {
            let code = Option::<usize>::try_from(&sliced_code.scalar_at(0)?)?;
            return if let Some(code) = code {
                Ok(
                    ConstantArray::new(self.values().scalar_at(code)?, sliced_code.len())
                        .to_array(),
                )
            } else {
                let dtype = self.values().dtype().with_nullability(
                    self.values().dtype().nullability() | self.codes().dtype().nullability(),
                );
                Ok(ConstantArray::new(Scalar::null(dtype), sliced_code.len()).to_array())
            };
        }
        DictArray::try_new(sliced_code, self.values().clone()).map(|a| a.into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = self.codes().scalar_at(index)?.as_ref().try_into()?;
        self.values().scalar_at(dict_index)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{Array, ArrayStatistics};
    use vortex_scalar::Scalar;

    use crate::DictArray;

    #[test]
    fn test_slice_into_const_dict() {
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter(vec![Some(0u32), None, Some(1)]).to_array(),
            PrimitiveArray::from_option_iter(vec![Some(0i32), Some(1), Some(2)]).to_array(),
        )
        .unwrap();

        assert_eq!(
            Some(Scalar::new(dict.dtype().clone(), 0i32.into())),
            dict.slice(0, 1).unwrap().as_constant()
        );

        assert_eq!(
            Some(Scalar::null(dict.dtype().clone())),
            dict.slice(1, 2).unwrap().as_constant()
        );
    }
}
