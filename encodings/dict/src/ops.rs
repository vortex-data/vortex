use vortex_array::arrays::{ConstantArray, ConstantVTable};
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayExt, ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DictArray, DictVTable};

impl OperationsVTable<DictVTable> for DictVTable {
    fn slice(array: &DictArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let sliced_code = array.codes().slice(start, stop)?;
        if sliced_code.is::<ConstantVTable>() {
            let code = Option::<usize>::try_from(&sliced_code.scalar_at(0)?)?;
            return if let Some(code) = code {
                Ok(
                    ConstantArray::new(array.values().scalar_at(code)?, sliced_code.len())
                        .to_array(),
                )
            } else {
                let dtype = array.values().dtype().with_nullability(
                    array.values().dtype().nullability() | array.codes().dtype().nullability(),
                );
                Ok(ConstantArray::new(Scalar::null(dtype), sliced_code.len()).to_array())
            };
        }
        DictArray::try_new(sliced_code, array.values().clone()).map(|a| a.into_array())
    }

    fn scalar_at(array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = array.codes().scalar_at(index)?.as_ref().try_into()?;
        array.values().scalar_at(dict_index)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayStatistics;
    use vortex_array::arrays::PrimitiveArray;
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
