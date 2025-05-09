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
            let code = usize::try_from(&sliced_code.scalar_at(0)?)?;
            return Ok(
                ConstantArray::new(self.values().scalar_at(code)?, sliced_code.len()).to_array(),
            );
        }
        DictArray::try_new(sliced_code, self.values().clone()).map(|a| a.into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = self.codes().scalar_at(index)?.as_ref().try_into()?;
        self.values().scalar_at(dict_index)
    }
}
