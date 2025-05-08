use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::DictArray;

impl ArrayOperationsImpl for DictArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DictArray::try_new(self.codes().slice(start, stop)?, self.values().clone())
            .map(|a| a.into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = self.codes().scalar_at(index)?.as_ref().try_into()?;
        self.values().scalar_at(dict_index)
    }
}
