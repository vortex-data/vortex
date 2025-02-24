use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::IsConstantFn;
use crate::Array;

impl IsConstantFn<&BoolArray> for BoolEncoding {
    fn is_constant(&self, array: &BoolArray) -> VortexResult<Option<bool>> {
        let true_count = array.boolean_buffer().count_set_bits();

        Ok(Some((true_count == 0) | (true_count == array.len())))
    }
}
