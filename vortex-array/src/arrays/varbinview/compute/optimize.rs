use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::compute::OptimizeFn;
use crate::{Array, ArrayRef};

impl OptimizeFn<&VarBinViewArray> for VarBinViewEncoding {
    fn optimize(&self, array: &VarBinViewArray) -> VortexResult<ArrayRef> {
        let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());

        for idx in 0..array.len() {
            let value = array.is_valid(idx)?.then(|| array.slice_at(idx));
            builder.append_option(value);
        }

        Ok(builder.finish())
    }
}
