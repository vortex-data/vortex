use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::compute::OptimizeFn;
use crate::{Array, ArrayRef};

impl OptimizeFn<&VarBinViewArray> for VarBinViewEncoding {
    fn optimize(&self, array: &VarBinViewArray) -> VortexResult<ArrayRef> {
        let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());

        array.with_iterator(|iter| {
            for item in iter {
                builder.append_option(item);
            }
        })?;

        Ok(builder.finish())
    }
}
