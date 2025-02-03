use itertools::Itertools;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::array::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::{MinMaxFn, MinMaxResult};

impl MinMaxFn<VarBinViewArray> for VarBinViewEncoding {
    fn min_max(&self, array: &VarBinViewArray) -> VortexResult<MinMaxResult> {
        let minmax = array.with_iterator(|iter| match iter.flatten().minmax() {
            itertools::MinMaxResult::NoElements => (None, None),
            itertools::MinMaxResult::OneElement(value) => {
                let scalar = Scalar::new(DType::Utf8(NonNullable), value.into());
                (Some(scalar.clone()), Some(scalar))
            }
            itertools::MinMaxResult::MinMax(min, max) => (
                Some(Scalar::new(DType::Utf8(NonNullable), min.into())),
                Some(Scalar::new(DType::Utf8(NonNullable), max.into())),
            ),
        })?;

        Ok(minmax)
    }
}
