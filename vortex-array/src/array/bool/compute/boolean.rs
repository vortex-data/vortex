use arrow_arith::boolean;
use arrow_array::cast::AsArray as _;
use arrow_array::{Array as _, BooleanArray};
use arrow_schema::ArrowError;
use vortex_error::VortexResult;

use crate::array::BoolArray;
use crate::arrow::FromArrowArray as _;
use crate::compute::{AndFn, OrFn};
use crate::{ArrayData, IntoCanonical};

impl BoolArray {
    /// Lift an Arrow binary boolean kernel function to Vortex arrays.
    fn lift_arrow<F>(&self, arrow_fun: F, other: &ArrayData) -> VortexResult<ArrayData>
    where
        F: FnOnce(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>,
    {
        let lhs = self.clone().into_canonical()?.into_arrow()?;
        let lhs = lhs.as_boolean();

        let rhs = other.clone().into_canonical()?.into_arrow()?;
        let rhs = rhs.as_boolean();

        let array = arrow_fun(lhs, rhs)?;

        Ok(ArrayData::from_arrow(&array, array.is_nullable()))
    }
}

impl OrFn for BoolArray {
    fn or(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        self.lift_arrow(boolean::or, array)
    }

    fn or_kleene(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        self.lift_arrow(boolean::or_kleene, array)
    }
}

impl AndFn for BoolArray {
    fn and(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        self.lift_arrow(boolean::and, array)
    }

    fn and_kleene(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        self.lift_arrow(boolean::and_kleene, array)
    }
}
