use arrow_arith::boolean;
use arrow_array::cast::AsArray as _;
use arrow_array::{Array as _, BooleanArray};
use arrow_schema::ArrowError;
use vortex_error::VortexResult;

use crate::array::BoolArray;
use crate::arrow::FromArrowArray as _;
use crate::compute::{AndFn, OrFn};
use crate::{Array, IntoCanonical};

impl BoolArray {
    /// Lift an Arrow binary boolean kernel function to Vortex arrays.
    fn lift_arrow<F>(&self, arrow_fun: F, other: &Array) -> VortexResult<Array>
    where
        F: FnOnce(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>,
    {
        let lhs = self.clone().into_canonical()?.into_arrow()?;
        let lhs = lhs.as_boolean();

        let rhs = other.clone().into_canonical()?.into_arrow()?;
        let rhs = rhs.as_boolean();

        let array = arrow_fun(lhs, rhs)?;

        Ok(Array::from_arrow(&array, array.is_nullable()))
    }
}

impl OrFn for BoolArray {
    fn or(&self, array: &Array) -> VortexResult<Array> {
        self.lift_arrow(boolean::or, array)
    }

    fn or_kleene(&self, array: &Array) -> VortexResult<Array> {
        self.lift_arrow(boolean::or_kleene, array)
    }
}

impl AndFn for BoolArray {
    fn and(&self, array: &Array) -> VortexResult<Array> {
        self.lift_arrow(boolean::and, array)
    }

    fn and_kleene(&self, array: &Array) -> VortexResult<Array> {
        self.lift_arrow(boolean::and_kleene, array)
    }
}
