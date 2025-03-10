use vortex_array::ArrayStatistics;
use vortex_array::compute::{IsConstantFn, IsConstantOpts};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl IsConstantFn<&DateTimePartsArray> for DateTimePartsEncoding {
    fn is_constant(
        &self,
        array: &DateTimePartsArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        Ok(Some(
            array.days().is_constant()
                && array.seconds().is_constant()
                && array.subseconds().is_constant(),
        ))
    }
}
