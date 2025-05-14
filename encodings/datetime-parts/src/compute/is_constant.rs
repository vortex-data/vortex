use vortex_array::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl IsConstantKernel for DateTimePartsVTable {
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

register_kernel!(IsConstantKernelAdapter(DateTimePartsVTable).lift());
