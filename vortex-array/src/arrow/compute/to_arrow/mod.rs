// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrow::ArrowArrayExecutor;
use crate::compute::Options;

/// Convert a Vortex array to an Arrow array with the encoding's preferred `DataType`.
///
/// For example, a `VarBinArray` will be converted to an Arrow `VarBin` array, instead of the
/// canonical `VarBinViewArray`.
///
/// Warning: do not use this to convert a Vortex [`crate::stream::ArrayStream`] since each array
/// may have a different preferred Arrow type. Use [`to_arrow`] instead.
#[deprecated(note = "Use ArrowArrayExecutor::execute_arrow instead")]
#[expect(deprecated)]
pub fn to_arrow_preferred(array: &ArrayRef) -> VortexResult<ArrowArrayRef> {
    to_arrow_opts(array, &ToArrowOptions { arrow_type: None })
}

/// Convert a Vortex array to an Arrow array of the given type.
#[deprecated(note = "Use ArrowArrayExecutor::execute_arrow instead")]
pub fn to_arrow(array: &ArrayRef, arrow_type: &DataType) -> VortexResult<ArrowArrayRef> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array.to_array().execute_arrow(Some(arrow_type), &mut ctx)
}

#[deprecated(note = "Use ArrowArrayExecutor::execute_arrow instead")]
#[expect(deprecated)]
pub fn to_arrow_opts(array: &ArrayRef, options: &ToArrowOptions) -> VortexResult<ArrowArrayRef> {
    let data_type = if let Some(data_type) = &options.arrow_type {
        data_type.clone()
    } else {
        array.dtype().to_arrow_dtype()?
    };
    let arrow = to_arrow(array, &data_type)?;

    vortex_ensure!(
        &data_type == arrow.data_type(),
        "to arrow returned array with data_type {}, expected {}",
        arrow.data_type(),
        data_type
    );

    Ok(arrow)
}

pub struct ToArrowOptions {
    /// The Arrow data type to convert to, if specified.
    pub arrow_type: Option<DataType>,
}

impl Options for ToArrowOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
