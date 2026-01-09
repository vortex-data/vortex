// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;

pub(super) fn to_arrow_run_end(
    array: ArrayRef,
    _ends_type: &DataType,
    _values_type: &Field, // Take values as a field to capture nullability
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // Check if we have a Vortex run-end array.
    // NOTE(ngates): while this module still lives in vortex-array, we cannot depend on the
    //  Vortex run-end crate. Therefore, we extract the children of the array directly.
    if array.encoding_id().as_ref() == "vortex.runend" {
        // TODO(ngates): we actually need to grab the run end metadata in order to check if
        //  there's a non-zero offset. We cannot just grab children which would be easy!
    }

    vortex_bail!("Run-end arrays are not yet supported in Arrow execution");
}
