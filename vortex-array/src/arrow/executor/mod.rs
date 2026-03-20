// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bool;
pub(crate) mod byte;
pub mod byte_view;
pub(crate) mod decimal;
pub(crate) mod dictionary;
pub(crate) mod fixed_size_list;
pub(crate) mod list;
pub(crate) mod list_view;
pub mod null;
pub mod primitive;
pub(crate) mod run_end;
pub(crate) mod struct_;
mod validity;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Schema;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::ArrowExportArray;
use crate::arrays::NativeArrow;
use crate::executor::ExecutionCtx;

/// Trait for executing a Vortex array to produce an Arrow array.
pub trait ArrowArrayExecutor: Sized {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    /// If `None`, the array's preferred (cheapest) Arrow type will be used.
    fn execute_arrow(
        self,
        data_type: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef>;

    /// Execute the array to produce an Arrow `RecordBatch` with the given schema.
    fn execute_record_batch(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RecordBatch> {
        let array = self.execute_arrow(Some(&DataType::Struct(schema.fields.clone())), ctx)?;
        Ok(RecordBatch::from(array.as_struct()))
    }

    /// Execute the array to produce Arrow `RecordBatch`'s with the given schema.
    fn execute_record_batches(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<RecordBatch>>;
}

impl ArrowArrayExecutor for ArrayRef {
    fn execute_arrow(
        self,
        data_type: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let len = self.len();

        let target = match data_type {
            Some(dt) => dt.clone(),
            None => preferred_arrow_type(&self)?,
        };

        let export = ArrowExportArray::new(self, target).into_array();
        let result = export.execute_until::<NativeArrow>(ctx)?;

        let native = result.as_opt::<NativeArrow>().ok_or_else(|| {
            vortex_err!(
                "Arrow export did not produce NativeArrowArray, got {}",
                result.encoding_id()
            )
        })?;
        let arrow = native.arrow_array().clone();
        vortex_ensure!(
            arrow.len() == len,
            "Arrow array length does not match Vortex array length"
        );
        Ok(arrow)
    }

    fn execute_record_batches(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<RecordBatch>> {
        self.to_array_iterator()
            .map(|a| a?.execute_record_batch(schema, ctx))
            .try_collect()
    }
}

/// Determine the preferred (cheapest) Arrow type for an array.
///
/// Checks the encoding's VTable first, then falls back to the dtype's canonical Arrow type.
pub(crate) fn preferred_arrow_type(array: &ArrayRef) -> VortexResult<DataType> {
    if let Some(dt) = array.preferred_arrow_data_type() {
        return Ok(dt);
    }
    array.dtype().to_arrow_dtype()
}
