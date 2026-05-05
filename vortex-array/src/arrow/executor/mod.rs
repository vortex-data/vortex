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
pub(crate) mod temporal;
pub(crate) mod validity;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use arrow_schema::Schema;
use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrow::ArrowSessionExt;
use crate::executor::ExecutionCtx;

/// Trait for executing a Vortex array to produce an Arrow array.
///
/// Prefer [`crate::arrow::ArrowSession`] porcelain on the active
/// [`vortex_session::VortexSession`] (e.g. `session.arrow().to_arrow_array(...)` via
/// [`crate::arrow::ArrowSessionExt`]). This trait is the underlying shim that delegates
/// to that porcelain and is retained for callers that already hold an [`ArrayRef`].
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
    ) -> VortexResult<RecordBatch>;

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
        let session = ctx.session().clone();
        session.arrow().to_arrow_array(self, data_type, ctx)
    }

    fn execute_record_batch(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RecordBatch> {
        let session = ctx.session().clone();
        session.arrow().to_arrow_record_batch(self, schema, ctx)
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
