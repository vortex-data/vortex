use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::SchemaRef;
use datafusion_common::{exec_datafusion_err, DataFusionError, Result as DFResult};
use datafusion_execution::RecordBatchStream;
use futures::Stream;
use vortex_array::array::ChunkedArray;
use vortex_array::{IntoArrayVariant, IntoCanonical};
use vortex_expr::ExprRef;

pub(crate) struct VortexRecordBatchStream {
    pub(crate) schema_ref: SchemaRef,

    pub(crate) idx: usize,
    pub(crate) num_chunks: usize,
    pub(crate) chunks: ChunkedArray,

    // The projection expressions stored as tuples of (expression, output column name)
    pub(crate) projection: Vec<(ExprRef, String)>,
}

impl Stream for VortexRecordBatchStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.idx >= self.num_chunks {
            return Poll::Ready(None);
        }

        // Grab next chunk, project and convert to Arrow.
        let chunk = self.chunks.chunk(self.idx)?;
        self.idx += 1;

        let struct_array = chunk
            .into_struct()
            .map_err(|vortex_error| DataFusionError::Execution(format!("{}", vortex_error)))?;

        let mut projection_arrays: Vec<(&str, ArrayRef)> = Vec::with_capacity(self.projection.len());
        for (expr, name) in &self.projection {
            let projected_array = expr.evaluate(struct_array.as_ref())
                .map_err(|vortex_err| {
                    exec_datafusion_err!("projection pushdown to Vortex failed: {vortex_err}")
                })?;
            let projected_array_ref = projected_array.into_canonical().map_err(|e| {
                exec_datafusion_err!("projection array into canonical failed: {e}")
            })?.into_arrow().map_err(|e| {
                exec_datafusion_err!("projection array into arrow failed: {e}")
            })?;
            projection_arrays.push((name.as_str(), projected_array_ref));
        }
        let projected_struct = StructArray::try_from(projection_arrays)?;

        Poll::Ready(Some(Ok(projected_struct.into())))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_chunks, Some(self.num_chunks))
    }
}

impl RecordBatchStream for VortexRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema_ref)
    }
}
