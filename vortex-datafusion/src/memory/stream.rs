use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::{exec_datafusion_err, DataFusionError, Result as DFResult};
use datafusion_execution::RecordBatchStream;
use futures::Stream;
use vortex_array::array::ChunkedArray;
use vortex_array::IntoArrayVariant;
use vortex_dtype::field::Field;

pub(crate) struct VortexRecordBatchStream {
    pub(crate) schema_ref: SchemaRef,

    pub(crate) idx: usize,
    pub(crate) num_chunks: usize,
    pub(crate) chunks: ChunkedArray,

    pub(crate) projection: Vec<Field>,
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

        let projected_struct = struct_array
            .project(&self.projection)
            .map_err(|vortex_err| {
                exec_datafusion_err!("projection pushdown to Vortex failed: {vortex_err}")
            })?;

        Poll::Ready(Some(Ok(projected_struct.into_record_batch()?)))
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
