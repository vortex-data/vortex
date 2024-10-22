use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use futures::StreamExt;
use tokio::runtime::Runtime;
use vortex::arrow::infer_schema;
use vortex::Array;
use vortex_error::{VortexError, VortexResult};

use super::LayoutBatchStream;
use crate::io::VortexReadAt;

fn vortex_to_arrow_error(error: VortexError) -> ArrowError {
    ArrowError::ExternalError(Box::new(error))
}

fn vortex_to_arrow(result: VortexResult<Array>) -> Result<RecordBatch, ArrowError> {
    result
        .and_then(RecordBatch::try_from)
        .map_err(vortex_to_arrow_error)
}

pub struct VortexRecordBatchReader<'a, R: VortexReadAt + Unpin + 'static> {
    stream: LayoutBatchStream<R>,
    arrow_schema: SchemaRef,
    runtime: &'a Runtime,
}

impl<'a, R: VortexReadAt + Unpin + 'static> VortexRecordBatchReader<'a, R> {
    pub fn new(
        stream: LayoutBatchStream<R>,
        runtime: &Runtime,
    ) -> VortexResult<VortexRecordBatchReader<R>> {
        let arrow_schema = Arc::new(infer_schema(stream.schema().dtype())?);
        Ok(VortexRecordBatchReader {
            stream,
            arrow_schema,
            runtime,
        })
    }
}

impl<'a, R: VortexReadAt + Unpin + 'static> Iterator for VortexRecordBatchReader<'a, R> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_result = self.runtime.block_on(self.stream.next());
        maybe_result.map(vortex_to_arrow)
    }
}

impl<'a, R: VortexReadAt + Unpin + 'static> RecordBatchReader for VortexRecordBatchReader<'a, R> {
    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        self.next().transpose()
    }
}
