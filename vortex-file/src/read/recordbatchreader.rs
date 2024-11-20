use std::future::Future;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use futures::StreamExt;
use vortex_array::arrow::infer_schema;
use vortex_array::ArrayData;
use vortex_error::{VortexError, VortexResult};
use vortex_io::VortexReadAt;

use super::VortexFileArrayStream;

fn vortex_to_arrow_error(error: VortexError) -> ArrowError {
    ArrowError::ExternalError(Box::new(error))
}

fn vortex_to_arrow(result: VortexResult<ArrayData>) -> Result<RecordBatch, ArrowError> {
    result
        .and_then(RecordBatch::try_from)
        .map_err(vortex_to_arrow_error)
}

pub trait AsyncRuntime {
    fn block_on<F: Future>(&self, fut: F) -> F::Output;
}

impl AsyncRuntime for tokio::runtime::Runtime {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.block_on(fut)
    }
}

pub struct VortexRecordBatchReader<'a, R, AR> {
    stream: VortexFileArrayStream<R>,
    arrow_schema: SchemaRef,
    runtime: &'a AR,
}

impl<'a, R, AR> VortexRecordBatchReader<'a, R, AR>
where
    R: VortexReadAt + Unpin + 'static,
    AR: AsyncRuntime,
{
    pub fn try_new(
        stream: VortexFileArrayStream<R>,
        runtime: &'a AR,
    ) -> VortexResult<VortexRecordBatchReader<'a, R, AR>> {
        let arrow_schema = Arc::new(infer_schema(stream.dtype())?);
        Ok(VortexRecordBatchReader {
            stream,
            arrow_schema,
            runtime,
        })
    }
}

impl<R, AR> Iterator for VortexRecordBatchReader<'_, R, AR>
where
    R: VortexReadAt + Unpin + 'static,
    AR: AsyncRuntime,
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_result = self.runtime.block_on(self.stream.next());
        maybe_result.map(vortex_to_arrow)
    }
}

impl<R, AR> RecordBatchReader for VortexRecordBatchReader<'_, R, AR>
where
    R: VortexReadAt + Unpin + 'static,
    AR: AsyncRuntime,
{
    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        self.next().transpose()
    }
}
