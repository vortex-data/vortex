use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use futures::StreamExt;
use vortex::arrow::infer_schema;
use vortex::error::{VortexError, VortexResult};
use vortex::stream::ArrayStream;
use vortex::Array;

fn vortex_to_arrow_error(error: VortexError) -> ArrowError {
    ArrowError::ExternalError(Box::new(error))
}

fn vortex_to_arrow(result: VortexResult<Array>) -> Result<RecordBatch, ArrowError> {
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

pub struct VortexRecordBatchReader<'a, S, AR> {
    stream: Pin<Box<S>>,
    arrow_schema: SchemaRef,
    runtime: &'a AR,
}

impl<'a, S, AR> VortexRecordBatchReader<'a, S, AR>
where
    S: ArrayStream,
    AR: AsyncRuntime,
{
    pub fn try_new(stream: S, runtime: &'a AR) -> VortexResult<Self> {
        let arrow_schema = Arc::new(infer_schema(stream.dtype())?);
        let stream = Box::pin(stream);
        Ok(VortexRecordBatchReader {
            stream,
            arrow_schema,
            runtime,
        })
    }
}

impl<S, AR> Iterator for VortexRecordBatchReader<'_, S, AR>
where
    S: ArrayStream,
    AR: AsyncRuntime,
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_result = self.runtime.block_on(self.stream.next());
        maybe_result.map(vortex_to_arrow)
    }
}

impl<S, AR> RecordBatchReader for VortexRecordBatchReader<'_, S, AR>
where
    S: ArrayStream,
    AR: AsyncRuntime,
{
    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }
}
