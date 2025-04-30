use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch, RecordBatchReader};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::error::ArrowError;
use vortex::arrow::compute::to_arrow;
use vortex::error::VortexResult;
use vortex::iter::ArrayIterator;

/// Adapter for converting a [`ArrayIterator`] into an Arrow [`RecordBatchReader`].
pub struct VortexRecordBatchReader<I> {
    iter: I,
    arrow_schema: SchemaRef,
    arrow_dtype: DataType,
}

impl<I: ArrayIterator> VortexRecordBatchReader<I> {
    pub fn try_new(iter: I) -> VortexResult<Self> {
        let arrow_schema = Arc::new(iter.dtype().to_arrow_schema()?);
        let arrow_dtype = DataType::Struct(arrow_schema.fields().clone());
        Ok(VortexRecordBatchReader {
            iter,
            arrow_schema,
            arrow_dtype,
        })
    }
}

impl<I: ArrayIterator> Iterator for VortexRecordBatchReader<I> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|result| {
            result
                .and_then(|array| to_arrow(&array, &self.arrow_dtype))
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
                .map(|array| RecordBatch::from(array.as_struct()))
        })
    }
}

impl<I: ArrayIterator> RecordBatchReader for VortexRecordBatchReader<I> {
    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }
}
