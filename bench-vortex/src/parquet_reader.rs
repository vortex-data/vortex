use std::fs::File;
use std::path::PathBuf;

use arrow_array::RecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::TryIntoArray;
use vortex::arrow::FromArrowType;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult};
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::stream::ArrayStream;

pub async fn parquet_to_vortex(parquet_path: PathBuf) -> VortexResult<impl ArrayStream> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path)?)?.build()?;

    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|br| {
            br.map_err(VortexError::from)
                .and_then(|b| b.try_into_array())
        }),
    );

    Ok(array_iter.into_stream())
}
