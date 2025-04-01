use std::fs::File;
use std::path::PathBuf;

use arrow_array::RecordBatchReader;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::{CsvReadOptions, SessionContext};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::TryIntoArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexResult};
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::stream::ArrayStream;

pub fn parquet_to_vortex(parquet_path: PathBuf) -> VortexResult<impl ArrayStream> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path)?)?.build()?;

    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|br| {
            br.map_err(VortexError::from)
                .and_then(|b| b.try_into_array())
        }),
    );

    Ok(array_iter.into_array_stream())
}

pub async fn csv_to_parquet_file(
    session: &SessionContext,
    options: CsvReadOptions<'_>,
    csv_path: &str,
    parquet_path: &str,
) -> VortexResult<()> {
    let df = session.read_csv(csv_path, options).await?;

    df.write_parquet(parquet_path, DataFrameWriteOptions::default(), None)
        .await?;
    Ok(())
}
