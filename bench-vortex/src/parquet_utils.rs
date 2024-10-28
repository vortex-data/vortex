use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::arrow::FromArrowType;
use vortex::dtype::DType;
use vortex::error::VortexResult;

use crate::data_downloads::FileType;
use crate::CompressionRunStats;

pub fn sum_column_chunk_sizes(path: &Path) -> VortexResult<CompressionRunStats> {
    let file = File::open(path)?;
    let total_compressed_size = file.metadata()?.size();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let mut compressed_sizes = vec![0; builder.parquet_schema().num_columns()];

    for row_group_metadata in builder.metadata().row_groups() {
        // For each row group, iterate over its columns
        compressed_sizes
            .iter_mut()
            .enumerate()
            .for_each(|(i, bytes)| *bytes += row_group_metadata.column(i).compressed_size() as u64);
    }

    let stats = CompressionRunStats {
        schema: DType::from_arrow(builder.schema().clone()),
        file_type: FileType::Parquet,
        total_compressed_size: Some(total_compressed_size),
        compressed_sizes,
        file_name: path
            .with_extension("")
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
    };
    Ok(stats)
}
