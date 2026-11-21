// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared Parquet writer settings for generated benchmark data.

use std::fs::File;
use std::path::Path;

use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterPropertiesBuilder;

/// Rows in each row group of generated benchmark Parquet that queries read from local disk:
/// DuckDB's default. Small row groups give the query engines more units of work to run in parallel.
pub const LOCAL_ROW_GROUP_ROWS: usize = 122_880;

/// Rows in each row group of generated benchmark Parquet that queries read from S3: parquet-rs's
/// default. Large row groups need fewer requests to object storage.
pub const S3_ROW_GROUP_ROWS: usize = 1024 * 1024;

/// Columns whose dictionary outgrows this fall back to plain encoding. The parquet-rs default
/// (1 MiB) keeps dictionaries for unique keys, which compress far worse than plain values. A small
/// limit approximates the per-column choice that DuckDB's writer makes.
const DICTIONARY_PAGE_SIZE_LIMIT: usize = 64 * 1024;

/// Writer settings for generated benchmark Parquet: zstd level 3, pages bounded by bytes only,
/// and small dictionaries.
///
/// Row group size is left to the caller.
pub fn parquet_writer_properties() -> WriterPropertiesBuilder {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).unwrap_or_else(|_| unreachable!("3 is a valid zstd level")),
        ))
        .set_dictionary_page_size_limit(DICTIONARY_PAGE_SIZE_LIMIT)
        // The default 20,000-row page limit splits row groups into many small pages, which gives
        // zstd too little context to compress repeating columns.
        .set_data_page_row_count_limit(usize::MAX)
}

/// Return whether every column chunk in `metadata` is zstd.
pub fn is_zstd(metadata: &ParquetMetaData) -> bool {
    metadata
        .row_groups()
        .iter()
        .flat_map(|row_group| row_group.columns())
        .all(|column| matches!(column.compression(), Compression::ZSTD(_)))
}

/// Return whether every column chunk of the Parquet file at `path` is zstd. It reads only the
/// footer. A benchmark uses it to rewrite data that it cached before it wrote zstd.
pub fn is_zstd_file(path: &Path) -> anyhow::Result<bool> {
    let metadata = ParquetMetaDataReader::new().parse_and_finish(&File::open(path)?)?;
    Ok(is_zstd(&metadata))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use super::*;

    fn write(path: &Path, props: Option<WriterProperties>) -> anyhow::Result<()> {
        let batch = RecordBatch::try_from_iter([(
            "x",
            Arc::new(Int64Array::from_iter_values(0..1000)) as _,
        )])?;
        let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), props)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    #[test]
    fn zstd_file_check_reads_the_codec() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let uncompressed = dir.path().join("uncompressed.parquet");
        let zstd = dir.path().join("zstd.parquet");
        write(&uncompressed, None)?;
        write(&zstd, Some(parquet_writer_properties().build()))?;

        assert!(!is_zstd_file(&uncompressed)?);
        assert!(is_zstd_file(&zstd)?);
        Ok(())
    }
}
