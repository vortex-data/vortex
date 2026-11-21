// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Copies of a benchmark's base Parquet files in other formats, one output file for each input
//! file, in a directory of the format's name beside the base `parquet` directory.
//!
//! - [`Format::ParquetSnappy`]: the same rows as DuckDB writes them by default, with Snappy,
//!   DuckDB's row groups, and its dictionaries, statistics, and Bloom filters. This is Parquet as
//!   an analytics engine writes it for speed, where the base files are zstd level 3 for size. The
//!   query engines read it like Parquet. `data-gen --formats parquet-snappy` writes it with the
//!   `duckdb` CLI.
//! - [`Format::ParquetDuckdbZstd`]: the same rows as DuckDB writes them by default, but with zstd
//!   level 3. Beside the two others, it tells the writer's effect from the codec's.
//! - [`Format::ParquetDuckdbZstd1m`]: the same, but in row groups of 1,048,576 rows, the row group
//!   size of the base TPC-H files. It tells the row group size's effect from the rest of the writer's.
//! - [`SizeFormat::ArrowIpc`]: uncompressed Arrow IPC, which no query reads. `data-gen
//!   --size-formats arrow-ipc` writes it only so that `scripts/capture-file-sizes.py` records its
//!   size. Lance is also written for size only, by `lance-bench --data-only`, because only that
//!   binary depends on Lance.

use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use arrow_ipc::writer::FileWriter;
use clap::ValueEnum;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tracing::info;

use crate::Format;
use crate::idempotent;

/// A format that is written only to record its file size. No query reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SizeFormat {
    /// Uncompressed Arrow IPC, as `compress-bench` writes it.
    #[clap(name = "arrow-ipc")]
    ArrowIpc,
}

/// Write `format` from the base Parquet files in `base_path`.
pub fn write_size_format(base_path: &Path, format: SizeFormat) -> Result<()> {
    match format {
        SizeFormat::ArrowIpc => {
            copy_parquet_directory(base_path, Format::ArrowIpc, write_arrow_ipc)
        }
    }
}

/// Write every Parquet file in `{base_path}/parquet/` again with DuckDB, to
/// `{base_path}/parquet-snappy/`. See [`write_duckdb_snappy_parquet`].
pub fn write_snappy_parquet_directory(base_path: &Path) -> Result<()> {
    copy_parquet_directory(
        base_path,
        Format::ParquetSnappy,
        write_duckdb_snappy_parquet,
    )
}

/// Write every Parquet file in `{base_path}/parquet/` again with DuckDB and zstd level 3, to
/// `{base_path}/parquet-duckdb-zstd/`. See [`write_duckdb_zstd_parquet`].
pub fn write_duckdb_zstd_parquet_directory(base_path: &Path) -> Result<()> {
    copy_parquet_directory(
        base_path,
        Format::ParquetDuckdbZstd,
        write_duckdb_zstd_parquet,
    )
}

/// Write every Parquet file in `{base_path}/parquet/` again with DuckDB, zstd level 3, and row
/// groups of 1,048,576 rows, to `{base_path}/parquet-duckdb-zstd-1m/`. See
/// [`write_duckdb_zstd_1m_parquet`].
pub fn write_duckdb_zstd_1m_parquet_directory(base_path: &Path) -> Result<()> {
    copy_parquet_directory(
        base_path,
        Format::ParquetDuckdbZstd1m,
        write_duckdb_zstd_1m_parquet,
    )
}

/// Write each Parquet file in `{base_path}/parquet/` with `write`, to `{base_path}/{format}/`.
///
/// The write is idempotent: an output file that exists is not written again.
fn copy_parquet_directory(
    base_path: &Path,
    format: Format,
    write: fn(&Path, &Path) -> Result<()>,
) -> Result<()> {
    let parquet_dir = base_path.join(Format::Parquet.name());
    let output_dir = base_path.join(format.name());
    let mut inputs = fs::read_dir(&parquet_dir)
        .with_context(|| format!("reading {}", parquet_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    inputs.retain(|path| path.extension().is_some_and(|ext| ext == "parquet"));
    inputs.sort();

    info!(
        "Writing {} files as {} in {}",
        inputs.len(),
        format.name(),
        output_dir.display()
    );
    for input in inputs {
        let stem = input
            .file_stem()
            .context("a Parquet file has no name")?
            .to_os_string();
        let mut output = output_dir.join(stem);
        output.set_extension(format.ext());
        idempotent(output.as_path(), |temp| write(&input, temp))
            .with_context(|| format!("writing {} as {}", input.display(), format.name()))?;
    }
    Ok(())
}

/// Write the Parquet file `input` as an uncompressed Arrow IPC file.
fn write_arrow_ipc(input: &Path, output: &Path) -> Result<()> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(input)?)?;
    let schema = Arc::clone(reader.schema());
    let mut writer = FileWriter::try_new(BufWriter::new(File::create(output)?), &schema)?;
    for batch in reader.build()? {
        writer.write(&batch?)?;
    }
    writer.finish()?;
    Ok(())
}

/// Write the Parquet file `input` again as DuckDB writes Parquet: `COPY ... (FORMAT parquet,
/// COMPRESSION snappy)` in the `duckdb` CLI, which must be on `PATH`. Every setting other than the
/// codec is DuckDB's default, so the file is what a DuckDB user gets. The codec is named because
/// it is the one the `parquet-snappy` name promises, and DuckDB's default could change.
pub fn write_duckdb_snappy_parquet(input: &Path, output: &Path) -> Result<()> {
    write_duckdb_parquet(input, output, "COMPRESSION snappy")
}

/// Write the Parquet file `input` again as DuckDB writes Parquet, but with zstd level 3: the
/// codec and level of the base files, with every other setting at DuckDB's default.
pub fn write_duckdb_zstd_parquet(input: &Path, output: &Path) -> Result<()> {
    write_duckdb_parquet(input, output, "COMPRESSION zstd, COMPRESSION_LEVEL 3")
}

/// Write the Parquet file `input` again as [`write_duckdb_zstd_parquet`] does, but in row groups of
/// 1,048,576 rows, which is how parquet-rs writes the base TPC-H files.
pub fn write_duckdb_zstd_1m_parquet(input: &Path, output: &Path) -> Result<()> {
    write_duckdb_parquet(
        input,
        output,
        "COMPRESSION zstd, COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 1048576",
    )
}

/// Write the Parquet file `input` to `output` with the `duckdb` CLI, with the `COPY` options
/// `codec`.
fn write_duckdb_parquet(input: &Path, output: &Path, codec: &str) -> Result<()> {
    let sql = format!(
        "COPY (FROM read_parquet({})) TO {} (FORMAT parquet, {codec})",
        sql_string(input)?,
        sql_string(output)?,
    );
    let result = Command::new("duckdb")
        .arg("-c")
        .arg(&sql)
        .output()
        .context("running the duckdb CLI, which writes DuckDB's Parquet")?;
    if !result.status.success() {
        bail!(
            "the duckdb CLI failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    Ok(())
}

/// `path` as a SQL string literal.
fn sql_string(path: &Path) -> Result<String> {
    let path = path
        .to_str()
        .with_context(|| format!("{} is not UTF-8", path.display()))?;
    Ok(format!("'{}'", path.replace('\'', "''")))
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_ipc::reader::FileReader;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::reader::FileReader as _;
    use parquet::file::reader::SerializedFileReader;

    use super::*;
    use crate::utils::parquet::parquet_writer_properties;

    /// Write `rows` rows to `{base}/parquet/t.parquet`, in row groups of `row_group_rows`.
    fn write_base(base: &Path, rows: i64, row_group_rows: usize) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..rows)),
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|i| format!("name-{}", i % 7)),
                )),
            ],
        )?;
        let dir = base.join(Format::Parquet.name());
        fs::create_dir_all(&dir)?;
        let props = parquet_writer_properties()
            .set_max_row_group_row_count(Some(row_group_rows))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(dir.join("t.parquet"))?, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(batch)
    }

    #[test]
    #[ignore = "needs the duckdb CLI on PATH, which the benchmark runners install"]
    fn snappy_parquet_is_duckdbs_parquet_with_the_same_rows() -> Result<()> {
        let base = tempfile::tempdir()?;
        let expected = write_base(base.path(), 1000, 300)?;

        write_snappy_parquet_directory(base.path())?;

        let output = base.path().join("parquet-snappy").join("t.parquet");
        let metadata = SerializedFileReader::new(File::open(&output)?)?
            .metadata()
            .clone();
        assert!(
            metadata
                .file_metadata()
                .created_by()
                .is_some_and(|created_by| created_by.starts_with("DuckDB"))
        );
        // DuckDB's own row groups, not the base file's 300 rows.
        assert_eq!(metadata.num_row_groups(), 1);
        for column in metadata.row_groups().iter().flat_map(|rg| rg.columns()) {
            assert_eq!(column.compression(), Compression::SNAPPY);
        }
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&output)?)?.build()?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        let actual = arrow_select::concat::concat_batches(&batches[0].schema(), &batches)?;
        assert_eq!(actual.num_rows(), expected.num_rows());
        assert_eq!(actual.columns(), expected.columns());
        Ok(())
    }

    #[test]
    fn sql_strings_escape_quotes() -> Result<()> {
        assert_eq!(
            sql_string(Path::new("/data/o'brien.parquet"))?,
            "'/data/o''brien.parquet'"
        );
        Ok(())
    }

    #[test]
    fn arrow_ipc_holds_the_same_rows() -> Result<()> {
        let base = tempfile::tempdir()?;
        let expected = write_base(base.path(), 100, 64)?;

        write_size_format(base.path(), SizeFormat::ArrowIpc)?;

        let output = base.path().join("arrow-ipc").join("t.arrow");
        let batches =
            FileReader::try_new(File::open(output)?, None)?.collect::<Result<Vec<_>, _>>()?;
        let actual = arrow_select::concat::concat_batches(&expected.schema(), &batches)?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
