// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Copies of a benchmark's base Parquet files in other formats, one output file for each input
//! file, in a directory of the format's name beside the base `parquet` directory.
//!
//! - [`Format::ParquetSnappy`]: the same files written with Snappy in place of zstd. The query
//!   engines read it like Parquet. `data-gen --formats parquet-snappy` writes it.
//! - [`SizeFormat::ArrowIpc`]: uncompressed Arrow IPC, which no query reads. `data-gen
//!   --size-formats arrow-ipc` writes it only so that `scripts/capture-file-sizes.py` records its
//!   size. Lance is also written for size only, by `lance-bench --data-only`, because only that
//!   binary depends on Lance.

use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use arrow_ipc::writer::FileWriter;
use clap::ValueEnum;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use tracing::info;

use crate::Format;
use crate::idempotent;
use crate::utils::parquet::parquet_writer_properties;

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

/// Write every Parquet file in `{base_path}/parquet/` again with Snappy, to
/// `{base_path}/parquet-snappy/`.
pub fn write_snappy_parquet_directory(base_path: &Path) -> Result<()> {
    copy_parquet_directory(base_path, Format::ParquetSnappy, write_snappy_parquet)
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

/// Write the Parquet file `input` again with Snappy. Every other setting is the base files':
/// [`parquet_writer_properties`], and the row group size of `input`'s first row group.
fn write_snappy_parquet(input: &Path, output: &Path) -> Result<()> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(input)?)?;
    let schema = Arc::clone(reader.schema());
    let row_group_rows = reader
        .metadata()
        .row_groups()
        .first()
        .map(|row_group| usize::try_from(row_group.num_rows()))
        .transpose()?;
    let props = parquet_writer_properties()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(row_group_rows)
        .build();
    let mut writer = ArrowWriter::try_new(File::create(output)?, schema, Some(props))?;
    for batch in reader.build()? {
        writer.write(&batch?)?;
    }
    writer.close()?;
    Ok(())
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
    use parquet::file::reader::FileReader as _;
    use parquet::file::reader::SerializedFileReader;

    use super::*;

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
    fn snappy_parquet_keeps_the_row_groups_and_changes_only_the_codec() -> Result<()> {
        let base = tempfile::tempdir()?;
        write_base(base.path(), 1000, 300)?;

        write_snappy_parquet_directory(base.path())?;

        let output = base.path().join("parquet-snappy").join("t.parquet");
        let metadata = SerializedFileReader::new(File::open(output)?)?
            .metadata()
            .clone();
        let row_groups: Vec<i64> = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows())
            .collect();
        assert_eq!(row_groups, [300, 300, 300, 100]);
        for row_group in metadata.row_groups() {
            for column in row_group.columns() {
                assert_eq!(column.compression(), Compression::SNAPPY);
            }
        }
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
