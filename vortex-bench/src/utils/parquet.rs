// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared Parquet writer settings for generated benchmark data.

use std::fs;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::metadata::KeyValue;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::EnabledStatistics;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterPropertiesBuilder;
use tracing::info;

/// Rows in each row group of generated benchmark Parquet that queries read from local disk:
/// DuckDB's default. Small row groups give the query engines more units of work to run in parallel.
pub const LOCAL_ROW_GROUP_ROWS: usize = 122_880;

/// Rows in each row group of generated benchmark Parquet that queries read from S3: parquet-rs's
/// default. Large row groups need fewer requests to object storage.
pub const S3_ROW_GROUP_ROWS: usize = 1024 * 1024;

/// Columns whose dictionary outgrows this fall back to plain encoding. The parquet-rs default
/// (1 MiB) keeps dictionaries for unique keys, which compress far worse than plain values. A small
/// limit approximates the per-column choice that DuckDB's writer makes.
pub const DICTIONARY_PAGE_SIZE_LIMIT: usize = 64 * 1024;

/// Most rows in a data page: parquet-rs's default. parquet-rs checks it after every 1,024 rows, so
/// full pages hold 20,480 rows.
pub const DATA_PAGE_ROW_COUNT_LIMIT: usize = 20_000;

/// Most bytes in a data page: parquet-rs's default.
pub const DATA_PAGE_SIZE_LIMIT: usize = 1024 * 1024;

/// The version of the layout that [`parquet_writer_properties`] writes. Raise it when a setting
/// changes, so that local data in the old layout is generated again: [`prepare_parquet_dir`]
/// removes a directory in the old layout, and [`remove_stale_parquet`] removes a file.
pub const LAYOUT_VERSION: u32 = 2;

/// The key of the file metadata entry in which each generated Parquet file records the
/// [`LAYOUT_VERSION`] it was written in.
pub const LAYOUT_KEY: &str = "vortex_bench_layout";

/// Writer settings for all Parquet that the benchmarks generate:
///
/// - zstd level 3. parquet-rs's default is no compression.
/// - Dictionaries of at most [`DICTIONARY_PAGE_SIZE_LIMIT`].
/// - Page statistics, so each file has a page index: the column index and the offset index. With
///   the offset index, a point lookup fetches and decodes only the pages that hold its rows.
/// - Data pages of at most [`DATA_PAGE_SIZE_LIMIT`] and about [`DATA_PAGE_ROW_COUNT_LIMIT`] rows.
/// - The [`LAYOUT_VERSION`] in the file metadata, under [`LAYOUT_KEY`]. A caller that sets its own
///   key-value metadata on the builder replaces it, so append other entries to the writer instead.
///
/// The page settings are parquet-rs's defaults, set here so that a change of default cannot change
/// the benchmarks. Against pages bounded by bytes only, the row limit made random access lookups
/// up to 4 times faster, and cost no space or scan time on TPC-H SF10.
///
/// Row group size is left to the caller: [`LOCAL_ROW_GROUP_ROWS`] or [`S3_ROW_GROUP_ROWS`].
pub fn parquet_writer_properties() -> WriterPropertiesBuilder {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).unwrap_or_else(|_| unreachable!("3 is a valid zstd level")),
        ))
        .set_dictionary_page_size_limit(DICTIONARY_PAGE_SIZE_LIMIT)
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_data_page_size_limit(DATA_PAGE_SIZE_LIMIT)
        .set_data_page_row_count_limit(DATA_PAGE_ROW_COUNT_LIMIT)
        .set_key_value_metadata(Some(vec![KeyValue::new(
            LAYOUT_KEY.to_string(),
            LAYOUT_VERSION.to_string(),
        )]))
}

/// Return whether `metadata` records the current [`LAYOUT_VERSION`].
pub fn has_current_layout(metadata: &ParquetMetaData) -> bool {
    let version = LAYOUT_VERSION.to_string();
    metadata
        .file_metadata()
        .key_value_metadata()
        .is_some_and(|kvs| {
            kvs.iter()
                .any(|kv| kv.key == LAYOUT_KEY && kv.value.as_deref() == Some(version.as_str()))
        })
}

/// Remove the generated Parquet file at `path` if it exists and does not record the current
/// [`LAYOUT_VERSION`], so that its generator, which skips a file that exists, writes it again. It
/// reads only the footer.
pub fn remove_stale_parquet(path: &Path) -> anyhow::Result<()> {
    let Ok(file) = File::open(path) else {
        return Ok(());
    };
    let metadata = ParquetMetaDataReader::new().parse_and_finish(&file)?;
    if !has_current_layout(&metadata) {
        info!(
            "Removing {}: it was not written in Parquet layout {LAYOUT_VERSION}",
            path.display()
        );
        fs::remove_file(path)?;
    }
    Ok(())
}

/// The text of the marker beside a `parquet` directory for row groups of `row_group_rows` rows.
fn layout(row_group_rows: usize) -> String {
    format!("layout {LAYOUT_VERSION}, row groups of {row_group_rows} rows\n")
}

/// Make `parquet_dir` ready for generated Parquet in row groups of `row_group_rows` rows.
///
/// The generators do not write a file that exists, so Parquet generated before in another layout
/// would stay. A marker file beside the directory, `parquet.layout`, records the layout of its
/// files: the [`LAYOUT_VERSION`] and the row groups. If the directory exists and the marker is
/// missing or records another layout, this removes it, so that the generator writes it again. It
/// also removes the other format directories beside it, such as `vortex-file-compressed`, because
/// they were converted from the old Parquet and their conversions skip files that exist.
pub fn prepare_parquet_dir(parquet_dir: &Path, row_group_rows: usize) -> anyhow::Result<()> {
    prepare_parquet_dir_keeping(parquet_dir, row_group_rows, |_| false)
}

/// [`prepare_parquet_dir`], but it keeps the entries of `parquet_dir` for which `keep` returns
/// true, such as input files that no generator writes.
pub fn prepare_parquet_dir_keeping(
    parquet_dir: &Path,
    row_group_rows: usize,
    keep: impl Fn(&Path) -> bool,
) -> anyhow::Result<()> {
    let marker = parquet_dir.with_extension("layout");
    let layout = layout(row_group_rows);
    let recorded = fs::read_to_string(&marker).ok();
    if parquet_dir.exists() && recorded.as_deref() != Some(layout.as_str()) {
        let data_dir = parquet_dir
            .parent()
            .ok_or_else(|| anyhow::anyhow!("{} has no parent", parquet_dir.display()))?;
        info!(
            "Removing the data in {}: its Parquet layout is not {}",
            data_dir.display(),
            layout.trim()
        );
        for entry in fs::read_dir(data_dir)? {
            let path = entry?.path();
            if path.is_dir() && path != parquet_dir {
                fs::remove_dir_all(path)?;
            }
        }
        for entry in fs::read_dir(parquet_dir)? {
            let path = entry?.path();
            if keep(&path) {
                continue;
            }
            if path.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
    }
    fs::create_dir_all(parquet_dir)?;
    fs::write(&marker, layout)?;
    Ok(())
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

/// A reader for the Parquet file at `path` that reads Arrow's view types as their plain types:
/// `Utf8View` as `Utf8`, and `BinaryView` as `Binary`.
///
/// The TPC-H and TPC-DS generators write `Utf8View` columns, and the Parquet file keeps that type
/// in its Arrow schema. An Arrow IPC writer writes every buffer that a view array refers to, and
/// the batches that a Parquet reader returns share their buffers, so an IPC file written from
/// view arrays holds the same bytes many times. Arrow IPC sizes are of the plain types.
pub fn plain_arrow_reader(path: &Path) -> anyhow::Result<ParquetRecordBatchReaderBuilder<File>> {
    let file = File::open(path)?;
    let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
    let schema = schema_without_views(metadata.schema());
    if schema == **metadata.schema() {
        return Ok(ParquetRecordBatchReaderBuilder::new_with_metadata(
            file, metadata,
        ));
    }
    let metadata = ArrowReaderMetadata::try_new(
        Arc::clone(metadata.metadata()),
        ArrowReaderOptions::new().with_schema(Arc::new(schema)),
    )?;
    Ok(ParquetRecordBatchReaderBuilder::new_with_metadata(
        file, metadata,
    ))
}

fn schema_without_views(schema: &Schema) -> Schema {
    Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(field_without_views)
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    )
}

fn field_without_views(field: &FieldRef) -> FieldRef {
    Arc::new(Field::clone(field).with_data_type(type_without_views(field.data_type())))
}

fn type_without_views(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Utf8View => DataType::Utf8,
        DataType::BinaryView => DataType::Binary,
        DataType::List(field) => DataType::List(field_without_views(field)),
        DataType::LargeList(field) => DataType::LargeList(field_without_views(field)),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(field_without_views(field), *size)
        }
        DataType::Struct(fields) => {
            DataType::Struct(fields.iter().map(field_without_views).collect())
        }
        DataType::Map(field, sorted) => DataType::Map(field_without_views(field), *sorted),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringViewArray;
    use arrow_array::cast::AsArray;
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
    fn the_plain_reader_reads_view_types_as_plain_types() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("views.parquet");
        let strings = ["a short one", "a string longer than twelve bytes"];
        let batch = RecordBatch::try_from_iter([(
            "s",
            Arc::new(StringViewArray::from_iter_values(strings)) as _,
        )])?;
        let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;

        let builder = plain_arrow_reader(&path)?;
        assert_eq!(builder.schema().field(0).data_type(), &DataType::Utf8);
        let read = builder.build()?.collect::<Result<Vec<_>, _>>()?;
        let column = read[0].column(0).as_string::<i32>();
        assert_eq!(column.iter().flatten().collect::<Vec<_>>(), strings);
        Ok(())
    }

    #[test]
    fn parquet_in_an_old_layout_is_removed() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let old = dir.path().join("old.parquet");
        let current = dir.path().join("current.parquet");
        write(&old, Some(WriterProperties::builder().build()))?;
        write(&current, Some(parquet_writer_properties().build()))?;

        remove_stale_parquet(&old)?;
        remove_stale_parquet(&current)?;
        remove_stale_parquet(&dir.path().join("missing.parquet"))?;
        assert!(!old.exists());
        assert!(current.exists());
        Ok(())
    }

    #[test]
    fn parquet_with_other_row_groups_is_generated_again() -> anyhow::Result<()> {
        let base = tempfile::tempdir()?;
        let parquet_dir = base.path().join("parquet");
        let file = parquet_dir.join("t.parquet");
        let vortex_dir = base.path().join("vortex-file-compressed");
        let converted = vortex_dir.join("t.vortex");

        // Parquet without a marker, as generated before the marker existed, and a file converted
        // from it.
        fs::create_dir_all(&parquet_dir)?;
        fs::write(&file, b"old")?;
        fs::create_dir_all(&vortex_dir)?;
        fs::write(&converted, b"old")?;
        prepare_parquet_dir(&parquet_dir, LOCAL_ROW_GROUP_ROWS)?;
        assert!(!file.exists());
        assert!(!converted.exists());

        fs::write(&file, b"local")?;
        prepare_parquet_dir(&parquet_dir, LOCAL_ROW_GROUP_ROWS)?;
        assert!(file.exists());

        prepare_parquet_dir(&parquet_dir, S3_ROW_GROUP_ROWS)?;
        assert!(parquet_dir.exists());
        assert!(!file.exists());
        assert_eq!(
            fs::read_to_string(base.path().join("parquet.layout"))?,
            layout(S3_ROW_GROUP_ROWS)
        );
        Ok(())
    }

    #[test]
    fn kept_inputs_stay_when_parquet_is_generated_again() -> anyhow::Result<()> {
        let base = tempfile::tempdir()?;
        let parquet_dir = base.path().join("parquet");
        let generated = parquet_dir.join("trip_0.parquet");
        let input = parquet_dir.join("zone_0.parquet");
        let vortex_dir = base.path().join("vortex-file-compressed");
        fs::create_dir_all(&parquet_dir)?;
        fs::write(&generated, b"old")?;
        fs::write(&input, b"input")?;
        fs::create_dir_all(&vortex_dir)?;

        prepare_parquet_dir_keeping(&parquet_dir, LOCAL_ROW_GROUP_ROWS, |path| path == input)?;
        assert!(!generated.exists());
        assert!(input.exists());
        assert!(!vortex_dir.exists());
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
