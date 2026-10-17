// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared Parquet writer settings for generated benchmark data.

use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::anyhow;
use parking_lot::Mutex;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::column::writer::ColumnCloseResult;
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterPropertiesBuilder;
use parquet::file::writer::SerializedFileWriter;

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

/// Writes part `index` (zero-based) of a table as a standalone Parquet file at the given path.
pub type WriteParquetPart = Box<dyn Fn(usize, &Path) -> anyhow::Result<()> + Send + Sync>;

/// One Parquet file to generate, split into parts that can be written independently.
pub struct ParquetTableJob {
    /// The final file. Jobs whose file already exists are skipped.
    pub path: PathBuf,
    /// Number of parts. Concatenated in index order, they make up the whole table.
    pub num_parts: usize,
    pub write_part: WriteParquetPart,
}

/// Generate Parquet tables on one shared pool of `workers` threads.
///
/// Every part of every table is a unit on a single work queue, so one large table spreads over all
/// workers and small tables fill in around it. Tables with the most parts are queued first. The
/// worker that finishes a table's last part copies the parts' row groups into the final file, in
/// part order and without re-encoding. Each part ends with its own final row group, so a table has
/// up to `num_parts - 1` more row groups than a single writer would produce.
pub fn write_parquet_tables(mut jobs: Vec<ParquetTableJob>, workers: usize) -> anyhow::Result<()> {
    jobs.retain(|job| !job.path.exists());
    jobs.sort_by_key(|job| std::cmp::Reverse(job.num_parts));
    for job in &jobs {
        if let Some(parent) = job.path.parent() {
            fs::create_dir_all(parent)?;
        }
    }

    let queue: Vec<(usize, usize)> = jobs
        .iter()
        .enumerate()
        .flat_map(|(job_idx, job)| (0..job.num_parts.max(1)).map(move |part| (job_idx, part)))
        .collect();
    let remaining: Vec<AtomicUsize> = jobs
        .iter()
        .map(|job| AtomicUsize::new(job.num_parts.max(1)))
        .collect();
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let first_error = Mutex::new(None);

    let run_unit = |job_idx: usize, part: usize| -> anyhow::Result<()> {
        let job = &jobs[job_idx];
        (job.write_part)(part, &part_path(&job.path, part))?;
        if remaining[job_idx].fetch_sub(1, Ordering::AcqRel) == 1 {
            finish_table(job)?;
        }
        Ok(())
    };

    std::thread::scope(|scope| {
        for _ in 0..workers.clamp(1, queue.len().max(1)) {
            scope.spawn(|| {
                while !failed.load(Ordering::Acquire) {
                    let Some(&(job_idx, part)) = queue.get(next.fetch_add(1, Ordering::AcqRel))
                    else {
                        break;
                    };
                    if let Err(err) = run_unit(job_idx, part) {
                        failed.store(true, Ordering::Release);
                        first_error.lock().get_or_insert(err);
                    }
                }
            });
        }
    });

    match first_error.into_inner() {
        Some(err) => {
            // Best effort: leave no part files behind for a later run to trip over.
            for job in &jobs {
                for part in 0..job.num_parts.max(1) {
                    drop(fs::remove_file(part_path(&job.path, part)));
                }
            }
            Err(err)
        }
        None => Ok(()),
    }
}

fn part_path(path: &Path, part: usize) -> PathBuf {
    PathBuf::from(format!("{}.part{part}", path.display()))
}

/// Combine a finished table's parts into its final file, then remove the parts.
fn finish_table(job: &ParquetTableJob) -> anyhow::Result<()> {
    let parts: Vec<PathBuf> = (0..job.num_parts.max(1))
        .map(|part| part_path(&job.path, part))
        .collect();
    if let [single] = parts.as_slice() {
        fs::rename(single, &job.path)?;
        return Ok(());
    }
    // Write next to the destination and rename, so an interrupted run never leaves a partial file
    // that a later run would skip.
    let tmp = PathBuf::from(format!("{}.tmp", job.path.display()));
    concat_parquet_files(&parts, &tmp)?;
    fs::rename(&tmp, &job.path)?;
    for part in &parts {
        fs::remove_file(part)?;
    }
    Ok(())
}

/// Concatenate Parquet files with identical schemas by copying their encoded column chunks.
///
/// The first input's key-value metadata, including the embedded Arrow schema, is carried over.
fn concat_parquet_files(inputs: &[PathBuf], output: &Path) -> anyhow::Result<()> {
    let inputs = inputs
        .iter()
        .map(|path| {
            let file = File::open(path)?;
            let metadata = ParquetMetaDataReader::new()
                .with_page_index_policy(PageIndexPolicy::Optional)
                .parse_and_finish(&file)?;
            Ok((file, metadata))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let first = &inputs
        .first()
        .ok_or_else(|| anyhow!("no Parquet files to concatenate"))?
        .1;
    for (_, metadata) in &inputs[1..] {
        anyhow::ensure!(
            metadata.file_metadata().schema() == first.file_metadata().schema(),
            "Parquet parts have different schemas"
        );
    }

    let props = WriterProperties::builder()
        .set_key_value_metadata(first.file_metadata().key_value_metadata().cloned())
        .build();
    let mut writer = SerializedFileWriter::new(
        File::create(output)?,
        first.file_metadata().schema_descr().root_schema_ptr(),
        Arc::new(props),
    )?;

    for (input, metadata) in &inputs {
        for (rg_idx, row_group) in metadata.row_groups().iter().enumerate() {
            let column_indexes = metadata.column_index().and_then(|ci| ci.get(rg_idx));
            let offset_indexes = metadata.offset_index().and_then(|oi| oi.get(rg_idx));
            let mut rg_writer = writer.next_row_group()?;
            for (col_idx, column) in row_group.columns().iter().enumerate() {
                let close = ColumnCloseResult {
                    bytes_written: u64::try_from(column.compressed_size())?,
                    rows_written: u64::try_from(row_group.num_rows())?,
                    metadata: column.clone(),
                    bloom_filter: None,
                    column_index: column_indexes.and_then(|row| row.get(col_idx)).cloned(),
                    offset_index: offset_indexes.and_then(|row| row.get(col_idx)).cloned(),
                };
                rg_writer.append_column(input, close)?;
            }
            rg_writer.close()?;
        }
    }
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;

    const ROWS_PER_PART: i64 = 1000;

    /// A job whose part `i` holds the values `i * ROWS_PER_PART..(i + 1) * ROWS_PER_PART`.
    fn sequence_job(path: PathBuf, num_parts: usize) -> ParquetTableJob {
        ParquetTableJob {
            path,
            num_parts,
            write_part: Box::new(|part, path| {
                let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
                let start = i64::try_from(part)? * ROWS_PER_PART;
                let values = Int64Array::from_iter_values(start..start + ROWS_PER_PART);
                let mut writer =
                    ArrowWriter::try_new(File::create(path)?, Arc::clone(&schema), None)?;
                writer.write(&RecordBatch::try_new(schema, vec![Arc::new(values)])?)?;
                writer.close()?;
                Ok(())
            }),
        }
    }

    fn read_values(path: &Path) -> anyhow::Result<Vec<i64>> {
        let mut values = Vec::new();
        for batch in ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()? {
            let batch = batch?;
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("expected an Int64 column"))?;
            values.extend(column.values().iter().copied());
        }
        Ok(values)
    }

    #[test]
    fn tables_share_the_queue_and_keep_part_order() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let existing = dir.path().join("existing.parquet");
        fs::write(&existing, b"not regenerated")?;
        let jobs = vec![
            sequence_job(dir.path().join("one.parquet"), 1),
            sequence_job(dir.path().join("many.parquet"), 9),
            sequence_job(existing.clone(), 4),
        ];

        write_parquet_tables(jobs, 4)?;

        assert_eq!(
            read_values(&dir.path().join("one.parquet"))?,
            (0..ROWS_PER_PART).collect::<Vec<_>>()
        );
        assert_eq!(
            read_values(&dir.path().join("many.parquet"))?,
            (0..9 * ROWS_PER_PART).collect::<Vec<_>>()
        );
        assert_eq!(fs::read(&existing)?, b"not regenerated");
        // Part and temporary files are gone.
        assert_eq!(fs::read_dir(dir.path())?.count(), 3);
        Ok(())
    }

    #[test]
    fn a_failed_part_fails_the_run() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("broken.parquet");
        let job = ParquetTableJob {
            path: path.clone(),
            num_parts: 3,
            write_part: Box::new(|part, _| {
                anyhow::ensure!(part != 1, "part 1 failed");
                Ok(())
            }),
        };

        let err = write_parquet_tables(vec![job], 2)
            .err()
            .ok_or_else(|| anyhow!("expected an error"))?;
        assert!(err.to_string().contains("part 1 failed"));
        assert!(!path.exists());
        assert_eq!(fs::read_dir(dir.path())?.count(), 0);
        Ok(())
    }
}
