// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::PathBuf;

use arrow_array::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::FromArrowArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::DatasetFixture;

// TODO: Upload the pre-sampled 5k parquet to R2 and download it in a build.rs instead of
// downloading the full ~112MB partition 0 at runtime.
const CLICKBENCH_URL: &str =
    "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_0.parquet";

/// Deterministic offsets (seed=42) into clickbench hits partition 0.
const SAMPLE_OFFSETS: [usize; 5] = [26225, 116739, 288389, 670487, 777572];
const ROWS_PER_OFFSET: usize = 1000;

const MAX_RETRIES: u32 = 3;

/// Returns the path to `data/clickbench_hits_5k.parquet` relative to the crate root,
/// downloading and sampling from the full dataset if it doesn't already exist.
fn cached_clickbench_parquet() -> VortexResult<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = crate_dir.join("data");
    let dest = data_dir.join("clickbench_hits_5k.parquet");

    if dest.exists() {
        return Ok(dest);
    }

    fs::create_dir_all(&data_dir).map_err(|e| vortex_err!("failed to create data dir: {e}"))?;

    // Download full partition 0 to a temp file.
    let source_bytes = download_with_retries(CLICKBENCH_URL)?;

    // Sample 5k rows and write to dest.
    sample_and_write(&source_bytes, &dest)?;

    Ok(dest)
}

fn download_with_retries(url: &str) -> VortexResult<Bytes> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| vortex_err!("failed to build HTTP client: {e}"))?;

    for attempt in 1..=MAX_RETRIES {
        match client.get(url).send() {
            Ok(response) if response.status().is_success() => {
                return response
                    .bytes()
                    .map_err(|e| vortex_err!("failed to read response body: {e}"));
            }
            Ok(response) if response.status().is_client_error() => {
                return Err(vortex_err!(
                    "HTTP {}: failed to download {url}",
                    response.status()
                ));
            }
            Ok(response) => {
                eprintln!(
                    "Download attempt {attempt}/{MAX_RETRIES} failed: HTTP {} for {url}",
                    response.status()
                );
            }
            Err(e) => {
                eprintln!("Download attempt {attempt}/{MAX_RETRIES} failed: {e}");
            }
        }

        if attempt < MAX_RETRIES {
            let delay = std::time::Duration::from_secs(2u64.pow(attempt));
            std::thread::sleep(delay);
        }
    }

    Err(vortex_err!(
        "failed to download {url} after {MAX_RETRIES} attempts"
    ))
}

#[expect(clippy::cast_possible_truncation)]
fn sample_and_write(source_bytes: &[u8], dest: &std::path::Path) -> VortexResult<()> {
    let source_bytes = Bytes::copy_from_slice(source_bytes);
    let builder = ParquetRecordBatchReaderBuilder::try_new(source_bytes.clone())
        .map_err(|e| vortex_err!("failed to open source parquet: {e}"))?;
    let metadata = builder.metadata().clone();

    let total_rows: usize = metadata
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as usize)
        .sum();

    // Build (row_group_index, local_offset, count) ranges for each sample window.
    let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
    for &offset in &SAMPLE_OFFSETS {
        let end = (offset + ROWS_PER_OFFSET).min(total_rows);
        let mut remaining = end - offset;
        let mut global_pos = 0usize;

        for (rg_idx, rg_meta) in metadata.row_groups().iter().enumerate() {
            let rg_rows = rg_meta.num_rows() as usize;
            let rg_end = global_pos + rg_rows;

            if offset < rg_end && global_pos < end {
                let local_start = offset.saturating_sub(global_pos);
                let local_end = (local_start + remaining).min(rg_rows);
                let count = local_end - local_start;
                if count > 0 {
                    ranges.push((rg_idx, local_start, count));
                    remaining -= count;
                }
            }
            global_pos = rg_end;
            if remaining == 0 {
                break;
            }
        }
    }

    // Read each range and collect batches.
    let mut sampled_batches: Vec<RecordBatch> = Vec::new();
    for &(rg_idx, local_offset, count) in &ranges {
        let reader = ParquetRecordBatchReaderBuilder::try_new(source_bytes.clone())
            .map_err(|e| vortex_err!("failed to open parquet for sampling: {e}"))?
            .with_row_groups(vec![rg_idx])
            .with_offset(local_offset)
            .with_limit(count)
            .with_batch_size(count)
            .build()
            .map_err(|e| vortex_err!("failed to build parquet reader: {e}"))?;

        for batch in reader {
            sampled_batches
                .push(batch.map_err(|e| vortex_err!("failed to read parquet batch: {e}"))?);
        }
    }

    // Write sampled batches to a parquet file.
    let schema = sampled_batches[0].schema();
    let combined = arrow_select::concat::concat_batches(&schema, &sampled_batches)
        .map_err(|e| vortex_err!("failed to concat batches: {e}"))?;

    let file =
        fs::File::create(dest).map_err(|e| vortex_err!("failed to create output parquet: {e}"))?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None)
        .map_err(|e| vortex_err!("failed to create parquet writer: {e}"))?;
    writer
        .write(&combined)
        .map_err(|e| vortex_err!("failed to write parquet: {e}"))?;
    writer
        .close()
        .map_err(|e| vortex_err!("failed to close parquet writer: {e}"))?;

    Ok(())
}

struct ClickBenchHits5kFixture;

impl DatasetFixture for ClickBenchHits5kFixture {
    fn name(&self) -> &str {
        "clickbench_hits_5k"
    }

    fn description(&self) -> &str {
        "5000 rows (5x1000 from random offsets) of ClickBench hits dataset with wide schema of primitives and strings"
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let path = cached_clickbench_parquet()?;
        let file_bytes = fs::read(&path)
            .map_err(|e| vortex_err!("failed to read cached parquet at {}: {e}", path.display()))?;
        let bytes = Bytes::from(file_bytes);

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .map_err(|e| vortex_err!("failed to open parquet: {e}"))?
            .with_batch_size(1000)
            .build()
            .map_err(|e| vortex_err!("failed to build parquet reader: {e}"))?;

        let batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| vortex_err!("failed to read parquet batches: {e}"))?;

        Ok(ChunkedArray::from_iter(
            batches
                .into_iter()
                .map(|batch| ArrayRef::from_arrow(batch, false))
                .collect::<VortexResult<Vec<_>>>()?,
        )
        .into_array())
    }
}

pub fn fixtures() -> Vec<Box<dyn DatasetFixture>> {
    vec![Box::new(ClickBenchHits5kFixture)]
}
