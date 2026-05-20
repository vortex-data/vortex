// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generate the TPC-H lineitem table as a single Parquet file.
//!
//! Streams `tpchgen` partitions in parallel and writes them into a single
//! Parquet file at the target path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use arrow_array::RecordBatch;
use parquet::arrow::AsyncArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use tokio::sync::mpsc;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::RecordBatchIterator;

const SCALE_FACTOR: f64 = 10.0;
const OUTPUT_PATH: &str = "/home/joe/data/tpch_sf10_lineitem.parquet";
const NUM_PARTS: i32 = 32;
const BATCH_SIZE: usize = 8192 * 8;
const CHANNEL_CAPACITY: usize = 4;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let start = Instant::now();

    let output_path = PathBuf::from(OUTPUT_PATH);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Build the schema from a dummy generator to pass to the writer.
    let probe_gen = LineItemGenerator::new(SCALE_FACTOR, 1, NUM_PARTS);
    let probe_iter = LineItemArrow::new(probe_gen).with_batch_size(BATCH_SIZE);
    let schema = Arc::clone(probe_iter.schema());
    drop(probe_iter);

    println!("Schema: {:#?}", schema);

    let file = TokioFile::create(&output_path).await?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(1_000_000))
        .build();
    let mut writer = AsyncArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;

    // Spawn a producer task per partition. Each pushes RecordBatches into a
    // shared channel. The single writer consumes serially.
    let (tx, mut rx) = mpsc::channel::<RecordBatch>(CHANNEL_CAPACITY);

    let mut producers = Vec::with_capacity(NUM_PARTS as usize);
    for part in 1..=NUM_PARTS {
        let tx = tx.clone();
        let handle = tokio::task::spawn_blocking(move || -> Result<()> {
            let lgen = LineItemGenerator::new(SCALE_FACTOR, part, NUM_PARTS);
            let iter = LineItemArrow::new(lgen).with_batch_size(BATCH_SIZE);
            for batch in iter {
                // blocking_send is okay inside spawn_blocking.
                if tx.blocking_send(batch).is_err() {
                    break;
                }
            }
            Ok(())
        });
        producers.push(handle);
    }
    drop(tx);

    let mut total_rows: u64 = 0;
    let mut batch_count: u64 = 0;
    while let Some(batch) = rx.recv().await {
        total_rows += batch.num_rows() as u64;
        batch_count += 1;
        writer.write(&batch).await?;
        if batch_count % 64 == 0 {
            println!(
                "wrote {} batches, {} rows so far ({:?} elapsed)",
                batch_count,
                total_rows,
                start.elapsed()
            );
        }
    }

    for handle in producers {
        handle.await??;
    }

    let metadata = writer.close().await?;

    let file_size = std::fs::metadata(&output_path)?.len();
    let elapsed = start.elapsed();

    println!("---");
    println!("Output: {}", output_path.display());
    println!(
        "File size: {} bytes ({:.2} GiB)",
        file_size,
        file_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("Rows: {}", total_rows);
    println!("Row groups: {}", metadata.row_groups().len());
    println!("Wall time: {:?}", elapsed);
    println!("Schema columns:");
    for f in schema.fields() {
        println!("  {} : {}", f.name(), f.data_type());
    }

    Ok(())
}
