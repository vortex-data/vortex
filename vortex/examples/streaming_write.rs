// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This example demonstrates writing large datasets to arbitrary sinks using streaming writes.
//!
//! Key concepts:
//! - Writing to any sink that implements `AsyncWrite` using `AsyncWriteAdapter`
//! - Streaming large amounts of data without buffering everything in memory
//! - Using the `Writer` API for incremental/push-based writing
//! - Custom sink implementations for non-file outputs
//! - Progress tracking with detailed summaries (total vs current metrics)
//! - Multi-flush patterns for batch processing with detailed reporting
//!
//! Run with: cargo run -p vortex --example streaming_write

use std::io;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::AsyncWrite;
use vortex::VortexSessionDefault;
use vortex::array::Array;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::AsyncWriteAdapter;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// A custom sink that collects data in memory.
///
/// This demonstrates how you can write to any custom sink, not just files.
/// In production, this could be network streams, S3 uploads, compression streams, etc.
struct MemorySink {
    buffer: Vec<u8>,
}

impl MemorySink {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn into_inner(self) -> Vec<u8> {
        self.buffer
    }
}

impl AsyncWrite for MemorySink {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.buffer.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Generate chunk of sample data, mocking 'user event'.
/// This simulates a streaming data source where chunks arrive over time.
///
fn generate_chunk(start_id: i64, count: usize) -> VortexResult<ArrayRef> {
    // Generate user IDs
    let user_ids: PrimitiveArray = (start_id..start_id + count as i64).collect();

    // Generate event types (cycling through a few types)
    let event_types: Vec<&str> = (0..count)
        .map(|i| match i % 4 {
            0 => "page_view",
            1 => "click",
            2 => "purchase",
            _ => "signup",
        })
        .collect();
    let event_array = VarBinViewArray::from_iter_str(event_types);

    // Generate scores
    let scores: PrimitiveArray = (0..count)
        .map(|i| ((start_id + i as i64) * 17 % 100) as f64 / 10.0)
        .collect();

    // Generate active flags
    let active = BoolArray::from_iter((0..count).map(|i| (i % 3) != 0));

    // Create struct array
    let struct_array = StructArray::try_new(
        FieldNames::from(vec![
            FieldName::from("user_id"),
            FieldName::from("event_type"),
            FieldName::from("score"),
            FieldName::from("active"),
        ]),
        vec![
            user_ids.into_array(),
            event_array.into_array(),
            scores.into_array(),
            active.into_array(),
        ],
        count,
        Validity::NonNullable,
    )?;

    Ok(struct_array.into_array())
}

/// Get DType for our event data schema
fn event_dtype() -> DType {
    let fields = StructFields::new(
        FieldNames::from(vec![
            FieldName::from("user_id"),
            FieldName::from("event_type"),
            FieldName::from("score"),
            FieldName::from("active"),
        ]),
        vec![
            DType::Primitive(PType::I64, Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
            DType::Primitive(PType::F64, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
    );
    DType::Struct(fields, Nullability::NonNullable)
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    println!("=== Vortex Streaming Write Examples ===\n");

    // Write to memory using AsyncWriteAdapter
    write_to_memory_sink().await?;

    // Incremental writing stream
    incremental_streaming_writer().await?;

    // Example 4: Multiple flush operations with detailed summaries
    example_4_multi_flush_with_summaries().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Write to in-memory sink using AsyncWriteAdapter.
///
/// This demonstrates writing to any AsyncWrite implementation, not just files.
async fn write_to_memory_sink() -> VortexResult<()> {
    println!("Writing to 'MemorySink':");
    println!("----------------------------------");

    let session: VortexSession = VortexSession::default().with_tokio();

    // Create chunks of data to simulate a stream
    let chunk1 = generate_chunk(0, 1000)?;
    let chunk2 = generate_chunk(1000, 1000)?;
    let chunk3 = generate_chunk(2000, 1000)?;

    let dtype = chunk1.dtype().clone();
    let data = ChunkedArray::try_new(vec![chunk1, chunk2, chunk3], dtype)?;

    // Create a custom memory sink
    let sink = MemorySink::new();

    // Wrap it with AsyncWriteAdapter to implement VortexWrite
    let mut adapter = AsyncWriteAdapter(sink);

    println!("Writing 3,000 rows to memory sink...");

    // Write the data
    let summary = session
        .write_options()
        .write(&mut adapter, data.to_array_stream())
        .await?;

    println!("✓ Wrote {} bytes", summary.size());
    println!("✓ Total rows: {}", summary.row_count());

    // Access the underlying buffer
    let bytes = adapter.0.into_inner();
    println!("✓ Memory buffer contains {} bytes\n", bytes.len());

    Ok(())
}

/// Incremental writing with chunk accumulation and auto-flush.
///
/// This shows:
/// - Push-based writing with chunks added incrementally
/// - Auto-flush logic when buffer threshold is reached
async fn incremental_streaming_writer() -> VortexResult<()> {
    println!("Incremental Writing streaming data, with flush");
    println!("----------------------------------------------");

    let session = VortexSession::default().with_tokio();

    // Create a sink
    let sink = MemorySink::new();
    let adapter = AsyncWriteAdapter(sink);

    // Create a Writer with our schema
    let mut writer = session.write_options().writer(adapter, event_dtype());

    println!("Pushing chunks incrementally with progress tracking...");

    let mut total_chunks_pushed = 0;
    let mut total_rows_pushed = 0;

    // Simulate streaming data: push chunks as they arrive
    for i in 0..5 {
        let chunk = generate_chunk(i * 500, 500)?;
        total_chunks_pushed += 1;
        total_rows_pushed += 500;

        println!(
            "  Chunk {}: Pushing 500 rows | Progress: bytes_written={}, buffered={} | Total: chunks={}, rows={}",
            i + 1,
            writer.bytes_written(),
            writer.buffered_bytes(),
            total_chunks_pushed,
            total_rows_pushed
        );

        writer.push(chunk).await?;
    }

    // Finish writing and get summary
    let summary = writer.finish().await?;

    println!("Total bytes written: {}", summary.size());
    println!("Total rows: {}", summary.row_count());
    println!("Chunks processed: {}", total_chunks_pushed);
    println!();

    Ok(())
}

/// Example 4: Multiple flush operations with detailed summaries.
///
/// This demonstrates:
/// - Writing multiple batches with separate flush operations
/// - Tracking detailed metrics per flush (similar to streaming_writer_v2's FlushSummary)
/// - Monitoring cumulative progress across multiple writes
/// - Understanding when to use explicit flushes vs single write
async fn example_4_multi_flush_with_summaries() -> VortexResult<()> {
    println!("Example 4: Multi-Flush with Detailed Summaries");
    println!("-----------------------------------------------");

    let session = VortexSession::default().with_tokio();

    // Create temporary directory for output files
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path();

    println!("Writing multiple batches to separate files with detailed tracking...\n");

    let mut total_bytes_written = 0u64;
    let mut total_rows_written = 0u64;
    let mut files_created = Vec::new();

    // Simulate streaming data with periodic flushes
    for batch_num in 0..3 {
        let start_id = batch_num * 1000;
        let chunk1 = generate_chunk(start_id, 500)?;
        let chunk2 = generate_chunk(start_id + 500, 500)?;

        let dtype = chunk1.dtype().clone();
        let batch_data = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)?;

        // Create a new file for each batch
        let file_path = base_path.join(format!("batch_{}.vortex", batch_num));
        let file = async_fs::File::create(&file_path).await?;

        // Write and get summary
        let summary = session
            .write_options()
            .write(file, batch_data.to_array_stream())
            .await?;

        // Track cumulative metrics
        total_bytes_written += summary.size();
        total_rows_written += summary.row_count();
        files_created.push(file_path.clone());

        // Print detailed summary for this flush (inspired by FlushSummary pattern)
        println!("📊 Batch {} Summary:", batch_num + 1);
        println!("   ├─ Bytes written: {} bytes", summary.size());
        println!("   ├─ Rows written: {}", summary.row_count());
        println!("   ├─ File: {}", file_path.display());
        println!(
            "   └─ Cumulative: {} bytes, {} rows across {} files\n",
            total_bytes_written,
            total_rows_written,
            files_created.len()
        );
    }

    println!("✅ Multi-Flush Complete:");
    println!("   ├─ Total bytes: {}", total_bytes_written);
    println!("   ├─ Total rows: {}", total_rows_written);
    println!("   └─ Files created: {}", files_created.len());

    // Verify all files exist
    for path in &files_created {
        let metadata = std::fs::metadata(path)?;
        println!("   ✓ {} ({} bytes)", path.display(), metadata.len());
    }
    println!();

    Ok(())
}
