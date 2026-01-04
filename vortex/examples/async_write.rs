// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This example demonstrates creating custom AsyncWrite implementations for Vortex.
//!
//! Key concepts:
//! - Implementing the `AsyncWrite` trait for custom sinks
//! - Using `AsyncWriteAdapter` to wrap custom AsyncWrite implementations
//! - Writing to in-memory buffers, network streams, or other custom destinations
//! - Testing custom AsyncWrite implementations
//!
//! Run with: cargo run -p vortex --example async_write

use std::io;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::AsyncWrite;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
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

/// A simple AsyncWrite wrapper around Vec<u8>.
///
/// This demonstrates how to create a custom AsyncWrite implementation
/// that can be used with Vortex's AsyncWriteAdapter.
struct VecAsyncWrite {
    inner: Vec<u8>,
}

impl VecAsyncWrite {
    fn new() -> Self {
        Self { inner: Vec::new() }
    }

    fn into_inner(self) -> Vec<u8> {
        self.inner
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl AsyncWrite for VecAsyncWrite {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.inner.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Generate sample data to test.
fn generate_sample_data(count: usize) -> VortexResult<ArrayRef> {
    let user_ids: PrimitiveArray = (0..count as i64).collect();

    let event_types: Vec<&str> = (0..count)
        .map(|i| match i % 3 {
            0 => "login",
            1 => "click",
            _ => "logout",
        })
        .collect();
    let event_array = VarBinViewArray::from_iter_str(event_types);

    let scores: PrimitiveArray = (0..count).map(|i| (i as f64) * 1.5).collect();
    let active = BoolArray::from_iter((0..count).map(|i| i % 2 == 0));

    let struct_array = StructArray::try_new(
        FieldNames::from(vec![
            FieldName::from("user_id"),
            FieldName::from("event"),
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

fn sample_dtype() -> DType {
    let fields = StructFields::new(
        FieldNames::from(vec![
            FieldName::from("user_id"),
            FieldName::from("event"),
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
    println!("=== Custom AsyncWrite Examples ===\n");

    // Write to custom Vec wrapper
    vec_async_write().await?;

    // Example 2: Use Writer API with custom sink
    writer_api_sink().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Writing to custom 'VecAsyncWrite' implementation.
async fn vec_async_write() -> VortexResult<()> {
    println!("VecAsyncWrite example: ");
    println!("-------------------------");
    let sample_data_size: usize = 100;

    let session: VortexSession = VortexSession::default().with_tokio();
    let data = generate_sample_data(sample_data_size)?;

    // Create custom sink
    let sink = VecAsyncWrite::new();
    let mut adapter = AsyncWriteAdapter(sink);

    println!("Start writing rows to VecAsyncWrite");

    let summary = session
        .write_options()
        .write(&mut adapter, data.to_array_stream())
        .await?;

    println!("✓ Bytes written: {}", summary.size());
    println!("✓ Rows written: {}", summary.row_count());

    // Access the underlying buffer
    let buffer = adapter.0.into_inner();
    println!("✓ Buffer contains {} bytes\n", buffer.len());

    Ok(())
}

/// Use Writer API with custom sink
async fn writer_api_sink() -> VortexResult<()> {
    println!("Writer API with Custom Sink");
    println!("---------------------------------------");
    let sample_data_size: usize = 200;

    let session = VortexSession::default().with_tokio();

    let sink = VecAsyncWrite::new();
    let adapter = AsyncWriteAdapter(sink);

    let mut writer = session.write_options().writer(adapter, sample_dtype());

    println!("Push data chunks");
    for i in 0..3 {
        let chunk = generate_sample_data(sample_data_size)?;
        println!(
            "  Chunk {}: bytes_written={}, buffered={}",
            i + 1,
            writer.bytes_written(),
            writer.buffered_bytes()
        );
        writer.push(chunk).await?;
    }

    println!("Result: ");
    let summary = writer.finish().await?;

    println!("  Total bytes: {}", summary.size());
    println!("  Total rows: {}", summary.row_count());
    println!();

    Ok(())
}
