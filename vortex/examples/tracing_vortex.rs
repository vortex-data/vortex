// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tracing Subscriber with Vortex Backend
//!
//! This example demonstrates a real-world use case: implementing a `tracing` subscriber
//! that writes all log events and spans to Vortex files.
//!
//! Run with: cargo run --example tracing_vortex --features tokio

#![allow(
    clippy::disallowed_types,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinSet;
use tracing::Level;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::span;
use tracing::warn;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::layer::SubscriberExt;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::file::WriteStrategyBuilder;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_session::VortexSession;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Vortex Tracing Subscriber Example ===\n");
    println!("This example demonstrates using Vortex as a backend for structured logging.\n");

    let session = VortexSession::default();

    // Create output directory
    let output_dir: PathBuf = "vortex-traces/".into();
    std::fs::create_dir_all(&output_dir)?;

    // Create the Vortex tracing layer
    let (vortex_layer, writer_handle, shutdown) =
        VortexLayer::new(session.clone(), output_dir.clone(), 100_000).await;

    // Set up the subscriber to write all spans and logs to Vortex
    let subscriber = tracing_subscriber::registry().with(vortex_layer);

    tracing::subscriber::set_global_default(subscriber)?;

    println!("Step 1: Simulating 100,000 user interactions...\n");

    // Generate some traced activity
    let mut tasks = JoinSet::new();
    for user_id in 0..100_000 {
        tasks.spawn(simulate_application_activity(user_id));
    }
    tasks.join_all().await;

    println!("\nStep 2: Flushing remaining events to disk...");

    // Flush and shutdown
    shutdown.signal();
    writer_handle.shutdown().await?;

    println!("\nStep 3: Reading back trace data from Vortex files...");

    // Read back and display the logged events
    read_trace_files(&session, &output_dir).await?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Simulates application activity with various log levels and spans
#[allow(clippy::cognitive_complexity)]
async fn simulate_application_activity(user_id: u32) {
    // Simulate HTTP request handling
    let request_span = span!(
        Level::INFO,
        "http_request",
        method = "GET",
        path = "/api/users",
        user_id = user_id
    );
    let _enter = request_span.enter();

    info!("Handling incoming request");

    // Simulate database query
    {
        let db_span = span!(
            Level::DEBUG,
            "database_query",
            table = "users",
            op = "SELECT"
        );
        let _db_enter = db_span.enter();

        debug!(rows = 42, "Query executed successfully");
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Simulate some business logic
    info!("Processing user data");

    // Simulate a warning
    warn!(cache_hit = false, "Cache miss, fetching from database");

    // Simulate another request
    drop(_enter);
    let request_span2 = span!(
        Level::INFO,
        "http_request",
        method = "POST",
        path = "/api/orders"
    );
    let _enter2 = request_span2.enter();

    info!(order_id = user_id + 10_000, "Creating new order");

    // Simulate an error condition
    {
        let validation_span = span!(Level::ERROR, "validation");
        let _val_enter = validation_span.enter();

        error!(
            field = "email",
            reason = "invalid format",
            "Validation failed"
        );
    }

    // Generate more events for demonstration
    for i in 0..20 {
        debug!(iteration = i, "Processing batch item");
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
    }

    info!("Request completed successfully");
}

/// A tracing Layer that writes events to Vortex files
#[derive(Clone)]
struct VortexLayer {
    sender: Arc<Mutex<Option<UnboundedSender<TraceEvent>>>>,
}

/// Represents a captured trace event
#[derive(Debug, Clone)]
struct TraceEvent {
    timestamp: i64,
    level: String,
    target: String,
    message: String,
    span_name: Option<String>,
    fields: Vec<(String, String)>,
}

struct ShutdownSignal {
    inner: Arc<Mutex<Option<UnboundedSender<TraceEvent>>>>,
}

impl ShutdownSignal {
    fn signal(self) {
        // Drop the sender, signaling to any receiver that it is finished writing.
        drop(self.inner.lock().unwrap().take());
    }
}

impl VortexLayer {
    async fn new(
        session: VortexSession,
        output_dir: PathBuf,
        batch_size: usize,
    ) -> (Self, WriterHandle, ShutdownSignal) {
        let (tx, rx) = mpsc::unbounded_channel();
        let signal = Arc::new(Mutex::new(Some(tx)));
        let handle = WriterHandle::spawn(session, rx, output_dir, batch_size);
        (
            Self {
                sender: Arc::clone(&signal),
            },
            handle,
            ShutdownSignal { inner: signal },
        )
    }
}

impl<S> Layer<S> for VortexLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut guard = self.sender.lock().unwrap();
        let Some(ref mut sender) = guard.as_mut() else {
            return;
        };

        let metadata = event.metadata();

        // Extract fields from the event
        let mut visitor = FieldVisitor::new();
        event.record(&mut visitor);

        let trace_event = TraceEvent {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64,
            level: metadata.level().to_string(),
            target: metadata.target().to_string(),
            message: visitor.message.unwrap_or_default(),
            span_name: None, // Could extract current span info here
            fields: visitor.fields,
        };

        // Send to async writer (non-blocking)
        let _unused = sender.send(trace_event);
    }
}

/// Visitor to extract fields from tracing events
struct FieldVisitor {
    message: Option<String>,
    fields: Vec<(String, String)>,
}

impl FieldVisitor {
    fn new() -> Self {
        Self {
            message: None,
            fields: Vec::new(),
        }
    }
}

impl tracing::field::Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let value_str = format!("{:?}", value);

        if field.name() == "message" {
            self.message = Some(value_str);
        } else {
            self.fields.push((field.name().to_string(), value_str));
        }
    }
}

/// Handle for managing the async writer task
struct WriterHandle {
    task: tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>,
}

impl WriterHandle {
    fn spawn(
        session: VortexSession,
        mut rx: mpsc::UnboundedReceiver<TraceEvent>,
        output_dir: PathBuf,
        batch_size: usize,
    ) -> Self {
        let task = tokio::spawn(async move {
            let mut buffer = Vec::new();
            let mut file_counter = 0;

            while let Some(event) = rx.recv().await {
                buffer.push(event);

                if buffer.len() >= batch_size {
                    write_batch_to_vortex(session.clone(), &output_dir, &buffer, file_counter)
                        .await?;
                    file_counter += 1;
                    buffer.clear();
                }
            }

            if !buffer.is_empty() {
                write_batch_to_vortex(session, &output_dir, &buffer, file_counter).await?;
            }

            Ok(())
        });

        Self { task }
    }

    async fn shutdown(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.task.await.unwrap()
    }
}

/// Writes a batch of events to a Vortex file
async fn write_batch_to_vortex(
    session: VortexSession,
    output_dir: &Path,
    events: &[TraceEvent],
    file_index: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if events.is_empty() {
        return Ok(());
    }

    // Extract columns
    let span_names = VarBinArray::from_iter(
        events.iter().map(|e| e.span_name.clone()),
        DType::Utf8(Nullability::Nullable),
    );
    let timestamps: PrimitiveArray = events.iter().map(|e| e.timestamp).collect();

    let levels = VarBinArray::from_iter(
        events.iter().map(|e| Some(e.level.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );

    let targets = VarBinArray::from_iter(
        events.iter().map(|e| Some(e.target.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );

    let messages = VarBinArray::from_iter(
        events.iter().map(|e| Some(e.message.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );

    // Serialize fields as JSON strings
    let fields_json: Vec<String> = events
        .iter()
        .map(|e| {
            if e.fields.is_empty() {
                "{}".to_string()
            } else {
                let map: HashMap<_, _> = e.fields.iter().cloned().collect();
                serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
            }
        })
        .collect();

    let fields = VarBinArray::from_iter(
        fields_json.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );

    // Create struct array
    let struct_array = StructArray::try_new(
        [
            "span_names",
            "timestamp",
            "level",
            "target",
            "message",
            "fields",
        ]
        .into(),
        vec![
            span_names.into_array(),
            timestamps.into_array(),
            levels.into_array(),
            targets.into_array(),
            messages.into_array(),
            fields.into_array(),
        ],
        events.len(),
        Validity::NonNullable,
    )?;

    // Write to file with compression
    let file_path = output_dir.join(format!("traces_{:04}.vortex", file_index));
    let mut file = tokio::fs::File::create(&file_path).await?;

    // Use compact encodings (Pco + Zstd) for the telemetry files.
    let write_opts = session.write_options().with_strategy(
        WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact())
            .build(),
    );

    write_opts
        .write(&mut file, struct_array.into_array().to_array_stream())
        .await?;

    println!(
        "  Wrote {} events to {} ({} bytes)",
        events.len(),
        file_path.display(),
        tokio::fs::metadata(&file_path).await?.len()
    );

    Ok(())
}

/// Reads and displays trace files
async fn read_trace_files(
    session: &VortexSession,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut entries = tokio::fs::read_dir(output_dir).await?;
    let mut file_count = 0;
    let mut total_events = 0;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("vortex") {
            file_count += 1;

            // Read the file
            let reader = session.open_options().open_path(path.clone()).await?;
            let array = reader.scan()?.into_array_stream()?.read_all().await?;

            total_events += array.len();
        }
    }

    let total_size = du(output_dir).await?;

    println!("  Found {} trace file(s)", file_count);
    println!("  Total events captured: {}", total_events);
    println!("  Vortex files size: {} bytes", total_size);

    // Demonstrate compression efficiency

    println!(
        "  Approximate bytes per event: {:.2}",
        total_size as f64 / total_events as f64
    );
    println!("\n  Note: Nested field data is stored in compressed columnar format");

    Ok(())
}

async fn du(path: impl AsRef<Path>) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let mut total_size = 0;
    let mut dirs = tokio::fs::read_dir(path.as_ref()).await?;
    while let Some(entry) = dirs.next_entry().await? {
        if !entry.file_type().await?.is_file() {
            continue;
        }
        total_size += entry.metadata().await?.len();
    }

    Ok(total_size)
}
