// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

use std::env::args;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use tracing::Instrument;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use vortex::VortexSessionDefault;
use vortex::array::ToCanonical;
use vortex::array::arrays::DictVTable;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::compressor::FloatCode;
use vortex::compressor::IntCode;
use vortex::compressor::StringCode;
use vortex::error::VortexResult;
use vortex::file::Footer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;
use vortex_cuda::CopyDeviceReadAt;
use vortex_cuda::CudaSession;
use vortex_cuda::TracingLaunchStrategy;
use vortex_cuda::VortexCudaStreamPool;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[cuda_not_available]
fn main() {}

#[cuda_available]
#[tokio::main]
async fn main() -> VortexResult<()> {
    let args: Vec<String> = args().collect();
    let json_output = args.iter().any(|arg| arg == "--json");

    let perfetto_file = File::create("trace.pb")?;

    let perfetto_layer = PerfettoLayer::new(perfetto_file).with_debug_annotations(true);

    if json_output {
        let log_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false);

        tracing_subscriber::registry()
            .with(perfetto_layer)
            .with(log_layer.with_filter(EnvFilter::from_default_env()))
            .init();
    } else {
        let log_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false)
            .event_format(tracing_subscriber::fmt::format().with_target(true));

        tracing_subscriber::registry()
            .with(perfetto_layer)
            .with(log_layer.with_filter(EnvFilter::from_default_env()))
            .init();
    }

    let session = VortexSession::default();
    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)?
        .with_launch_strategy(Arc::new(TracingLaunchStrategy));

    #[allow(clippy::expect_used, clippy::unwrap_in_result)]
    let input_path = args
        .iter()
        .skip(1)
        .find(|arg| !arg.starts_with("--"))
        .expect("must provide path to .vortex file");
    let input_path = PathBuf::from(input_path);

    assert!(input_path.exists(), "input path does not exist");

    let (recompressed, footer) = recompress_for_gpu(input_path, &session).await?;

    // Create a full scan that executes on the GPU
    let cuda_stream =
        VortexCudaStreamPool::new(Arc::clone(cuda_ctx.stream().context()), 1).get_stream()?;
    let gpu_reader = CopyDeviceReadAt::new(recompressed, cuda_stream);

    let gpu_file = session
        .open_options()
        .with_footer(footer)
        .open(Arc::new(gpu_reader))
        .await?;

    // execute_micros => µs to execute
    let mut batches = gpu_file.scan()?.into_array_stream()?;

    let mut chunk = 0;
    while let Some(next) = batches.next().await.transpose()? {
        let record = next.to_struct();

        for (field, field_name) in record
            .unmasked_fields()
            .iter()
            .zip(record.struct_fields().names().iter())
        {
            let field_name = field_name.to_string();
            // skip dict, varbin isn't properly implemented.
            if field.is::<DictVTable>() {
                continue;
            }

            let len = field.len();

            let span = tracing::info_span!(
                "array execution",
                chunk = chunk,
                field_name = field_name,
                len = len,
            );

            async {
                if field.clone().execute_cuda(&mut cuda_ctx).await.is_err() {
                    tracing::error!("failed to execute_cuda on column");
                }
            }
            .instrument(span)
            .await;
        }

        chunk += 1;
    }

    Ok(())
}

// Dump the values out as a new Vortex file for analysis.

/// Recompress the input file using only GPU-executable encodings, returning the file as an
/// in-memory byte array.
#[cuda_available]
async fn recompress_for_gpu(
    input_path: impl AsRef<Path>,
    session: &VortexSession,
) -> VortexResult<(ByteBuffer, Footer)> {
    // Setup the reader
    let input = session.open_options().open_path(input_path).await?;

    // Build a scan to read all columns from the input, and recompress them using only GPU-compatible
    // encodings.
    let scan = input.scan()?.into_array_stream()?;

    // Rebuild a copy of the file that only uses GPU-compatible compression algorithms.
    let compressor = BtrBlocksCompressorBuilder::empty()
        .include_int([
            IntCode::Uncompressed,
            IntCode::Constant,
            IntCode::BitPacking,
            IntCode::For,
            IntCode::Sequence,
            IntCode::ZigZag,
            IntCode::Dict,
        ])
        .include_float([
            FloatCode::Uncompressed,
            FloatCode::Constant,
            FloatCode::Alp,
            FloatCode::AlpRd,
            FloatCode::RunEnd,
        ])
        // Don't compress strings, this is b/c we don't have any BtrBlocks encodings that support
        // strings.
        .include_string([
            StringCode::Uncompressed,
            StringCode::Constant,
            StringCode::Dict,
            StringCode::Zstd,
            StringCode::ZstdBuffers,
        ])
        .build();

    // Read an input stream from a Vortex file.
    let writer = WriteStrategyBuilder::default()
        .with_compressor(compressor)
        .build();

    // Segment sink?
    let mut out = ByteBufferMut::empty();
    let result = session
        .write_options()
        .with_strategy(writer)
        .write(&mut out, scan)
        .await?;

    Ok((out.freeze(), result.footer().clone()))
}
