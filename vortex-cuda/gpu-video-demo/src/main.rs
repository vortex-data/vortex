// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU Video Demo: scans a Vortex file containing RGB video frames, converts to
//! NV12 on GPU, encodes to H.264 via NVENC, and streams over SRT.
//!
//! Usage:
//!   cargo run -p gpu-video-demo -- s3://bucket/video.vortex --width 1920 --height 1080
//!
//! Then connect with VLC:
//!   vlc srt://<host>:9000

#![allow(unused_imports)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::sys::CUdeviceptr;
use futures::StreamExt;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_BUFFER_FORMAT;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_CODEC_H264_GUID;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_INPUT_RESOURCE_TYPE;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_PRESET_P4_GUID;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_TUNING_INFO;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use tokio::time::Duration;
use tokio::time::sleep_until;
use tracing::Instrument;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaSession;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PooledFileReadAt;
use vortex_cuda::PooledObjectStoreReadAt;
use vortex_cuda::TracingLaunchStrategy;
use vortex_cuda::VortexCudaStreamPool;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::register_cuda_layout;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

mod mux;
mod nv12;
mod transport;

#[derive(Parser)]
#[command(
    name = "gpu-video-demo",
    about = "Stream a Vortex file as live H.264 video via GPU scan + NVENC"
)]
struct Cli {
    /// S3 URI (s3://bucket/path) or local path to a Vortex file with RGB frame data.
    source: String,

    /// Frame width in pixels.
    #[arg(long)]
    width: u32,

    /// Frame height in pixels.
    #[arg(long)]
    height: u32,

    /// Target frame rate.
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Target bitrate in Mbps.
    #[arg(long, default_value_t = 20)]
    bitrate_mbps: u32,

    /// SRT listener port.
    #[arg(long, default_value_t = 9000)]
    srt_port: u16,

    /// Path to write Perfetto trace output.
    #[arg(long)]
    perfetto: Option<PathBuf>,

    /// Loop the video file continuously.
    #[arg(long)]
    loop_playback: bool,
}

#[cuda_not_available]
fn main() {}

#[cuda_available]
#[tokio::main]
async fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    // Setup tracing
    let perfetto_guard = if let Some(ref perfetto_path) = cli.perfetto {
        let perfetto_file = File::create(perfetto_path)?;
        Some(PerfettoLayer::new(perfetto_file).with_debug_annotations(true))
    } else {
        None
    };

    let log_layer = tracing_subscriber::fmt::layer()
        .pretty()
        .with_span_events(FmtSpan::NONE)
        .with_ansi(false)
        .event_format(tracing_subscriber::fmt::format().with_target(true));

    let registry =
        tracing_subscriber::registry().with(log_layer.with_filter(EnvFilter::from_default_env()));

    if let Some(perfetto) = perfetto_guard {
        registry.with(perfetto).init();
    } else {
        registry.init();
    }

    // CUDA setup
    let session = VortexSession::default().with_tokio();
    register_cuda_layout(&session);

    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)?
        .with_launch_strategy(Arc::new(TracingLaunchStrategy));

    let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(
        cuda_ctx.stream().context(),
    )));
    let cuda_stream =
        VortexCudaStreamPool::new(Arc::clone(cuda_ctx.stream().context()), 1).get_stream()?;
    let handle = session.handle();

    // Parse source and create reader
    let reader: Arc<dyn vortex::io::VortexReadAt> = if cli.source.starts_with("s3://") {
        let url = Url::parse(&cli.source).map_err(|e| vortex_err!("Invalid S3 URL: {e}"))?;
        let bucket = url
            .host_str()
            .ok_or_else(|| vortex_err!("S3 URL missing bucket name"))?;
        let path = ObjectPath::from(url.path());
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?,
        );
        Arc::new(PooledObjectStoreReadAt::new(
            store,
            path,
            handle,
            Arc::clone(&pool),
            cuda_stream,
        ))
    } else {
        let path = PathBuf::from(&cli.source);
        Arc::new(PooledFileReadAt::open(
            &path,
            handle,
            Arc::clone(&pool),
            cuda_stream,
        )?)
    };

    let width = cli.width;
    let height = cli.height;
    let fps = cli.fps;
    let bitrate = cli.bitrate_mbps * 1_000_000;

    // Load RGB→NV12 kernel
    let nv12_kernel = nv12::load_rgb_to_nv12_kernel(&session)?;

    // Allocate NV12 buffer on GPU (width * height * 3/2 for Y + UV planes)
    let nv12_size = (width as usize) * (height as usize) * 3 / 2;
    let nv12_device: CudaSlice<u8> = cuda_ctx.device_alloc(nv12_size)?;
    // Extract the raw device pointer and immediately drop the SyncOnDrop guard
    // so it doesn't hold an immutable borrow on cuda_ctx.
    let nv12_ptr = {
        let (ptr, _sync) = nv12_device.device_ptr(cuda_ctx.stream());
        ptr
    };

    // Create NVENC encoder
    let cuda_context = cuda_ctx.stream().context().clone();
    cuda_context
        .bind_to_thread()
        .map_err(|e| vortex_err!("Failed to bind CUDA context: {e}"))?;

    let encoder = nvidia_video_codec_sdk::Encoder::initialize_with_cuda(cuda_context)
        .map_err(|e| vortex_err!("NVENC init failed: {e}"))?;

    let mut init_params =
        nvidia_video_codec_sdk::EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init_params
        .preset_guid(NV_ENC_PRESET_P4_GUID)
        .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOW_LATENCY)
        .display_aspect_ratio(width, height)
        .framerate(fps, 1)
        .enable_picture_type_decision();

    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12, init_params)
        .map_err(|e| vortex_err!("NVENC session start failed: {e}"))?;

    // Register NV12 buffer with NVENC as an external CUDA resource
    let mut registered_resource = session
        .register_generic_resource(
            (),
            NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            nv12_ptr as *mut std::ffi::c_void,
            width,
        )
        .map_err(|e| vortex_err!("NVENC register failed: {e}"))?;

    let mut output_bitstream = session
        .create_output_bitstream()
        .map_err(|e| vortex_err!("NVENC create bitstream failed: {e}"))?;

    // Create MPEG-TS muxer
    let mut mux = mux::TsMuxer::new(fps);

    // Wait for SRT connection
    let mut srt_sender = transport::SrtSender::listen(cli.srt_port).await?;

    tracing::info!(
        "Streaming {}x{} @ {}fps, bitrate={}Mbps",
        width,
        height,
        fps,
        cli.bitrate_mbps
    );

    let frame_duration = Duration::from_secs_f64(1.0 / f64::from(fps));
    let stream_start = tokio::time::Instant::now();
    let mut frame_idx: u64 = 0;

    loop {
        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;
        let mut batches = gpu_file.scan()?.into_array_stream()?;

        while let Some(batch) = batches.next().await.transpose()? {
            let span = tracing::info_span!("frame", frame = frame_idx);

            async {
                // Execute on GPU to get canonical form
                let canonical = batch.execute_cuda(&mut cuda_ctx).await?;
                let struct_arr = canonical.into_struct();

                // Extract R, G, B device pointers — flat u8 columns, no list wrapping.
                let r_prim = struct_arr
                    .unmasked_field_by_name("R")?
                    .to_canonical()?
                    .into_primitive();
                let r_ptr = r_prim.buffer_handle().cuda_device_ptr()?;

                let g_prim = struct_arr
                    .unmasked_field_by_name("G")?
                    .to_canonical()?
                    .into_primitive();
                let g_ptr = g_prim.buffer_handle().cuda_device_ptr()?;

                let b_prim = struct_arr
                    .unmasked_field_by_name("B")?
                    .to_canonical()?
                    .into_primitive();
                let b_ptr = b_prim.buffer_handle().cuda_device_ptr()?;

                // Launch RGB→NV12 kernel
                nv12::rgb_to_nv12_launch(
                    cuda_ctx.stream(),
                    &nv12_kernel,
                    r_ptr,
                    g_ptr,
                    b_ptr,
                    nv12_ptr,
                    width,
                    height,
                )?;

                // Sync stream before NVENC reads the NV12 buffer
                cuda_ctx
                    .stream()
                    .synchronize()
                    .map_err(|e| vortex_err!("CUDA stream sync failed: {e}"))?;

                // Encode frame to H.264
                session
                    .encode_picture(
                        &mut registered_resource,
                        &mut output_bitstream,
                        nvidia_video_codec_sdk::EncodePictureParams {
                            input_timestamp: frame_idx,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| vortex_err!("NVENC encode failed: {e}"))?;
                let lock = output_bitstream
                    .lock()
                    .map_err(|e| vortex_err!("NVENC lock bitstream failed: {e}"))?;
                let h264_nals = lock.data().to_vec();
                drop(lock);

                // Mux to MPEG-TS
                let ts_packets = mux.write_access_unit(&h264_nals, frame_idx);

                // Send over SRT
                srt_sender.send(ts_packets).await?;

                VortexResult::Ok(())
            }
            .instrument(span)
            .await?;

            // Pace to target FPS
            frame_idx += 1;
            let target = stream_start + frame_duration * frame_idx as u32;
            sleep_until(target).await;
        }

        if !cli.loop_playback {
            break;
        }
        tracing::info!("Looping back to start of file");
    }

    // Flush encoder
    session
        .end_of_stream()
        .map_err(|e| vortex_err!("NVENC flush failed: {e}"))?;

    srt_sender.close().await?;

    tracing::info!("Streamed {frame_idx} frames");
    Ok(())
}
