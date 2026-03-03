// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU Video Demo: scans a Vortex file containing RGB video frames, converts to
//! NV12 on GPU, encodes to H.264 via NVENC, and streams over TCP.
//!
//! Usage:
//!   cargo run -p gpu-video-demo -- s3://bucket/video.vortex --width 1920 --height 1080
//!
//! Then connect with ffplay:
//!   ffplay tcp://localhost:9000

#![allow(unused_imports)]

use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use futures::StreamExt;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_BUFFER_FORMAT;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_CODEC_H264_GUID;
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

    /// TCP listener port for MPEG-TS streaming.
    #[arg(long, default_value_t = 9000)]
    port: u16,

    /// Path to write Perfetto trace output.
    #[arg(long)]
    perfetto: Option<PathBuf>,

    /// Loop the video file continuously.
    #[arg(long)]
    loop_playback: bool,
}

/// Parse H.264 NAL unit types from Annex B bitstream for debug logging.
#[cuda_available]
fn parse_nal_types(data: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                types.push(data[i + 3] & 0x1F);
                i += 4;
            } else if data[i + 2] == 0
                && i + 3 < data.len()
                && data[i + 3] == 1
                && i + 4 < data.len()
            {
                types.push(data[i + 4] & 0x1F);
                i += 5;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    types
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

    let cuda_context = cuda_ctx.stream().context().clone();

    // Allocate NV12 buffer on GPU (width * height * 3/2 for Y + UV planes)
    let pixels_per_frame = (width as usize) * (height as usize);
    let nv12_size = pixels_per_frame * 3 / 2;
    let nv12_device: CudaSlice<u8> = cuda_ctx.device_alloc(nv12_size)?;
    let nv12_ptr = {
        let (ptr, _sync) = nv12_device.device_ptr(cuda_ctx.stream());
        ptr
    };

    // Allocate RGB frame buffers on GPU for accumulating pixels across scan batches.
    // Scan batches may not align to frame boundaries (width * height pixels).
    let r_frame: CudaSlice<u8> = cuda_ctx.device_alloc(pixels_per_frame)?;
    let g_frame: CudaSlice<u8> = cuda_ctx.device_alloc(pixels_per_frame)?;
    let b_frame: CudaSlice<u8> = cuda_ctx.device_alloc(pixels_per_frame)?;
    let r_frame_ptr = {
        let (ptr, _sync) = r_frame.device_ptr(cuda_ctx.stream());
        ptr
    };
    let g_frame_ptr = {
        let (ptr, _sync) = g_frame.device_ptr(cuda_ctx.stream());
        ptr
    };
    let b_frame_ptr = {
        let (ptr, _sync) = b_frame.device_ptr(cuda_ctx.stream());
        ptr
    };
    let cu_stream = cuda_ctx.stream().cu_stream();

    // Create NVENC encoder.
    // NVENC internally pushes/pops the CUDA context, so we rebind it afterward
    // to ensure the pinned buffer pool on worker threads can access CUDA memory.
    let encoder = nvidia_video_codec_sdk::Encoder::initialize_with_cuda(cuda_context.clone())
        .map_err(|e| vortex_err!("NVENC init failed: {e}"))?;

    // Get preset config as baseline, then customize GOP and SPS/PPS repeat.
    let preset_config = encoder
        .get_preset_config(
            NV_ENC_CODEC_H264_GUID,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOW_LATENCY,
        )
        .map_err(|e| vortex_err!("NVENC get preset config failed: {e}"))?;

    let mut encode_config = preset_config.presetCfg;
    // IDR every 1 second so ffplay can start decoding mid-stream
    encode_config.gopLength = fps;
    unsafe {
        encode_config.encodeCodecConfig.h264Config.idrPeriod = fps;
        // Repeat SPS/PPS with every IDR so the decoder can join at any keyframe
        encode_config
            .encodeCodecConfig
            .h264Config
            .set_repeatSPSPPS(1);
    }

    let mut init_params =
        nvidia_video_codec_sdk::EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init_params
        .preset_guid(NV_ENC_PRESET_P4_GUID)
        .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOW_LATENCY)
        .display_aspect_ratio(width, height)
        .framerate(fps, 1)
        .enable_picture_type_decision()
        .encode_config(&mut encode_config);

    let nvenc_session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12, init_params)
        .map_err(|e| vortex_err!("NVENC session start failed: {e}"))?;

    // Create NVENC-managed input and output buffers
    let mut input_buffer = nvenc_session
        .create_input_buffer()
        .map_err(|e| vortex_err!("NVENC create input buffer failed: {e}"))?;

    let mut output_bitstream = nvenc_session
        .create_output_bitstream()
        .map_err(|e| vortex_err!("NVENC create bitstream failed: {e}"))?;

    // Rebind CUDA context after NVENC init to restore context stack.
    cuda_context
        .bind_to_thread()
        .map_err(|e| vortex_err!("Failed to rebind CUDA context: {e}"))?;

    // Create MPEG-TS muxer
    let mut mux = mux::TsMuxer::new(fps);

    // Debug: save first 5 seconds of TS output to file
    let mut debug_file = File::create("/tmp/gpu-video-debug.ts")?;

    // Wait for TCP connection
    let mut sender = transport::TcpSender::listen(cli.port).await?;

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
    let mut frame_fill: usize = 0;

    loop {
        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;
        let mut batches = gpu_file.scan()?.into_array_stream()?;

        while let Some(batch) = batches.next().await.transpose()? {
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

            let batch_pixels = r_prim.len();
            if frame_idx < 3 {
                tracing::info!(
                    frame = frame_idx,
                    batch_pixels,
                    pixels_per_frame,
                    frame_fill,
                    "batch received"
                );
            }

            // Copy batch pixels into frame accumulation buffers, processing
            // complete frames as they fill up.
            let mut batch_offset: usize = 0;
            while batch_offset < batch_pixels {
                let needed = pixels_per_frame - frame_fill;
                let available = batch_pixels - batch_offset;
                let copy_count = needed.min(available);

                // D2D copy from batch into frame accumulation buffers
                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        r_frame_ptr + frame_fill as u64,
                        r_ptr + batch_offset as u64,
                        copy_count,
                        cu_stream,
                    )
                    .result()
                    .map_err(|e| vortex_err!("D2D copy R failed: {e}"))?;
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        g_frame_ptr + frame_fill as u64,
                        g_ptr + batch_offset as u64,
                        copy_count,
                        cu_stream,
                    )
                    .result()
                    .map_err(|e| vortex_err!("D2D copy G failed: {e}"))?;
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        b_frame_ptr + frame_fill as u64,
                        b_ptr + batch_offset as u64,
                        copy_count,
                        cu_stream,
                    )
                    .result()
                    .map_err(|e| vortex_err!("D2D copy B failed: {e}"))?;
                }

                frame_fill += copy_count;
                batch_offset += copy_count;

                if frame_fill == pixels_per_frame {
                    // Full frame accumulated — convert, encode, and send.
                    let _span =
                        tracing::info_span!("encode_frame", frame = frame_idx).entered();

                    // Launch RGB→NV12 kernel on the complete frame
                    nv12::rgb_to_nv12_launch(
                        cuda_ctx.stream(),
                        &nv12_kernel,
                        r_frame_ptr,
                        g_frame_ptr,
                        b_frame_ptr,
                        nv12_ptr,
                        width,
                        height,
                    )?;

                    cuda_ctx
                        .stream()
                        .synchronize()
                        .map_err(|e| vortex_err!("CUDA stream sync failed: {e}"))?;

                    // Download NV12 data from GPU to host
                    let nv12_host: Vec<u8> = cuda_ctx
                        .stream()
                        .clone_dtoh(&nv12_device)
                        .map_err(|e| vortex_err!("NV12 dtoh copy failed: {e}"))?;

                    // Debug dumps for first frame
                    if frame_idx == 0 {
                        std::fs::write("/tmp/frame0.nv12", &nv12_host)?;
                        tracing::info!(
                            nv12_bytes = nv12_host.len(),
                            "saved raw NV12 frame 0"
                        );
                    }

                    // Write to NVENC input buffer (pitch-aware for NV12)
                    unsafe {
                        let mut lock = input_buffer
                            .lock()
                            .map_err(|e| vortex_err!("NVENC lock input failed: {e}"))?;
                        if frame_idx == 0 {
                            tracing::info!(
                                pitch = lock.pitch(),
                                width,
                                "NVENC input buffer pitch"
                            );
                        }
                        lock.write_nv12(&nv12_host, width, height);
                    }
                    nvenc_session
                        .encode_picture(
                            &mut input_buffer,
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

                    if frame_idx < 5 {
                        let nal_types = parse_nal_types(&h264_nals);
                        tracing::info!(
                            frame = frame_idx,
                            h264_bytes = h264_nals.len(),
                            ?nal_types,
                            "encoded frame"
                        );
                    }

                    // Mux to MPEG-TS
                    let ts_packets = mux.write_access_unit(&h264_nals, frame_idx);

                    if frame_idx < 300 {
                        debug_file.write_all(&ts_packets)?;
                    }

                    // Send over TCP
                    sender.send(ts_packets).await?;

                    frame_fill = 0;
                    frame_idx += 1;
                    let target = stream_start + frame_duration * frame_idx as u32;
                    sleep_until(target).await;
                }
            }
        }

        if !cli.loop_playback {
            break;
        }
        tracing::info!("Looping back to start of file");
    }

    // Flush encoder
    nvenc_session
        .end_of_stream()
        .map_err(|e| vortex_err!("NVENC flush failed: {e}"))?;

    sender.close().await?;

    tracing::info!("Streamed {frame_idx} frames");
    Ok(())
}
