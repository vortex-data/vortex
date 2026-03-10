// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU Video Demo: scans a Vortex file containing RGB video frames, converts to
//! NV12 on GPU, encodes to H.264 via NVENC, and streams over TCP.
//!
//! The pipeline uses two threads for parallelism:
//! - **Main async task**: scans batches → GPU decompression → RGB→NV12 kernel
//! - **Encoder thread**: NVENC encode → H264 bytes back to main
//!
//! Double-buffered NV12 lets the main task write the next frame while the encoder
//! reads the current one.
//!
//! Usage:
//!   cargo run -p gpu-video-demo -- s3://bucket/video.vortex --width 1920 --height 1080
//!
//! Then connect with ffplay:
//!   ffplay tcp://localhost:9000

#![allow(unused_imports)]

use std::ffi::c_void;
use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use clap::Parser;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::sys::CUdeviceptr;
use futures::StreamExt;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_BUFFER_FORMAT;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_CODEC_H264_GUID;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_INPUT_RESOURCE_TYPE;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_PARAMS_RC_MODE;
use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_PRESET_P1_GUID;
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
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::col;
use vortex::expr::eq;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::select;
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

/// RAII wrapper for GPU memory allocated with `cuMemAlloc` (synchronous).
///
/// NVENC requires `cuMemAlloc`-allocated memory; stream-ordered allocations
/// from `cuMemAllocAsync` are not supported for resource registration.
#[cuda_available]
struct SyncDeviceBuf {
    ptr: CUdeviceptr,
    _len: usize,
}

#[cuda_available]
impl SyncDeviceBuf {
    fn alloc(len: usize) -> VortexResult<Self> {
        let ptr = unsafe { cudarc::driver::result::malloc_sync(len) }
            .map_err(|e| vortex_err!("cuMemAlloc failed: {e}"))?;
        Ok(Self { ptr, _len: len })
    }

    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

#[cuda_available]
impl Drop for SyncDeviceBuf {
    fn drop(&mut self) {
        unsafe {
            // Ignore errors during cleanup — the CUDA context may already be torn down.
            drop(cudarc::driver::result::free_sync(self.ptr));
        }
    }
}

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

    /// Output MPEG-TS file path. If set, writes to file instead of TCP streaming.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Path to write Perfetto trace output.
    #[arg(long)]
    perfetto: Option<PathBuf>,

    /// Loop the video file continuously.
    #[arg(long)]
    loop_playback: bool,

    /// Comma-separated list of columns to project (e.g. "R,G,B" or "G").
    /// Unprojected channels are zero-filled (black). Reduces S3 I/O.
    #[arg(long, value_delimiter = ',')]
    columns: Option<Vec<String>>,

    /// Posterize (discretize) each color channel to N levels on the GPU.
    /// E.g. --posterize 4 maps each channel to {0, 85, 170, 255}.
    #[arg(long)]
    posterize: Option<u32>,

    /// Filter to only frames where the given boolean column is true.
    /// E.g. --filter detected. Requires the Vortex file to have that column
    /// (generated by video-prep with --detections).
    #[arg(long)]
    filter: Option<String>,
}

/// Sent from main task → encoder thread when an NV12 buffer is ready to encode.
struct Nv12ReadyFrame {
    /// Which of the double buffers contains the NV12 data (0 or 1).
    buf_idx: usize,
    frame_idx: u64,
}

/// Sent from encoder thread → main task with the encoded H.264 bitstream.
struct EncodedFrame {
    h264_nals: Vec<u8>,
    frame_idx: u64,
}

/// Signals the encoder thread to shut down gracefully.
enum EncoderMsg {
    Frame(Nv12ReadyFrame),
    Shutdown,
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
    let is_file_output = cli.output.is_some();

    // Determine which columns to project.
    let projected_columns: Vec<String> = cli
        .columns
        .unwrap_or_else(|| vec!["R".into(), "G".into(), "B".into()]);
    let has_r = projected_columns.iter().any(|c| c == "R");
    let has_g = projected_columns.iter().any(|c| c == "G");
    let has_b = projected_columns.iter().any(|c| c == "B");
    tracing::info!(
        ?projected_columns,
        "column projection (unprojected channels will be black)"
    );

    // Load GPU kernels
    let nv12_kernel = nv12::load_rgb_to_nv12_kernel(&session)?;

    let cuda_context = cuda_ctx.stream().context().clone();

    // Allocate double-buffered NV12 on GPU (width * height * 3/2 for Y + UV planes).
    // Use cuMemAlloc (synchronous) because NVENC cannot register stream-ordered
    // allocations from cuMemAllocAsync.
    let pixels_per_frame = (width as usize) * (height as usize);
    let nv12_size = pixels_per_frame * 3 / 2;

    let nv12_bufs = [
        SyncDeviceBuf::alloc(nv12_size)?,
        SyncDeviceBuf::alloc(nv12_size)?,
    ];
    let nv12_ptrs = [nv12_bufs[0].device_ptr(), nv12_bufs[1].device_ptr()];

    // Allocate RGB frame buffers on GPU for accumulating pixels across scan batches.
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

    // Zero-fill unprojected channel buffers so they render as black.
    if !has_r {
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(r_frame_ptr, 0, pixels_per_frame, cu_stream)
                .result()
                .map_err(|e| vortex_err!("memset R failed: {e}"))?;
        }
    }
    if !has_g {
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(g_frame_ptr, 0, pixels_per_frame, cu_stream)
                .result()
                .map_err(|e| vortex_err!("memset G failed: {e}"))?;
        }
    }
    if !has_b {
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(b_frame_ptr, 0, pixels_per_frame, cu_stream)
                .result()
                .map_err(|e| vortex_err!("memset B failed: {e}"))?;
        }
    }

    // Create NVENC encoder.
    // NVENC internally pushes/pops the CUDA context, so we rebind it afterward
    // to ensure the pinned buffer pool on worker threads can access CUDA memory.
    let encoder = nvidia_video_codec_sdk::Encoder::initialize_with_cuda(cuda_context.clone())
        .map_err(|e| vortex_err!("NVENC init failed: {e}"))?;

    let preset_guid = NV_ENC_PRESET_P1_GUID;
    let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOW_LATENCY;

    let preset_config = encoder
        .get_preset_config(NV_ENC_CODEC_H264_GUID, preset_guid, tuning_info)
        .map_err(|e| vortex_err!("NVENC get preset config failed: {e}"))?;

    let mut encode_config = preset_config.presetCfg;
    // IDR every 1 second so ffplay can start decoding mid-stream
    encode_config.gopLength = fps;
    // Set VBR rate control with the user-specified bitrate
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_VBR;
    encode_config.rcParams.averageBitRate = bitrate;
    encode_config.rcParams.maxBitRate = bitrate * 2;
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
        .preset_guid(preset_guid)
        .tuning_info(tuning_info)
        .display_aspect_ratio(width, height)
        .framerate(fps, 1)
        .enable_picture_type_decision()
        .encode_config(&mut encode_config);

    let nvenc_session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12, init_params)
        .map_err(|e| vortex_err!("NVENC session start failed: {e}"))?;

    // Rebind CUDA context after NVENC init to restore context stack.
    cuda_context
        .bind_to_thread()
        .map_err(|e| vortex_err!("Failed to rebind CUDA context: {e}"))?;

    // Channels: bounded(1) for backpressure so main blocks if encoder is behind.
    let (frame_tx, frame_rx) = mpsc::sync_channel::<EncoderMsg>(1);
    let (encoded_tx, encoded_rx) = mpsc::channel::<EncodedFrame>();
    // Oneshot for the encoder thread to report init success/failure.
    let (init_tx, init_rx) = mpsc::sync_channel::<VortexResult<()>>(0);

    // Pass raw NV12 device pointers to the encoder thread. Registration must
    // happen on the thread that owns the Session to satisfy borrow lifetimes.
    let nv12_ptr_0 = nv12_ptrs[0];
    let nv12_ptr_1 = nv12_ptrs[1];

    // Spawn encoder thread — owns Session, registers resources, encodes frames.
    let encoder_cuda_ctx = cuda_context.clone();
    let encoder_handle = std::thread::spawn(move || -> VortexResult<()> {
        // Run init in a closure so we can report errors back before exiting.
        let init = || -> VortexResult<()> {
            encoder_cuda_ctx
                .bind_to_thread()
                .map_err(|e| vortex_err!("Encoder thread: failed to bind CUDA context: {e}"))?;

            // Register both NV12 GPU buffers directly with NVENC (zero-copy).
            // The `()` marker is safe because the main task keeps `nv12_bufs` alive
            // for the entire duration of the encoder thread.
            let mut nv12_resource_0 = nvenc_session
                .register_generic_resource(
                    (),
                    NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    nv12_ptr_0 as *mut c_void,
                    width,
                )
                .map_err(|e| vortex_err!("NVENC register NV12 buf 0 failed: {e}"))?;

            let mut nv12_resource_1 = nvenc_session
                .register_generic_resource(
                    (),
                    NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    nv12_ptr_1 as *mut c_void,
                    width,
                )
                .map_err(|e| vortex_err!("NVENC register NV12 buf 1 failed: {e}"))?;

            let mut output_bitstream = nvenc_session
                .create_output_bitstream()
                .map_err(|e| vortex_err!("NVENC create bitstream failed: {e}"))?;

            // Signal init success to the main thread.
            init_tx
                .send(Ok(()))
                .map_err(|_| vortex_err!("Main thread not waiting for init"))?;

            loop {
                let msg = frame_rx
                    .recv()
                    .map_err(|e| vortex_err!("Encoder thread: channel recv failed: {e}"))?;

                let frame = match msg {
                    EncoderMsg::Frame(f) => f,
                    EncoderMsg::Shutdown => break,
                };

                let _span = tracing::info_span!("nvenc_encode", frame = frame.frame_idx).entered();

                let input = if frame.buf_idx == 0 {
                    &mut nv12_resource_0
                } else {
                    &mut nv12_resource_1
                };
                nvenc_session
                    .encode_picture(
                        input,
                        &mut output_bitstream,
                        nvidia_video_codec_sdk::EncodePictureParams {
                            input_timestamp: frame.frame_idx,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| vortex_err!("NVENC encode failed: {e}"))?;

                let lock = output_bitstream
                    .lock()
                    .map_err(|e| vortex_err!("NVENC lock bitstream failed: {e}"))?;
                let h264_nals = lock.data().to_vec();
                drop(lock);

                encoded_tx
                    .send(EncodedFrame {
                        h264_nals,
                        frame_idx: frame.frame_idx,
                    })
                    .map_err(|e| vortex_err!("Encoder thread: send encoded failed: {e}"))?;
            }

            // Drop resources before flushing (NVENC requires this ordering)
            drop(nv12_resource_0);
            drop(nv12_resource_1);
            drop(output_bitstream);

            // Flush encoder
            nvenc_session
                .end_of_stream()
                .map_err(|e| vortex_err!("NVENC flush failed: {e}"))?;

            Ok(())
        };

        let result = init();
        if let Err(ref e) = result {
            tracing::error!("Encoder thread failed: {e}");
            // Try to report the error; ignore if main thread already moved on.
            drop(init_tx.send(Err(vortex_err!("Encoder thread init failed: {e}"))));
        }
        result
    });

    // Wait for encoder thread to finish initialization.
    init_rx
        .recv()
        .map_err(|_| vortex_err!("Encoder thread exited before signaling init"))?
        .map_err(|e| vortex_err!("Encoder thread init error: {e}"))?;

    tracing::info!("Encoder thread initialized successfully");

    // Create MPEG-TS muxer
    let mut mux = mux::TsMuxer::new(fps);

    // Output: either file or TCP stream.
    let mut output_file = if let Some(ref path) = cli.output {
        tracing::info!("Writing MPEG-TS to {}", path.display());
        Some(std::io::BufWriter::new(File::create(path)?))
    } else {
        None
    };
    let mut tcp_sender = if cli.output.is_none() {
        Some(transport::TcpSender::listen(cli.port).await?)
    } else {
        None
    };

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
    let mut buf_idx: usize = 0;

    let mut stage_scan_ns: u64 = 0;
    let mut stage_cuda_ns: u64 = 0;
    let mut stage_extract_ns: u64 = 0;
    let mut stage_d2d_ns: u64 = 0;
    let mut stage_nv12_ns: u64 = 0;
    let mut stage_sync_ns: u64 = 0;
    let mut stage_recv_ns: u64 = 0;
    let mut stage_mux_ns: u64 = 0;
    let mut stage_send_ns: u64 = 0;
    let mut batch_count: u64 = 0;

    loop {
        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;
        // Build projection: optionally wrap each column with posterize.
        let projection = if let Some(levels) = cli.posterize {
            use vortex_cuda::scalar_fn::posterize::posterize;
            pack(
                projected_columns
                    .iter()
                    .map(|c| (c.as_str(), posterize(col(c.as_str()), levels))),
                Nullability::NonNullable,
            )
        } else {
            select(
                projected_columns
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                root(),
            )
        };

        let filter = cli
            .filter
            .as_ref()
            .map(|column| eq(col(column.as_str()), lit(true)));

        let mut batches = gpu_file
            .scan()?
            .with_projection(projection)
            .with_some_filter(filter)
            .with_concurrency(16)
            .into_array_stream()?;

        while let Some(batch) = {
            let t = std::time::Instant::now();
            let b = batches.next().await.transpose()?;
            stage_scan_ns += t.elapsed().as_nanos() as u64;
            b
        } {
            // Execute on GPU to get canonical form
            let t = std::time::Instant::now();
            let canonical = batch.execute_cuda(&mut cuda_ctx).await?;
            let struct_arr = canonical.into_struct();
            stage_cuda_ns += t.elapsed().as_nanos() as u64;

            // Extract projected channel device pointers.
            let t = std::time::Instant::now();
            let r_ptr = if has_r {
                let p = struct_arr
                    .unmasked_field_by_name("R")?
                    .to_canonical()?
                    .into_primitive();
                Some(p)
            } else {
                None
            };
            let g_ptr = if has_g {
                let p = struct_arr
                    .unmasked_field_by_name("G")?
                    .to_canonical()?
                    .into_primitive();
                Some(p)
            } else {
                None
            };
            let b_ptr = if has_b {
                let p = struct_arr
                    .unmasked_field_by_name("B")?
                    .to_canonical()?
                    .into_primitive();
                Some(p)
            } else {
                None
            };
            stage_extract_ns += t.elapsed().as_nanos() as u64;

            // Use any projected channel to determine batch size.
            let batch_pixels = r_ptr
                .as_ref()
                .or(g_ptr.as_ref())
                .or(b_ptr.as_ref())
                .map(|p| p.len())
                .ok_or_else(|| vortex_err!("no columns projected"))?;
            batch_count += 1;
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

                // D2D copy from batch into frame accumulation buffers (projected channels only)
                let t = std::time::Instant::now();
                unsafe {
                    if let Some(ref rp) = r_ptr {
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            r_frame_ptr + frame_fill as u64,
                            rp.buffer_handle().cuda_device_ptr()? + batch_offset as u64,
                            copy_count,
                            cu_stream,
                        )
                        .result()
                        .map_err(|e| vortex_err!("D2D copy R failed: {e}"))?;
                    }
                    if let Some(ref gp) = g_ptr {
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            g_frame_ptr + frame_fill as u64,
                            gp.buffer_handle().cuda_device_ptr()? + batch_offset as u64,
                            copy_count,
                            cu_stream,
                        )
                        .result()
                        .map_err(|e| vortex_err!("D2D copy G failed: {e}"))?;
                    }
                    if let Some(ref bp) = b_ptr {
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            b_frame_ptr + frame_fill as u64,
                            bp.buffer_handle().cuda_device_ptr()? + batch_offset as u64,
                            copy_count,
                            cu_stream,
                        )
                        .result()
                        .map_err(|e| vortex_err!("D2D copy B failed: {e}"))?;
                    }
                }

                stage_d2d_ns += t.elapsed().as_nanos() as u64;

                frame_fill += copy_count;
                batch_offset += copy_count;

                if frame_fill == pixels_per_frame {
                    // Full frame accumulated — convert and hand off to encoder.
                    let _span = tracing::info_span!("produce_frame", frame = frame_idx).entered();

                    // Launch RGB→NV12 kernel into current double buffer
                    let t = std::time::Instant::now();
                    nv12::rgb_to_nv12_launch(
                        cuda_ctx.stream(),
                        &nv12_kernel,
                        r_frame_ptr,
                        g_frame_ptr,
                        b_frame_ptr,
                        nv12_ptrs[buf_idx],
                        width,
                        height,
                    )?;
                    stage_nv12_ns += t.elapsed().as_nanos() as u64;

                    // Wait for NV12 kernel to finish before encoder reads the buffer
                    let t = std::time::Instant::now();
                    cuda_ctx
                        .stream()
                        .synchronize()
                        .map_err(|e| vortex_err!("CUDA stream sync failed: {e}"))?;
                    stage_sync_ns += t.elapsed().as_nanos() as u64;

                    // Collect the PREVIOUS frame's encoded output. The encoder
                    // was processing it in parallel while we scanned this frame.
                    if frame_idx > 0 {
                        let t = std::time::Instant::now();
                        let encoded = encoded_rx
                            .recv()
                            .map_err(|e| vortex_err!("Recv encoded frame failed: {e}"))?;
                        stage_recv_ns += t.elapsed().as_nanos() as u64;

                        if encoded.frame_idx < 5 {
                            let nal_types = parse_nal_types(&encoded.h264_nals);
                            tracing::info!(
                                frame = encoded.frame_idx,
                                h264_bytes = encoded.h264_nals.len(),
                                ?nal_types,
                                "encoded frame"
                            );
                        }

                        // Send current frame to encoder BEFORE mux+write so
                        // encoding overlaps with output I/O.
                        frame_tx
                            .send(EncoderMsg::Frame(Nv12ReadyFrame { buf_idx, frame_idx }))
                            .map_err(|e| vortex_err!("Send to encoder failed: {e}"))?;

                        let t = std::time::Instant::now();
                        let ts_packets =
                            mux.write_access_unit(&encoded.h264_nals, encoded.frame_idx);
                        stage_mux_ns += t.elapsed().as_nanos() as u64;

                        let t = std::time::Instant::now();
                        if let Some(ref mut f) = output_file {
                            f.write_all(&ts_packets)?;
                        } else if let Some(ref mut s) = tcp_sender {
                            s.send(ts_packets).await?;
                        }
                        stage_send_ns += t.elapsed().as_nanos() as u64;

                        // Pace to target FPS only when streaming live.
                        if output_file.is_none() {
                            let target =
                                stream_start + frame_duration * (encoded.frame_idx + 1) as u32;
                            sleep_until(target).await;
                        }
                    } else {
                        // First frame: no previous frame to collect, just send to encoder.
                        frame_tx
                            .send(EncoderMsg::Frame(Nv12ReadyFrame { buf_idx, frame_idx }))
                            .map_err(|e| vortex_err!("Send to encoder failed: {e}"))?;
                    }

                    // Flip to the other NV12 buffer for the next frame
                    buf_idx = 1 - buf_idx;
                    frame_fill = 0;
                    frame_idx += 1;
                }
            }
        }

        if !cli.loop_playback {
            break;
        }
        tracing::info!("Looping back to start of file");
    }

    // Drain the last encoded frame (was submitted but not yet collected)
    if frame_idx > 0 {
        let encoded = encoded_rx
            .recv()
            .map_err(|e| vortex_err!("Recv final encoded frame failed: {e}"))?;
        let ts_packets = mux.write_access_unit(&encoded.h264_nals, encoded.frame_idx);
        if let Some(ref mut f) = output_file {
            f.write_all(&ts_packets)?;
        } else if let Some(ref mut s) = tcp_sender {
            s.send(ts_packets).await?;
        }
    }

    // Signal encoder thread to shut down and wait for it
    frame_tx
        .send(EncoderMsg::Shutdown)
        .map_err(|e| vortex_err!("Send shutdown to encoder failed: {e}"))?;

    encoder_handle
        .join()
        .map_err(|_| vortex_err!("Encoder thread panicked"))?
        .map_err(|e| vortex_err!("Encoder thread error: {e}"))?;

    // Flush file or close TCP connection.
    if let Some(mut f) = output_file {
        f.flush()?;
    } else if let Some(s) = tcp_sender {
        s.close().await?;
    }

    let elapsed = stream_start.elapsed();
    let avg_fps = frame_idx as f64 / elapsed.as_secs_f64();
    let bc = batch_count.max(1) as f64;
    let fc = (frame_idx.max(1)) as f64;
    tracing::info!(
        frames = frame_idx,
        batches = batch_count,
        elapsed_secs = format!("{:.2}", elapsed.as_secs_f64()),
        avg_fps = format!("{:.1}", avg_fps),
        "streaming complete"
    );
    tracing::info!(
        scan_ms = format!("{:.1}", stage_scan_ns as f64 / 1e6),
        cuda_ms = format!("{:.1}", stage_cuda_ns as f64 / 1e6),
        extract_ms = format!("{:.1}", stage_extract_ns as f64 / 1e6),
        d2d_ms = format!("{:.1}", stage_d2d_ns as f64 / 1e6),
        nv12_ms = format!("{:.1}", stage_nv12_ns as f64 / 1e6),
        sync_ms = format!("{:.1}", stage_sync_ns as f64 / 1e6),
        recv_ms = format!("{:.1}", stage_recv_ns as f64 / 1e6),
        mux_ms = format!("{:.1}", stage_mux_ns as f64 / 1e6),
        send_ms = format!("{:.1}", stage_send_ns as f64 / 1e6),
        "total stage time (ms)"
    );
    tracing::info!(
        scan_per_batch_us = format!("{:.0}", stage_scan_ns as f64 / 1e3 / bc),
        cuda_per_batch_us = format!("{:.0}", stage_cuda_ns as f64 / 1e3 / bc),
        extract_per_batch_us = format!("{:.0}", stage_extract_ns as f64 / 1e3 / bc),
        d2d_per_frame_us = format!("{:.0}", stage_d2d_ns as f64 / 1e3 / fc),
        nv12_per_frame_us = format!("{:.0}", stage_nv12_ns as f64 / 1e3 / fc),
        sync_per_frame_us = format!("{:.0}", stage_sync_ns as f64 / 1e3 / fc),
        recv_per_frame_us = format!("{:.0}", stage_recv_ns as f64 / 1e3 / fc),
        mux_per_frame_us = format!("{:.0}", stage_mux_ns as f64 / 1e3 / fc),
        send_per_frame_us = format!("{:.0}", stage_send_ns as f64 / 1e3 / fc),
        "per-unit stage time (us)"
    );
    Ok(())
}
