// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts a video file to a Vortex file with flat RGB frame data.
//!
//! Uses ffmpeg to decode the video into raw RGB24 pixels, then deinterleaves
//! into separate R, G, B planes and writes a Vortex file with schema:
//!
//! ```text
//! Struct {
//!   R: u8,
//!   G: u8,
//!   B: u8,
//! }
//! ```
//!
//! Each chunk corresponds to one frame, with width*height rows (one per pixel).
//! This flat layout avoids nested List arrays and maps directly to GPU buffers
//! after a CUDA scan.
//!
//! Usage:
//!   cargo run -p video-prep -- input.mp4 --output video.vortex --width 1920 --height 1080

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;

use clap::Parser;
use futures::stream;
use tracing_subscriber::EnvFilter;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
use vortex_cuda::layout::register_cuda_layout;

#[derive(Parser)]
#[command(
    name = "video-prep",
    about = "Convert a video file to Vortex format with RGB frame data"
)]
struct Cli {
    /// Path to input video file.
    input: PathBuf,

    /// Path to output Vortex file.
    #[arg(long)]
    output: PathBuf,

    /// Frame width in pixels.
    #[arg(long)]
    width: u32,

    /// Frame height in pixels.
    #[arg(long)]
    height: u32,

    /// Maximum number of frames to process (0 = all).
    #[arg(long, default_value_t = 0)]
    max_frames: usize,

    /// Path to a JSON file with per-frame detection booleans (from detect.py).
    /// If provided, adds a `has_hot_dog` bool column to the output.
    #[arg(long)]
    detections: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let pixels_per_frame = cli.width as usize * cli.height as usize;
    let rgb_frame_bytes = pixels_per_frame * 3; // RGB24

    // Load per-frame detection booleans if provided.
    let detections: Option<Vec<bool>> = if let Some(ref path) = cli.detections {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| vortex_err!("Failed to read detections JSON: {e}"))?;
        let bools: Vec<bool> = serde_json::from_str(&json_str)
            .map_err(|e| vortex_err!("Failed to parse detections JSON: {e}"))?;
        tracing::info!(
            "Loaded {} frame detections from {}",
            bools.len(),
            path.display()
        );
        Some(bools)
    } else {
        None
    };

    // Flat schema: each row is one pixel, each chunk is one frame.
    let u8_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
    let bool_dtype = DType::Bool(Nullability::NonNullable);

    let (field_names, field_dtypes) = if detections.is_some() {
        (
            FieldNames::from(["R", "G", "B", "has_hot_dog"]),
            vec![u8_dtype.clone(), u8_dtype.clone(), u8_dtype, bool_dtype],
        )
    } else {
        (
            FieldNames::from(["R", "G", "B"]),
            vec![u8_dtype.clone(), u8_dtype.clone(), u8_dtype],
        )
    };

    let struct_dtype = DType::Struct(
        StructFields::new(field_names, field_dtypes),
        Nullability::NonNullable,
    );

    // Launch ffmpeg to decode video into raw RGB24
    tracing::info!("Decoding {} with ffmpeg", cli.input.display());

    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-i",
            cli.input
                .to_str()
                .ok_or_else(|| vortex_err!("Invalid input path"))?,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", cli.width, cli.height),
            "-v",
            "error",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| vortex_err!("Failed to spawn ffmpeg: {e}"))?;

    let mut stdout = ffmpeg
        .stdout
        .take()
        .ok_or_else(|| vortex_err!("Failed to capture ffmpeg stdout"))?;

    let session = VortexSession::default();
    register_cuda_layout(&session);

    let write_strategy = WriteStrategyBuilder::default()
        .with_cuda_compatible_encodings()
        .with_row_block_size(pixels_per_frame)
        .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
        .build();

    let output_path = cli.output.clone();
    let mut output = async_fs::File::create(&output_path).await?;

    let mut frame_buf = vec![0u8; rgb_frame_bytes];
    let mut r_plane = vec![0u8; pixels_per_frame];
    let mut g_plane = vec![0u8; pixels_per_frame];
    let mut b_plane = vec![0u8; pixels_per_frame];
    let mut frame_count: usize = 0;

    let mut arrays = Vec::new();

    loop {
        // Read one frame from ffmpeg
        match stdout.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => vortex_bail!("Failed to read frame from ffmpeg: {e}"),
        }

        // Deinterleave RGB24 into separate planes
        for (i, chunk) in frame_buf.chunks_exact(3).enumerate() {
            r_plane[i] = chunk[0];
            g_plane[i] = chunk[1];
            b_plane[i] = chunk[2];
        }

        // Build flat PrimitiveArray<u8> per channel — one row per pixel.
        let r_arr = PrimitiveArray::new(r_plane.clone(), Validity::NonNullable);
        let g_arr = PrimitiveArray::new(g_plane.clone(), Validity::NonNullable);
        let b_arr = PrimitiveArray::new(b_plane.clone(), Validity::NonNullable);

        let (names, fields) = if let Some(ref dets) = detections {
            let has_hot_dog = dets.get(frame_count).copied().unwrap_or(false);
            let bool_arr: BoolArray = std::iter::repeat_n(has_hot_dog, pixels_per_frame).collect();
            (
                FieldNames::from(["R", "G", "B", "has_hot_dog"]),
                vec![
                    r_arr.into_array(),
                    g_arr.into_array(),
                    b_arr.into_array(),
                    bool_arr.into_array(),
                ],
            )
        } else {
            (
                FieldNames::from(["R", "G", "B"]),
                vec![r_arr.into_array(), g_arr.into_array(), b_arr.into_array()],
            )
        };

        let struct_arr =
            StructArray::try_new(names, fields, pixels_per_frame, Validity::NonNullable)?;

        arrays.push(struct_arr.into_array());

        frame_count += 1;
        if frame_count.is_multiple_of(100) {
            tracing::info!("Processed {frame_count} frames");
        }

        if cli.max_frames > 0 && frame_count >= cli.max_frames {
            break;
        }
    }

    // Wait for ffmpeg to finish
    let status = ffmpeg
        .wait()
        .map_err(|e| vortex_err!("Failed to wait for ffmpeg: {e}"))?;
    if !status.success() {
        tracing::warn!("ffmpeg exited with status: {status}");
    }

    if arrays.is_empty() {
        vortex_bail!("No frames decoded from input video");
    }

    tracing::info!("Writing {frame_count} frames to {}", output_path.display());

    // Write all arrays as a Vortex file
    let array_stream =
        ArrayStreamAdapter::new(struct_dtype, stream::iter(arrays.into_iter().map(Ok)));

    session
        .write_options()
        .with_strategy(write_strategy)
        .write(&mut output, array_stream)
        .await?;

    tracing::info!(
        "Done: {frame_count} frames written to {}",
        output_path.display()
    );

    Ok(())
}
