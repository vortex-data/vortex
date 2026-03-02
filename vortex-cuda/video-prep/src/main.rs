// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts a video file to a Vortex file with RGB frame data.
//!
//! Uses ffmpeg to decode the video into raw RGB24 pixels, then deinterleaves
//! into separate R, G, B planes and writes a Vortex file with schema:
//!
//! ```text
//! Struct {
//!   R: List<u8>,
//!   G: List<u8>,
//!   B: List<u8>,
//! }
//! ```
//!
//! Each row is one frame, each list has width*height elements.
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
use vortex::array::arrays::ListArray;
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
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let pixels_per_frame = cli.width as usize * cli.height as usize;
    let rgb_frame_bytes = pixels_per_frame * 3; // RGB24

    // Build the DType for our schema
    let list_u8_dtype = DType::List(
        Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
        Nullability::NonNullable,
    );
    let struct_dtype = DType::Struct(
        StructFields::new(
            FieldNames::from(["R", "G", "B"]),
            vec![list_u8_dtype.clone(), list_u8_dtype.clone(), list_u8_dtype],
        ),
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
        .with_row_block_size(1)
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

        // Build arrays directly for each plane
        let offsets =
            PrimitiveArray::new(vec![0u64, pixels_per_frame as u64], Validity::NonNullable);

        let r_elements = PrimitiveArray::new(r_plane.clone(), Validity::NonNullable);
        let r_list = ListArray::try_new(
            r_elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )?;

        let offsets =
            PrimitiveArray::new(vec![0u64, pixels_per_frame as u64], Validity::NonNullable);
        let g_elements = PrimitiveArray::new(g_plane.clone(), Validity::NonNullable);
        let g_list = ListArray::try_new(
            g_elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )?;

        let offsets =
            PrimitiveArray::new(vec![0u64, pixels_per_frame as u64], Validity::NonNullable);
        let b_elements = PrimitiveArray::new(b_plane.clone(), Validity::NonNullable);
        let b_list = ListArray::try_new(
            b_elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )?;

        let struct_arr = StructArray::try_new(
            FieldNames::from(["R", "G", "B"]),
            vec![
                r_list.into_array(),
                g_list.into_array(),
                b_list.into_array(),
            ],
            1,
            Validity::NonNullable,
        )?;

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
