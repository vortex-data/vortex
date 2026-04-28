// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

//! Convert a tiled GeoTIFF into a Vortex file with one row per tile.
//!
//! The output schema is:
//!
//! ```text
//! struct {
//!   minx:   f64,
//!   miny:   f64,
//!   maxx:   f64,
//!   maxy:   f64,
//!   band_0: list<u8>,
//!   band_1: list<u8>,
//!   ...
//! }
//! ```
//!
//! Each row holds one tile's footprint (in CRS coordinates derived from the GeoTIFF
//! geotransform) and the raw bytes of every band, separated. `List<u8>` is used over
//! `ListView<u8>` because tile bytes are written contiguously and we never reorder elements.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example geotiff_to_vortex -p vortex --release -- \
//!     --input m_3211428_sw_11_030_20230617_20240228.tif \
//!     --output naip.vortex \
//!     --compression compact \
//!     --block-size-bytes 4194304
//! ```

use std::fs::File;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use tiff::decoder::Decoder;
use tiff::decoder::DecodingResult;
use tiff::tags::Tag;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CompressionMode {
    /// Default BtrBlocks cascade (no PCO, no Zstd).
    Btrblocks,
    /// BtrBlocks with the "compact" extras: PCO (numerics) and Zstd (strings/buffers).
    Compact,
}

#[derive(Parser, Debug)]
#[command(about = "Convert a tiled GeoTIFF into a Vortex file (one row per tile).")]
struct Args {
    /// Path to the source tiled GeoTIFF.
    #[arg(long, short)]
    input: PathBuf,

    /// Output Vortex file path.
    #[arg(long, short)]
    output: PathBuf,

    /// Compression strategy for the file write pipeline.
    #[arg(long, value_enum, default_value_t = CompressionMode::Btrblocks)]
    compression: CompressionMode,

    /// Target row-group block size in bytes. The default of 2 MiB matches Vortex's
    /// `BufferedStrategy` default. The example translates this byte budget into a row count
    /// using the per-tile size, since `WriteStrategyBuilder::with_row_block_size` is
    /// expressed in rows.
    #[arg(long, default_value_t = 2 * 1024 * 1024)]
    block_size_bytes: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let tiff_size = std::fs::metadata(&args.input)?.len();
    let raster = read_geotiff(&args.input)?;

    println!(
        "loaded {tiles} tiles ({tile_w}x{tile_h}, {bands} bands, {raster_w}x{raster_h} raster)",
        tiles = raster.num_tiles(),
        tile_w = raster.tile_w,
        tile_h = raster.tile_h,
        bands = raster.bands,
        raster_w = raster.image_w,
        raster_h = raster.image_h,
    );
    println!(
        "geotransform: origin=({:.3}, {:.3}) scale=({:.6}, {:.6})",
        raster.origin_x, raster.origin_y, raster.scale_x, raster.scale_y,
    );

    let array = build_struct_array(&raster)?;

    // Pick a row-group size in rows that approximates the requested byte budget.
    let row_bytes_estimate =
        (raster.tile_w as usize).saturating_mul(raster.tile_h as usize) + 4 * 8;
    let rows_per_block = args
        .block_size_bytes
        .div_ceil(row_bytes_estimate.max(1))
        .max(1);

    let compressor = match args.compression {
        CompressionMode::Btrblocks => BtrBlocksCompressorBuilder::default(),
        CompressionMode::Compact => BtrBlocksCompressorBuilder::default().with_compact(),
    };

    let strategy = WriteStrategyBuilder::default()
        .with_btrblocks_builder(compressor)
        .with_row_block_size(rows_per_block)
        .build();

    let session = VortexSession::default();
    let len = array.len();
    let stream = array.into_array().to_array_stream();

    let summary = session
        .write_options()
        .with_strategy(strategy)
        .write(tokio::fs::File::create(&args.output).await?, stream)
        .await?;

    let vortex_size = summary.size();
    println!(
        "wrote {} rows to {} ({} bytes)",
        len,
        args.output.display(),
        vortex_size,
    );
    println!(
        "geotiff: {tiff} bytes  vortex: {vx} bytes  ratio: {ratio:.3}x  ({delta:+.1}%)",
        tiff = tiff_size,
        vx = vortex_size,
        ratio = tiff_size as f64 / vortex_size as f64,
        delta = (vortex_size as f64 - tiff_size as f64) * 100.0 / tiff_size as f64,
    );

    Ok(())
}

/// All the per-raster information the converter needs in one pass: tile dimensions, bands,
/// geotransform, and the raw bytes for every (band, tile) pair.
struct Raster {
    image_w: u32,
    image_h: u32,
    tile_w: u32,
    tile_h: u32,
    bands: usize,
    origin_x: f64,
    origin_y: f64,
    scale_x: f64,
    scale_y: f64,
    /// `tile_bands[band][tile]` = raw bytes of that band's tile (always `tile_w * tile_h` bytes).
    tile_bands: Vec<Vec<Vec<u8>>>,
}

impl Raster {
    fn num_tiles(&self) -> usize {
        self.tile_bands.first().map(Vec::len).unwrap_or(0)
    }

    fn tiles_per_row(&self) -> u32 {
        self.image_w.div_ceil(self.tile_w)
    }
}

fn read_geotiff(path: &PathBuf) -> Result<Raster> {
    let mut decoder = Decoder::new(File::open(path).with_context(|| {
        format!("failed to open GeoTIFF at {}", path.display())
    })?)?;

    // Tile sizes; bail if the file is striped instead of tiled.
    let tile_w = decoder
        .get_tag_u32(Tag::TileWidth)
        .context("input is not a tiled GeoTIFF (missing TileWidth tag)")?;
    let tile_h = decoder
        .get_tag_u32(Tag::TileLength)
        .context("input is not a tiled GeoTIFF (missing TileLength tag)")?;

    let (image_w, image_h) = decoder.dimensions()?;

    // Probe the first chunk to determine how many bands the tiff crate is actually decoding.
    // `SamplesPerPixel` may exceed this when the file declares extra samples (e.g. NIR alongside
    // RGB) that get dropped to fit the photometric `ColorType` reported by the decoder.
    let first_chunk = read_chunk_u8(&mut decoder, 0)?;
    let tile_pixels = (tile_w as usize) * (tile_h as usize);
    if first_chunk.is_empty() || first_chunk.len() % tile_pixels != 0 {
        bail!(
            "first chunk size {} is not a multiple of tile pixel count {}",
            first_chunk.len(),
            tile_pixels
        );
    }
    let bands = first_chunk.len() / tile_pixels;

    // Default planar configuration is 1 (chunky / interleaved). Both 1 and 2 are useful,
    // but the deinterleave path differs, so we handle them separately.
    let planar_cfg = decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1);

    // Geotransform via ModelTiepoint + ModelPixelScale. The full ModelTransformation tag is rarer
    // and not handled here; users with rotated/skewed CRS transforms can extend this branch.
    let tiepoint = decoder.get_tag_f64_vec(Tag::ModelTiepointTag)?;
    let scale = decoder.get_tag_f64_vec(Tag::ModelPixelScaleTag)?;
    if tiepoint.len() < 6 || scale.len() < 2 {
        bail!(
            "GeoTIFF is missing a usable geotransform (tiepoint len {}, scale len {})",
            tiepoint.len(),
            scale.len()
        );
    }
    // Tiepoint format: (i, j, k, x, y, z). Standard north-up convention; `scale.1` is the
    // *positive* pixel height in CRS units, so y decreases as j increases.
    let origin_x = tiepoint[3] - tiepoint[0] * scale[0];
    let origin_y = tiepoint[4] + tiepoint[1] * scale[1];
    let scale_x = scale[0];
    let scale_y = scale[1];

    let tiles_x = image_w.div_ceil(tile_w);
    let tiles_y = image_h.div_ceil(tile_h);
    let tile_count = (tiles_x as usize) * (tiles_y as usize);
    let tile_bytes = (tile_w as usize) * (tile_h as usize);

    // Eagerly decode every tile. With a few hundred tiles this is fine; for very large rasters
    // the converter would want to stream instead.
    let mut tile_bands: Vec<Vec<Vec<u8>>> = (0..bands)
        .map(|_| Vec::with_capacity(tile_count))
        .collect();

    let mut first_chunk_opt = Some(first_chunk);
    for tile_idx in 0..tile_count {
        let tile_idx_u32 =
            u32::try_from(tile_idx).context("tile count exceeds u32 range")?;
        // Edge tiles are clipped by the tiff crate's `read_chunk`, so the chunk's actual
        // data dimensions can be smaller than the nominal tile dimensions. We always copy
        // into a full tile_w x tile_h buffer per band, leaving the padded region as zeros.
        let (data_w, data_h) = decoder.chunk_data_dimensions(tile_idx_u32);
        let data_w = data_w as usize;
        let data_h = data_h as usize;
        match planar_cfg {
            1 => {
                let bytes = if tile_idx == 0 {
                    first_chunk_opt
                        .take()
                        .unwrap_or_else(Vec::new)
                } else {
                    read_chunk_u8(&mut decoder, tile_idx_u32)?
                };
                let expected = data_w * data_h * bands;
                if bytes.len() != expected {
                    bail!(
                        "unexpected chunk size at tile {}: got {} bytes, expected {} ({}x{}x{})",
                        tile_idx,
                        bytes.len(),
                        expected,
                        data_w,
                        data_h,
                        bands
                    );
                }
                for band in 0..bands {
                    let mut out = vec![0u8; tile_bytes];
                    for y in 0..data_h {
                        for x in 0..data_w {
                            out[y * (tile_w as usize) + x] = bytes[(y * data_w + x) * bands + band];
                        }
                    }
                    tile_bands[band].push(out);
                }
            }
            2 => {
                for band in 0..bands {
                    let band_u32 =
                        u32::try_from(band).context("band index exceeds u32 range")?;
                    let chunk_idx = band_u32
                        .checked_mul(u32::try_from(tile_count).context("tile count too large")?)
                        .and_then(|v| v.checked_add(tile_idx_u32))
                        .context("planar chunk index overflowed u32")?;
                    let bytes = read_chunk_u8(&mut decoder, chunk_idx)?;
                    let expected = data_w * data_h;
                    if bytes.len() != expected {
                        bail!(
                            "planar chunk size mismatch: got {} bytes, expected {}",
                            bytes.len(),
                            expected
                        );
                    }
                    let mut out = vec![0u8; tile_bytes];
                    for y in 0..data_h {
                        for x in 0..data_w {
                            out[y * (tile_w as usize) + x] = bytes[y * data_w + x];
                        }
                    }
                    tile_bands[band].push(out);
                }
            }
            other => bail!("unsupported PlanarConfiguration tag value {}", other),
        }
    }

    Ok(Raster {
        image_w,
        image_h,
        tile_w,
        tile_h,
        bands,
        origin_x,
        origin_y,
        scale_x,
        scale_y,
        tile_bands,
    })
}

fn read_chunk_u8(decoder: &mut Decoder<File>, chunk_idx: u32) -> Result<Vec<u8>> {
    match decoder.read_chunk(chunk_idx)? {
        DecodingResult::U8(bytes) => Ok(bytes),
        _ => bail!("only 8-bit GeoTIFFs are supported"),
    }
}

fn build_struct_array(raster: &Raster) -> Result<StructArray> {
    let tile_count = raster.num_tiles();
    let tiles_per_row = raster.tiles_per_row();

    // Bounds: one f64 per tile per axis.
    let mut minx = Vec::with_capacity(tile_count);
    let mut miny = Vec::with_capacity(tile_count);
    let mut maxx = Vec::with_capacity(tile_count);
    let mut maxy = Vec::with_capacity(tile_count);

    for tile_idx in 0..tile_count {
        let tx = (tile_idx as u32) % tiles_per_row;
        let ty = (tile_idx as u32) / tiles_per_row;
        let x0 = raster.origin_x + (tx * raster.tile_w) as f64 * raster.scale_x;
        let x1 = raster.origin_x + ((tx + 1) * raster.tile_w) as f64 * raster.scale_x;
        let y0 = raster.origin_y - ((ty + 1) * raster.tile_h) as f64 * raster.scale_y;
        let y1 = raster.origin_y - (ty * raster.tile_h) as f64 * raster.scale_y;
        minx.push(x0);
        maxx.push(x1);
        miny.push(y0);
        maxy.push(y1);
    }

    let mut fields: Vec<(String, ArrayRef)> = Vec::with_capacity(4 + raster.bands);
    fields.push(("minx".to_string(), f64_column(minx)));
    fields.push(("miny".to_string(), f64_column(miny)));
    fields.push(("maxx".to_string(), f64_column(maxx)));
    fields.push(("maxy".to_string(), f64_column(maxy)));

    for (band_idx, tiles) in raster.tile_bands.iter().enumerate() {
        let list = build_list_u8(tiles)?;
        fields.push((format!("band_{band_idx}"), list));
    }

    let items: Vec<(&str, ArrayRef)> =
        fields.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    StructArray::from_fields(&items).map_err(Into::into)
}

fn f64_column(values: Vec<f64>) -> ArrayRef {
    PrimitiveArray::new(Buffer::<f64>::from(values), Validity::NonNullable).into_array()
}

fn build_list_u8(tiles: &[Vec<u8>]) -> Result<ArrayRef> {
    // Concatenate all tile bytes into a single contiguous buffer; offsets carry the per-tile
    // boundaries. u64 offsets handle rasters where total bytes exceed 2^32.
    let total: usize = tiles.iter().map(Vec::len).sum();
    let mut elements: Vec<u8> = Vec::with_capacity(total);
    let mut offsets: Vec<u64> = Vec::with_capacity(tiles.len() + 1);
    offsets.push(0);
    for tile in tiles {
        elements.extend_from_slice(tile);
        offsets.push(elements.len() as u64);
    }

    let elements_array =
        PrimitiveArray::new(Buffer::<u8>::from(elements), Validity::NonNullable).into_array();
    let offsets_array =
        PrimitiveArray::new(Buffer::<u64>::from(offsets), Validity::NonNullable).into_array();
    Ok(ListArray::try_new(elements_array, offsets_array, Validity::NonNullable)?.into_array())
}
