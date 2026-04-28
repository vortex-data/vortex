// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation, clippy::type_complexity)]

//! Benchmark raster access between a tiled GeoTIFF and the matching Vortex file produced by
//! the `geotiff_to_vortex` example.
//!
//! Two access patterns are measured on both backends:
//!
//! 1. **Single-tile decode** — pick a random tile and decode it into an `ndarray::Array3<u8>`
//!    with shape `(bands, tile_h, tile_w)`. On the GeoTIFF side this is one `read_chunk` call.
//!    On the Vortex side this is a row-index `Selection` pushed into the file scan; the result
//!    arrives as `List<u8>` chunks that are reshaped into the ndarray.
//! 2. **Random crop batch** — sample `--num-crops` random crops at `--crop-size` pixels and
//!    decode every tile that overlaps each crop. On the GeoTIFF side this is a per-crop loop of
//!    `read_chunk` calls. On the Vortex side this is a single OR-of-bbox filter expression
//!    pushed through `with_filter`, so tile rows are pruned by the file's zone-map statistics.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example raster_bench -p vortex --release -- \
//!     --geotiff m_3211428_sw_11_030_20230617_20240228.tif \
//!     --vortex naip.vortex \
//!     --num-crops 16 \
//!     --crop-size 1024 \
//!     --iterations 5
//! ```

use std::fs::File;
use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use futures::TryStreamExt;
use ndarray::Array3;
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tiff::decoder::Decoder;
use tiff::decoder::DecodingResult;
use tiff::tags::Tag;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::List;
use vortex::array::arrays::ListView;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::list::ListArrayExt;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::expr::Expression;
use vortex::array::expr::and;
use vortex::array::expr::col;
use vortex::array::expr::gt;
use vortex::array::expr::lit;
use vortex::array::expr::lt;
use vortex::array::expr::or;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_array::ExecutionCtx;

#[derive(Parser, Debug)]
#[command(about = "Benchmark single-tile and random-crop reads against GeoTIFF and Vortex.")]
struct Args {
    /// Path to the source tiled GeoTIFF.
    #[arg(long)]
    geotiff: PathBuf,

    /// Path to the matching Vortex file (one row per tile).
    #[arg(long)]
    vortex: PathBuf,

    /// How many random crops to sample in the batch benchmark.
    #[arg(long, default_value_t = 16)]
    num_crops: usize,

    /// Crop edge length in pixels.
    #[arg(long, default_value_t = 1024)]
    crop_size: u32,

    /// Iterations per benchmark to average over.
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// PRNG seed for reproducible tile / crop selection.
    #[arg(long, default_value_t = 0xa10c_u64)]
    seed: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let session = VortexSession::default();

    let meta = read_tiff_metadata(&args.geotiff)?;
    println!(
        "geotiff: {w}x{h}, {bands} bands, tiles {tw}x{th} ({tx}x{ty} grid)",
        w = meta.image_w,
        h = meta.image_h,
        bands = meta.bands,
        tw = meta.tile_w,
        th = meta.tile_h,
        tx = meta.tiles_x(),
        ty = meta.tiles_y(),
    );

    let mut rng = StdRng::seed_from_u64(args.seed);

    // ---- Single-tile decode ------------------------------------------------------------------
    let tile_count = meta.num_tiles();
    let tile_indices: Vec<u32> = (0..args.iterations)
        .map(|_| rng.random_range(0..tile_count) as u32)
        .collect();

    let geotiff_single = bench(args.iterations, || {
        for &tile_idx in &tile_indices {
            let _arr = decode_tiff_tile(&args.geotiff, tile_idx, &meta)?;
        }
        Ok(())
    })?;

    let vortex_single = bench_async(args.iterations, || async {
        for &tile_idx in &tile_indices {
            let _arr = decode_vortex_tile(&session, &args.vortex, tile_idx, &meta).await?;
        }
        Ok(())
    })
    .await?;

    println!(
        "\nsingle-tile decode ({} iterations, {} tiles each):\n  geotiff: {:>8.3} ms/iter\n  vortex:  {:>8.3} ms/iter  ({:.2}x)",
        args.iterations,
        tile_indices.len(),
        geotiff_single.as_secs_f64() * 1e3,
        vortex_single.as_secs_f64() * 1e3,
        geotiff_single.as_secs_f64() / vortex_single.as_secs_f64(),
    );

    // ---- Random crop batch -------------------------------------------------------------------
    let crops: Vec<Crop> = (0..args.num_crops)
        .map(|_| Crop::random(&mut rng, &meta, args.crop_size))
        .collect();

    let geotiff_crops = bench(args.iterations, || {
        for crop in &crops {
            drop(decode_tiff_crop(&args.geotiff, crop, &meta)?);
        }
        Ok(())
    })?;

    let vortex_crops = bench_async(args.iterations, || async {
        for crop in &crops {
            drop(decode_vortex_crop(&session, &args.vortex, crop, &meta).await?);
        }
        Ok(())
    })
    .await?;

    let vortex_crops_combined = bench_async(args.iterations, || async {
        drop(decode_vortex_crops_combined(&session, &args.vortex, &crops, &meta).await?);
        Ok(())
    })
    .await?;

    println!(
        "\nrandom crop batch ({} crops at {}px, {} iterations):\n  \
         geotiff (per-crop loop):     {:>8.3} ms/iter\n  \
         vortex  (per-crop scan):     {:>8.3} ms/iter\n  \
         vortex  (single OR scan):    {:>8.3} ms/iter",
        args.num_crops,
        args.crop_size,
        args.iterations,
        geotiff_crops.as_secs_f64() * 1e3,
        vortex_crops.as_secs_f64() * 1e3,
        vortex_crops_combined.as_secs_f64() * 1e3,
    );

    Ok(())
}

// =================================================================================================
// Shared metadata
// =================================================================================================

struct RasterMeta {
    image_w: u32,
    image_h: u32,
    tile_w: u32,
    tile_h: u32,
    bands: usize,
    origin_x: f64,
    origin_y: f64,
    scale_x: f64,
    scale_y: f64,
    planar_cfg: u32,
}

impl RasterMeta {
    fn tiles_x(&self) -> u32 {
        self.image_w.div_ceil(self.tile_w)
    }

    fn tiles_y(&self) -> u32 {
        self.image_h.div_ceil(self.tile_h)
    }

    fn num_tiles(&self) -> usize {
        (self.tiles_x() as usize) * (self.tiles_y() as usize)
    }

    /// Map a pixel coordinate to CRS coordinates using the image-origin geotransform.
    fn pixel_to_world(&self, px: u32, py: u32) -> (f64, f64) {
        let x = self.origin_x + (px as f64) * self.scale_x;
        let y = self.origin_y - (py as f64) * self.scale_y;
        (x, y)
    }
}

fn read_tiff_metadata(path: &PathBuf) -> Result<RasterMeta> {
    let mut decoder = Decoder::new(File::open(path)?)?;
    let (image_w, image_h) = decoder.dimensions()?;
    let tile_w = decoder.get_tag_u32(Tag::TileWidth)?;
    let tile_h = decoder.get_tag_u32(Tag::TileLength)?;
    let planar_cfg = decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1);

    let tiepoint = decoder.get_tag_f64_vec(Tag::ModelTiepointTag)?;
    let scale = decoder.get_tag_f64_vec(Tag::ModelPixelScaleTag)?;
    if tiepoint.len() < 6 || scale.len() < 2 {
        bail!("GeoTIFF is missing a usable geotransform");
    }
    let origin_x = tiepoint[3] - tiepoint[0] * scale[0];
    let origin_y = tiepoint[4] + tiepoint[1] * scale[1];

    // Probe the first chunk to discover how many bands the decoder actually returns. The
    // converter does the same probe, so the resulting Vortex schema and the GeoTIFF decode
    // path agree on band count even when `SamplesPerPixel` includes extra samples that get
    // dropped to fit the photometric `ColorType`.
    let probe = match decoder.read_chunk(0)? {
        DecodingResult::U8(v) => v,
        _ => bail!("only 8-bit GeoTIFFs are supported"),
    };
    let (data_w, data_h) = decoder.chunk_data_dimensions(0);
    let probe_pixels = (data_w as usize) * (data_h as usize);
    if probe_pixels == 0 || probe.len() % probe_pixels != 0 {
        bail!(
            "first chunk size {} not divisible by chunk pixel count {}",
            probe.len(),
            probe_pixels
        );
    }
    let bands = probe.len() / probe_pixels;

    Ok(RasterMeta {
        image_w,
        image_h,
        tile_w,
        tile_h,
        bands,
        origin_x,
        origin_y,
        scale_x: scale[0],
        scale_y: scale[1],
        planar_cfg,
    })
}

// =================================================================================================
// GeoTIFF decode paths
// =================================================================================================

fn decode_tiff_tile(path: &PathBuf, tile_idx: u32, meta: &RasterMeta) -> Result<Array3<u8>> {
    let mut decoder = Decoder::new(File::open(path)?)?;
    let tw = meta.tile_w as usize;
    let th = meta.tile_h as usize;
    let bands = meta.bands;
    // Edge tiles are clipped: data dims may be smaller than (tile_w, tile_h). We always emit a
    // full-size ndarray and zero-fill the padded region to keep crop/tile shapes uniform.
    let (data_w, data_h) = decoder.chunk_data_dimensions(tile_idx);
    let dw = data_w as usize;
    let dh = data_h as usize;

    let mut out = Array3::<u8>::zeros((bands, th, tw));
    match meta.planar_cfg {
        1 => {
            let bytes = match decoder.read_chunk(tile_idx)? {
                DecodingResult::U8(v) => v,
                _ => bail!("only 8-bit GeoTIFFs are supported"),
            };
            for b in 0..bands {
                for y in 0..dh {
                    for x in 0..dw {
                        out[(b, y, x)] = bytes[(y * dw + x) * bands + b];
                    }
                }
            }
        }
        2 => {
            for b in 0..bands {
                let chunk_idx =
                    u32::try_from(b)?.saturating_mul(u32::try_from(meta.num_tiles())?) + tile_idx;
                let bytes = match decoder.read_chunk(chunk_idx)? {
                    DecodingResult::U8(v) => v,
                    _ => bail!("only 8-bit GeoTIFFs are supported"),
                };
                for y in 0..dh {
                    for x in 0..dw {
                        out[(b, y, x)] = bytes[y * dw + x];
                    }
                }
            }
        }
        other => bail!("unsupported PlanarConfiguration {}", other),
    }
    Ok(out)
}

fn decode_tiff_crop(path: &PathBuf, crop: &Crop, meta: &RasterMeta) -> Result<Vec<Array3<u8>>> {
    // Read every tile that overlaps the crop. We don't bother clipping into a final crop array,
    // since the access cost we want to measure is "fetch all required tile bytes".
    let tile_indices = tiles_overlapping_crop(crop, meta);
    let mut out = Vec::with_capacity(tile_indices.len());
    for tile_idx in tile_indices {
        out.push(decode_tiff_tile(path, tile_idx, meta)?);
    }
    Ok(out)
}

// =================================================================================================
// Vortex decode paths
// =================================================================================================

async fn decode_vortex_tile(
    session: &VortexSession,
    path: &PathBuf,
    tile_idx: u32,
    meta: &RasterMeta,
) -> Result<Array3<u8>> {
    let file = session.open_options().open_path(path).await?;
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_row_indices(Buffer::<u64>::from(vec![tile_idx as u64]))
        .into_array_stream()?
        .try_collect()
        .await?;

    let mut ctx = session.create_execution_ctx();
    let tiles = decode_vortex_chunks(chunks, meta, &mut ctx)?;
    tiles
        .into_iter()
        .next()
        .context("scan produced no tiles for the requested row index")
}

async fn decode_vortex_crop(
    session: &VortexSession,
    path: &PathBuf,
    crop: &Crop,
    meta: &RasterMeta,
) -> Result<Vec<Array3<u8>>> {
    let file = session.open_options().open_path(path).await?;
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_filter(crop_filter(crop))
        .into_array_stream()?
        .try_collect()
        .await?;

    let mut ctx = session.create_execution_ctx();
    decode_vortex_chunks(chunks, meta, &mut ctx)
}

async fn decode_vortex_crops_combined(
    session: &VortexSession,
    path: &PathBuf,
    crops: &[Crop],
    meta: &RasterMeta,
) -> Result<Vec<Array3<u8>>> {
    // Combine all crop predicates into one OR-tree so we can read every overlapping tile
    // through a single scan and let the file's zone map prune the rest.
    let Some(filter) = crops
        .iter()
        .map(crop_filter)
        .reduce(or)
    else {
        return Ok(vec![]);
    };

    let file = session.open_options().open_path(path).await?;
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .try_collect()
        .await?;

    let mut ctx = session.create_execution_ctx();
    decode_vortex_chunks(chunks, meta, &mut ctx)
}

fn decode_vortex_chunks(
    chunks: Vec<ArrayRef>,
    meta: &RasterMeta,
    ctx: &mut ExecutionCtx,
) -> Result<Vec<Array3<u8>>> {
    let total_rows: usize = chunks.iter().map(|c| c.len()).sum();
    if total_rows == 0 || chunks.is_empty() {
        return Ok(vec![]);
    }

    let dtype = chunks[0].dtype().clone();
    let chunked: StructArray = ChunkedArray::try_new(chunks, dtype)?
        .into_array()
        .execute(ctx)?;

    let tile_pixels = (meta.tile_w as usize) * (meta.tile_h as usize);

    // For each band column, materialize the elements + per-row [start, end) offsets. Lists may
    // come back canonicalized to a `ListView` (`ListView` is the canonical form for list dtypes),
    // or they may stay as `List` if the file kept that encoding — handle both.
    let mut band_buffers: Vec<(Vec<u8>, Vec<(u64, u64)>)> = Vec::with_capacity(meta.bands);
    for b in 0..meta.bands {
        let name = format!("band_{b}");
        let field: ArrayRef = chunked.unmasked_field_by_name(&name)?.clone().execute(ctx)?;
        let (elements, ranges) = if field.is::<List>() {
            list_ranges_from_list(&field, ctx)?
        } else if field.is::<ListView>() {
            list_ranges_from_view(&field, ctx)?
        } else {
            bail!(
                "expected band column to canonicalize to List or ListView, got {}",
                field.encoding_id()
            );
        };
        band_buffers.push((elements, ranges));
    }

    let mut out = Vec::with_capacity(total_rows);
    for row in 0..total_rows {
        let mut arr = Array3::<u8>::zeros((meta.bands, meta.tile_h as usize, meta.tile_w as usize));
        for (b, (elements, ranges)) in band_buffers.iter().enumerate() {
            let (start, end) = ranges[row];
            let start = start as usize;
            let end = end as usize;
            if end - start != tile_pixels {
                bail!(
                    "band {b} tile {row} has {} bytes, expected {tile_pixels}",
                    end - start
                );
            }
            let tile = &elements[start..end];
            for y in 0..(meta.tile_h as usize) {
                for x in 0..(meta.tile_w as usize) {
                    arr[(b, y, x)] = tile[y * (meta.tile_w as usize) + x];
                }
            }
        }
        out.push(arr);
    }
    Ok(out)
}

fn list_ranges_from_list(
    field: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Result<(Vec<u8>, Vec<(u64, u64)>)> {
    let list = field.as_::<List>();
    let elements: PrimitiveArray = list.elements().clone().execute(ctx)?;
    let offsets: PrimitiveArray = list.offsets().clone().execute(ctx)?;
    let elem_bytes = elements.as_slice::<u8>().to_vec();
    let offsets_u64 = offsets_to_u64(&offsets)?;
    let ranges: Vec<(u64, u64)> = offsets_u64
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();
    Ok((elem_bytes, ranges))
}

fn list_ranges_from_view(
    field: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Result<(Vec<u8>, Vec<(u64, u64)>)> {
    let view = field.as_::<ListView>();
    let elements: PrimitiveArray = view.elements().clone().execute(ctx)?;
    let offsets: PrimitiveArray = view.offsets().clone().execute(ctx)?;
    let sizes: PrimitiveArray = view.sizes().clone().execute(ctx)?;
    let elem_bytes = elements.as_slice::<u8>().to_vec();
    let offsets_u64 = offsets_to_u64(&offsets)?;
    let sizes_u64 = offsets_to_u64(&sizes)?;
    let ranges: Vec<(u64, u64)> = offsets_u64
        .iter()
        .zip(sizes_u64.iter())
        .map(|(&o, &s)| (o, o + s))
        .collect();
    Ok((elem_bytes, ranges))
}

fn offsets_to_u64(offsets: &PrimitiveArray) -> Result<Vec<u64>> {
    use vortex::dtype::PType;
    let result = match offsets.ptype() {
        PType::I32 => offsets.as_slice::<i32>().iter().map(|&v| v as u64).collect(),
        PType::U32 => offsets.as_slice::<u32>().iter().map(|&v| v as u64).collect(),
        PType::I64 => offsets.as_slice::<i64>().iter().map(|&v| v as u64).collect(),
        PType::U64 => offsets.as_slice::<u64>().to_vec(),
        other => bail!("unsupported list offset ptype: {other:?}"),
    };
    Ok(result)
}

// =================================================================================================
// Crops + filter expressions
// =================================================================================================

#[derive(Clone, Debug)]
struct Crop {
    pixel_x: u32,
    pixel_y: u32,
    size: u32,
    minx: f64,
    miny: f64,
    maxx: f64,
    maxy: f64,
}

impl Crop {
    fn random<R: Rng>(rng: &mut R, meta: &RasterMeta, size: u32) -> Self {
        let max_x = meta.image_w.saturating_sub(size);
        let max_y = meta.image_h.saturating_sub(size);
        let pixel_x = if max_x > 0 { rng.random_range(0..max_x) } else { 0 };
        let pixel_y = if max_y > 0 { rng.random_range(0..max_y) } else { 0 };
        let (x0, y1) = meta.pixel_to_world(pixel_x, pixel_y);
        let (x1, y0) = meta.pixel_to_world(pixel_x + size, pixel_y + size);
        Crop {
            pixel_x,
            pixel_y,
            size,
            minx: x0.min(x1),
            maxx: x0.max(x1),
            miny: y0.min(y1),
            maxy: y0.max(y1),
        }
    }
}

fn crop_filter(crop: &Crop) -> Expression {
    // A tile overlaps the crop when its bounds are not strictly to one side. With the schema
    // produced by the converter (`minx <= maxx`, `miny <= maxy` per row) the overlap check is:
    //   tile.minx < crop.maxx AND tile.maxx > crop.minx AND
    //   tile.miny < crop.maxy AND tile.maxy > crop.miny
    let x_overlap = and(lt(col("minx"), lit(crop.maxx)), gt(col("maxx"), lit(crop.minx)));
    let y_overlap = and(lt(col("miny"), lit(crop.maxy)), gt(col("maxy"), lit(crop.miny)));
    and(x_overlap, y_overlap)
}

fn tiles_overlapping_crop(crop: &Crop, meta: &RasterMeta) -> Vec<u32> {
    let tx_start = crop.pixel_x / meta.tile_w;
    let tx_end = (crop.pixel_x + crop.size - 1) / meta.tile_w;
    let ty_start = crop.pixel_y / meta.tile_h;
    let ty_end = (crop.pixel_y + crop.size - 1) / meta.tile_h;
    let tiles_x = meta.tiles_x();

    let mut out = Vec::new();
    for ty in ty_start..=ty_end.min(meta.tiles_y().saturating_sub(1)) {
        for tx in tx_start..=tx_end.min(tiles_x.saturating_sub(1)) {
            out.push(ty * tiles_x + tx);
        }
    }
    out
}

// =================================================================================================
// Timing helpers
// =================================================================================================

fn bench<F: FnMut() -> Result<()>>(iterations: usize, mut f: F) -> Result<Duration> {
    // Warm up once so the first run doesn't inflate the average with cold caches.
    f()?;
    let start = Instant::now();
    for _ in 0..iterations {
        f()?;
    }
    Ok(start.elapsed() / iterations.max(1) as u32)
}

async fn bench_async<F, Fut>(iterations: usize, mut f: F) -> Result<Duration>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<()>>,
{
    f().await?;
    let start = Instant::now();
    for _ in 0..iterations {
        f().await?;
    }
    Ok(start.elapsed() / iterations.max(1) as u32)
}
