// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-column RLE vs RunEnd size ratios, plus inspection of how the RLE inner
//! children (values / indices / offsets) are compressed by the cascade.
//!
//! The `indices` path mirrors `IntRLEScheme` under `unstable_encodings`, which
//! forces a Delta wrap on the (narrowed) indices before compressing bases and
//! deltas. We report whether `fastlanes.delta` actually ends up in the indices
//! tree and how much it buys.
//!
//! ```bash
//! cargo run -p compress-bench --release --features unstable_encodings \
//!     --example rle_cascade_inspect
//! ```

use std::hint::black_box;

use anyhow::Result;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressor;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::RLE;
use vortex::encodings::fastlanes::RLEArrayExt;
use vortex::encodings::fastlanes::delta_compress;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::session::VortexSession;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::taxi_data::taxi_data_parquet;

const DELTA_ID: &str = "fastlanes.delta";

fn compress(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    Ok(BtrBlocksCompressor::default().compress(array, ctx)?)
}

/// True if `id` occurs anywhere in the tree.
fn contains(array: &ArrayRef, id: &str) -> bool {
    array.encoding_id().to_string() == id || array.children().iter().any(|c| contains(c, id))
}

/// Reproduces `IntRLEScheme`'s unstable indices path: Delta-wrap then compress.
fn forced_delta_indices(indices: PrimitiveArray, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    let len = indices.len();
    let (bases, deltas) = delta_compress(&indices, ctx)?;
    let bases = compress(&bases.into_array(), ctx)?;
    let deltas = compress(&deltas.into_array(), ctx)?;
    Ok(Delta::try_new(bases, deltas, 0, len)?.into_array())
}

#[tokio::main]
async fn main() -> Result<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();

    println!("Loading NYC taxi dataset (downloads ~56MB on first run)...");
    let parquet = taxi_data_parquet().await?;
    let chunked = parquet_to_vortex_chunks(parquet).await?;
    let struct_array = chunked
        .into_array()
        .execute::<Canonical>(&mut ctx)?
        .into_struct();
    let names = struct_array.names().clone();

    println!("Rows: {}\n", struct_array.len());
    println!(
        "{:<20} {:>7} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7} | {:>13} {:>6}",
        "column",
        "avg_run",
        "RE 1lvl",
        "RLE 1lvl",
        "RLE/RE",
        "RE casc",
        "RLE casc",
        "RLE/RE",
        "idx_enc",
        "delta?",
    );
    println!("{}", "-".repeat(110));

    for (idx, name) in names.iter().enumerate() {
        let field = struct_array.unmasked_field(idx).clone();
        if !(field.dtype().is_int() || field.dtype().is_float()) {
            continue;
        }
        let prim = field.clone().execute::<PrimitiveArray>(&mut ctx)?;

        // --- RunEnd ---
        let re = RunEnd::encode(prim.clone().into_array(), &mut ctx)?;
        let num_runs = re.ends().len();
        let avg_run = prim.len() as f64 / num_runs.max(1) as f64;
        let re_ends = re.ends().clone();
        let re_values = re.values().clone();
        let re_1lvl = re.clone().into_array().nbytes();
        // Cascade: compress ends and values.
        let re_casc =
            compress(&re_ends, &mut ctx)?.nbytes() + compress(&re_values, &mut ctx)?.nbytes();

        // --- FastLanes RLE ---
        let rle = RLE::encode(prim.as_view(), &mut ctx)?;
        let rle_1lvl = rle.clone().into_array().nbytes();

        let values_child = rle.values().clone();
        let indices_prim = rle
            .indices()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .narrow(&mut ctx)?;
        let offsets_prim = rle
            .values_idx_offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .narrow(&mut ctx)?;

        let c_values = compress(&values_child, &mut ctx)?;
        let c_indices = forced_delta_indices(indices_prim, &mut ctx)?;
        let c_offsets = compress(&offsets_prim.into_array(), &mut ctx)?;
        let rle_casc = c_values.nbytes() + c_indices.nbytes() + c_offsets.nbytes();

        let idx_enc = c_indices.encoding_id().to_string();
        let delta_used = contains(&c_indices, DELTA_ID);

        // Keep the optimizer honest about the work above.
        black_box((&c_values, &c_offsets));

        println!(
            "{:<20} {:>7.1} | {:>9} {:>9} {:>6.2}x | {:>9} {:>9} {:>6.2}x | {:>13} {:>6}",
            truncate(name.as_ref(), 20),
            avg_run,
            re_1lvl,
            rle_1lvl,
            rle_1lvl as f64 / re_1lvl.max(1) as f64,
            re_casc,
            rle_casc,
            rle_casc as f64 / re_casc.max(1) as f64,
            idx_enc.trim_start_matches("fastlanes."),
            if delta_used { "yes" } else { "NO" },
        );
    }

    println!("\nRLE/RE > 1 means RLE is larger. 1lvl = encoding only; casc = children compressed.");
    println!("idx_enc / delta? = top encoding of the RLE indices child and whether Delta appears.");

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
