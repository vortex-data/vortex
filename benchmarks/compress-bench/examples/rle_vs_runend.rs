// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Head-to-head: FastLanes RLE vs RunEnd on real data.
//!
//! Both are run-length-family encodings with different representations:
//!  * `RunEnd` stores one `(end, value)` pair per run — O(num_runs) storage,
//!    random access via binary search over the ends.
//!  * FastLanes `RLE` stores a per-1024-chunk dictionary plus a full-width u16
//!    index per element — O(N) indices before cascading, but branchless,
//!    SIMD-friendly fixed-width chunk decoding.
//!
//! This encodes each numeric taxi column with both (single level, no cascade)
//! and reports run statistics, compression ratio, and decode throughput.
//!
//! ```bash
//! cargo run -p compress-bench --release --example rle_vs_runend
//! ```

use std::hint::black_box;
use std::time::Instant;

use anyhow::Result;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::encodings::fastlanes::RLE;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::session::VortexSession;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::taxi_data::taxi_data_parquet;

/// Number of decode iterations used to estimate throughput.
const DECODE_ITERS: u32 = 20;

struct Measure {
    encoded_bytes: u64,
    ratio: f64,
    decode_mbps: f64,
}

fn measure_decode(encoded: &ArrayRef, decoded_bytes: u64, session: &VortexSession) -> Result<f64> {
    let mut ctx = session.create_execution_ctx();
    let start = Instant::now();
    for _ in 0..DECODE_ITERS {
        black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
    }
    let per_iter = start.elapsed() / DECODE_ITERS;
    Ok((decoded_bytes as f64 / (1024.0 * 1024.0)) / per_iter.as_secs_f64())
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
        "{:<22} {:>9} {:>8} | {:>10} {:>7} {:>10} | {:>10} {:>7} {:>10}",
        "column",
        "rows",
        "avg_run",
        "RE bytes",
        "RE x",
        "RE MB/s",
        "RLE bytes",
        "RLE x",
        "RLE MB/s"
    );
    println!("{}", "-".repeat(108));

    for (idx, name) in names.iter().enumerate() {
        let field = struct_array.unmasked_field(idx).clone();
        if !(field.dtype().is_int() || field.dtype().is_float()) {
            continue;
        }

        let prim = field.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let uncompressed = prim.clone().into_array().nbytes();

        // RunEnd.
        let re = RunEnd::encode(prim.clone().into_array(), &mut ctx)?;
        let num_runs = re.ends().len();
        let avg_run = prim.len() as f64 / num_runs.max(1) as f64;
        let re_array = re.into_array();
        let re = Measure {
            encoded_bytes: re_array.nbytes(),
            ratio: uncompressed as f64 / re_array.nbytes().max(1) as f64,
            decode_mbps: measure_decode(&re_array, uncompressed, &session)?,
        };

        // FastLanes RLE.
        let rle_array = RLE::encode(prim.as_view(), &mut ctx)?.into_array();
        let rle = Measure {
            encoded_bytes: rle_array.nbytes(),
            ratio: uncompressed as f64 / rle_array.nbytes().max(1) as f64,
            decode_mbps: measure_decode(&rle_array, uncompressed, &session)?,
        };

        println!(
            "{:<22} {:>9} {:>8.1} | {:>10} {:>6.1}x {:>10.0} | {:>10} {:>6.1}x {:>10.0}",
            truncate(name.as_ref(), 22),
            prim.len(),
            avg_run,
            re.encoded_bytes,
            re.ratio,
            re.decode_mbps,
            rle.encoded_bytes,
            rle.ratio,
            rle.decode_mbps,
        );
    }

    println!(
        "\nRE = RunEnd (one (end,value) per run). RLE = FastLanes RLE (per-chunk dict + u16 indices)."
    );
    println!("Sizes are single-level (no cascade); the default compressor cascades both further.");

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
