// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-data benchmark for the NeaTS encoding.
//!
//! Loads the taxi dataset that `vortex-bench` already wires up, finds every f32/f64 column, and
//! reports compressed size and round-trip error for NeaTS at a handful of error bounds. This lets
//! us see how the encoding behaves on actual time-series-shaped columns (`fare_amount`,
//! `trip_distance`, `total_amount`, etc.) without standing up a new data loader.
//!
//! Usage:
//!   cargo run -p vortex-bench --release --bin neats-real [-- /path/to/your.parquet]
//!
//! If no path is provided, the binary downloads (or reuses) the Yellow Taxi parquet file via the
//! same `taxi_data_parquet` helper used elsewhere in `vortex-bench`. The download is cached under
//! the project's idempotent data path.

#![expect(clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::taxi_data::taxi_data_parquet;
use vortex_neats::NeaTSOptions;
use vortex_neats::neats_encode;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let parquet_path = match env::args().nth(1) {
        Some(path) => PathBuf::from(path),
        None => {
            println!("# no path provided, downloading the taxi parquet (cached on disk)");
            taxi_data_parquet().await?
        }
    };
    println!("# loading parquet: {}", parquet_path.display());
    let chunked = parquet_to_vortex_chunks(parquet_path).await?;

    let mut ctx = SESSION.create_execution_ctx();
    let chunks: Vec<ArrayRef> = chunked.chunks().to_vec();

    // Collect every fp column across all chunks into a single concatenated Vec<f64> per column.
    let columns = extract_float_columns(&chunks, &mut ctx)?;
    if columns.is_empty() {
        println!("# no f32/f64 columns found in this parquet file");
        return Ok(());
    }

    println!(
        "# {:<24} {:>12} {:>16} {:>16} {:>16} {:>14}",
        "column", "rows", "raw_bytes", "neats_bytes", "ratio", "max_abs_err",
    );

    for (name, values) in columns {
        let array = PrimitiveArray::from_iter(values.iter().copied());
        let raw_bytes = (values.len() * size_of::<f64>()) as f64;

        for epsilon in [None, Some(1e-6), Some(1e-3)] {
            let opts = NeaTSOptions {
                epsilon,
                ..NeaTSOptions::default()
            };
            let t0 = Instant::now();
            let encoded = neats_encode(array.as_view(), opts)?;
            let encode_time = t0.elapsed();
            let neats_bytes = encoded.as_ref().nbytes() as f64;

            let mut ctx2 = SESSION.create_execution_ctx();
            let decoded = encoded
                .clone()
                .into_array()
                .execute::<PrimitiveArray>(&mut ctx2)?;
            let decoded_slice = decoded.as_slice::<f64>();
            let max_abs_err = values
                .iter()
                .zip(decoded_slice.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            let label = match epsilon {
                None => "lossless".to_string(),
                Some(e) => format!("eps={e:.0e}"),
            };
            println!(
                "{:<24} {:>12} {:>16.0} {:>16.0} {:>15.3}x {:>14.3e}  ({label}, encode {} us)",
                name,
                values.len(),
                raw_bytes,
                neats_bytes,
                raw_bytes / neats_bytes,
                max_abs_err,
                encode_time.as_micros(),
            );
        }
    }

    Ok(())
}

fn extract_float_columns(
    chunks: &[ArrayRef],
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    // Each chunk is a StructArray; pull its float fields and concat across chunks.
    let mut out: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for chunk in chunks {
        let DType::Struct(..) = chunk.dtype() else {
            continue;
        };
        let s = chunk
            .as_opt::<Struct>()
            .expect("dtype said struct but array is not StructArray");
        let names = s.names().clone();
        for (i, name) in names.iter().enumerate() {
            let field = s.unmasked_field(i).clone();
            match field.dtype() {
                DType::Primitive(PType::F32, _) => {
                    let p = field.execute::<PrimitiveArray>(ctx)?;
                    let entry = out.entry(name.to_string()).or_default();
                    entry.extend(p.as_slice::<f32>().iter().map(|v| *v as f64));
                }
                DType::Primitive(PType::F64, _) => {
                    let p = field.execute::<PrimitiveArray>(ctx)?;
                    let entry = out.entry(name.to_string()).or_default();
                    entry.extend(p.as_slice::<f64>().iter().copied());
                }
                _ => {}
            }
        }
    }
    Ok(out.into_iter().collect())
}
