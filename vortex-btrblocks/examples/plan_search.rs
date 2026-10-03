// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the estimate-based compressor with the exhaustive and genetic plan searches on a
//! few synthetic columns.
//!
//! ```text
//! cargo run --release -p vortex-btrblocks --example plan_search
//! ```

use std::time::Duration;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_compressor::search::Candidate;
use vortex_compressor::search::CostModel;
use vortex_compressor::search::ExhaustiveSearch;
use vortex_compressor::search::GeneticSearch;
use vortex_error::VortexResult;

const ROWS: usize = 64 * 1024;
const ITERATIONS: usize = 5;

fn columns() -> Vec<(&'static str, ArrayRef)> {
    let mut rng = StdRng::seed_from_u64(42);

    let statuses = ["pending", "shipped", "delivered", "returned", "cancelled"];
    let status = VarBinViewArray::from_iter(
        (0..ROWS).map(|_| Some(statuses[rng.random_range(0..statuses.len())])),
        DType::Utf8(Nullability::NonNullable),
    );

    let quantity = PrimitiveArray::from_iter((0..ROWS).map(|_| rng.random_range(1i32..=50)));

    let mut now = 1_700_000_000_000i64;
    let event_time = PrimitiveArray::from_iter((0..ROWS).map(|_| {
        now += rng.random_range(0i64..2_000);
        now
    }));

    let price = PrimitiveArray::from_iter(
        (0..ROWS).map(|_| f64::from(rng.random_range(100u32..100_000)) / 100.0),
    );

    let mut store = 0i32;
    let store_id = PrimitiveArray::from_iter((0..ROWS).map(|_| {
        if rng.random_bool(0.002) {
            store = rng.random_range(0..500);
        }
        store
    }));

    let url = VarBinViewArray::from_iter(
        (0..ROWS).map(|_| {
            Some(format!(
                "https://shop.example.com/product/{}?ref={}",
                rng.random_range(0..20_000),
                ["email", "search", "social"][rng.random_range(0..3)]
            ))
        }),
        DType::Utf8(Nullability::NonNullable),
    );

    vec![
        ("status", status.into_array()),
        ("quantity", quantity.into_array()),
        ("event_time", event_time.into_array()),
        ("price", price.into_array()),
        ("store_id", store_id.into_array()),
        ("url", url.into_array()),
    ]
}

fn micros(duration: Duration) -> String {
    format!("{:.0}", duration.as_secs_f64() * 1e6)
}

fn print_row(label: &str, candidate: &Candidate) {
    let m = &candidate.measurement;
    println!(
        "| {label} | {} | {} | {} | `{}` |",
        m.nbytes,
        micros(m.compress_time),
        micros(m.decompress_time),
        candidate.plan
    );
}

fn main() -> VortexResult<()> {
    let session = vortex_array::array_session();
    let compressor = BtrBlocksCompressorBuilder::from_session(&session)
        .unrestricted()
        .build();
    let mut ctx = session.create_execution_ctx();

    for (name, column) in columns() {
        println!("\n### {name} ({} bytes uncompressed)\n", column.nbytes());
        println!("| search | bytes | compress µs | decompress µs | plan |");
        println!("|---|---:|---:|---:|---|");

        let (_, estimated) = compressor.compress_recording_plan(&column, &mut ctx)?;
        print_row(
            "estimate (default)",
            &compressor.measure_plan(&column, &estimated, ITERATIONS, &mut ctx)?,
        );

        let smallest = ExhaustiveSearch::new(CostModel::size())
            .with_iterations(ITERATIONS)
            .run(&compressor, &column, &mut ctx)?;
        print_row("exhaustive, size", &smallest);

        for (label, bandwidth) in [
            ("exhaustive, read at 100 MB/s", 100e6),
            ("exhaustive, read at 10 GB/s", 10e9),
        ] {
            let best = ExhaustiveSearch::new(CostModel::read_time(bandwidth))
                .with_iterations(ITERATIONS)
                .run(&compressor, &column, &mut ctx)?;
            print_row(label, &best);
        }

        let front = GeneticSearch {
            iterations: ITERATIONS,
            ..GeneticSearch::default()
        }
        .with_initial_plans([smallest.plan])
        .run(&compressor, &column, &mut ctx)?;
        for candidate in &front {
            print_row("genetic front", candidate);
        }
    }
    Ok(())
}
