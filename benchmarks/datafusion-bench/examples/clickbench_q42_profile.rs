// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic ClickBench Query 42 profiling benchmark.
//!
//! This generates synthetic data matching the ClickBench schema for query 42
//! and runs it through DataFusion with Vortex format for profiling.
//!
//! Query 42:
//! SELECT "WindowClientWidth", "WindowClientHeight", COUNT(*) AS PageViews
//! FROM hits
//! WHERE "CounterID" = 62
//!   AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31'
//!   AND "IsRefresh" = 0 AND "DontCountHits" = 0
//!   AND "URLHash" = 2868770270353813622
//! GROUP BY "WindowClientWidth", "WindowClientHeight"
//! ORDER BY PageViews DESC
//! LIMIT 10 OFFSET 10000;

use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::{
    ArrayRef, Int16Array, Int32Array, Int64Array, RecordBatch, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::datasource::MemTable;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::*;
use datafusion_datasource::ListingTableUrl;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef as VortexArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::ObjectStoreWriter;
use vortex::io::VortexWrite;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexFormatFactory;

const NUM_ROWS: usize = 10_000_000; // 10M rows for realistic profiling
const BATCH_SIZE: usize = 1_000_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

fn hits_schema() -> Schema {
    Schema::new(vec![
        Field::new("CounterID", DataType::Int32, false),
        Field::new(
            "EventDate",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("IsRefresh", DataType::Int16, false),
        Field::new("DontCountHits", DataType::Int16, false),
        Field::new("URLHash", DataType::Int64, false),
        Field::new("WindowClientWidth", DataType::Int16, false),
        Field::new("WindowClientHeight", DataType::Int16, false),
    ])
}

fn generate_batch(rng: &mut StdRng, batch_size: usize) -> RecordBatch {
    // Generate realistic data distributions
    let counter_ids: Vec<i32> = (0..batch_size)
        .map(|_| {
            // Make CounterID=62 appear ~1% of the time
            if rng.random::<f32>() < 0.01 {
                62
            } else {
                rng.random_range(1..1000)
            }
        })
        .collect();

    // EventDate in July 2013 range (as microseconds since epoch)
    // July 1, 2013 00:00:00 UTC = 1372636800 seconds = 1372636800000000 microseconds
    let july_start: i64 = 1372636800_000_000;
    let july_end: i64 = july_start + 31 * 24 * 3600 * 1_000_000; // 31 days

    let event_dates: Vec<i64> = (0..batch_size)
        .map(|_| {
            // 80% in July 2013, 20% outside
            if rng.random::<f32>() < 0.8 {
                rng.random_range(july_start..july_end)
            } else {
                rng.random_range(
                    july_start - 90 * 24 * 3600 * 1_000_000..july_end + 90 * 24 * 3600 * 1_000_000,
                )
            }
        })
        .collect();

    // IsRefresh: mostly 0
    let is_refresh: Vec<i16> = (0..batch_size)
        .map(|_| if rng.random::<f32>() < 0.95 { 0 } else { 1 })
        .collect();

    // DontCountHits: mostly 0
    let dont_count_hits: Vec<i16> = (0..batch_size)
        .map(|_| if rng.random::<f32>() < 0.98 { 0 } else { 1 })
        .collect();

    // URLHash: specific value appears rarely
    let target_hash: i64 = 2868770270353813622;
    let url_hashes: Vec<i64> = (0..batch_size)
        .map(|_| {
            // Make target hash appear ~0.1% of the time
            if rng.random::<f32>() < 0.001 {
                target_hash
            } else {
                rng.random::<i64>()
            }
        })
        .collect();

    // Window dimensions: common screen sizes
    let common_widths: [i16; 10] = [1920, 1366, 1536, 1440, 1280, 1600, 1024, 800, 768, 360];
    let common_heights: [i16; 10] = [1080, 768, 864, 900, 720, 900, 768, 600, 1024, 640];

    let window_widths: Vec<i16> = (0..batch_size)
        .map(|_| common_widths[rng.random_range(0..common_widths.len())])
        .collect();

    let window_heights: Vec<i16> = (0..batch_size)
        .map(|_| common_heights[rng.random_range(0..common_heights.len())])
        .collect();

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(counter_ids)),
        Arc::new(TimestampMicrosecondArray::from(event_dates)),
        Arc::new(Int16Array::from(is_refresh)),
        Arc::new(Int16Array::from(dont_count_hits)),
        Arc::new(Int64Array::from(url_hashes)),
        Arc::new(Int16Array::from(window_widths)),
        Arc::new(Int16Array::from(window_heights)),
    ];

    RecordBatch::try_new(Arc::new(hits_schema()), arrays).unwrap()
}

fn register_vortex_format_factory(
    factory: VortexFormatFactory,
    session_state_builder: &mut SessionStateBuilder,
) {
    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            datafusion::common::GetExt::get_ext(&factory).to_uppercase(),
            Arc::new(datafusion::datasource::provider::DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(Arc::new(factory));
    }
}

async fn write_vortex_file(
    store: &Arc<dyn ObjectStore>,
    path: &str,
    batches: &[RecordBatch],
) -> anyhow::Result<()> {
    // Concatenate all batches
    let schema = batches[0].schema();
    let combined = arrow_select::concat::concat_batches(&schema, batches)?;
    let array = VortexArrayRef::from_arrow(&combined, false)?;

    let path = Path::from_url_path(path)?;
    let mut write = ObjectStoreWriter::new(store.clone(), &path).await?;
    SESSION
        .write_options()
        .write(&mut write, array.to_array_stream())
        .await?;
    write.shutdown().await?;

    Ok(())
}

const QUERY_42: &str = r#"
SELECT "WindowClientWidth", "WindowClientHeight", COUNT(*) AS PageViews
FROM hits
WHERE "CounterID" = 62
  AND "EventDate" >= '2013-07-01'
  AND "EventDate" <= '2013-07-31'
  AND "IsRefresh" = 0
  AND "DontCountHits" = 0
  AND "URLHash" = 2868770270353813622
GROUP BY "WindowClientWidth", "WindowClientHeight"
ORDER BY PageViews DESC
LIMIT 10 OFFSET 10000
"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("ClickBench Query 42 Profiling Benchmark");
    println!("========================================");
    println!("Generating {} rows of synthetic data...", NUM_ROWS);

    let mut rng = StdRng::seed_from_u64(42);
    let num_batches = NUM_ROWS / BATCH_SIZE;

    let gen_start = Instant::now();
    let batches: Vec<RecordBatch> = (0..num_batches)
        .map(|i| {
            if i % 5 == 0 {
                println!("  Generating batch {}/{}", i + 1, num_batches);
            }
            generate_batch(&mut rng, BATCH_SIZE)
        })
        .collect();
    println!("Data generation took: {:?}", gen_start.elapsed());

    // Run query with Arrow in-memory (baseline)
    println!("\n--- Arrow In-Memory Baseline ---");
    let ctx = SessionContext::new();
    let schema = Arc::new(hits_schema());
    let mem_table = MemTable::try_new(schema.clone(), vec![batches.clone()])?;
    ctx.register_table("hits", Arc::new(mem_table))?;

    // Warmup
    let _warmup = ctx.sql(QUERY_42).await?.collect().await?;

    let arrow_times: Vec<_> = (0..5)
        .map(|i| {
            let ctx = ctx.clone();
            async move {
                let start = Instant::now();
                let result = ctx.sql(QUERY_42).await.unwrap().collect().await.unwrap();
                let elapsed = start.elapsed();
                println!(
                    "  Iteration {}: {:?} ({} rows)",
                    i + 1,
                    elapsed,
                    result.iter().map(|b| b.num_rows()).sum::<usize>()
                );
                elapsed
            }
        })
        .collect();

    let mut arrow_results = Vec::new();
    for fut in arrow_times {
        arrow_results.push(fut.await);
    }
    let arrow_avg = arrow_results.iter().sum::<std::time::Duration>() / 5;
    println!("Arrow average: {:?}", arrow_avg);

    // Create in-memory object store for Vortex files
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

    // Write Vortex file
    println!("\nWriting Vortex file to in-memory store...");
    let write_start = Instant::now();
    write_vortex_file(&store, "data/hits.vortex", &batches).await?;
    println!("Vortex write took: {:?}", write_start.elapsed());

    // Run query with Vortex
    println!("\n--- Vortex File ---");
    let factory = VortexFormatFactory::new();
    let mut session_state_builder = SessionStateBuilder::new().with_default_features();
    register_vortex_format_factory(factory, &mut session_state_builder);
    let vortex_ctx = SessionContext::new_with_state(session_state_builder.build());
    vortex_ctx.register_object_store(&Url::parse("s3://in-memory/")?, store.clone());

    let table_url = ListingTableUrl::parse("s3://in-memory/data/")?;
    let list_opts = ListingOptions::new(Arc::new(VortexFormat::new(SESSION.clone())))
        .with_session_config_options(vortex_ctx.state().config())
        .with_file_extension("vortex");

    let table = ListingTable::try_new(
        ListingTableConfig::new(table_url)
            .with_listing_options(list_opts)
            .with_schema(schema),
    )?;

    vortex_ctx.register_table("hits", Arc::new(table))?;

    // Warmup
    let _warmup = vortex_ctx.sql(QUERY_42).await?.collect().await?;

    let mut vortex_results = Vec::new();
    for i in 0..5 {
        let start = Instant::now();
        let result = vortex_ctx.sql(QUERY_42).await?.collect().await?;
        let elapsed = start.elapsed();
        println!(
            "  Iteration {}: {:?} ({} rows)",
            i + 1,
            elapsed,
            result.iter().map(|b| b.num_rows()).sum::<usize>()
        );
        vortex_results.push(elapsed);
    }
    let vortex_avg = vortex_results.iter().sum::<std::time::Duration>() / 5;
    println!("Vortex average: {:?}", vortex_avg);

    println!("\n========================================");
    println!("Summary:");
    println!("  Arrow in-memory: {:?}", arrow_avg);
    println!("  Vortex file:     {:?}", vortex_avg);
    println!(
        "  Ratio (Vortex/Arrow): {:.2}x",
        vortex_avg.as_secs_f64() / arrow_avg.as_secs_f64()
    );

    Ok(())
}
