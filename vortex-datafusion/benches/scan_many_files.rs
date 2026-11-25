// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scanning many small files of one schema, where preparing each file's scan dominates.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;

use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use divan::Bencher;
use mimalloc::MiMalloc;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use object_store::path::Path;
use tokio::runtime::Runtime;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::FieldNames;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const COLUMNS: usize = 32;
const ROWS: usize = 16;

/// A comparison of every column, and a disjunction over all of them. Every file holds the same
/// values and each has one matching row, so no file is pruned.
fn query() -> String {
    let comparisons = (0..COLUMNS)
        .map(|idx| format!("c{idx} >= {idx}"))
        .collect::<Vec<_>>()
        .join(" AND ");
    let disjunction = (0..COLUMNS)
        .map(|idx| format!("c{idx} + 1 = {}", idx + 1))
        .collect::<Vec<_>>()
        .join(" OR ");
    format!("SELECT c0, c1 + c2 AS sum FROM tbl WHERE {comparisons} AND ({disjunction})")
}

/// A context whose table `tbl` has `files` files.
async fn context(files: usize) -> SessionContext {
    let session = VortexSession::default().with_tokio();
    let store = Arc::new(InMemory::new());
    let names = FieldNames::from_iter((0..COLUMNS).map(|idx| format!("c{idx}")));
    for file in 0..files {
        let columns = (0..COLUMNS)
            .map(|idx| Buffer::from_iter((0..ROWS).map(|row| (row + idx) as i64)).into_array())
            .collect::<Vec<_>>();
        let array =
            StructArray::try_new(names.clone(), columns, ROWS, Validity::NonNullable).unwrap();
        let mut bytes = Vec::new();
        session
            .write_options()
            .write(&mut bytes, array.into_array().to_array_stream())
            .await
            .unwrap();
        store
            .put(&Path::from(format!("tbl/{file:05}.vortex")), bytes.into())
            .await
            .unwrap();
    }

    // One partition, so that the files open one after another on the measuring thread.
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    ctx.register_object_store(
        &Url::parse("memory://bench").unwrap(),
        store as Arc<dyn ObjectStore>,
    );
    let config = ListingTableConfig::new(ListingTableUrl::parse("memory://bench/tbl/").unwrap())
        .with_listing_options(ListingOptions::new(Arc::new(VortexFormat::new(session))))
        .infer_schema(&ctx.state())
        .await
        .unwrap();
    ctx.register_table("tbl", Arc::new(ListingTable::try_new(config).unwrap()))
        .unwrap();
    ctx
}

/// A plan executes once, so each sample gets its own.
async fn plan(ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    ctx.sql(&query())
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}

/// Simulation runs one small table: it counts instructions, which do not need the larger one.
#[cfg(not(codspeed))]
const FILES: &[usize] = &[16, 256];
#[cfg(codspeed)]
const FILES: &[usize] = &[16];

#[divan::bench(args = FILES, sample_count = 50)]
fn scan_many_files(bencher: Bencher, files: usize) {
    let runtime = Runtime::new().unwrap();
    let ctx = runtime.block_on(context(files));
    // The first execution fills the footer cache, as a table that is queried again has.
    let batches = runtime
        .block_on(async { collect(plan(&ctx).await, ctx.task_ctx()).await })
        .unwrap();
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        files
    );
    bencher
        .with_inputs(|| runtime.block_on(plan(&ctx)))
        .bench_values(|plan| runtime.block_on(collect(plan, ctx.task_ctx())).unwrap());
}
