// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use futures::StreamExt;
use mimalloc::MiMalloc;
use vortex_datafusion::VortexFormatFactory;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn get_session_context() -> SessionContext {
    let rt_builder = RuntimeEnvBuilder::new();

    let rt = rt_builder
        .build_arc()
        .expect("could not build runtime environment");

    let factory = VortexFormatFactory::new();

    let mut session_state_builder = SessionStateBuilder::new()
        .with_config(SessionConfig::default())
        .with_runtime_env(rt)
        .with_default_features();

    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            GetExt::get_ext(&factory).to_uppercase(), // Has to be uppercase
            Arc::new(DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(Arc::new(factory));
    }

    SessionContext::new_with_state(session_state_builder.build())
}

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let ctx = get_session_context();

    ctx.sql(
        r#"
    CREATE EXTERNAL TABLE hits
    STORED AS VORTEX
    LOCATION '/Volumes/Code/vortex/bench-vortex/data/clickbench_partitioned/vortex-file-compressed/'
    "#,
    )
    .await?;

    let start = std::time::Instant::now();

    let mut stream = ctx
        .sql("select * from hits")
        .await?
        .execute_stream()
        .await?;

    while let Some(batch) = stream.next().await.transpose()? {
        // Discard the batches
        drop(batch);
    }

    let elapsed = start.elapsed();
    println!("scanned 14GB of clickbench data in {elapsed:?}");

    Ok(())
}
