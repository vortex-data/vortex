// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::stream::StreamExt;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use parquet::arrow::async_reader::ParquetObjectReader;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::arrow::FromArrowArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::vortex_err;
use vortex_file::WriteOptionsSessionExt;
use vortex_session::VortexSession;

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    // console_subscriber::init();
    // let store = LocalFileSystem::new();
    let store = HttpBuilder::new()
        .with_url("https://vortex-benchmark-results-database.s3.amazonaws.com")
        .build()?;
    let store = Arc::new(store);
    let reader = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new(
        ParquetObjectReader::new(store, Path::from("testing/hits.parquet")),
        // ParquetObjectReader::new(store, Path::from("/Users/aduffy/Downloads/hits.parquet")),
    )
    .await?;

    let stream = reader.build()?;

    // Turn into a Vortex record batch stream
    let vx_stream = ArrayStreamAdapter::new(
        DType::from_arrow(stream.schema().as_ref()),
        stream.map(|br| {
            br.map_err(|e| vortex_err!("error: {e}"))
                .map(|b| ArrayRef::from_arrow(b, false))
        }),
    );

    // Write to file
    let output = tokio::fs::File::create("/tmp/output.vortex").await?;
    let session = VortexSession::default();
    println!("begin writing");
    let start = std::time::Instant::now();
    session.write_options().write(output, vx_stream).await?;
    let duration = start.elapsed();

    println!("rewrote in {duration:?}");

    Ok(())
}
