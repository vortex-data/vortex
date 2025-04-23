use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_schema::Schema;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path as ObjectStorePath;
use tokio::fs::OpenOptions;
use tracing::info;
use url::Url;
use vortex::TryIntoArray;
use vortex::dtype::arrow::FromArrowType;
use vortex::file::{VORTEX_FILE_EXTENSION, VortexWriteOptions};
use vortex_datafusion::persistent::VortexFormat;

use crate::conversions::parquet_to_vortex;
use crate::datasets::BenchmarkDataset;
use crate::idempotent_async;
use crate::tpch::named_locks::with_lock;

pub async fn convert_parquet_to_vortex(input_path: &Path, dataset: BenchmarkDataset) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH => {
            // This is done on-demand by the register_vortex_file function
            Ok(())
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::convert_parquet_to_vortex(input_path).await
        }
    }
}

pub async fn register_parquet_files(
    session: &SessionContext,
    object_store: Arc<dyn ObjectStore>,
    table_name: &str,
    file_url: &Url,
    schema: &Schema,
    dataset: BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH => {
            let mut parquet_url = file_url.clone();
            if file_url.path().ends_with(".tbl") {
                // Replace .tbl with .parquet in the path
                let parquet_path = file_url.path().replace(".tbl", ".parquet");
                parquet_url.set_path(&parquet_path);
            }

            ensure_parquet_file_exists(
                session,
                object_store.as_ref(),
                file_url,
                &parquet_url,
                schema,
            )
            .await?;

            session
                .register_parquet(
                    table_name,
                    parquet_url.as_str(),
                    ParquetReadOptions::default(),
                )
                .await?;
        }
        BenchmarkDataset::ClickBench { single_file } => {
            // For ClickBench, we use simplified pre-built Parquet registration
            let format = Arc::new(ParquetFormat::new());
            let mut parquet_path = dataset.parquet_path(file_url)?;

            if single_file {
                parquet_path = parquet_path.join("hits.parquet")?;
            }

            info!("Registering table from {}", &parquet_path);
            let table_url = ListingTableUrl::parse(parquet_path)?;

            let config = ListingTableConfig::new(table_url)
                .with_listing_options(ListingOptions::new(format))
                .with_schema(schema.clone().into());

            let listing_table = Arc::new(ListingTable::try_new(config)?);
            session.register_table(table_name, listing_table)?;
        }
    }

    Ok(())
}

pub async fn register_vortex_files(
    session: &SessionContext,
    object_store: Arc<dyn ObjectStore>,
    table_name: &str,
    file_url: &Url,
    schema: &Schema,
    dataset: BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH => {
            // Extract the filename from the URL
            let csv_basename = file_url
                .path_segments()
                .expect("url path not empty")
                .next_back();
            let vortex_basename = csv_basename
                .unwrap()
                .replace(".tbl", (".".to_owned() + VORTEX_FILE_EXTENSION).as_ref());

            // Calculate vortex directory path
            let vortex_dir = dataset.vortex_path(file_url)?;
            let vtx_file = &vortex_dir.join(vortex_basename.as_ref())?;

            if let Err(e) = object_store
                .head(&ObjectStorePath::parse(vtx_file.path())?)
                .await
            {
                info!(
                    "Checking if file exists: File {} doesn't exist because {e}",
                    vtx_file
                );

                if vtx_file.scheme() == "file" {
                    // Create directory if it doesn't exist
                    if let Some(parent) = Path::new(vtx_file.path()).parent() {
                        fs::create_dir_all(parent)?;
                    }

                    with_lock(vtx_file.path().to_owned(), async move || {
                        // First convert CSV to record batches
                        let record_batches = session
                            .read_csv(
                                file_url.as_str(),
                                datafusion::prelude::CsvReadOptions::default()
                                    .delimiter(b'|')
                                    .has_header(false)
                                    .file_extension("tbl")
                                    .schema(schema),
                            )
                            .await?
                            .execute_stream()
                            .await?;

                        let adapter = vortex::stream::ArrayStreamAdapter::new(
                            vortex::dtype::DType::from_arrow(record_batches.schema()),
                            record_batches.then(|batch| async move {
                                batch
                                    .map_err(vortex::error::VortexError::from)
                                    .and_then(|b| b.try_into_array())
                            }),
                        );

                        let array_stream = Box::pin(adapter)
                            as std::pin::Pin<Box<dyn vortex::stream::ArrayStream + Send>>;

                        // Write to file
                        let f = OpenOptions::new()
                            .write(true)
                            .truncate(true)
                            .create(true)
                            .open(vtx_file.path())
                            .await?;
                        VortexWriteOptions::default().write(f, array_stream).await?;

                        anyhow::Ok(())
                    })
                    .await?;
                } else {
                    anyhow::bail!("Writing to remote storage not supported");
                }
            }

            // Register the Vortex file
            let format = Arc::new(VortexFormat::default());
            let table_url = ListingTableUrl::parse(vtx_file.as_str())?;
            let config = ListingTableConfig::new(table_url)
                .with_listing_options(ListingOptions::new(format))
                .infer_schema(&session.state())
                .await?;

            let listing_table = Arc::new(ListingTable::try_new(config)?);
            session.register_table(table_name, listing_table)?;
        }
        BenchmarkDataset::ClickBench { single_file } => {
            crate::clickbench::register_vortex_files(
                session.clone(),
                table_name,
                file_url,
                schema,
                single_file,
            )?;
        }
    }

    Ok(())
}

async fn ensure_parquet_file_exists(
    session: &SessionContext,
    object_store: &dyn ObjectStore,
    file_url: &Url,
    parquet_url: &Url,
    schema: &Schema,
) -> Result<()> {
    let parquet_path = parquet_url.path();

    if let Err(e) = object_store
        .head(&ObjectStorePath::parse(parquet_path)?)
        .await
    {
        info!(
            "Checking if file exist: File {} doesn't exist because {e}",
            parquet_url.as_str()
        );

        if parquet_url.scheme() != "file" {
            anyhow::bail!("Writing to S3 does not seem to work!");
        }

        with_lock(parquet_path.to_owned(), async move || {
            crate::conversions::csv_to_parquet_file(
                session,
                datafusion::prelude::CsvReadOptions::default()
                    .delimiter(b'|')
                    .has_header(false)
                    .file_extension("tbl")
                    .schema(schema),
                file_url.as_str(),
                parquet_path,
            )
            .await
        })
        .await?;
    }

    Ok(())
}

pub async fn parquet_file_to_vortex(parquet_path: &Path, vortex_path: &PathBuf) -> Result<()> {
    idempotent_async(vortex_path, async |vtx_file| {
        info!("Converting {:?} to Vortex format", parquet_path);

        let array_stream = parquet_to_vortex(parquet_path.to_path_buf())?;

        let f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&vtx_file)
            .await?;

        VortexWriteOptions::default().write(f, array_stream).await?;

        anyhow::Ok(())
    })
    .await?;

    Ok(())
}
