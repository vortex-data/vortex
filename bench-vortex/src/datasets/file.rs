use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_schema::Schema;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use object_store::ObjectStore;
use object_store::path::Path as ObjectStorePath;
use tokio::fs::OpenOptions;
use tracing::info;
use url::Url;
use vortex::error::VortexExpect;
use vortex::file::VortexWriteOptions;
use vortex_datafusion::persistent::VortexFormat;

use crate::conversions::parquet_to_vortex;
use crate::datasets::BenchmarkDataset;
use crate::idempotent_async;

pub async fn convert_parquet_to_vortex(input_path: &Path, dataset: BenchmarkDataset) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH => {
            // This is done on-demand by the register_vortex_file function
            Ok(())
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::convert_parquet_to_vortex(input_path).await
        }
        _ => todo!(),
    }
}

pub async fn register_parquet_files(
    session: &SessionContext,
    object_store: Arc<dyn ObjectStore>,
    table_name: &str,
    file_url: &Url,
    schema: Option<Schema>,
    dataset: BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH => {
            let parquet_url = file_url.clone();
            ensure_parquet_file_exists(object_store.as_ref(), &parquet_url).await?;

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
                .with_listing_options(ListingOptions::new(format));

            let config = if let Some(schema) = schema {
                config.with_schema(schema.into())
            } else {
                config
                    .infer_schema(&session.state())
                    .await
                    .vortex_expect("cannot infer schema")
            };

            let listing_table = Arc::new(ListingTable::try_new(config)?);
            session.register_table(table_name, listing_table)?;
        }
        _ => todo!(),
    }

    Ok(())
}

pub async fn register_vortex_files(
    session: &SessionContext,
    _object_store: Arc<dyn ObjectStore>,
    table_name: &str,
    file_url: &Url,
    schema: Option<Schema>,
    dataset: BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH | BenchmarkDataset::TpcDS => {
            // Register the Vortex file
            let format = Arc::new(VortexFormat::default());
            let table_url = ListingTableUrl::parse(file_url.as_str())?;
            let config = ListingTableConfig::new(table_url)
                .with_listing_options(ListingOptions::new(format));

            let config = if let Some(schema) = schema {
                config.with_schema(schema.into())
            } else {
                config.infer_schema(&session.state()).await?
            };

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
            )
            .await?;
        }
    }

    Ok(())
}

async fn ensure_parquet_file_exists(
    object_store: &dyn ObjectStore,
    parquet_url: &Url,
) -> Result<()> {
    let parquet_path = parquet_url.path();

    if let Err(e) = object_store
        .head(&ObjectStorePath::parse(parquet_path)?)
        .await
    {
        info!(
            "Asserting file exist: File {} doesn't exist because {e}",
            parquet_url.as_str()
        );

        if parquet_url.scheme() != "file" {
            anyhow::bail!("Writing to S3 does not seem to work!");
        }
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
