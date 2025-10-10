// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use std::{fmt, fs};

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use clap::ValueEnum;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use futures::{StreamExt, TryStreamExt, stream};
use glob::Pattern;
use lance::datafusion::LanceTableProvider;
use lance::dataset::{Dataset as LanceDataset, WriteParams};
use log::trace;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use reqwest::IntoUrl;
use reqwest::blocking::Response;
use serde::Serialize;
use tokio::fs::{OpenOptions, create_dir_all};
use tracing::{Instrument, info, warn};
use url::Url;
use vortex::error::VortexExpect;
use vortex::file::VortexWriteOptions;
use vortex_datafusion::VortexFormat;

use crate::conversions::parquet_to_vortex;
use crate::utils::file_utils::{idempotent, idempotent_async};
use crate::{CompactionStrategy, Format};

pub static HITS_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    use DataType::*;
    Schema::new(vec![
        Field::new("WatchID", Int64, false),
        Field::new("JavaEnable", Int16, false),
        Field::new("Title", Utf8View, false),
        Field::new("GoodEvent", Int16, false),
        Field::new("EventTime", Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("EventDate", Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("CounterID", Int32, false),
        Field::new("ClientIP", Int32, false),
        Field::new("RegionID", Int32, false),
        Field::new("UserID", Int64, false),
        Field::new("CounterClass", Int16, false),
        Field::new("OS", Int16, false),
        Field::new("UserAgent", Int16, false),
        Field::new("URL", Utf8View, false),
        Field::new("Referer", Utf8View, false),
        Field::new("IsRefresh", Int16, false),
        Field::new("RefererCategoryID", Int16, false),
        Field::new("RefererRegionID", Int32, false),
        Field::new("URLCategoryID", Int16, false),
        Field::new("URLRegionID", Int32, false),
        Field::new("ResolutionWidth", Int16, false),
        Field::new("ResolutionHeight", Int16, false),
        Field::new("ResolutionDepth", Int16, false),
        Field::new("FlashMajor", Int16, false),
        Field::new("FlashMinor", Int16, false),
        Field::new("FlashMinor2", Utf8View, false),
        Field::new("NetMajor", Int16, false),
        Field::new("NetMinor", Int16, false),
        Field::new("UserAgentMajor", Int16, false),
        Field::new("UserAgentMinor", Utf8View, false),
        Field::new("CookieEnable", Int16, false),
        Field::new("JavascriptEnable", Int16, false),
        Field::new("IsMobile", Int16, false),
        Field::new("MobilePhone", Int16, false),
        Field::new("MobilePhoneModel", Utf8View, false),
        Field::new("Params", Utf8View, false),
        Field::new("IPNetworkID", Int32, false),
        Field::new("TraficSourceID", Int16, false),
        Field::new("SearchEngineID", Int16, false),
        Field::new("SearchPhrase", Utf8View, false),
        Field::new("AdvEngineID", Int16, false),
        Field::new("IsArtifical", Int16, false),
        Field::new("WindowClientWidth", Int16, false),
        Field::new("WindowClientHeight", Int16, false),
        Field::new("ClientTimeZone", Int16, false),
        Field::new(
            "ClientEventTime",
            Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("SilverlightVersion1", Int16, false),
        Field::new("SilverlightVersion2", Int16, false),
        Field::new("SilverlightVersion3", Int32, false),
        Field::new("SilverlightVersion4", Int16, false),
        Field::new("PageCharset", Utf8View, false),
        Field::new("CodeVersion", Int32, false),
        Field::new("IsLink", Int16, false),
        Field::new("IsDownload", Int16, false),
        Field::new("IsNotBounce", Int16, false),
        Field::new("FUniqID", Int64, false),
        Field::new("OriginalURL", Utf8View, false),
        Field::new("HID", Int32, false),
        Field::new("IsOldCounter", Int16, false),
        Field::new("IsEvent", Int16, false),
        Field::new("IsParameter", Int16, false),
        Field::new("DontCountHits", Int16, false),
        Field::new("WithHash", Int16, false),
        Field::new("HitColor", Utf8View, false),
        Field::new(
            "LocalEventTime",
            Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("Age", Int16, false),
        Field::new("Sex", Int16, false),
        Field::new("Income", Int16, false),
        Field::new("Interests", Int16, false),
        Field::new("Robotness", Int16, false),
        Field::new("RemoteIP", Int32, false),
        Field::new("WindowName", Int32, false),
        Field::new("OpenerName", Int32, false),
        Field::new("HistoryLength", Int16, false),
        Field::new("BrowserLanguage", Utf8View, false),
        Field::new("BrowserCountry", Utf8View, false),
        Field::new("SocialNetwork", Utf8View, false),
        Field::new("SocialAction", Utf8View, false),
        Field::new("HTTPError", Int16, false),
        Field::new("SendTiming", Int32, false),
        Field::new("DNSTiming", Int32, false),
        Field::new("ConnectTiming", Int32, false),
        Field::new("ResponseStartTiming", Int32, false),
        Field::new("ResponseEndTiming", Int32, false),
        Field::new("FetchTiming", Int32, false),
        Field::new("SocialSourceNetworkID", Int16, false),
        Field::new("SocialSourcePage", Utf8View, false),
        Field::new("ParamPrice", Int64, false),
        Field::new("ParamOrderID", Utf8View, false),
        Field::new("ParamCurrency", Utf8View, false),
        Field::new("ParamCurrencyID", Int16, false),
        Field::new("OpenstatServiceName", Utf8View, false),
        Field::new("OpenstatCampaignID", Utf8View, false),
        Field::new("OpenstatAdID", Utf8View, false),
        Field::new("OpenstatSourceID", Utf8View, false),
        Field::new("UTMSource", Utf8View, false),
        Field::new("UTMMedium", Utf8View, false),
        Field::new("UTMCampaign", Utf8View, false),
        Field::new("UTMContent", Utf8View, false),
        Field::new("UTMTerm", Utf8View, false),
        Field::new("FromTag", Utf8View, false),
        Field::new("HasGCLID", Int16, false),
        Field::new("RefererHash", Int64, false),
        Field::new("URLHash", Int64, false),
        Field::new("CLID", Int32, false),
    ])
});

pub async fn convert_parquet_to_vortex(
    input_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let (format, dir_name) = match compaction {
        CompactionStrategy::Compact => (Format::VortexCompact, Format::VortexCompact.name()),
        CompactionStrategy::Default => (Format::OnDiskVortex, Format::OnDiskVortex.name()),
    };

    let vortex_dir = input_path.join(dir_name);
    let parquet_path = input_path.join(Format::Parquet.name());
    create_dir_all(&vortex_dir).await?;

    let parquet_inputs = fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;

    trace!(
        "Found {} parquet files in {}",
        parquet_inputs.len(),
        parquet_path.to_str().unwrap()
    );

    let iter = parquet_inputs
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"));

    stream::iter(iter)
        .map(|dir_entry| {
            let filename = {
                let mut temp = dir_entry.path();
                temp.set_extension("");
                temp.file_name().unwrap().to_str().unwrap().to_string()
            };
            let parquet_file_path = parquet_path.join(format!("{filename}.parquet"));
            let output_path = vortex_dir.join(format!("{filename}.{}", format.ext()));

            tokio::spawn(
                async move {
                    idempotent_async(&output_path, move |vtx_file| async move {
                        info!(
                            "Processing file '{filename}' with {:?} strategy",
                            compaction
                        );
                        let array_stream = parquet_to_vortex(parquet_file_path)?;
                        let mut f = OpenOptions::new()
                            .write(true)
                            .truncate(true)
                            .create(true)
                            .open(&vtx_file)
                            .await?;

                        let write_options = compaction.apply_options(VortexWriteOptions::default());

                        write_options.write(&mut f, array_stream).await?;

                        anyhow::Ok(())
                    })
                    .await
                    .expect("Failed to write Vortex file")
                }
                .in_current_span(),
            )
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(())
}

/// Convert Parquet files to Lance format.
/// Lance manages its own internal partitioning, so we convert all Parquet files
/// (whether Single or Partitioned flavor) into a single Lance dataset.
pub async fn convert_parquet_to_lance(input_path: &Path) -> anyhow::Result<()> {
    let lance_dir = input_path.join(Format::Lance.name());
    let parquet_path = input_path.join(Format::Parquet.name());
    let dataset_path = lance_dir.join("hits.lance");

    // Use idempotent pattern to avoid reprocessing.
    idempotent_async(&dataset_path, move |lance_path| async move {
        create_dir_all(&lance_dir).await?;

        // Collect all Parquet files in the directory.
        let parquet_files: Vec<_> = fs::read_dir(&parquet_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"))
            .map(|entry| entry.path())
            .collect();

        if parquet_files.is_empty() {
            anyhow::bail!("No Parquet files found in {}", parquet_path.display());
        }

        info!(
            "Converting {} Parquet file(s) to Lance format",
            parquet_files.len()
        );

        // Get schema from the first Parquet file without loading all data.
        let first_file = File::open(&parquet_files[0])?;
        let first_builder = ParquetRecordBatchReaderBuilder::try_new(first_file)?;
        let schema = first_builder.schema().clone();

        // Create a streaming iterator that reads from all Parquet files sequentially.
        let batch_iter = ParquetFilesIterator::new(parquet_files, schema.clone())?;

        info!("Starting streaming write to Lance");

        // Write all batches to a single Lance dataset using streaming.
        let lance_path_str = lance_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Lance dataset path is not valid UTF-8"))?;
        LanceDataset::write(batch_iter, lance_path_str, Some(WriteParams::default())).await?;

        info!(
            "Successfully created Lance dataset at {}",
            lance_path.display()
        );

        anyhow::Ok(())
    })
    .await?;

    Ok(())
}

/// A streaming iterator that reads RecordBatches from multiple Parquet files sequentially.
///
/// We need this because we cannot gather all of the data in memory before writing to lance due to
/// running out of memory on CI.
struct ParquetFilesIterator {
    files: Vec<PathBuf>,
    schema: Arc<Schema>,
    current_file_index: usize,
    current_reader: Option<Box<dyn RecordBatchReader + Send>>,
}

impl ParquetFilesIterator {
    fn new(files: Vec<PathBuf>, schema: Arc<Schema>) -> anyhow::Result<Self> {
        let mut iter = Self {
            files,
            schema,
            current_file_index: 0,
            current_reader: None,
        };
        iter.advance_to_next_file()
            .map_err(|e| anyhow::anyhow!("Failed to open first Parquet file: {}", e))?;
        Ok(iter)
    }

    fn advance_to_next_file(&mut self) -> Result<(), ArrowError> {
        if self.current_file_index < self.files.len() {
            let file = File::open(&self.files[self.current_file_index]).map_err(|e| {
                ArrowError::IoError(format!("Failed to open Parquet file: {}", e), e)
            })?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            self.current_reader = Some(Box::new(builder.build()?));
            self.current_file_index += 1;
        } else {
            self.current_reader = None;
        }
        Ok(())
    }
}

impl Iterator for ParquetFilesIterator {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.current_reader {
                Some(reader) => {
                    match reader.next() {
                        Some(Ok(batch)) => return Some(Ok(batch)),
                        Some(Err(e)) => return Some(Err(e)),
                        None => {
                            // Current file is exhausted, move to the next file.
                            if let Err(e) = self.advance_to_next_file() {
                                return Some(Err(e));
                            }
                        }
                    }
                }
                None => return None,
            }
        }
    }
}

impl RecordBatchReader for ParquetFilesIterator {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

pub async fn register_vortex_files(
    session: SessionContext,
    table_name: &str,
    input_path: &Url,
    schema: Option<Schema>,
    glob_pattern: Option<Pattern>,
) -> anyhow::Result<()> {
    let vortex_path = input_path.join(&format!("{}/", Format::OnDiskVortex.name()))?;
    let format = Arc::new(VortexFormat::default());

    info!(
        "Registering table from {vortex_path} with glob {:?}",
        glob_pattern.as_ref().map(|p| p.as_str()).unwrap_or("")
    );

    let table_url = ListingTableUrl::try_new(vortex_path, glob_pattern)?;

    let config = ListingTableConfig::new(table_url).with_listing_options(
        ListingOptions::new(format).with_session_config_options(session.state().config()),
    );

    let config = if let Some(schema) = schema {
        config.with_schema(schema.into())
    } else {
        config.infer_schema(&session.state()).await?
    };

    let listing_table = Arc::new(ListingTable::try_new(config)?);
    session.register_table(table_name, listing_table)?;

    Ok(())
}

pub async fn register_vortex_compact_files(
    session: SessionContext,
    table_name: &str,
    input_path: &Url,
    schema: Option<Schema>,
    glob_pattern: Option<Pattern>,
) -> anyhow::Result<()> {
    let vortex_compact_path = input_path.join(&format!("{}/", Format::VortexCompact.name()))?;
    let format = Arc::new(VortexFormat::default());

    info!(
        "Registering vortex-compact table from {vortex_compact_path} with glob {:?}",
        glob_pattern.as_ref().map(|p| p.as_str()).unwrap_or("")
    );

    let table_url = ListingTableUrl::try_new(vortex_compact_path, glob_pattern)?;

    let config = ListingTableConfig::new(table_url).with_listing_options(
        ListingOptions::new(format).with_session_config_options(session.state().config()),
    );

    let config = if let Some(schema) = schema {
        config.with_schema(schema.into())
    } else {
        config.infer_schema(&session.state()).await?
    };

    let listing_table = Arc::new(ListingTable::try_new(config)?);
    session.register_table(table_name, listing_table)?;

    Ok(())
}

pub fn register_parquet_files(
    session: &SessionContext,
    table_name: &str,
    input_path: &Url,
    schema: &Schema,
    glob_pattern: Option<Pattern>,
) -> anyhow::Result<()> {
    let format = Arc::new(ParquetFormat::new());
    let table_path = input_path.join(&format!("{}/", Format::Parquet))?;

    info!(
        "Registering table from {} with glob {:?}",
        &table_path,
        glob_pattern.as_ref().map(|p| p.as_str()).unwrap_or("")
    );
    let table_url = ListingTableUrl::try_new(table_path, glob_pattern)?;

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(
            ListingOptions::new(format).with_session_config_options(session.state().config()),
        )
        .with_schema(schema.clone().into());

    let listing_table = Arc::new(ListingTable::try_new(config)?);

    session.register_table(table_name, listing_table)?;

    Ok(())
}

/// Register Lance files with DataFusion.
/// Lance manages its own internal partitioning, so there's always a single dataset
/// regardless of the ClickBench flavor.
pub async fn register_lance_files(
    session: &SessionContext,
    table_name: &str,
    input_path: &Url,
) -> anyhow::Result<()> {
    let dataset_path = input_path.join(&format!("{}/hits.lance/", Format::Lance.name()))?;

    info!("Registering Lance table from {}", &dataset_path);

    let path_str = dataset_path
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Failed to convert URL to file path: {}", dataset_path))?;
    if !path_str.exists() {
        anyhow::bail!(
            "Lance dataset not found at {}. Run data generation with --generate-data first.",
            path_str.display()
        );
    }

    // Open the single Lance dataset.
    let path_str_utf8 = path_str
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Lance dataset path is not valid UTF-8"))?;
    let dataset = LanceDataset::open(path_str_utf8).await?;
    let provider = LanceTableProvider::new(
        Arc::new(dataset),
        false, // with_row_id
        false, // with_row_addr
    );

    // Register the table with DataFusion.
    session.register_table(table_name, Arc::new(provider))?;

    info!("Successfully registered Lance table '{}'", table_name);

    Ok(())
}

pub fn clickbench_queries(queries_file_path: PathBuf) -> Vec<(usize, String)> {
    fs::read_to_string(queries_file_path)
        .unwrap()
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .enumerate()
        .collect()
}

/// Clickbench has two different flavors:
/// - Singe - 1 file containing the whole dataset, just under 100 million rows.
/// - Partitioned (which we run by default) - 100 files, each containing ~1 million rows, all sharing the same schema.
#[derive(ValueEnum, Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize)]
pub enum Flavor {
    #[default]
    Partitioned,
    Single,
}

impl Display for Flavor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.to_possible_value()
                .vortex_expect("Invalid flavour value")
                .get_name()
                .to_lowercase()
        )
    }
}

impl Flavor {
    // TODO(joe): move these elsewhere.
    pub fn download(
        &self,
        client: &reqwest::blocking::Client,
        basepath: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let basepath = basepath.as_ref();
        match self {
            Flavor::Single => {
                let output_path = basepath.join(Format::Parquet.name()).join("hits.parquet");
                idempotent(&output_path, |output_path| {
                    info!("Downloading single clickbench file");
                    let url = "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_single/hits.parquet";
                    let mut response = retry_get(client, url)?;
                    let mut file = File::create(output_path)?;
                    response.copy_to(&mut file)?;

                    anyhow::Ok(())
                })?;
            }
            Flavor::Partitioned => {
                // The clickbench-provided file is missing some higher-level type info, so we reprocess it
                // to add that info, see https://github.com/ClickHouse/ClickBench/issues/7.
                let pool = rayon::ThreadPoolBuilder::new()
                    .thread_name(|i| format!("clickbench download {i}"))
                    .build()?;
                let _ = pool.install(|| (0_u32..100).into_par_iter().map(|idx| {
                    let output_path = basepath.join(Format::Parquet.name()).join(format!("hits_{idx}.parquet"));
                    idempotent(&output_path, |output_path| {
                        info!("Downloading file {idx}");
                        let url = format!("https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_{idx}.parquet");
                        let mut response = retry_get(client, url)?;
                        let mut file = File::create(output_path)?;
                        response.copy_to(&mut file)?;

                        anyhow::Ok(())
                    })
                }).collect::<anyhow::Result<Vec<_>>>())?;
            }
        }
        Ok(())
    }
}

fn retry_get(client: &reqwest::blocking::Client, url: impl IntoUrl) -> anyhow::Result<Response> {
    let url = url.as_str();
    let make_req = || client.get(url).send();

    let mut output = None;

    for attempt in 1..4 {
        match make_req().and_then(|r| r.error_for_status()) {
            Ok(r) => {
                output = Some(r);
                break;
            }
            Err(e) => {
                warn!("Request errored with {e}, retying for the {attempt} time");
            }
        }

        // Very basic backoff mechanism
        std::thread::sleep(Duration::from_secs(attempt));
    }

    match output {
        Some(v) => Ok(v),
        None => anyhow::bail!("Exahusted retry attempts for {url}"),
    }
}
