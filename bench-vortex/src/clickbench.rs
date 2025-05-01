use std::fmt::Display;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use std::{fmt, fs};

use arrow_schema::{DataType, Field, Schema, TimeUnit};
use clap::ValueEnum;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use futures::{StreamExt, TryStreamExt, stream};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use reqwest::IntoUrl;
use reqwest::blocking::Response;
use tokio::fs::{OpenOptions, create_dir_all};
use tracing::{debug, info, warn};
use url::Url;
use vortex::error::VortexExpect;
use vortex::file::{VORTEX_FILE_EXTENSION, VortexWriteOptions};
use vortex_datafusion::persistent::VortexFormat;

use crate::conversions::parquet_to_vortex;
use crate::utils::file_utils::{idempotent, idempotent_async};

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

pub async fn convert_parquet_to_vortex(input_path: &Path) -> anyhow::Result<()> {
    let vortex_dir = input_path.join("vortex");
    let parquet_path = input_path.join("parquet");
    create_dir_all(&vortex_dir).await?;

    let parquet_inputs = fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;

    debug!(
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
            let output_path = vortex_dir.join(format!("{filename}.{VORTEX_FILE_EXTENSION}"));

            tokio::spawn(async move {
                idempotent_async(&output_path, move |vtx_file| async move {
                    info!("Processing file '{filename}'");
                    let array_stream = parquet_to_vortex(parquet_file_path)?;
                    let f = OpenOptions::new()
                        .write(true)
                        .truncate(true)
                        .create(true)
                        .open(&vtx_file)
                        .await?;

                    VortexWriteOptions::default().write(f, array_stream).await?;

                    anyhow::Ok(())
                })
                .await
                .expect("Failed to write Vortex file")
            })
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(())
}

pub async fn register_vortex_files(
    session: SessionContext,
    table_name: &str,
    input_path: &Url,
    schema: Option<Schema>,
    single_file: bool,
) -> anyhow::Result<()> {
    let mut vortex_path = input_path.join("vortex/")?;
    if single_file {
        vortex_path = vortex_path.join("hits_0.vortex")?;
    }

    let format = Arc::new(VortexFormat::default());

    info!("Registering table from {vortex_path}");

    let table_url = ListingTableUrl::parse(vortex_path)?;

    let config =
        ListingTableConfig::new(table_url).with_listing_options(ListingOptions::new(format));

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

    Ok(())
}

pub fn register_parquet_files(
    session: &SessionContext,
    table_name: &str,
    input_path: &Url,
    schema: &Schema,
    single_file: bool,
) -> anyhow::Result<()> {
    let format = Arc::new(ParquetFormat::new());
    let mut table_path = input_path.join("parquet/")?;
    if single_file {
        table_path = table_path.join("hits_0.parquet")?;
    }

    info!("Registering table from {}", &table_path);
    let table_url = ListingTableUrl::parse(table_path)?;

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(ListingOptions::new(format))
        .with_schema(schema.clone().into());

    let listing_table = Arc::new(ListingTable::try_new(config)?);

    session.register_table(table_name, listing_table)?;

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

#[derive(ValueEnum, Default, Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        )
    }
}

impl Flavor {
    pub fn download(
        &self,
        client: &reqwest::blocking::Client,
        basepath: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let basepath = basepath.as_ref();
        match self {
            Flavor::Single => {
                let output_path = basepath.join("parquet").join("hits.parquet");
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
                let _ = (0_u32..100).into_par_iter().map(|idx| {
                    let output_path = basepath.join("parquet").join(format!("hits_{idx}.parquet"));
                    idempotent(&output_path, |output_path| {
                        info!("Downloading file {idx}");
                        let url = format!("https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_{idx}.parquet");
                        let mut response = retry_get(client, url)?;
                        let mut file = File::create(output_path)?;
                        response.copy_to(&mut file)?;

                        anyhow::Ok(())
                    })
                }).collect::<anyhow::Result<Vec<_>>>()?;
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
