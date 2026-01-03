// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;
use std::sync::LazyLock;
use std::time::Duration;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::TimeUnit;
use bytes::Bytes;
use clap::ValueEnum;
use reqwest::IntoUrl;
use serde::Deserialize;
use serde::Serialize;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::task::JoinSet;
use tracing::info;
use tracing::warn;
use vortex::error::VortexExpect;

use crate::Format;
// Re-export for use by clickbench_benchmark
pub use crate::conversions::convert_parquet_to_vortex;
use crate::idempotent_async;

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

/// Clickbench has two different flavors:
/// - Singe - 1 file containing the whole dataset, just under 100 million rows.
/// - Partitioned (which we run by default) - 100 files, each containing ~1 million rows, all sharing the same schema.
#[derive(ValueEnum, Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Flavor {
    #[default]
    Partitioned,
    Single,
}

impl FromStr for Flavor {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "partitioned" => Ok(Flavor::Partitioned),
            "single" => Ok(Flavor::Single),
            _ => anyhow::bail!(
                "Failed to parse {s}, only valid flavor values are `partitioned` or `single`"
            ),
        }
    }
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
    pub async fn download(
        &self,
        client: reqwest::Client,
        basepath: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let basepath = basepath.as_ref();
        match self {
            Flavor::Single => {
                let output_path = basepath.join(Format::Parquet.name()).join("hits.parquet");
                idempotent_async(output_path.as_path(), |output_path| async {
                    info!("Downloading single clickbench file");
                    let url = "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_single/hits.parquet";
                    let  body = retry_get(&client, url).await?;
                    let mut file = File::create(output_path).await?;

                    file.write_all(&body).await?;

                    anyhow::Ok(())
                }).await?;
            }
            Flavor::Partitioned => {
                // The clickbench-provided file is missing some higher-level type info, so we reprocess it
                // to add that info, see https://github.com/ClickHouse/ClickBench/issues/7.

                let mut tasks = (0_u32..100).map(|idx| {
                    let output_path = basepath.join(Format::Parquet.name()).join(format!("hits_{idx}.parquet"));
                    let client = client.clone();

                    idempotent_async(output_path,  move|output_path| async move {
                        info!("Downloading file {idx}");
                        let url = format!("https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_{idx}.parquet");
                        let  body = retry_get(&client, url).await?;
                        let mut file = File::create(output_path).await?;
                        file.write_all(&body).await?;

                        anyhow::Ok(())
                    })
                }).collect::<JoinSet<_>>();

                while let Some(task) = tasks.join_next().await {
                    task??;
                }
            }
        }
        Ok(())
    }
}

async fn retry_get(client: &reqwest::Client, url: impl IntoUrl) -> anyhow::Result<Bytes> {
    let url = url.as_str();
    let make_req = || async { client.get(url).send().await };

    let mut body = None;

    for attempt in 0..3 {
        match make_req().await.and_then(|r| r.error_for_status()) {
            Ok(r) => match r.bytes().await {
                Ok(b) => {
                    body = Some(b);
                    break;
                }
                Err(e) => {
                    warn!("Request errored with {e}, retying for the {attempt} time");
                }
            },
            Err(e) => {
                warn!("Request errored with {e}, retying for the {attempt} time");
            }
        }

        // Very basic backoff mechanism
        tokio::time::sleep(Duration::from_secs(attempt + 1)).await;
    }

    match body {
        Some(v) => Ok(v),
        None => anyhow::bail!("Exahusted retry attempts for {url}"),
    }
}
