// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;
use std::sync::LazyLock;
use std::time::Duration;

use arrow_schema::Schema;
use bytes::Bytes;
use clap::ValueEnum;
use futures::StreamExt;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
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
pub use crate::conversions::convert_parquet_directory_to_vortex;
use crate::idempotent_async;
use crate::schema_from_ddl;

pub static HITS_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        WatchID BIGINT NOT NULL,
        JavaEnable SMALLINT NOT NULL,
        Title VARCHAR NOT NULL,
        GoodEvent SMALLINT NOT NULL,
        EventTime TIMESTAMP NOT NULL,
        EventDate TIMESTAMP NOT NULL,
        CounterID INTEGER NOT NULL,
        ClientIP INTEGER NOT NULL,
        RegionID INTEGER NOT NULL,
        UserID BIGINT NOT NULL,
        CounterClass SMALLINT NOT NULL,
        OS SMALLINT NOT NULL,
        UserAgent SMALLINT NOT NULL,
        URL VARCHAR NOT NULL,
        Referer VARCHAR NOT NULL,
        IsRefresh SMALLINT NOT NULL,
        RefererCategoryID SMALLINT NOT NULL,
        RefererRegionID INTEGER NOT NULL,
        URLCategoryID SMALLINT NOT NULL,
        URLRegionID INTEGER NOT NULL,
        ResolutionWidth SMALLINT NOT NULL,
        ResolutionHeight SMALLINT NOT NULL,
        ResolutionDepth SMALLINT NOT NULL,
        FlashMajor SMALLINT NOT NULL,
        FlashMinor SMALLINT NOT NULL,
        FlashMinor2 VARCHAR NOT NULL,
        NetMajor SMALLINT NOT NULL,
        NetMinor SMALLINT NOT NULL,
        UserAgentMajor SMALLINT NOT NULL,
        UserAgentMinor VARCHAR NOT NULL,
        CookieEnable SMALLINT NOT NULL,
        JavascriptEnable SMALLINT NOT NULL,
        IsMobile SMALLINT NOT NULL,
        MobilePhone SMALLINT NOT NULL,
        MobilePhoneModel VARCHAR NOT NULL,
        Params VARCHAR NOT NULL,
        IPNetworkID INTEGER NOT NULL,
        TraficSourceID SMALLINT NOT NULL,
        SearchEngineID SMALLINT NOT NULL,
        SearchPhrase VARCHAR NOT NULL,
        AdvEngineID SMALLINT NOT NULL,
        IsArtifical SMALLINT NOT NULL,
        WindowClientWidth SMALLINT NOT NULL,
        WindowClientHeight SMALLINT NOT NULL,
        ClientTimeZone SMALLINT NOT NULL,
        ClientEventTime TIMESTAMP NOT NULL,
        SilverlightVersion1 SMALLINT NOT NULL,
        SilverlightVersion2 SMALLINT NOT NULL,
        SilverlightVersion3 INTEGER NOT NULL,
        SilverlightVersion4 SMALLINT NOT NULL,
        PageCharset VARCHAR NOT NULL,
        CodeVersion INTEGER NOT NULL,
        IsLink SMALLINT NOT NULL,
        IsDownload SMALLINT NOT NULL,
        IsNotBounce SMALLINT NOT NULL,
        FUniqID BIGINT NOT NULL,
        OriginalURL VARCHAR NOT NULL,
        HID INTEGER NOT NULL,
        IsOldCounter SMALLINT NOT NULL,
        IsEvent SMALLINT NOT NULL,
        IsParameter SMALLINT NOT NULL,
        DontCountHits SMALLINT NOT NULL,
        WithHash SMALLINT NOT NULL,
        HitColor VARCHAR NOT NULL,
        LocalEventTime TIMESTAMP NOT NULL,
        Age SMALLINT NOT NULL,
        Sex SMALLINT NOT NULL,
        Income SMALLINT NOT NULL,
        Interests SMALLINT NOT NULL,
        Robotness SMALLINT NOT NULL,
        RemoteIP INTEGER NOT NULL,
        WindowName INTEGER NOT NULL,
        OpenerName INTEGER NOT NULL,
        HistoryLength SMALLINT NOT NULL,
        BrowserLanguage VARCHAR NOT NULL,
        BrowserCountry VARCHAR NOT NULL,
        SocialNetwork VARCHAR NOT NULL,
        SocialAction VARCHAR NOT NULL,
        HTTPError SMALLINT NOT NULL,
        SendTiming INTEGER NOT NULL,
        DNSTiming INTEGER NOT NULL,
        ConnectTiming INTEGER NOT NULL,
        ResponseStartTiming INTEGER NOT NULL,
        ResponseEndTiming INTEGER NOT NULL,
        FetchTiming INTEGER NOT NULL,
        SocialSourceNetworkID SMALLINT NOT NULL,
        SocialSourcePage VARCHAR NOT NULL,
        ParamPrice BIGINT NOT NULL,
        ParamOrderID VARCHAR NOT NULL,
        ParamCurrency VARCHAR NOT NULL,
        ParamCurrencyID SMALLINT NOT NULL,
        OpenstatServiceName VARCHAR NOT NULL,
        OpenstatCampaignID VARCHAR NOT NULL,
        OpenstatAdID VARCHAR NOT NULL,
        OpenstatSourceID VARCHAR NOT NULL,
        UTMSource VARCHAR NOT NULL,
        UTMMedium VARCHAR NOT NULL,
        UTMCampaign VARCHAR NOT NULL,
        UTMContent VARCHAR NOT NULL,
        UTMTerm VARCHAR NOT NULL,
        FromTag VARCHAR NOT NULL,
        HasGCLID SMALLINT NOT NULL,
        RefererHash BIGINT NOT NULL,
        URLHash BIGINT NOT NULL,
        CLID INTEGER NOT NULL,
    ",
    )
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
                idempotent_async(output_path.as_path(), |output_path| async move {
                    info!("Downloading single clickbench file");
                    let url = "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_single/hits.parquet";
                    download_large_file(&client, url, &output_path).await?;
                    anyhow::Ok(())
                })
                .await?;
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

/// Downloads a large file with streaming and progress indication.
async fn download_large_file(
    client: &reqwest::Client,
    url: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    let response = client.get(url).send().await?.error_for_status()?;

    let total_size = response.content_length().unwrap_or(0);

    let progress = ProgressBar::new(total_size);
    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec})",
        )
        .expect("valid template"),
    );

    let mut file = File::create(output_path).await?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        progress.inc(chunk.len() as u64);
    }

    progress.finish();
    Ok(())
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
