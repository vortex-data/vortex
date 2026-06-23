// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::str::FromStr;
use std::sync::LazyLock;

use anyhow::Context;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::TimeUnit;
use clap::ValueEnum;
use parquet::file::reader::FileReader;
use parquet::file::reader::SerializedFileReader;
use serde::Deserialize;
use serde::Serialize;
use tracing::info;
use vortex::error::VortexExpect;

use crate::Format;
use crate::IdempotentPath;
// Re-export for use by clickbench_benchmark
pub use crate::conversions::convert_parquet_directory_to_vortex;
use crate::datasets::data_downloads::download_data;
use crate::datasets::data_downloads::download_many;
use crate::utils::file::temp_download_filepath;

/// Benchmark and local data directory name for ClickBench sorted by event date/time.
pub const CLICKBENCH_SORTED_NAME: &str = "clickbench-sorted";
const CLICKBENCH_PARTITIONED_NAME: &str = "clickbench_partitioned";
const SORTED_SHARD_COUNT: usize = 100;
const SORTED_SHARD_COUNT_U64: u64 = 100;

/// Zero-based ClickBench query IDs that filter by or order/group on `EventDate`/`EventTime`.
pub const CLICKBENCH_SORTED_QUERY_IDS: &[usize] = &[23, 24, 26, 36, 37, 38, 39, 40, 41, 42];

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

/// Expected result row counts for the 43 upstream ClickBench queries.
pub fn clickbench_expected_row_counts() -> Vec<usize> {
    vec![
        1, 1, 1, 1, 1, 1, 1, 18, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 1, 10, 10, 10, 10,
        10, 10, 25, 25, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    ]
}

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
    pub async fn download(&self, basepath: impl AsRef<Path>) -> anyhow::Result<()> {
        let basepath = basepath.as_ref();
        match self {
            Flavor::Single => {
                let output_path = basepath.join(Format::Parquet.name()).join("hits.parquet");
                info!("Downloading single clickbench file");
                let url = "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_single/hits.parquet";
                download_data(output_path, url).await?;
            }
            Flavor::Partitioned => {
                // The clickbench-provided file is missing some higher-level type info, so we reprocess it
                // to add that info, see https://github.com/ClickHouse/ClickBench/issues/7.
                info!("Downloading 100 ClickBench parquet shards");
                let parquet_dir = basepath.join(Format::Parquet.name());
                let downloads = (0_u32..100).map(|idx| {
                    let output_path = parquet_dir.join(format!("hits_{idx}.parquet"));
                    let url = format!("https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_{idx}.parquet");
                    (output_path, url)
                });
                download_many(downloads).await?;
            }
        }
        Ok(())
    }
}

/// Generate globally sorted ClickBench Parquet shards under `basepath`.
pub async fn generate_sorted_clickbench(basepath: impl AsRef<Path>) -> anyhow::Result<()> {
    let source_base = CLICKBENCH_PARTITIONED_NAME.to_data_path();
    Flavor::Partitioned.download(&source_base).await?;

    let source_parquet_dir = source_base.join(Format::Parquet.name());
    let output_parquet_dir = basepath.as_ref().join(Format::Parquet.name());

    if output_parquet_dir.exists() {
        info!(
            "Sorted ClickBench parquet already exists at {}",
            output_parquet_dir.display()
        );
        return Ok(());
    }

    let temp_root = temp_download_filepath();
    let result =
        generate_sorted_clickbench_inner(&source_parquet_dir, &output_parquet_dir, &temp_root);
    if result.is_err() {
        drop(fs::remove_dir_all(&temp_root));
    }
    result
}

fn generate_sorted_clickbench_inner(
    source_parquet_dir: &Path,
    output_parquet_dir: &Path,
    temp_root: &Path,
) -> anyhow::Result<()> {
    let source_rows = parquet_dir_row_count(source_parquet_dir)?;
    anyhow::ensure!(
        source_rows > 0,
        "ClickBench source parquet directory has no rows: {}",
        source_parquet_dir.display()
    );

    fs::create_dir_all(temp_root)
        .with_context(|| format!("Failed to create temp dir {}", temp_root.display()))?;

    let temp_output_dir = temp_root.join(Format::Parquet.name());
    let duckdb_temp_dir = temp_root.join("duckdb-tmp");
    fs::create_dir_all(&temp_output_dir)
        .with_context(|| format!("Failed to create temp dir {}", temp_output_dir.display()))?;
    fs::create_dir_all(&duckdb_temp_dir)
        .with_context(|| format!("Failed to create temp dir {}", duckdb_temp_dir.display()))?;

    let script = sorted_clickbench_duckdb_script(
        source_parquet_dir,
        &temp_output_dir,
        &duckdb_temp_dir,
        source_rows,
    );
    let db_path = temp_root.join("sort.duckdb");

    info!(
        "Generating globally sorted ClickBench parquet in {}",
        temp_output_dir.display()
    );

    let mut child = Command::new("duckdb")
        .arg(&db_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to run DuckDB CLI while generating sorted ClickBench data")?;

    let mut stdin = child
        .stdin
        .take()
        .context("Failed to open DuckDB stdin while generating sorted ClickBench data")?;
    stdin
        .write_all(script.as_bytes())
        .context("Failed to write sorted ClickBench SQL to DuckDB stdin")?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .context("Failed to wait for DuckDB while generating sorted ClickBench data")?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "DuckDB failed generating sorted ClickBench data: stdout=\"{stdout}\", stderr=\"{stderr}\""
        );
    }

    let output_files = parquet_files(&temp_output_dir)?;
    anyhow::ensure!(
        output_files.len() == SORTED_SHARD_COUNT,
        "Expected {SORTED_SHARD_COUNT} sorted ClickBench shards, got {}",
        output_files.len()
    );

    let output_rows = parquet_files_row_count(output_files)?;
    anyhow::ensure!(
        output_rows == source_rows,
        "Sorted ClickBench row-count mismatch: source={source_rows}, output={output_rows}"
    );

    if let Some(parent) = output_parquet_dir.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output dir {}", parent.display()))?;
    }
    fs::rename(&temp_output_dir, output_parquet_dir).with_context(|| {
        format!(
            "Failed to move sorted ClickBench parquet from {} to {}",
            temp_output_dir.display(),
            output_parquet_dir.display()
        )
    })?;

    drop(fs::remove_dir_all(temp_root));
    Ok(())
}

fn sorted_clickbench_duckdb_script(
    source_parquet_dir: &Path,
    output_parquet_dir: &Path,
    duckdb_temp_dir: &Path,
    source_rows: u64,
) -> String {
    let source_glob = source_parquet_dir.join("hits_*.parquet");
    let rows_per_shard = source_rows.div_ceil(SORTED_SHARD_COUNT_U64);
    let columns = HITS_SCHEMA
        .fields()
        .iter()
        .map(|field| quote_identifier(field.name()))
        .collect::<Vec<_>>()
        .join(", ");

    let mut script = format!(
        "\
PRAGMA temp_directory={temp_dir};
CREATE TABLE hits_sorted AS
    SELECT *
    FROM read_parquet({source_glob})
    ORDER BY \"EventDate\", \"EventTime\", \"WatchID\";
",
        temp_dir = sql_string_literal(&duckdb_temp_dir.display().to_string()),
        source_glob = sql_string_literal(&source_glob.display().to_string()),
    );

    for shard_idx in 0..SORTED_SHARD_COUNT_U64 {
        let start = shard_idx * rows_per_shard;
        let end = (start + rows_per_shard).min(source_rows);
        let output_path = output_parquet_dir.join(format!("hits_{shard_idx}.parquet"));
        script.push_str(&format!(
            "\
COPY (
    SELECT {columns}
    FROM hits_sorted
    WHERE rowid >= {start} AND rowid < {end}
    ORDER BY rowid
) TO {output_path} (FORMAT parquet, COMPRESSION zstd);
",
            output_path = sql_string_literal(&output_path.display().to_string()),
        ));
    }

    script
}

fn parquet_dir_row_count(parquet_dir: &Path) -> anyhow::Result<u64> {
    let files = parquet_files(parquet_dir)?;
    anyhow::ensure!(
        !files.is_empty(),
        "No Parquet files found in {}",
        parquet_dir.display()
    );
    parquet_files_row_count(files)
}

fn parquet_files(parquet_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(parquet_dir)
        .with_context(|| format!("Failed to read parquet dir {}", parquet_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("Failed to list parquet dir {}", parquet_dir.display()))?;

    files.retain(|path| path.extension().is_some_and(|ext| ext == "parquet"));
    files.sort();
    Ok(files)
}

fn parquet_files_row_count(files: Vec<PathBuf>) -> anyhow::Result<u64> {
    let mut total = 0_u64;
    for file_path in files {
        let file = File::open(&file_path)
            .with_context(|| format!("Failed to open parquet file {}", file_path.display()))?;
        let reader = SerializedFileReader::new(file)
            .with_context(|| format!("Failed to read parquet metadata {}", file_path.display()))?;
        let rows = reader.metadata().file_metadata().num_rows();
        let rows = u64::try_from(rows).with_context(|| {
            format!("Parquet row count was negative in {}", file_path.display())
        })?;
        total = total.checked_add(rows).with_context(|| {
            format!(
                "Parquet row count overflow while reading {}",
                file_path.display()
            )
        })?;
    }
    Ok(total)
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}
