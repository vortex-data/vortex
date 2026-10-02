// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `cpu-only` use case of the [Time Series Benchmark Suite](https://github.com/timescale/tsbs).
//!
//! TSBS simulates CPU metrics for a fleet of hosts: one row per host every 10 seconds, ten host
//! tags and ten CPU usage metrics. This suite uses TSBS's canonical configuration of 4,000 hosts
//! over three days, and runs one query of each of its 15 DevOps query types.

use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::builder::Float64Builder;
use arrow_array::builder::StringBuilder;
use arrow_array::builder::TimestampMicrosecondBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use arrow_schema::TimeUnit;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::utils::file::resolve_data_url;
use crate::utils::file::temp_download_filepath;
use crate::workspace_root;

/// Benchmark and local data directory name.
pub const TSBS_NAME: &str = "tsbs";

/// TSBS commit the data generator is built from, pinned so the data cannot drift.
const TSBS_COMMIT: &str = "8323e59c74027b108f4ad5ec5d3e498b0101a02e";

/// Generator arguments. The queries in `sql/tsbs.sql` were generated for this exact configuration.
const GENERATOR_ARGS: &[&str] = &[
    "--use-case=cpu-only",
    "--seed=123",
    "--scale=4000",
    "--timestamp-start=2016-01-01T00:00:00Z",
    "--timestamp-end=2016-01-04T00:00:00Z",
    "--log-interval=10s",
    "--format=timescaledb",
];

/// 4,000 hosts with one reading every 10 seconds for three days, end exclusive.
const EXPECTED_ROWS: u64 = 4_000 * 3 * 24 * 60 * 6;

const ROWS_PER_BATCH: usize = 128 * 1024;
const ROWS_PER_FILE: u64 = EXPECTED_ROWS / 10;

const TAGS: [&str; 10] = [
    "hostname",
    "region",
    "datacenter",
    "rack",
    "os",
    "arch",
    "team",
    "service",
    "service_version",
    "service_environment",
];

const METRICS: [&str; 10] = [
    "usage_user",
    "usage_system",
    "usage_idle",
    "usage_nice",
    "usage_iowait",
    "usage_irq",
    "usage_softirq",
    "usage_steal",
    "usage_guest",
    "usage_guest_nice",
];

/// The `cpu` table, typed as TSBS's TimescaleDB loader creates it: tags as text, metrics as
/// double precision.
static CPU_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    let fields = [Field::new(
        "time",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]
    .into_iter()
    .chain(
        TAGS.iter()
            .map(|tag| Field::new(*tag, DataType::Utf8, false)),
    )
    .chain(
        METRICS
            .iter()
            .map(|metric| Field::new(*metric, DataType::Float64, false)),
    )
    .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
});

/// TSBS `cpu-only` with the 15 DevOps queries.
pub struct TsbsBenchmark {
    data_url: Url,
}

impl TsbsBenchmark {
    /// Create the benchmark, optionally using a remote data directory.
    pub fn new(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            data_url: resolve_data_url(use_remote_data_dir.as_deref(), TSBS_NAME)?,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for TsbsBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/tsbs.md"
    }

    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("tsbs.sql");
        Ok(split_queries(&fs::read_to_string(queries_file)?))
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let base_path = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url))?;
        let parquet_dir = base_path.join(Format::Parquet.name());
        if parquet_dir.exists() {
            info!("TSBS parquet already exists at {}", parquet_dir.display());
            return Ok(());
        }

        let temp_dir = temp_download_filepath();
        let result = generate_cpu_parquet(&temp_dir, &parquet_dir);
        if result.is_err() {
            drop(fs::remove_dir_all(&temp_dir));
        }
        result
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Tsbs
    }

    fn dataset_name(&self) -> &str {
        TSBS_NAME
    }

    fn dataset_display(&self) -> String {
        TSBS_NAME.to_string()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("cpu", Some(CPU_SCHEMA.as_ref().clone()))]
    }
}

/// Split `;`-separated SQL into numbered queries. A `;` must not appear in a comment.
fn split_queries(sql: &str) -> Vec<(usize, String)> {
    sql.split_terminator(';')
        .map(str::trim)
        .filter(|stmt| !stmt.is_empty())
        .map(str::to_string)
        .enumerate()
        .collect()
}

fn generate_cpu_parquet(temp_dir: &Path, parquet_dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(temp_dir)
        .with_context(|| format!("Failed to create temp dir {}", temp_dir.display()))?;

    info!("Generating TSBS cpu-only data with tsbs_generate_data at {TSBS_COMMIT}");
    let mut child = Command::new("go")
        .arg("run")
        .arg(format!(
            "github.com/timescale/tsbs/cmd/tsbs_generate_data@{TSBS_COMMIT}"
        ))
        .args(GENERATOR_ARGS)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to run `go`. Generating TSBS data needs a Go toolchain on PATH")?;
    let stdout = child
        .stdout
        .take()
        .context("Failed to open tsbs_generate_data stdout")?;

    let mut writer = ParquetFiles::new(temp_dir);
    let rows = parse_tsbs_output(BufReader::new(stdout), ROWS_PER_BATCH, |batch| {
        writer.write(&batch)
    })?;
    writer.finish()?;

    let status = child.wait()?;
    anyhow::ensure!(status.success(), "tsbs_generate_data failed with {status}");
    anyhow::ensure!(
        rows == EXPECTED_ROWS,
        "TSBS row-count mismatch: expected {EXPECTED_ROWS}, got {rows}"
    );

    if let Some(parent) = parquet_dir.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(temp_dir, parquet_dir).with_context(|| {
        format!(
            "Failed to move TSBS parquet from {} to {}",
            temp_dir.display(),
            parquet_dir.display()
        )
    })?;
    Ok(())
}

/// Writes batches to `cpu_NNN.parquet` files of about [`ROWS_PER_FILE`] rows each.
struct ParquetFiles<'a> {
    dir: &'a Path,
    writer: Option<ArrowWriter<File>>,
    file_idx: usize,
    file_rows: u64,
}

impl<'a> ParquetFiles<'a> {
    fn new(dir: &'a Path) -> Self {
        Self {
            dir,
            writer: None,
            file_idx: 0,
            file_rows: 0,
        }
    }

    fn write(&mut self, batch: &RecordBatch) -> anyhow::Result<()> {
        if self.writer.is_none() {
            let path = self.dir.join(format!("cpu_{:03}.parquet", self.file_idx));
            let props = WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
                .build();
            self.writer = Some(ArrowWriter::try_new(
                File::create(path)?,
                Arc::clone(&CPU_SCHEMA),
                Some(props),
            )?);
        }
        self.writer
            .as_mut()
            .context("Parquet writer was just created")?
            .write(batch)?;

        self.file_rows += batch.num_rows() as u64;
        if self.file_rows >= ROWS_PER_FILE {
            self.close_file()?;
        }
        Ok(())
    }

    fn close_file(&mut self) -> anyhow::Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
            self.file_idx += 1;
            self.file_rows = 0;
        }
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<()> {
        self.close_file()
    }
}

/// Parse `tsbs_generate_data --format=timescaledb` output into `cpu` record batches.
///
/// The output starts with header lines naming the tags and metrics, followed by pairs of lines:
/// a `tags,k=v,...` line for a host and a `cpu,<epoch nanos>,<metric values>` line.
fn parse_tsbs_output(
    reader: impl BufRead,
    rows_per_batch: usize,
    mut on_batch: impl FnMut(RecordBatch) -> anyhow::Result<()>,
) -> anyhow::Result<u64> {
    let mut builder = CpuBatchBuilder::new(rows_per_batch);
    // Reused across rows: every row repeats one of a few thousand tag sets.
    let mut tags = vec![String::new(); TAGS.len()];
    let mut have_tags = false;
    let mut rows = 0u64;

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let (kind, rest) = line.split_once(',').unwrap_or((line.as_str(), ""));
        match kind {
            "tags" if rest.contains(' ') => check_header(rest, &TAGS, " string")?,
            "tags" => {
                parse_tags(rest, &mut tags).with_context(|| format!("Invalid line '{line}'"))?;
                have_tags = true;
            }
            "cpu" => {
                let (first, metrics) = rest.split_once(',').unwrap_or((rest, ""));
                let Ok(nanos) = first.parse::<i64>() else {
                    check_header(rest, &METRICS, "")?;
                    continue;
                };
                anyhow::ensure!(have_tags, "cpu line before any tags: '{line}'");
                let values =
                    parse_metrics(metrics).with_context(|| format!("Invalid line '{line}'"))?;

                builder.append(nanos / 1_000, &tags, &values);
                rows += 1;
                if builder.len() == rows_per_batch {
                    on_batch(builder.finish()?)?;
                }
            }
            _ => anyhow::bail!("Unexpected TSBS line '{line}'"),
        }
    }

    if builder.len() > 0 {
        on_batch(builder.finish()?)?;
    }
    Ok(rows)
}

/// Parse `k=v,...` tag pairs, in [`TAGS`] order, into `tags`.
fn parse_tags(pairs: &str, tags: &mut [String]) -> anyhow::Result<()> {
    let mut count = 0;
    for ((pair, tag), value) in pairs.split(',').zip(TAGS).zip(tags.iter_mut()) {
        let parsed = pair
            .strip_prefix(tag)
            .and_then(|rest| rest.strip_prefix('='))
            .with_context(|| format!("Expected tag {tag}"))?;
        value.clear();
        value.push_str(parsed);
        count += 1;
    }
    anyhow::ensure!(count == TAGS.len(), "Expected {} tags", TAGS.len());
    Ok(())
}

fn parse_metrics(values: &str) -> anyhow::Result<[f64; METRICS.len()]> {
    let mut parsed = [0.0; METRICS.len()];
    let mut count = 0;
    for (value, slot) in values.split(',').zip(parsed.iter_mut()) {
        *slot = value.parse()?;
        count += 1;
    }
    anyhow::ensure!(count == METRICS.len(), "Expected {} metrics", METRICS.len());
    Ok(parsed)
}

fn check_header(names: &str, expected: &[&str], suffix: &str) -> anyhow::Result<()> {
    let matches = names
        .split(',')
        .map(|name| name.strip_suffix(suffix).unwrap_or(name))
        .eq(expected.iter().copied());
    anyhow::ensure!(
        matches,
        "Unexpected TSBS header '{names}', expected {}",
        expected.join(",")
    );
    Ok(())
}

struct CpuBatchBuilder {
    time: TimestampMicrosecondBuilder,
    tags: Vec<StringBuilder>,
    metrics: Vec<Float64Builder>,
    len: usize,
}

impl CpuBatchBuilder {
    fn new(capacity: usize) -> Self {
        Self {
            time: TimestampMicrosecondBuilder::with_capacity(capacity),
            tags: TAGS
                .iter()
                .map(|_| StringBuilder::with_capacity(capacity, capacity * 16))
                .collect(),
            metrics: METRICS
                .iter()
                .map(|_| Float64Builder::with_capacity(capacity))
                .collect(),
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn append(&mut self, micros: i64, tags: &[String], metrics: &[f64]) {
        self.time.append_value(micros);
        for (builder, tag) in self.tags.iter_mut().zip(tags) {
            builder.append_value(tag);
        }
        for (builder, metric) in self.metrics.iter_mut().zip(metrics) {
            builder.append_value(*metric);
        }
        self.len += 1;
    }

    fn finish(&mut self) -> anyhow::Result<RecordBatch> {
        let columns = std::iter::once(Arc::new(self.time.finish()) as ArrayRef)
            .chain(
                self.tags
                    .iter_mut()
                    .map(|builder| Arc::new(builder.finish()) as ArrayRef),
            )
            .chain(
                self.metrics
                    .iter_mut()
                    .map(|builder| Arc::new(builder.finish()) as ArrayRef),
            )
            .collect();
        self.len = 0;
        Ok(RecordBatch::try_new(Arc::clone(&CPU_SCHEMA), columns)?)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::Float64Array;
    use arrow_array::StringArray;
    use arrow_array::TimestampMicrosecondArray;

    use super::*;

    const SAMPLE: &str = "\
tags,hostname string,region string,datacenter string,rack string,os string,arch string,team string,service string,service_version string,service_environment string
cpu,usage_user,usage_system,usage_idle,usage_nice,usage_iowait,usage_irq,usage_softirq,usage_steal,usage_guest,usage_guest_nice

tags,hostname=host_0,region=eu-central-1,datacenter=eu-central-1a,rack=6,os=Ubuntu15.10,arch=x86,team=SF,service=19,service_version=1,service_environment=test
cpu,1451606400000000000,58,2,24,61,22,63,6,44,80,38
tags,hostname=host_1,region=us-west-1,datacenter=us-west-1a,rack=41,os=Ubuntu15.10,arch=x64,team=NYC,service=9,service_version=1,service_environment=staging
cpu,1451606400000000000,84,11,53,87,29,20,54,77,53,74
tags,hostname=host_0,region=eu-central-1,datacenter=eu-central-1a,rack=6,os=Ubuntu15.10,arch=x86,team=SF,service=19,service_version=1,service_environment=test
cpu,1451606410000000000,59,4,25,60,23,62,5,47,80,38
";

    #[test]
    fn parses_generator_output_into_cpu_batches() -> anyhow::Result<()> {
        let mut batches = Vec::new();
        let rows = parse_tsbs_output(SAMPLE.as_bytes(), 2, |batch| {
            batches.push(batch);
            Ok(())
        })?;

        assert_eq!(rows, 3);
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );

        let first = &batches[0];
        let time = first
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .context("time column")?;
        assert_eq!(time.value(0), 1_451_606_400_000_000);

        let hostname = first
            .column_by_name("hostname")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .context("hostname column")?;
        assert_eq!(hostname.value(1), "host_1");

        let usage_guest_nice = batches[1]
            .column_by_name("usage_guest_nice")
            .and_then(|column| column.as_any().downcast_ref::<Float64Array>())
            .context("usage_guest_nice column")?;
        assert_eq!(usage_guest_nice.len(), 1);
        assert_eq!(usage_guest_nice.value(0), 38.0);
        Ok(())
    }

    #[test]
    fn rejects_unexpected_headers() {
        let sample = SAMPLE.replace("usage_user,usage_system", "usage_system,usage_user");
        assert!(parse_tsbs_output(sample.as_bytes(), 2, |_| Ok(())).is_err());
    }

    #[test]
    fn queries_file_holds_one_query_per_devops_query_type() -> anyhow::Result<()> {
        let sql = fs::read_to_string(
            workspace_root()
                .join("vortex-bench")
                .join("sql")
                .join("tsbs.sql"),
        )?;
        let queries = split_queries(&sql);

        assert_eq!(queries.len(), 15);
        assert!(queries.iter().all(|(_, query)| query.contains("FROM cpu")));
        Ok(())
    }
}
