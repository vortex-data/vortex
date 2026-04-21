// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Download public datasets, extract string columns, and write Vortex files
//! with paired raw (VarBin) and FSST-compressed columns plus sidecar stats.

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use arrow_array::Array as ArrowArray;
use arrow_array::DictionaryArray;
use arrow_array::FixedSizeListArray;
use arrow_array::LargeListArray;
use arrow_array::LargeStringArray;
use arrow_array::ListArray;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use arrow_array::StructArray;
use arrow_array::types::Int32Type;
use arrow_array::types::UInt32Type;
use clap::Args;
use humansize::BINARY;
use humansize::SizeFormatter;
use serde::Deserialize;
use serde::Serialize;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::info;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_bench::Benchmark;
use vortex_bench::data_dir;
use vortex_bench::polarsignals::PolarSignalsBenchmark;
use vortex_bench::realnest::gharchive::GithubArchiveBenchmark;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

/// Known public datasets.
#[derive(Debug, Clone, Copy, clap::ValueEnum, Serialize, Deserialize)]
pub enum DatasetName {
    /// ClickBench hits URL column (~1M rows per shard)
    ClickbenchUrl,
    /// ClickBench hits Title column (~1M rows per shard)
    ClickbenchTitle,
    /// ClickBench hits Referer column (~1M rows per shard)
    ClickbenchReferer,
    /// ClickBench hits SearchPhrase column (~1M rows per shard)
    ClickbenchSearchPhrase,
    /// ClickBench hits Params column (~1M rows per shard)
    ClickbenchParams,
    /// Synthetic JSON lines (key-value NDJSON)
    JsonLines,
    /// FineWeb URL column (CommonCrawl web URLs)
    FinewebUrl,
    /// FineWeb text column (CommonCrawl web page text)
    FinewebText,
    /// GitHub Archive repo.name field
    GharchiveRepoName,
    /// GitHub Archive actor.login field
    GharchiveActorLogin,
    /// GitHub Archive payload.ref field
    GharchivePayloadRef,
    /// GitHub Archive actor.avatar_url field
    GharchiveActorAvatarUrl,
    /// PolarSignals labels.comm field
    PolarsignalsLabelsComm,
    /// PolarSignals labels.thread_name field
    PolarsignalsLabelsThreadName,
    /// PolarSignals locations.mapping_file field
    PolarsignalsMappingFile,
    /// PolarSignals locations.lines.function_name field
    PolarsignalsFunctionName,
    /// PolarSignals locations.lines.function_filename field
    PolarsignalsFunctionFilename,
    /// TPC-H lineitem l_comment column (SF1, ~6M rows)
    TpchLineitem,
}

impl std::fmt::Display for DatasetName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.file_stem())
    }
}

impl DatasetName {
    pub fn file_stem(&self) -> &'static str {
        match self {
            Self::ClickbenchUrl => "clickbench_url",
            Self::ClickbenchTitle => "clickbench_title",
            Self::ClickbenchReferer => "clickbench_referer",
            Self::ClickbenchSearchPhrase => "clickbench_search_phrase",
            Self::ClickbenchParams => "clickbench_params",
            Self::JsonLines => "json_lines",
            Self::FinewebUrl => "fineweb_url",
            Self::FinewebText => "fineweb_text",
            Self::GharchiveRepoName => "gharchive_repo_name",
            Self::GharchiveActorLogin => "gharchive_actor_login",
            Self::GharchivePayloadRef => "gharchive_payload_ref",
            Self::GharchiveActorAvatarUrl => "gharchive_actor_avatar_url",
            Self::PolarsignalsLabelsComm => "polarsignals_labels_comm",
            Self::PolarsignalsLabelsThreadName => "polarsignals_labels_thread_name",
            Self::PolarsignalsMappingFile => "polarsignals_mapping_file",
            Self::PolarsignalsFunctionName => "polarsignals_function_name",
            Self::PolarsignalsFunctionFilename => "polarsignals_function_filename",
            Self::TpchLineitem => "tpch_lineitem",
        }
    }
}

#[derive(Args)]
pub struct PrepArgs {
    /// Which dataset to prepare
    #[arg(value_enum)]
    pub dataset: DatasetName,

    /// Maximum number of strings to sample (0 = all)
    #[arg(long, default_value_t = 1_000_000)]
    pub max_rows: usize,
}

/// Sidecar statistics written alongside the Vortex file.
#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetStats {
    pub dataset: String,
    pub row_count: usize,
    pub raw_bytes: u64,
    pub fsst_compressed_bytes: u64,
    pub compression_ratio: f64,
    pub avg_string_len: f64,
}

pub async fn run(args: PrepArgs) -> Result<()> {
    let strings = fetch_strings(&args).await?;
    let row_count = strings.len();
    info!("Fetched {row_count} strings for {}", args.dataset);

    let raw_bytes: u64 = strings.iter().map(|s| s.len() as u64).sum();
    let avg_string_len = if row_count > 0 {
        raw_bytes as f64 / row_count as f64
    } else {
        0.0
    };

    // Build raw VarBin array
    let varbin = VarBinArray::from_iter_nonnull(
        strings.iter().map(|s| s.as_bytes()),
        DType::Utf8(Nullability::NonNullable),
    );

    // Train FSST and compress
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    let fsst_array: FSSTArray = fsst_compress(varbin, len, &dtype, &compressor);

    // Estimate compressed size from the codes buffer
    let fsst_compressed_bytes = estimate_fsst_bytes(&fsst_array);
    let compression_ratio = if fsst_compressed_bytes > 0 {
        raw_bytes as f64 / fsst_compressed_bytes as f64
    } else {
        0.0
    };

    let stats = DatasetStats {
        dataset: args.dataset.to_string(),
        row_count,
        raw_bytes,
        fsst_compressed_bytes,
        compression_ratio,
        avg_string_len,
    };

    // Write output files
    let out_dir = output_dir(&args.dataset);
    tokio::fs::create_dir_all(&out_dir).await?;

    let raw_path = out_dir.join(format!("{}_raw.vortex", args.dataset.file_stem()));
    let fsst_path = out_dir.join(format!("{}_fsst.vortex", args.dataset.file_stem()));
    let stats_path = out_dir.join(format!("{}_stats.json", args.dataset.file_stem()));
    let strings_path = out_dir.join(format!("{}_strings.txt", args.dataset.file_stem()));

    // Write raw strings as a Vortex file
    let raw_varbin = VarBinArray::from_iter_nonnull(
        strings.iter().map(|s| s.as_bytes()),
        DType::Utf8(Nullability::NonNullable),
    );
    write_array_file(&raw_path, raw_varbin.into_array()).await?;
    info!("Wrote raw Vortex file: {}", raw_path.display());

    // Write FSST array as a Vortex file
    write_array_file(&fsst_path, fsst_array.into_array()).await?;
    info!("Wrote FSST Vortex file: {}", fsst_path.display());

    // Write raw strings as newline-delimited text (for easy loading by mine/run)
    {
        let mut f = std::fs::File::create(&strings_path)?;
        for s in &strings {
            writeln!(f, "{s}")?;
        }
    }
    info!("Wrote strings file: {}", strings_path.display());

    // Write stats
    let stats_json = serde_json::to_string_pretty(&stats)?;
    tokio::fs::write(&stats_path, &stats_json).await?;

    println!("\n=== Dataset Stats ===");
    println!("  Dataset:           {}", args.dataset);
    println!("  Rows:              {row_count}");
    println!(
        "  Raw size:          {}",
        SizeFormatter::new(raw_bytes, BINARY)
    );
    println!(
        "  FSST size:         {}",
        SizeFormatter::new(fsst_compressed_bytes, BINARY)
    );
    println!("  Compression ratio: {:.2}x", compression_ratio);
    println!("  Avg string len:    {:.1} bytes", avg_string_len);
    println!("  Output dir:        {}", out_dir.display());

    Ok(())
}

pub fn output_dir(dataset: &DatasetName) -> PathBuf {
    data_dir()
        .join("string-filter-bench")
        .join(dataset.file_stem())
}

fn estimate_fsst_bytes(fsst: &FSSTArray) -> u64 {
    let codes = fsst.codes();
    codes.bytes().len() as u64
}

async fn write_array_file(path: &Path, array: ArrayRef) -> Result<()> {
    let mut file = File::create(path).await?;
    SESSION
        .write_options()
        .write(&mut file, array.to_array_stream())
        .await
        .context("Failed to write Vortex file")?;
    file.flush().await?;
    Ok(())
}

/// Fetch strings for the given dataset.
async fn fetch_strings(args: &PrepArgs) -> Result<Vec<String>> {
    match args.dataset {
        DatasetName::ClickbenchUrl => fetch_clickbench_column("URL", args.max_rows).await,
        DatasetName::ClickbenchTitle => fetch_clickbench_column("Title", args.max_rows).await,
        DatasetName::ClickbenchReferer => fetch_clickbench_column("Referer", args.max_rows).await,
        DatasetName::ClickbenchSearchPhrase => {
            fetch_clickbench_column("SearchPhrase", args.max_rows).await
        }
        DatasetName::ClickbenchParams => fetch_clickbench_column("Params", args.max_rows).await,
        DatasetName::JsonLines => Ok(generate_json_lines(args.max_rows)),
        DatasetName::FinewebUrl => fetch_fineweb_column("url", args.max_rows).await,
        DatasetName::FinewebText => fetch_fineweb_column("text", args.max_rows).await,
        DatasetName::GharchiveRepoName => {
            fetch_gharchive_field(&["repo", "name"], args.max_rows).await
        }
        DatasetName::GharchiveActorLogin => {
            fetch_gharchive_field(&["actor", "login"], args.max_rows).await
        }
        DatasetName::GharchivePayloadRef => {
            fetch_gharchive_field(&["payload", "ref"], args.max_rows).await
        }
        DatasetName::GharchiveActorAvatarUrl => {
            fetch_gharchive_field(&["actor", "avatar_url"], args.max_rows).await
        }
        DatasetName::PolarsignalsLabelsComm => {
            fetch_polarsignals_field(&["labels", "comm"], args.max_rows).await
        }
        DatasetName::PolarsignalsLabelsThreadName => {
            fetch_polarsignals_field(&["labels", "thread_name"], args.max_rows).await
        }
        DatasetName::PolarsignalsMappingFile => {
            fetch_polarsignals_field(&["locations", "mapping_file"], args.max_rows).await
        }
        DatasetName::PolarsignalsFunctionName => {
            fetch_polarsignals_field(&["locations", "lines", "function_name"], args.max_rows).await
        }
        DatasetName::PolarsignalsFunctionFilename => {
            fetch_polarsignals_field(&["locations", "lines", "function_filename"], args.max_rows)
                .await
        }
        DatasetName::TpchLineitem => fetch_tpch_lineitem(args.max_rows).await,
    }
}

/// Download the ClickBench hits dataset (Parquet, ~100MB) and extract one string column.
async fn fetch_clickbench_column(
    column_name: &'static str,
    max_rows: usize,
) -> Result<Vec<String>> {
    // Use a single shard (~100MB) rather than the full dataset (~11GB)
    let parquet_url = "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_0.parquet";
    let cache_path = data_dir()
        .join("string-filter-bench")
        .join("clickbench_hits.parquet");

    // Download if not cached
    if !cache_path.exists() {
        info!("Downloading ClickBench hits.parquet (~100MB)...");
        tokio::fs::create_dir_all(cache_path.parent().unwrap()).await?;
        download_file(parquet_url, &cache_path).await?;
        info!("Download complete: {}", cache_path.display());
    } else {
        info!("Using cached ClickBench data: {}", cache_path.display());
    }

    fetch_parquet_string_path(cache_path, &[column_name], max_rows).await
}

/// Extract a string column from the pre-downloaded FineWeb parquet file.
async fn fetch_fineweb_column(column_name: &str, max_rows: usize) -> Result<Vec<String>> {
    let parquet_path = data_dir()
        .join("fineweb")
        .join("parquet")
        .join("sample.parquet");

    anyhow::ensure!(
        parquet_path.exists(),
        "FineWeb parquet not found. Download it first with:\n  \
         cargo run --release --bin data-gen -- fineweb --formats parquet\n\
         Expected: {}",
        parquet_path.display()
    );

    let strings = fetch_parquet_string_path(parquet_path, &[column_name], max_rows).await?;

    info!("Extracted {} strings", strings.len());
    Ok(strings)
}

async fn fetch_gharchive_field(field_path: &[&str], max_rows: usize) -> Result<Vec<String>> {
    let parquet_path = data_dir()
        .join("gharchive")
        .join("parquet")
        .join("events.parquet");

    if !parquet_path.exists() {
        info!("Generating GitHub Archive parquet data...");
        GithubArchiveBenchmark::with_remote_data_dir(None)?
            .generate_base_data()
            .await?;
    }

    fetch_parquet_string_path(parquet_path, field_path, max_rows).await
}

async fn fetch_polarsignals_field(field_path: &[&str], max_rows: usize) -> Result<Vec<String>> {
    let parquet_path = data_dir()
        .join("polarsignals")
        .join("1000000")
        .join("parquet")
        .join("stacktraces.parquet");

    if !parquet_path.exists() {
        info!("Generating PolarSignals parquet data...");
        PolarSignalsBenchmark::new(1)?.generate_base_data().await?;
    }

    fetch_parquet_string_path(parquet_path, field_path, max_rows).await
}

async fn fetch_parquet_string_path(
    parquet_path: PathBuf,
    field_path: &[&str],
    max_rows: usize,
) -> Result<Vec<String>> {
    anyhow::ensure!(!field_path.is_empty(), "field path must not be empty");

    let field_path: Vec<String> = field_path
        .iter()
        .map(|field| (*field).to_string())
        .collect();
    let display_path = field_path.join(".");
    info!(
        "Extracting '{display_path}' from Parquet (up to {max_rows} rows) at {}",
        parquet_path.display()
    );

    tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
        use parquet::arrow::ProjectionMask;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = std::fs::File::open(&parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        let root_field = field_path.first().context("field path must not be empty")?;
        let schema = builder.schema();
        let root_idx = schema
            .fields()
            .iter()
            .position(|field| field.name() == root_field)
            .with_context(|| {
                format!(
                    "Root column '{root_field}' not found in {}",
                    parquet_path.display()
                )
            })?;

        let projection = ProjectionMask::roots(builder.parquet_schema(), [root_idx]);
        let reader = builder.with_projection(projection).build()?;

        let nested_path: Vec<&str> = field_path.iter().skip(1).map(String::as_str).collect();
        let mut strings = Vec::with_capacity(initial_capacity(max_rows, 1_000_000));
        for batch in reader {
            let batch = batch?;
            let column = batch.column(0);
            extract_strings_from_arrow_path(column.as_ref(), &nested_path, &mut strings, max_rows)?;
            if reached_limit(strings.len(), max_rows) {
                break;
            }
        }
        Ok(strings)
    })
    .await?
}

fn extract_strings_from_arrow_path(
    array: &dyn ArrowArray,
    field_path: &[&str],
    out: &mut Vec<String>,
    max_rows: usize,
) -> Result<()> {
    if reached_limit(out.len(), max_rows) {
        return Ok(());
    }

    if field_path.is_empty() {
        return append_string_values(array, out, max_rows);
    }

    if let Some(struct_array) = array.as_any().downcast_ref::<StructArray>() {
        let field_name = field_path[0];
        let field_idx = struct_array
            .fields()
            .iter()
            .position(|field| field.name() == field_name)
            .with_context(|| format!("Struct field '{field_name}' not found"))?;
        return extract_strings_from_arrow_path(
            struct_array.column(field_idx).as_ref(),
            &field_path[1..],
            out,
            max_rows,
        );
    }

    if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
        return extract_strings_from_arrow_path(
            list_array.values().as_ref(),
            field_path,
            out,
            max_rows,
        );
    }

    if let Some(list_array) = array.as_any().downcast_ref::<LargeListArray>() {
        return extract_strings_from_arrow_path(
            list_array.values().as_ref(),
            field_path,
            out,
            max_rows,
        );
    }

    if let Some(list_array) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        return extract_strings_from_arrow_path(
            list_array.values().as_ref(),
            field_path,
            out,
            max_rows,
        );
    }

    anyhow::bail!(
        "Cannot descend through {:?} while looking for '{}'",
        array.data_type(),
        field_path.join(".")
    );
}

fn append_string_values(
    array: &dyn ArrowArray,
    out: &mut Vec<String>,
    max_rows: usize,
) -> Result<()> {
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        for i in 0..array.len() {
            if reached_limit(out.len(), max_rows) {
                break;
            }
            if !array.is_null(i) {
                push_string(array.value(i), out, max_rows);
            }
        }
        return Ok(());
    }

    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..array.len() {
            if reached_limit(out.len(), max_rows) {
                break;
            }
            if !array.is_null(i) {
                push_string(array.value(i), out, max_rows);
            }
        }
        return Ok(());
    }

    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        for i in 0..array.len() {
            if reached_limit(out.len(), max_rows) {
                break;
            }
            if !array.is_null(i) {
                push_string(array.value(i), out, max_rows);
            }
        }
        return Ok(());
    }

    if let Some(array) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
        for i in 0..array.len() {
            if reached_limit(out.len(), max_rows) {
                break;
            }
            if let Some(value) = dictionary_string_value_u32(array, i)? {
                push_string(&value, out, max_rows);
            }
        }
        return Ok(());
    }

    if let Some(array) = array.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
        for i in 0..array.len() {
            if reached_limit(out.len(), max_rows) {
                break;
            }
            if let Some(value) = dictionary_string_value_i32(array, i)? {
                push_string(&value, out, max_rows);
            }
        }
        return Ok(());
    }

    anyhow::bail!(
        "Expected string-like Arrow array, got {:?}",
        array.data_type()
    )
}

fn dictionary_string_value_u32(
    array: &DictionaryArray<UInt32Type>,
    index: usize,
) -> Result<Option<String>> {
    if array.is_null(index) {
        return Ok(None);
    }
    string_value_at(array.values().as_ref(), array.keys().value(index) as usize)
}

fn dictionary_string_value_i32(
    array: &DictionaryArray<Int32Type>,
    index: usize,
) -> Result<Option<String>> {
    if array.is_null(index) {
        return Ok(None);
    }
    let key = array.keys().value(index);
    anyhow::ensure!(key >= 0, "dictionary key must be non-negative");
    string_value_at(array.values().as_ref(), key as usize)
}

fn string_value_at(array: &dyn ArrowArray, index: usize) -> Result<Option<String>> {
    if array.is_null(index) {
        return Ok(None);
    }

    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Some(array.value(index).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(Some(array.value(index).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(Some(array.value(index).to_string()));
    }

    anyhow::bail!(
        "Expected dictionary values to be string-like, got {:?}",
        array.data_type()
    )
}

fn push_string(value: &str, out: &mut Vec<String>, max_rows: usize) {
    if value.is_empty() || reached_limit(out.len(), max_rows) {
        return;
    }

    let mut end = value.len().min(4096);
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    let value = &value[..end];
    if value.is_empty() {
        return;
    }

    if value.contains('\n') || value.contains('\r') {
        out.push(value.replace(['\n', '\r'], " "));
    } else {
        out.push(value.to_string());
    }
}

fn reached_limit(len: usize, max_rows: usize) -> bool {
    max_rows != 0 && len >= max_rows
}

fn initial_capacity(max_rows: usize, default_capacity: usize) -> usize {
    if max_rows == 0 {
        default_capacity
    } else {
        max_rows.min(default_capacity)
    }
}

/// Generate synthetic JSON lines for benchmarking.
fn generate_json_lines(count: usize) -> Vec<String> {
    use rand::prelude::*;

    let mut rng = StdRng::seed_from_u64(42);

    let names = [
        "alice", "bob", "charlie", "diana", "eve", "frank", "grace", "heidi", "ivan", "judy",
    ];
    let cities = [
        "new_york",
        "san_francisco",
        "london",
        "tokyo",
        "berlin",
        "paris",
        "sydney",
        "toronto",
        "mumbai",
        "seoul",
    ];
    let tags = [
        "enterprise",
        "startup",
        "developer",
        "admin",
        "analyst",
        "manager",
        "intern",
        "contractor",
    ];
    let departments = [
        "engineering",
        "marketing",
        "sales",
        "support",
        "hr",
        "finance",
        "legal",
        "ops",
    ];

    (0..count)
        .map(|i| {
            let name = names[rng.random_range(0..names.len())];
            let city = cities[rng.random_range(0..cities.len())];
            let age = rng.random_range(20..65);
            let tag = tags[rng.random_range(0..tags.len())];
            let dept = departments[rng.random_range(0..departments.len())];
            let score = rng.random_range(0..1000);
            format!(
                r#"{{"id":{i},"name":"{name}","city":"{city}","age":{age},"tag":"{tag}","dept":"{dept}","score":{score}}}"#
            )
        })
        .collect()
}

/// Generate TPC-H lineitem l_comment strings using tpchgen (SF1).
async fn fetch_tpch_lineitem(max_rows: usize) -> Result<Vec<String>> {
    info!("Generating TPC-H lineitem l_comment strings (SF1, up to {max_rows} rows)...");
    let strings = tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
        use arrow_array::Array as _;
        use tpchgen::generators::LineItemGenerator;

        let scale_factor = 1.0;
        let part = 1;
        let num_parts = 1;
        let batch_size = 8192 * 64;

        let generator = LineItemGenerator::new(scale_factor, part, num_parts);
        let iter = tpchgen_arrow::LineItemArrow::new(generator).with_batch_size(batch_size);

        let mut strings = Vec::with_capacity(initial_capacity(max_rows, 6_000_000));
        for batch in iter {
            let schema = batch.schema();
            let col_idx = schema
                .fields()
                .iter()
                .position(|f| f.name() == "l_comment")
                .context("l_comment column not found in lineitem")?;

            let column = batch.column(col_idx);
            // tpchgen-arrow may produce StringArray, LargeStringArray, or StringViewArray
            if let Some(arr) = column.as_any().downcast_ref::<StringArray>() {
                for i in 0..arr.len() {
                    if reached_limit(strings.len(), max_rows) {
                        break;
                    }
                    if !arr.is_null(i) {
                        let s = arr.value(i);
                        if !s.is_empty() {
                            strings.push(s.to_string());
                        }
                    }
                }
            } else if let Some(arr) = column.as_any().downcast_ref::<StringViewArray>() {
                for i in 0..arr.len() {
                    if reached_limit(strings.len(), max_rows) {
                        break;
                    }
                    if !arr.is_null(i) {
                        let s = arr.value(i);
                        if !s.is_empty() {
                            strings.push(s.to_string());
                        }
                    }
                }
            } else if let Some(arr) = column.as_any().downcast_ref::<LargeStringArray>() {
                for i in 0..arr.len() {
                    if reached_limit(strings.len(), max_rows) {
                        break;
                    }
                    if !arr.is_null(i) {
                        let s = arr.value(i);
                        if !s.is_empty() {
                            strings.push(s.to_string());
                        }
                    }
                }
            } else {
                anyhow::bail!(
                    "l_comment column has unexpected type: {:?}",
                    column.data_type()
                );
            }
            if reached_limit(strings.len(), max_rows) {
                break;
            }
        }
        Ok(strings)
    })
    .await??;

    info!("Generated {} l_comment strings", strings.len());
    Ok(strings)
}

/// Simple HTTP download with progress indication.
async fn download_file(url: &str, dest: &Path) -> Result<()> {
    use futures::StreamExt;
    use indicatif::ProgressBar;
    use indicatif::ProgressStyle;

    let client = reqwest::Client::builder()
        .read_timeout(Duration::from_secs(120))
        .timeout(Duration::from_secs(60 * 30))
        .build()?;

    let response = client
        .get(url)
        .send()
        .await?
        .error_for_status()
        .context("HTTP request failed")?;

    let total = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec})",
        )
        .expect("valid template"),
    );

    let mut file = File::create(dest).await?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }
    file.flush().await?;
    pb.finish_and_clear();

    Ok(())
}
