// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Functions for reading benchmark data from S3.

use std::sync::OnceLock;

use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use wasm_bindgen::JsValue;

use super::entry::BenchmarkEntry;
use crate::website::charts::BenchmarkResponse;
use crate::website::charts::OwnedBenchmarks;
use crate::website::charts::extract_summary;
use crate::website::charts::process_benchmarks;
use crate::website::charts::process_benchmarks_owned;
use crate::website::commit::CommitInfo;

/// Log to the browser console (WASM) or stderr (native).
#[cfg(target_arch = "wasm32")]
macro_rules! log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into());
    }
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! log {
    ($($t:tt)*) => {
        eprintln!($($t)*);
    }
}

/// Base URL for the S3 bucket containing benchmark data.
const S3_BASE_URL: &str = "https://vortex-benchmark-results-database-test.s3.amazonaws.com";

// ============================================================================
// Static caches for data (fetched/processed once, reused across calls)
// ============================================================================

/// Processed benchmark data ready for serialization.
pub struct ProcessedData {
    /// Sorted commits.
    pub commits: Vec<CommitInfo>,
    /// All benchmarks with owned strings.
    pub benchmarks: OwnedBenchmarks,
}

/// Global cache for processed data.
static PROCESSED_DATA: OnceLock<ProcessedData> = OnceLock::new();

/// Ensures data is loaded and processed, returning a reference to the cached data.
///
/// This function fetches data from S3 and processes it on the first call, then returns the
/// cached result on subsequent calls.
pub async fn ensure_data_loaded(
    session: &VortexSession,
    commits_key: &str,
    data_key: &str,
) -> VortexResult<&'static ProcessedData> {
    // If already cached, return immediately.
    if let Some(data) = PROCESSED_DATA.get() {
        log!("[ensure_data_loaded] Returning cached data");
        return Ok(data);
    }

    log!("[ensure_data_loaded] Fetching and processing data...");

    // Fetch from S3.
    let (data_array, commits_array) = futures::try_join!(
        read_s3_array(session, data_key),
        read_s3_array(session, commits_key)
    )?;

    // Parse arrays.
    let entries = BenchmarkEntry::vec_from_array(&data_array)?;
    let mut commits = CommitInfo::vec_from_array(&commits_array)?;
    commits.sort_unstable();

    log!(
        "[ensure_data_loaded] Parsed {} entries, {} commits",
        entries.len(),
        commits.len()
    );

    // Process into owned structures.
    let benchmarks = process_benchmarks_owned(&entries, &commits)?;

    let processed = ProcessedData {
        commits,
        benchmarks,
    };

    // Store in cache (ignore error if another thread beat us to it).
    drop(PROCESSED_DATA.set(processed));

    Ok(PROCESSED_DATA.get().expect("just set"))
}

/// Returns the benchmark summary (metadata only, fast serialization).
pub async fn get_benchmark_summary(
    session: &VortexSession,
    commits_key: &str,
    data_key: &str,
) -> VortexResult<String> {
    let data = ensure_data_loaded(session, commits_key, data_key).await?;

    log!("[get_benchmark_summary] Building summary...");
    let summary = extract_summary(&data.benchmarks, data.commits.clone());

    log!("[get_benchmark_summary] Serializing with serde_json...");
    let json = serde_json::to_string(&summary)
        .map_err(|e| vortex_err!("Failed to serialize summary: {}", e))?;

    log!(
        "[get_benchmark_summary] Done, JSON size: {} bytes",
        json.len()
    );
    Ok(json)
}

/// Returns chart data for a specific group and chart.
pub async fn get_chart_data(
    session: &VortexSession,
    commits_key: &str,
    data_key: &str,
    group: &str,
    chart: &str,
) -> VortexResult<String> {
    let data = ensure_data_loaded(session, commits_key, data_key).await?;

    log!(
        "[get_chart_data] Looking up group='{}', chart='{}'",
        group,
        chart
    );

    let group_data = data
        .benchmarks
        .get(group)
        .ok_or_else(|| vortex_err!("Group '{}' not found", group))?;

    let chart_data = group_data
        .charts
        .get(chart)
        .ok_or_else(|| vortex_err!("Chart '{}' not found in group '{}'", chart, group))?;

    log!("[get_chart_data] Serializing chart data...");
    let json = serde_json::to_string(chart_data)
        .map_err(|e| vortex_err!("Failed to serialize chart data: {}", e))?;

    log!("[get_chart_data] Done, JSON size: {} bytes", json.len());
    Ok(json)
}

// ============================================================================
// S3 reading functions
// ============================================================================

/// Reads a Vortex array from an S3 object.
///
/// This function downloads the Vortex file from S3 using HTTP (the bucket is public) and
/// returns the parsed array.
///
/// # Arguments
///
/// * `session` - The Vortex session for reading files.
/// * `key` - The S3 object key (e.g., "test/random_access.vortex").
///
/// # Errors
///
/// Returns an error if:
/// - The HTTP request fails.
/// - The file is not a valid Vortex file.
pub async fn read_s3_array(session: &VortexSession, key: &str) -> VortexResult<ArrayRef> {
    let url = format!("{}/{}", S3_BASE_URL, key);
    log!("[read_s3_array] Fetching {}...", url);

    let response = reqwest::get(&url)
        .await
        .map_err(|e| vortex_err!("Failed to fetch {}: {}", url, e))?;

    if !response.status().is_success() {
        vortex_bail!(
            "HTTP error fetching {}: {} {}",
            url,
            response.status().as_u16(),
            response.status().as_str()
        );
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| vortex_err!("Failed to read response body: {}", e))?;

    log!(
        "[read_s3_array] Downloaded {} bytes, parsing Vortex file...",
        bytes.len()
    );

    // Parse as Vortex file and read all data.
    // Note: We use `open_read_at` directly instead of `open_buffer` because `open_buffer` uses
    // `futures::executor::block_on` which requires `std::time` (not available in WASM).
    let buffer: vortex::buffer::ByteBuffer = bytes.to_vec().into();
    let file = session
        .open_options()
        .with_initial_read_size(0)
        .without_segment_cache()
        .open_read_at(buffer)
        .await?;

    let array = file.scan()?.into_array_stream()?.read_all().await?;
    log!("[read_s3_array] Parsed array with {} rows", array.len());

    Ok(array)
}

/// Reads benchmark entries from an S3 object containing a Vortex file.
///
/// This function downloads the Vortex file from S3 using HTTP (the bucket is public), parses the
/// columnar struct array, and converts it to a vector of row-wise [`BenchmarkEntry`] structs.
///
/// # Arguments
///
/// * `session` - The Vortex session for reading files.
/// * `key` - The S3 object key (e.g., "test/random_access.vortex").
///
/// # Errors
///
/// Returns an error if:
/// - The HTTP request fails.
/// - The file is not a valid Vortex file.
/// - The schema does not match the expected [`BenchmarkEntry`] schema.
pub async fn read_benchmark_entries(
    session: &VortexSession,
    key: &str,
) -> VortexResult<Vec<BenchmarkEntry>> {
    let array = read_s3_array(session, key).await?;
    BenchmarkEntry::vec_from_array(&array)
}

/// Fetches benchmark data and commit metadata from S3 and returns them as a JavaScript object.
///
/// The returned object has the structure:
/// ```javascript
/// {
///   benchmarks: { [group_name]: { charts: { [chart_name]: { aligned_series: { [series_name]: [...] } } } } },
///   commits: [{ timestamp, author: { name, email }, message, commit_id }, ...]
/// }
/// ```
///
/// # Arguments
///
/// * `session` - The Vortex session for reading files.
/// * `commits_key` - S3 key for the commits Vortex file.
/// * `data_key` - S3 key for the benchmark data Vortex file.
///
/// # Errors
///
/// Returns an error if:
/// - Either S3 fetch fails.
/// - The files are not valid Vortex files.
/// - The schemas don't match expected formats.
/// - Validation fails (empty names, no data points, mismatched lengths).
pub async fn get_benchmark_data(
    session: &VortexSession,
    commits_key: &str,
    data_key: &str,
) -> VortexResult<JsValue> {
    log!("[get_benchmark_data] Fetching data and commits in parallel...");

    let (data_array, commits_array) = futures::try_join!(
        read_s3_array(session, data_key),
        read_s3_array(session, commits_key)
    )?;

    log!("[get_benchmark_data] Parsing benchmark entries...");
    let data = BenchmarkEntry::vec_from_array(&data_array)?;
    log!(
        "[get_benchmark_data] Parsed {} benchmark entries",
        data.len()
    );

    log!("[get_benchmark_data] Parsing commit info...");
    let mut commits = CommitInfo::vec_from_array(&commits_array)?;
    log!(
        "[get_benchmark_data] Parsed {} commits, sorting...",
        commits.len()
    );
    commits.sort_unstable();

    log!("[get_benchmark_data] Processing benchmarks...");
    let benchmarks = process_benchmarks(&data, &commits)?;

    let response = BenchmarkResponse {
        benchmarks,
        commits,
    };

    log!("[get_benchmark_data] Serializing response to JS...");
    let js_value = serde_wasm_bindgen::to_value(&response)
        .map_err(|e| vortex_err!("Failed to serialize benchmark response: {e}"))?;

    log!("[get_benchmark_data] Done!");
    Ok(js_value)
}
