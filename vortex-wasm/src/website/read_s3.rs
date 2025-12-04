// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Functions for reading benchmark data from S3.

use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use wasm_bindgen::JsValue;

use super::entry::BenchmarkEntry;
use crate::website::charts::process_benchmarks;
use crate::website::commit::CommitInfo;

/// Base URL for the S3 bucket containing benchmark data.
const S3_BASE_URL: &str = "https://vortex-benchmark-results-database.s3.amazonaws.com";

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

    file.scan()?.into_array_stream()?.read_all().await
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

pub async fn get_benchmark_data(
    session: &VortexSession,
    commits_key: &str,
    data_key: &str,
) -> VortexResult<JsValue> {
    let (data_array, commits_array) = futures::try_join!(
        read_s3_array(session, data_key),
        read_s3_array(session, commits_key)
    )?;

    let data = BenchmarkEntry::vec_from_array(&data_array)?;
    let mut commits = CommitInfo::vec_from_array(&commits_array)?;
    commits.sort_unstable();

    let benchmarks = process_benchmarks(&data, &commits);

    let js_value = serde_wasm_bindgen::to_value(&benchmarks)
        .map_err(|e| vortex_err!("Something happened during serialization: {e}"))?;

    Ok(js_value)
}
