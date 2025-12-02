// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Atomic S3 update operations for Vortex files using the AWS CLI.
//!
//! This module provides functions to read a Vortex file from S3, apply a transformation, and write
//! the result back atomically using optimistic concurrency control via ETags.

use std::fs;
use std::io::Write;
use std::process::Command;
use std::time::Duration;

use tempfile::NamedTempFile;
use vortex::array::ArrayRef;
use vortex::array::builders::builder_with_capacity;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

use super::entry::BenchmarkEntry;

const MAX_RETRIES: u32 = 5;

/// Internal error type for retry control.
enum UpdateError {
    /// The ETag has changed since we read the object. The operation should be retried.
    EtagMismatch,
    /// A non-retryable error occurred.
    Other(String),
}

/// Gets the current ETag of an S3 object using the AWS CLI.
fn get_etag(bucket: &str, key: &str) -> Result<String, UpdateError> {
    let output = Command::new("aws")
        .args([
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ETag",
            "--output",
            "text",
        ])
        .output()
        .map_err(|e| UpdateError::Other(format!("Failed to run aws CLI: {}", e)))?;

    if !output.status.success() {
        return Err(UpdateError::Other(format!(
            "aws s3api head-object failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let etag = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if etag.is_empty() || etag == "null" {
        return Err(UpdateError::Other("Failed to retrieve ETag".to_string()));
    }

    Ok(etag)
}

/// Downloads an S3 object to a local file using the AWS CLI with ETag matching.
fn download_object(
    bucket: &str,
    key: &str,
    etag: &str,
    dest_path: &str,
) -> Result<(), UpdateError> {
    let output = Command::new("aws")
        .args([
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--if-match",
            etag,
            dest_path,
        ])
        .output()
        .map_err(|e| UpdateError::Other(format!("Failed to run aws CLI: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("PreconditionFailed") || stderr.contains("412") {
            return Err(UpdateError::EtagMismatch);
        }
        return Err(UpdateError::Other(format!(
            "aws s3api get-object failed: {}",
            stderr
        )));
    }

    Ok(())
}

/// Uploads a local file to S3 using the AWS CLI with ETag matching.
fn upload_object(bucket: &str, key: &str, etag: &str, src_path: &str) -> Result<(), UpdateError> {
    let output = Command::new("aws")
        .args([
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--if-match",
            etag,
            "--body",
            src_path,
        ])
        .output()
        .map_err(|e| UpdateError::Other(format!("Failed to run aws CLI: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("PreconditionFailed") || stderr.contains("412") {
            return Err(UpdateError::EtagMismatch);
        }
        return Err(UpdateError::Other(format!(
            "aws s3api put-object failed: {}",
            stderr
        )));
    }

    Ok(())
}

/// Updates a Vortex file stored in S3 atomically using optimistic concurrency control.
///
/// This function reads the existing file from S3, applies a transformation, and writes it back
/// using conditional puts with ETags. If another process modifies the file between read and write,
/// the operation is automatically retried.
///
/// # Arguments
///
/// * `session` - The Vortex session for reading and writing files.
/// * `bucket` - The S3 bucket name.
/// * `key` - The S3 object key.
/// * `update_fn` - A function that takes the file's array data and returns the updated array.
///   The returned array must have the same dtype as the input. This function may be called
///   multiple times if retries are needed.
///
/// # Errors
///
/// Returns an error if:
/// - The S3 object does not exist.
/// - The update function returns an error.
/// - The update function returns an array with a different dtype.
/// - The retry limit is reached without success.
/// - An S3 operation fails with a non-retryable error.
pub fn update_s3_object<F>(
    session: &VortexSession,
    bucket: &str,
    key: &str,
    mut update_fn: F,
) -> VortexResult<()>
where
    F: FnMut(ArrayRef) -> VortexResult<ArrayRef>,
{
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| vortex_err!("Failed to create tokio runtime: {}", e))?;

    for attempt in 0..MAX_RETRIES {
        match try_update_s3_object(session, bucket, key, &mut update_fn, &runtime) {
            Ok(()) => return Ok(()),
            Err(UpdateError::EtagMismatch) => {
                eprintln!("ETag mismatch on attempt {}. Retrying...", attempt + 1);
                std::thread::sleep(Duration::from_millis(100 * (1 << attempt)));
            }
            Err(UpdateError::Other(e)) => {
                vortex_bail!("S3 update failed: {}", e);
            }
        }
    }

    vortex_bail!("Failed to update S3 object after {} attempts", MAX_RETRIES)
}

/// Attempts a single update of an S3 object.
fn try_update_s3_object<F>(
    session: &VortexSession,
    bucket: &str,
    key: &str,
    update_fn: &mut F,
    runtime: &tokio::runtime::Runtime,
) -> Result<(), UpdateError>
where
    F: FnMut(ArrayRef) -> VortexResult<ArrayRef>,
{
    // Get current ETag.
    let etag = get_etag(bucket, key)?;

    // Download to temp file.
    let download_file = NamedTempFile::new()
        .map_err(|e| UpdateError::Other(format!("Failed to create temp file: {}", e)))?;
    let download_path = download_file.path().to_string_lossy().to_string();

    download_object(bucket, key, &etag, &download_path)?;

    // Read and parse.
    let existing_bytes = fs::read(&download_path)
        .map_err(|e| UpdateError::Other(format!("Failed to read downloaded file: {}", e)))?;

    let file = session
        .open_options()
        .open_buffer(existing_bytes)
        .map_err(|e| UpdateError::Other(format!("Failed to open Vortex file: {}", e)))?;

    let original_dtype = file.dtype().clone();

    let existing_array = runtime
        .block_on(async { file.scan()?.into_array_stream()?.read_all().await })
        .map_err(|e| UpdateError::Other(format!("Failed to read array: {}", e)))?;

    // Apply the user's update function.
    let updated_array = update_fn(existing_array)
        .map_err(|e| UpdateError::Other(format!("Update function failed: {}", e)))?;

    // Validate that the dtype matches.
    if updated_array.dtype() != &original_dtype {
        return Err(UpdateError::Other(format!(
            "Update function changed dtype from {} to {}. \
             The updated array must have the same dtype as the input file.",
            original_dtype,
            updated_array.dtype()
        )));
    }

    // Serialize updated array to Vortex file bytes.
    let mut buffer = Vec::new();
    runtime
        .block_on(async {
            session
                .write_options()
                .write(&mut buffer, updated_array.to_array_stream())
                .await
        })
        .map_err(|e| UpdateError::Other(format!("Failed to serialize array: {}", e)))?;

    // Write to temp file for upload.
    let mut upload_file = NamedTempFile::new()
        .map_err(|e| UpdateError::Other(format!("Failed to create temp file: {}", e)))?;
    upload_file
        .write_all(&buffer)
        .map_err(|e| UpdateError::Other(format!("Failed to write temp file: {}", e)))?;
    upload_file
        .flush()
        .map_err(|e| UpdateError::Other(format!("Failed to flush temp file: {}", e)))?;

    let upload_path = upload_file.path().to_string_lossy().to_string();

    // Upload with if-match.
    upload_object(bucket, key, &etag, &upload_path)?;

    Ok(())
}

/// Appends a single [`BenchmarkEntry`] to a Vortex file stored in S3.
///
/// This function uses [`update_s3_object`] with optimistic concurrency control to atomically
/// append the entry to the existing data. If concurrent modifications are detected, the operation
/// is automatically retried.
///
/// # Arguments
///
/// * `session` - The Vortex session for reading and writing files.
/// * `bucket` - The S3 bucket name.
/// * `key` - The S3 object key.
/// * `entry` - The benchmark entry to append.
pub fn append_benchmark_entry(
    session: &VortexSession,
    bucket: &str,
    key: &str,
    entry: &BenchmarkEntry,
) -> VortexResult<()> {
    let scalar = entry.into_scalar();

    update_s3_object(session, bucket, key, |existing_array| {
        let existing_len = existing_array.len();
        let dtype = existing_array.dtype().clone();

        let mut builder = builder_with_capacity(&dtype, existing_len + 1);
        builder.extend_from_array(&existing_array);
        builder.append_scalar(&scalar)?;

        Ok(builder.finish())
    })
}
