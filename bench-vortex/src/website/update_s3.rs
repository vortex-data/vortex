// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Atomic S3 update operations for Vortex files.
//!
//! This module provides functions to read a Vortex file from S3, apply a transformation, and write
//! the result back atomically using optimistic concurrency control via ETags.

use std::future::Future;
use std::time::Duration;

use aws_sdk_s3::Client;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::primitives::ByteStream;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

const INITIAL_DELAY: Duration = Duration::from_millis(100);
const MAX_DELAY: Duration = Duration::from_secs(60);

/// Internal error type for retry control.
enum UpdateError {
    /// The ETag has changed since we read the object. The operation should be retried.
    EtagMismatch,
    /// A non-retryable error occurred.
    Other(VortexError),
}

impl From<VortexError> for UpdateError {
    fn from(e: VortexError) -> Self {
        UpdateError::Other(e)
    }
}

/// Updates a Vortex file stored in S3 atomically using optimistic concurrency control.
///
/// This function reads the existing file from S3, applies a transformation, and writes it back
/// using conditional puts with ETags. If another process modifies the file between read and write,
/// the operation is automatically retried with exponential backoff.
///
/// # Arguments
///
/// * `client` - The AWS S3 client to use for operations.
/// * `session` - The Vortex session for reading and writing files.
/// * `bucket` - The S3 bucket name.
/// * `key` - The S3 object key.
/// * `update_fn` - An async function that takes the file's array data and returns the updated
///   array. The returned array must have the same dtype as the input. This function may be called
///   multiple times if retries are needed.
///
/// # Errors
///
/// Returns an error if:
/// - The S3 object does not exist.
/// - The update function returns an error.
/// - The update function returns an array with a different dtype.
/// - The retry delay reaches the maximum (60 seconds) without success.
/// - An S3 operation fails with a non-retryable error.
#[expect(clippy::use_debug)]
pub async fn update_s3_object<F, Fut>(
    client: &Client,
    session: &VortexSession,
    bucket: &str,
    key: &str,
    mut update_fn: F,
) -> VortexResult<()>
where
    F: FnMut(ArrayRef) -> Fut,
    Fut: Future<Output = VortexResult<ArrayRef>>,
{
    let mut delay = INITIAL_DELAY;
    let mut attempt = 0;

    loop {
        match try_update_s3_object(client, session, bucket, key, &mut update_fn).await {
            Ok(()) => return Ok(()),
            Err(UpdateError::EtagMismatch) => {
                attempt += 1;
                tracing::debug!(
                    "ETag mismatch on attempt {}. Retrying after {:?}...",
                    attempt,
                    delay
                );
            }
            Err(UpdateError::Other(e)) => {
                attempt += 1;
                eprintln!(
                    "Error on attempt {}: {}. Retrying after {:?}...",
                    attempt, e, delay
                );
            }
        }

        // If we've reached max delay, fail.
        if delay >= MAX_DELAY {
            vortex_bail!(
                "Failed to update S3 object after {} attempts (delay reached {:?})",
                attempt,
                MAX_DELAY
            );
        }

        tokio::time::sleep(delay).await;

        // Exponential backoff: double the delay, capped at MAX_DELAY.
        delay = (delay * 2).min(MAX_DELAY);
    }
}

/// Attempts a single update of an S3 object.
async fn try_update_s3_object<F, Fut>(
    client: &Client,
    session: &VortexSession,
    bucket: &str,
    key: &str,
    update_fn: &mut F,
) -> Result<(), UpdateError>
where
    F: FnMut(ArrayRef) -> Fut,
    Fut: Future<Output = VortexResult<ArrayRef>>,
{
    // Get current ETag.
    let head = client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| vortex_err!("Failed to get object metadata: {}", e))?;

    let etag = head
        .e_tag()
        .ok_or_else(|| vortex_err!("No ETag returned from head_object"))?
        .to_string();

    // Download with if-match.
    let get_result = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .if_match(&etag)
        .send()
        .await;

    let get_output = match get_result {
        Ok(output) => output,
        Err(SdkError::ServiceError(err)) if err.err().code() == Some("PreconditionFailed") => {
            return Err(UpdateError::EtagMismatch);
        }
        Err(e) => {
            return Err(UpdateError::Other(vortex_err!(
                "Failed to download object: {}",
                e
            )));
        }
    };

    let existing_bytes = get_output
        .body
        .collect()
        .await
        .map_err(|e| vortex_err!("Failed to read object body: {}", e))?
        .into_bytes();

    // Parse as Vortex file and read all data.
    let file = session.open_options().open_buffer(existing_bytes)?;
    let original_dtype = file.dtype().clone();
    let existing_array = file.scan()?.into_array_stream()?.read_all().await?;

    // Apply the user's update function.
    let updated_array = update_fn(existing_array).await?;

    // Validate that the dtype matches.
    if updated_array.dtype() != &original_dtype {
        return Err(UpdateError::Other(vortex_err!(
            "Update function changed dtype from {} to {}. \
             The updated array must have the same dtype as the input file.",
            original_dtype,
            updated_array.dtype()
        )));
    }

    // Serialize updated array to Vortex file bytes.
    let mut buffer = Vec::new();
    session
        .write_options()
        .write(&mut buffer, updated_array.to_array_stream())
        .await?;

    // Upload with if-match.
    let put_result = client
        .put_object()
        .bucket(bucket)
        .key(key)
        .if_match(&etag)
        .body(ByteStream::from(buffer))
        .send()
        .await;

    match put_result {
        Ok(_) => Ok(()),
        Err(SdkError::ServiceError(err)) if err.err().code() == Some("PreconditionFailed") => {
            Err(UpdateError::EtagMismatch)
        }
        Err(e) => Err(UpdateError::Other(vortex_err!(
            "Failed to upload object: {}",
            e
        ))),
    }
}
