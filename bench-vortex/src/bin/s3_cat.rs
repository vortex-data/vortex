// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Appends a local file to an S3 object using optimistic concurrency control via ETags.
//!
//! This binary is a Rust port of `scripts/cat-s3.sh` and handles concurrent appends to S3 objects
//! by using conditional requests with ETags. If the object has been modified by another process
//! between read and write, the operation is retried.

use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use aws_sdk_s3::Client;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::primitives::ByteStream;
use clap::Parser;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

const MAX_RETRIES: u32 = 5;
const RETRY_DELAY: Duration = Duration::from_millis(100);

#[derive(Parser, Debug)]
#[command(
    name = "s3_cat",
    about = "Append a local file to an S3 object with optimistic concurrency control"
)]
struct Args {
    /// S3 bucket name.
    bucket: String,

    /// S3 object key.
    key: String,

    /// Path to the local file to append.
    local_file: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = Client::new(&config);

    let local_content =
        std::fs::read(&args.local_file).context("Failed to read local file to append")?;

    let is_gzipped = args.key.ends_with(".gz");

    for attempt in 0..MAX_RETRIES {
        match try_append(&client, &args.bucket, &args.key, &local_content, is_gzipped).await {
            Ok(()) => {
                println!("File updated and uploaded successfully.");
                return Ok(());
            }
            Err(AppendError::EtagMismatch) => {
                println!("ETag mismatch on attempt {}. Retrying...", attempt + 1);
                tokio::time::sleep(RETRY_DELAY).await;
            }
            Err(AppendError::Other(e)) => {
                return Err(e);
            }
        }
    }

    bail!("Too many failures: {MAX_RETRIES}");
}

enum AppendError {
    EtagMismatch,
    Other(anyhow::Error),
}

impl From<anyhow::Error> for AppendError {
    fn from(e: anyhow::Error) -> Self {
        AppendError::Other(e)
    }
}

async fn try_append(
    client: &Client,
    bucket: &str,
    key: &str,
    local_content: &[u8],
    is_gzipped: bool,
) -> Result<(), AppendError> {
    // Get current ETag.
    let head = client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .context("Failed to get object metadata")?;

    let etag = head
        .e_tag()
        .context("No ETag returned from head_object")?
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
            return Err(AppendError::EtagMismatch);
        }
        Err(e) => {
            return Err(AppendError::Other(
                anyhow::Error::new(e).context("Failed to download object"),
            ));
        }
    };

    let existing_bytes = get_output
        .body
        .collect()
        .await
        .context("Failed to read object body")?
        .into_bytes();

    // Concatenate contents.
    let new_content = if is_gzipped {
        // Decompress existing content.
        let mut decoder = GzDecoder::new(&existing_bytes[..]);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("Failed to decompress existing content")?;

        // Append new content.
        decompressed.extend_from_slice(local_content);

        // Recompress.
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&decompressed)
            .context("Failed to compress concatenated content")?;
        encoder.finish().context("Failed to finish compression")?
    } else {
        let mut combined = existing_bytes.to_vec();
        combined.extend_from_slice(local_content);
        combined
    };

    // Upload with if-match.
    let put_result = client
        .put_object()
        .bucket(bucket)
        .key(key)
        .if_match(&etag)
        .body(ByteStream::from(new_content))
        .send()
        .await;

    match put_result {
        Ok(_) => Ok(()),
        Err(SdkError::ServiceError(err)) if err.err().code() == Some("PreconditionFailed") => {
            Err(AppendError::EtagMismatch)
        }
        Err(e) => Err(AppendError::Other(
            anyhow::Error::new(e).context("Failed to upload object"),
        )),
    }
}
