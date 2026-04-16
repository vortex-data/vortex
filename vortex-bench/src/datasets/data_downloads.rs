// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use bzip2::read::BzDecoder;
use futures::StreamExt;
use futures::stream;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use parking_lot::RwLock;
use reqwest::Client;
use reqwest::Response;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use tracing::info;
use tracing::warn;

use crate::utils::file::idempotent;
use crate::utils::file::idempotent_async;

/// Default concurrency limit for bulk downloads. Keeps us polite to the upstream while still
/// saturating a typical 10 Gb link on a parquet-per-shard benchmark.
pub const DEFAULT_DOWNLOAD_CONCURRENCY: usize = 16;

/// Shared HTTP client used by every dataset download.
///
/// Reusing a single client gives us connection pooling, DNS caching, and consistent timeouts
/// across all callers. Each benchmark used to build its own `reqwest::Client` on every download,
/// which both wasted TLS handshakes and made it hard to reason about total in-flight concurrency.
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .read_timeout(Duration::from_secs(60))
        .timeout(Duration::from_secs(60 * 15))
        .build()
        .expect("failed to build shared benchmark HTTP client")
});

/// Access the shared HTTP client. Exposed for callers that need custom request shapes
/// (e.g. streaming VCF parsing) while still benefitting from pooled connections.
pub fn http_client() -> &'static Client {
    &HTTP_CLIENT
}

async fn retry_get<F: Future<Output = Result<Response>>, R: Fn() -> F>(
    make_req: R,
    tmp_path: PathBuf,
) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 3;
    let mut last_err: Option<Error> = None;
    let progress = RwLock::new(None::<ProgressBar>);

    let retry = async || -> Result<()> {
        let mut file = TokioFile::create(tmp_path)
            .await
            .context("Failed to create file")?;
        let response = make_req()
            .await
            .context("Failed to send HTTP request")?
            .error_for_status()
            .context("HTTP request returned error status")?;

        *progress.write() = response.content_length().map(|total| {
            let progress = ProgressBar::new(total);
            progress.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec})",
                )
                    .expect("valid template"),
            );
            progress
        });

        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            AsyncWriteExt::write_all(&mut file, &chunk)
                .await
                .context("Failed to write to file")?;
            if let Some(p) = progress.write().as_mut() {
                p.inc(chunk.len() as u64)
            }
        }

        AsyncWriteExt::flush(&mut file).await?;
        Ok(())
    };

    for attempt in 0..MAX_ATTEMPTS {
        let outcome = retry.clone()().await;

        match outcome {
            Ok(_) => {
                if let Some(p) = progress.write().take() {
                    p.finish_and_clear()
                }
                return Ok(());
            }
            Err(e) => {
                if let Some(p) = progress.write().take() {
                    p.abandon()
                }
                last_err = Some(e)
            }
        }
        let backoff = Duration::from_secs(1u64 << attempt);
        warn!(
            "download attempt {} failed; retrying in {:?}",
            attempt + 1,
            backoff
        );
        tokio::time::sleep(backoff).await;
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("retry_get exhausted with no recorded error")))
}

/// Idempotently download a single URL to `fname`.
///
/// Uses the shared HTTP client, a 3-attempt exponential backoff retry loop, and an `indicatif`
/// progress bar. If `fname` already exists, the download is skipped.
#[tracing::instrument(skip_all, fields(url = %data_url.as_ref(), path = %fname.display()))]
pub async fn download_data(fname: PathBuf, data_url: impl AsRef<str>) -> Result<PathBuf> {
    let client = http_client();

    idempotent_async(&fname, async |path| {
        let url = data_url.as_ref();
        info!(
            "Downloading {} from {}",
            fname.to_str().context("Failed to convert path to string")?,
            url
        );
        retry_get(
            async || {
                let res = client.get(url).send().await?;
                Ok(res)
            },
            path,
        )
        .await
    })
    .await
}

/// Idempotently download many `(path, url)` pairs with bounded parallelism.
///
/// This is the preferred way to fetch multi-shard datasets (ClickBench partitioned, vector
/// dataset train shards, Public BI tables, etc.) because it:
///
/// - caps in-flight HTTP requests at `max_concurrency` so we don't overwhelm the upstream
///   or our own network stack,
/// - reuses the shared HTTP client across every shard,
/// - short-circuits on the first error (the remaining in-flight downloads are dropped when
///   the returned future is dropped),
/// - returns the resolved on-disk paths in the same order they were submitted.
///
/// Pass `0` as `max_concurrency` to use [`DEFAULT_DOWNLOAD_CONCURRENCY`].
#[tracing::instrument(skip_all, fields(count = tracing::field::Empty, max_concurrency))]
pub async fn download_many<I>(downloads: I, max_concurrency: usize) -> Result<Vec<PathBuf>>
where
    I: IntoIterator,
    I::Item: IntoDownload,
{
    let downloads: Vec<(PathBuf, String)> = downloads
        .into_iter()
        .map(IntoDownload::into_download)
        .collect();
    tracing::Span::current().record("count", downloads.len());

    let concurrency = if max_concurrency == 0 {
        DEFAULT_DOWNLOAD_CONCURRENCY
    } else {
        max_concurrency
    };

    let results: Vec<Result<PathBuf>> = stream::iter(downloads)
        .map(|(path, url)| async move { download_data(path, url).await })
        .buffered(concurrency)
        .collect()
        .await;

    results.into_iter().collect()
}

/// Anything that can be described as a `(target_path, url)` pair accepted by [`download_many`].
pub trait IntoDownload {
    fn into_download(self) -> (PathBuf, String);
}

impl IntoDownload for (PathBuf, String) {
    fn into_download(self) -> (PathBuf, String) {
        self
    }
}

impl IntoDownload for (PathBuf, &str) {
    fn into_download(self) -> (PathBuf, String) {
        (self.0, self.1.to_owned())
    }
}

#[tracing::instrument(skip_all, fields(input = %input_path.display(), output = %output_path.display()))]
pub fn decompress_bz2(input_path: PathBuf, output_path: PathBuf) -> Result<PathBuf> {
    idempotent(&output_path, |path| {
        info!(
            "Decompressing bzip from {} to {}",
            input_path
                .to_str()
                .context("Failed to convert input path to string")?,
            output_path
                .to_str()
                .context("Failed to convert output path to string")?
        );
        let input_file = File::open(&input_path)
            .with_context(|| format!("Failed to open input file: {:?}", input_path))?;
        let mut decoder = BzDecoder::new(input_file);

        let mut buffer = Vec::new();
        decoder
            .read_to_end(&mut buffer)
            .context("Failed to decompress bzip2 data")?;

        let mut output_file = File::create(path)
            .with_context(|| format!("Failed to create output file: {:?}", path))?;
        output_file
            .write_all(&buffer)
            .context("Failed to write decompressed data")?;
        Ok(output_path.clone())
    })
}
