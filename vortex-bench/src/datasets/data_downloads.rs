// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use bzip2::read::BzDecoder;
use futures::StreamExt;
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

pub async fn download_data(fname: PathBuf, data_url: impl AsRef<str>) -> Result<PathBuf> {
    let client = Client::builder()
        .read_timeout(Duration::from_secs(60))
        .timeout(Duration::from_secs(60 * 15))
        .build()
        .context("Failed to build HTTP client")?;

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
