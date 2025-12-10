// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use bzip2::read::BzDecoder;
use reqwest::Client;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use tracing::info;

use crate::utils::file::idempotent;
use crate::utils::file::idempotent_async;

pub async fn download_data(fname: PathBuf, data_url: &str) -> Result<PathBuf> {
    idempotent_async(&fname, async |path| {
        info!(
            "Downloading {} from {}",
            fname.to_str().context("Failed to convert path to string")?,
            data_url
        );
        let mut file = TokioFile::create(path)
            .await
            .context("Failed to create file")?;
        let mut response = Client::builder()
            .read_timeout(Duration::from_secs(60))
            .timeout(Duration::from_secs(60 * 15))
            .build()
            .context("Failed to build HTTP client")?
            .get(data_url)
            .send()
            .await
            .context("Failed to send HTTP request")?
            .error_for_status()
            .context("HTTP request returned error status")?;

        while let Some(chunk) = response
            .chunk()
            .await
            .context("Failed to read response chunk")?
        {
            AsyncWriteExt::write_all(&mut file, &chunk)
                .await
                .context("Failed to write to file")?;
        }
        Ok(())
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
