use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use bzip2::read::BzDecoder;
use log::info;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use vortex::error::VortexError;

use crate::utils::file_utils::{idempotent, idempotent_async};

pub async fn download_data(fname: PathBuf, data_url: &str) -> PathBuf {
    idempotent_async(&fname, async |path| {
        info!("Downloading {} from {}", fname.to_str().unwrap(), data_url);
        let mut file = TokioFile::create(path).await?;
        let mut response = reqwest::get(data_url).await?;
        if !response.status().is_success() {
            anyhow::bail!("Failed to download data from {}", data_url);
        }
        while let Some(chunk) = response.chunk().await? {
            AsyncWriteExt::write_all(&mut file, &chunk).await?;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await
    .unwrap()
}

pub fn decompress_bz2(input_path: PathBuf, output_path: PathBuf) -> PathBuf {
    idempotent(&output_path, |path| {
        info!(
            "Decompressing bzip from {} to {}",
            input_path.to_str().unwrap(),
            output_path.to_str().unwrap()
        );
        let input_file = File::open(input_path).unwrap();
        let mut decoder = BzDecoder::new(input_file);

        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).unwrap();

        let mut output_file = File::create(path).unwrap();
        output_file.write_all(&buffer).unwrap();
        Ok::<PathBuf, VortexError>(output_path.clone())
    })
    .unwrap()
}
