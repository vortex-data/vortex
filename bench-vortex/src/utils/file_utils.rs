// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::create_dir_all;
use std::future::Future;
use std::path::{
    Path,
    PathBuf,
};

use anyhow::{
    Context,
    Result,
    bail,
};
use url::Url;

/// Creates a file if it doesn't already exist.
/// NB: Does NOT modify the given path to ensure that it resides in the data directory.
pub fn idempotent<T, P: IdempotentPath + ?Sized>(
    path: &P,
    f: impl FnOnce(&Path) -> Result<T>,
) -> Result<PathBuf> {
    let data_path = path.to_data_path();
    let temp_path = temp_download_filepath();
    if !data_path.exists() {
        // Ensure parent directory exists
        if let Some(parent) = data_path.parent() {
            create_dir_all(parent).context("Failed to create parent directories")?;
        }
        f(temp_path.as_path())?;
        std::fs::rename(temp_path, &data_path).context("Failed to rename temp file")?;
    }
    Ok(data_path)
}

pub async fn idempotent_async<T, F, P>(path: &P, f: impl FnOnce(PathBuf) -> F) -> Result<PathBuf>
where
    F: Future<Output = Result<T>>,
    P: IdempotentPath + ?Sized,
{
    let data_path = path.to_data_path();
    let temp_path = temp_download_filepath();
    if !data_path.exists() {
        // Ensure parent directory exists
        if let Some(parent) = data_path.parent() {
            create_dir_all(parent).context("Failed to create parent directories")?;
        }
        f(temp_path.clone()).await?;
        std::fs::rename(temp_path, &data_path).context("Failed to rename temp file")?;
    }
    Ok(data_path)
}

pub trait IdempotentPath {
    fn to_data_path(&self) -> PathBuf;
}

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

pub fn temp_download_filepath() -> PathBuf {
    data_dir().join(format!("download_{}.file", uuid::Uuid::new_v4()))
}

impl IdempotentPath for str {
    fn to_data_path(&self) -> PathBuf {
        data_dir().join(self)
    }
}

impl IdempotentPath for PathBuf {
    fn to_data_path(&self) -> PathBuf {
        self.to_path_buf()
    }
}

/// Convert a URL scheme to a storage type string
///
/// Maps URL schemes (s3, gcs, file) to storage type identifiers
/// for benchmark reporting.
///
/// # Returns
/// - A storage type string ("s3", "gcs", "nvme")
/// - Or an error if the scheme is unknown
pub fn url_scheme_to_storage(url: &Url) -> Result<String> {
    use super::constants::{
        STORAGE_GCS,
        STORAGE_NVME,
        STORAGE_S3,
    };

    match url.scheme() {
        STORAGE_S3 => Ok(STORAGE_S3.to_owned()),
        STORAGE_GCS => Ok(STORAGE_GCS.to_owned()),
        "file" => Ok(STORAGE_NVME.to_owned()),
        otherwise => {
            bail!("unknown URL scheme: {}", otherwise)
        }
    }
}
