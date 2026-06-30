// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use get_dir::FileTarget;
use get_dir::GetDir;
use get_dir::Target;
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
            fs::create_dir_all(parent).context("Failed to create parent directories")?;
        }
        f(temp_path.as_path())?;
        fs::rename(temp_path, &data_path).context("Failed to rename temp file")?;
    }
    Ok(data_path)
}

pub async fn idempotent_async<P, FN, T, FUT>(path: P, f: FN) -> Result<PathBuf>
where
    P: IdempotentPath,
    FN: FnOnce(PathBuf) -> FUT,
    FUT: Future<Output = Result<T>>,
{
    let data_path = path.to_data_path();
    let temp_path = temp_download_filepath();
    if !data_path.exists() {
        // Ensure parent directory exists
        if let Some(parent) = data_path.parent() {
            fs::create_dir_all(parent).context("Failed to create parent directories")?;
        }
        f(temp_path.clone()).await?;
        fs::rename(temp_path, &data_path).context("Failed to rename temp file")?;
    }
    Ok(data_path)
}

pub trait IdempotentPath {
    fn to_data_path(&self) -> PathBuf;
}

pub fn data_dir() -> PathBuf {
    workspace_root().join("vortex-bench").join("data")
}

/// Find the workspace's root by looking for Cargo's lock file
pub fn workspace_root() -> PathBuf {
    GetDir::new()
        .target(Target::File(FileTarget::new("Cargo.lock")))
        .run_reverse()
        .expect("Can't find workspace root")
}

pub fn temp_download_filepath() -> PathBuf {
    workspace_root()
        .join("target")
        .join(format!("download_{}.file", uuid::Uuid::new_v4()))
}

impl IdempotentPath for &str {
    fn to_data_path(&self) -> PathBuf {
        data_dir().join(self)
    }
}

impl IdempotentPath for String {
    fn to_data_path(&self) -> PathBuf {
        self.as_str().to_data_path()
    }
}

impl IdempotentPath for PathBuf {
    fn to_data_path(&self) -> PathBuf {
        self.to_path_buf()
    }
}

impl IdempotentPath for &PathBuf {
    fn to_data_path(&self) -> PathBuf {
        self.to_path_buf()
    }
}

impl IdempotentPath for Path {
    fn to_data_path(&self) -> PathBuf {
        self.to_path_buf()
    }
}

impl IdempotentPath for &Path {
    fn to_data_path(&self) -> PathBuf {
        self.to_path_buf()
    }
}

/// Resolve the `--use-remote-data-dir` CLI option to a `Url` for a named dataset.
///
/// When `remote_data_dir` is `None`, returns a `file://` URL pointing at the dataset's local cache
/// directory (`<data_dir>/<local_subdir>/`).
///
/// When `remote_data_dir` is `Some(...)`, parses it as a remote URL (typically `s3://` or `gs://`).
/// The user must have pre-uploaded the expected data layout; a warning is logged if the URL does
/// not end in `/`, and an informational message describes the expected layout.
///
/// This helper replaces the boilerplate `create_data_url()` that used to be duplicated across every
/// benchmark that supports remote data directories (ClickBench, Fineweb, GhArchive, ...).
pub fn resolve_data_url(remote_data_dir: Option<&str>, local_subdir: &str) -> Result<Url> {
    match remote_data_dir {
        None => {
            let data_dir = data_dir().join(local_subdir);
            Url::from_directory_path(&data_dir).map_err(|_| {
                anyhow::anyhow!("Failed to create URL from directory path: {:?}", &data_dir)
            })
        }
        Some(remote_data_dir) => {
            if !remote_data_dir.ends_with('/') {
                tracing::warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash \
                        e.g. s3://vortex-bench-dev-eu/develop/12345/{}/",
                    local_subdir,
                );
            }
            tracing::info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                    "If it does not, you should kill this command, locally generate the files ",
                    "(by running without\n",
                    "--use-remote-data-dir) and upload data/{}/ to some remote location.",
                ),
                remote_data_dir,
                local_subdir,
            );
            Ok(Url::parse(remote_data_dir)?)
        }
    }
}

/// Convert a URL scheme to a storage type string
///
/// Maps URL schemes (s3, file) to storage type identifiers
/// for benchmark reporting.
///
/// # Returns
/// - A storage type string ("s3", "nvme")
/// - Or an error if the scheme is unknown
pub fn url_scheme_to_storage(url: &Url) -> Result<String> {
    use super::constants::STORAGE_GCS;
    use super::constants::STORAGE_NVME;
    use super::constants::STORAGE_S3;

    match url.scheme() {
        STORAGE_S3 => Ok(STORAGE_S3.to_owned()),
        "gs" => Ok(STORAGE_GCS.to_owned()),
        "file" => Ok(STORAGE_NVME.to_owned()),
        otherwise => {
            bail!("unknown URL scheme: {}", otherwise)
        }
    }
}
