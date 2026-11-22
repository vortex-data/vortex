// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use get_dir::FileTarget;
use get_dir::GetDir;
use get_dir::Target;
use url::Url;

static DATA_GENERATION_FORBIDDEN: AtomicBool = AtomicBool::new(false);

/// Make [`idempotent`] and [`idempotent_async`] fail instead of creating a missing file.
///
/// Engine benchmark binaries call this so they only run against data prepared up front by
/// `data-gen` (`vx-bench prepare-data`), and never generate data while benchmarking.
pub fn forbid_data_generation() {
    DATA_GENERATION_FORBIDDEN.store(true, Ordering::Relaxed);
}

fn check_data_generation_allowed(data_path: &Path) -> Result<()> {
    if DATA_GENERATION_FORBIDDEN.load(Ordering::Relaxed) {
        bail!(
            "benchmark data is missing at {}, and this binary does not generate data. \
             Prepare it first with `vx-bench prepare-data` or `cargo run --bin data-gen`.",
            data_path.display()
        );
    }
    Ok(())
}

struct TempPathGuard {
    path: PathBuf,
    armed: bool,
}

impl TempPathGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for TempPathGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let result = match fs::symlink_metadata(&self.path) {
            Ok(meta) if meta.is_dir() => fs::remove_dir_all(&self.path),
            Ok(_) => fs::remove_file(&self.path),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
            Err(err) => Err(err),
        };
        if let Err(err) = result
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %self.path.display(),
                error = %err,
                "Failed to clean up temporary file"
            );
        }
    }
}

/// Creates a file if it doesn't already exist.
/// NB: Does NOT modify the given path to ensure that it resides in the data directory.
pub fn idempotent<T, P: IdempotentPath + ?Sized>(
    path: &P,
    f: impl FnOnce(&Path) -> Result<T>,
) -> Result<PathBuf> {
    let data_path = path.to_data_path();
    if !data_path.exists() {
        check_data_generation_allowed(&data_path)?;
        // Ensure parent directory exists
        if let Some(parent) = data_path.parent() {
            fs::create_dir_all(parent).context("Failed to create parent directories")?;
        }
        let temp_path = TempPathGuard::new(temp_download_filepath());
        f(temp_path.path())?;
        fs::rename(temp_path.path(), &data_path).context("Failed to rename temp file")?;
        temp_path.disarm();
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
    if !data_path.exists() {
        check_data_generation_allowed(&data_path)?;
        // Ensure parent directory exists
        if let Some(parent) = data_path.parent() {
            fs::create_dir_all(parent).context("Failed to create parent directories")?;
        }
        let temp_path = TempPathGuard::new(temp_download_filepath());
        f(temp_path.path().to_path_buf()).await?;
        fs::rename(temp_path.path(), &data_path).context("Failed to rename temp file")?;
        temp_path.disarm();
    }
    Ok(data_path)
}

pub trait IdempotentPath {
    fn to_data_path(&self) -> PathBuf;
}

pub fn bench_dir() -> PathBuf {
    workspace_root().join("vortex-bench")
}

pub fn data_dir() -> PathBuf {
    bench_dir().join("data")
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
                anyhow::anyhow!("Failed to create URL from directory path: {:?}", data_dir)
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
            tracing::debug!(
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;
    use std::future::Future;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::Context;
    use std::task::Poll;
    use std::task::Waker;

    use anyhow::Result;
    use anyhow::bail;
    use parking_lot::Mutex;
    use tempfile::tempdir;

    use super::idempotent;
    use super::idempotent_async;

    #[test]
    fn idempotent_cleans_up_temp_file_when_generator_fails() -> Result<()> {
        let dir = tempdir()?;
        let final_path = dir.path().join("output.parquet");
        let observed_temp: Rc<RefCell<Option<PathBuf>>> = Rc::new(RefCell::new(None));
        let observed_clone = Rc::clone(&observed_temp);

        let err = idempotent(&final_path, |temp_path| -> Result<()> {
            assert!(!temp_path.exists());
            *observed_clone.borrow_mut() = Some(temp_path.to_path_buf());
            fs::write(temp_path, b"partial-output")?;
            bail!("generator failed");
        })
        .unwrap_err();

        assert!(err.to_string().contains("generator failed"));
        let temp_path = observed_temp.borrow().clone().expect("temp path recorded");
        assert!(
            !temp_path.exists(),
            "temp file leaked: {}",
            temp_path.display()
        );
        assert!(!final_path.exists());
        Ok(())
    }

    #[test]
    fn idempotent_cleans_up_temp_file_when_rename_fails() -> Result<()> {
        let dir = tempdir()?;
        let final_path = dir.path().join("output.parquet");
        let observed_temp: Rc<RefCell<Option<PathBuf>>> = Rc::new(RefCell::new(None));
        let observed_clone = Rc::clone(&observed_temp);

        let err = idempotent(&final_path, |temp_path| -> Result<()> {
            *observed_clone.borrow_mut() = Some(temp_path.to_path_buf());
            fs::write(temp_path, b"ready-to-rename")?;
            // Create a non-empty directory at final_path so fs::rename fails.
            fs::create_dir(&final_path)?;
            fs::write(final_path.join("blocker"), b"x")?;
            Ok(())
        })
        .unwrap_err();

        assert!(err.to_string().contains("Failed to rename temp file"));
        let temp_path = observed_temp.borrow().clone().expect("temp path recorded");
        assert!(
            !temp_path.exists(),
            "temp file leaked: {}",
            temp_path.display()
        );
        Ok(())
    }

    #[test]
    fn idempotent_skips_existing_output_without_invoking_generator() -> Result<()> {
        let dir = tempdir()?;
        let final_path = dir.path().join("existing.parquet");
        fs::write(&final_path, b"existing")?;

        let result = idempotent(&final_path, |_temp_path| -> Result<()> {
            panic!("generator should not run when output already exists");
        })?;

        assert_eq!(result, final_path);
        assert_eq!(fs::read(&final_path)?, b"existing");
        Ok(())
    }

    #[test]
    fn idempotent_async_cleans_up_temp_file_on_error_and_retry_succeeds() -> Result<()> {
        let dir = tempdir()?;
        let final_path = dir.path().join("async_output.vortex");
        let leaked_temp = Arc::new(Mutex::new(None::<PathBuf>));
        let leaked_clone = Arc::clone(&leaked_temp);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let err = rt
            .block_on(idempotent_async(&final_path, move |temp_path| {
                let leaked_clone = Arc::clone(&leaked_clone);
                async move {
                    assert!(!temp_path.exists());
                    *leaked_clone.lock() = Some(temp_path.clone());
                    fs::write(&temp_path, b"partial-async")?;
                    bail!("async generator failed");
                    #[allow(unreachable_code)]
                    Ok::<(), anyhow::Error>(())
                }
            }))
            .unwrap_err();

        assert!(err.to_string().contains("async generator failed"));
        let first_temp = leaked_temp.lock().clone().expect("temp path recorded");
        assert!(
            !first_temp.exists(),
            "first attempt temp file leaked: {}",
            first_temp.display()
        );
        assert!(!final_path.exists());

        // Retry succeeds and publishes the final output.
        let published = rt.block_on(idempotent_async(&final_path, |temp_path| async move {
            fs::write(&temp_path, b"good-async")?;
            Ok(())
        }))?;
        assert_eq!(published, final_path);
        assert_eq!(fs::read(&final_path)?, b"good-async");
        Ok(())
    }

    #[test]
    fn idempotent_async_cleans_up_temp_file_when_future_is_dropped() -> Result<()> {
        let dir = tempdir()?;
        let final_path = dir.path().join("cancelled_output.vortex");
        let observed_temp = Arc::new(Mutex::new(None::<PathBuf>));
        let observed_clone = Arc::clone(&observed_temp);

        struct WriteThenPending {
            temp_path: PathBuf,
            recorded: Arc<Mutex<Option<PathBuf>>>,
            written: bool,
        }

        impl Future for WriteThenPending {
            type Output = Result<()>;

            fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                if !self.written {
                    fs::write(&self.temp_path, b"in-flight").expect("write temp file");
                    *self.recorded.lock() = Some(self.temp_path.clone());
                    self.written = true;
                }
                Poll::Pending
            }
        }

        {
            let mut fut = Box::pin(idempotent_async(&final_path, move |temp_path| {
                WriteThenPending {
                    temp_path,
                    recorded: Arc::clone(&observed_clone),
                    written: false,
                }
            }));
            let waker = Waker::noop();
            let mut cx = Context::from_waker(waker);
            assert!(fut.as_mut().poll(&mut cx).is_pending());
            // Drop fut while pending.
        }

        let temp_path = observed_temp.lock().clone().expect("temp path recorded");
        assert!(
            !temp_path.exists(),
            "temp file leaked after future drop: {}",
            temp_path.display()
        );
        assert!(!final_path.exists());
        Ok(())
    }
}
