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

/// Reserves a private per-attempt directory under `target/` and removes it on drop.
///
/// Using a dedicated directory instead of a bare temp file prevents a queued blocking
/// `File::create` from recreating a leaked file after an async future is dropped, since the
/// missing parent directory makes the late create fail with `NotFound`. Note that this only holds
/// for writers that do not create missing parent directories (e.g. `object_store::local::LocalFileSystem`
/// creates parent directories on `NotFound` and could still recreate the directory if a write is
/// already queued on cancellation).
struct TempPathGuard {
    dir: PathBuf,
    path: PathBuf,
}

impl TempPathGuard {
    fn new() -> Result<Self> {
        let dir = workspace_root()
            .join("target")
            .join(format!("download_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).context("Failed to create temp directory")?;
        let path = dir.join("payload.file");
        Ok(Self { dir, path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempPathGuard {
    fn drop(&mut self) {
        if let Err(err) = fs::remove_dir_all(&self.dir)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %self.dir.display(),
                error = %err,
                "Failed to clean up temporary directory"
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
        let temp_path = TempPathGuard::new()?;
        f(temp_path.path())?;
        fs::rename(temp_path.path(), &data_path).context("Failed to rename temp file")?;
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
        let temp_path = TempPathGuard::new()?;
        f(temp_path.path().to_path_buf()).await?;
        fs::rename(temp_path.path(), &data_path).context("Failed to rename temp file")?;
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
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::task::Context;
    use std::task::Waker;

    use anyhow::Result;
    use anyhow::bail;
    use parking_lot::Mutex;
    use tempfile::TempDir;
    use tempfile::tempdir_in;

    use super::idempotent;
    use super::idempotent_async;
    use super::workspace_root;

    fn test_tempdir() -> Result<TempDir> {
        let target_dir = workspace_root().join("target");
        fs::create_dir_all(&target_dir)?;
        Ok(tempdir_in(target_dir)?)
    }

    #[test]
    fn idempotent_cleans_up_temp_file_when_generator_fails() -> Result<()> {
        let dir = test_tempdir()?;
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
        let dir = test_tempdir()?;
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
    fn idempotent_async_cleans_up_temp_file_on_error_and_retry_succeeds() -> Result<()> {
        let dir = test_tempdir()?;
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
                    Err::<(), _>(anyhow::anyhow!("async generator failed"))
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
    fn idempotent_async_cancels_before_async_file_create_completes() -> Result<()> {
        let dir = test_tempdir()?;
        let final_path = dir.path().join("cancelled_before_create.vortex");
        let observed_temp = Arc::new(Mutex::new(None::<PathBuf>));
        let observed_clone = Arc::clone(&observed_temp);

        let rt = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_all()
            .build()?;

        rt.block_on(async {
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();

            // Occupy the single blocking worker so tokio::fs::File::create stays queued.
            let blocker = tokio::task::spawn_blocking(move || {
                let _ = entered_tx.send(());
                let _ = release_rx.recv();
            });
            entered_rx.recv().expect("blocking worker started");

            {
                let mut fut = Box::pin(idempotent_async(&final_path, move |temp_path| {
                    let observed_clone = Arc::clone(&observed_clone);
                    async move {
                        *observed_clone.lock() = Some(temp_path.clone());
                        tokio::fs::File::create(&temp_path).await?;
                        Ok::<(), anyhow::Error>(())
                    }
                }));
                let waker = Waker::noop();
                let mut cx = Context::from_waker(waker);
                assert!(fut.as_mut().poll(&mut cx).is_pending());
                // Drop while File::create is still queued on the blocking pool.
            }

            // Unblock the blocking pool and wait for the queued File::create to run.
            let _ = release_tx.send(());
            blocker.await.expect("blocker finished");
            tokio::task::spawn_blocking(|| ())
                .await
                .expect("blocking queue drained");
        });

        let temp_path = observed_temp.lock().clone().expect("temp path recorded");
        let temp_dir = temp_path.parent().expect("temp path has parent dir");
        assert!(
            !temp_path.exists(),
            "temp file leaked after queued File::create ran: {}",
            temp_path.display()
        );
        assert!(
            !temp_dir.exists(),
            "temp attempt directory leaked: {}",
            temp_dir.display()
        );
        assert!(!final_path.exists());
        Ok(())
    }
}
