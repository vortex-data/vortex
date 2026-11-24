// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fetching remote source files, by content hash, into a local cache or the store.
//!
//! A remote file is identified by the SHA-256 of its bytes, never by its URL, so a mirror on S3
//! or Hugging Face and the upstream copy give the same chunk keys. Where the bytes live depends
//! on the source's [`Storage`]:
//!
//! - `reference`: `<cache>/<hh>/<sha256>`. The cache can be deleted; files are fetched again.
//! - `copy`: `<store>/raw/<hh>/<sha256>`, so the store holds everything it needs.

use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::bail;
use anyhow::ensure;
use futures::StreamExt;
use glob::Pattern;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use sha2::Digest;
use sha2::Sha256;

use crate::spec::Storage;

/// The environment variable that overrides the default cache directory.
pub const CACHE_ENV: &str = "VX_LAB_CACHE";

/// A parsed remote file location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteUrl {
    /// An `http://` or `https://` URL. Globs are not supported.
    Http(String),
    /// `s3://bucket/key`, where the key may contain globs.
    S3 {
        /// The bucket.
        bucket: String,
        /// The object key or key glob.
        key: String,
    },
    /// `file:///absolute/path`, where the path may contain globs. Used for local mirrors and tests.
    File(PathBuf),
}

impl RemoteUrl {
    /// Parses a URL from a spec.
    pub fn parse(url: &str) -> anyhow::Result<Self> {
        if url.starts_with("http://") || url.starts_with("https://") {
            ensure!(
                !url.contains('*'),
                "`{url}`: globs are only supported for s3:// and file:// urls"
            );
            Ok(Self::Http(url.to_string()))
        } else if let Some(rest) = url.strip_prefix("s3://") {
            let (bucket, key) = rest
                .split_once('/')
                .with_context(|| format!("`{url}` has no object key"))?;
            ensure!(
                !bucket.is_empty() && !key.is_empty(),
                "`{url}` needs a bucket and a key"
            );
            Ok(Self::S3 {
                bucket: bucket.to_string(),
                key: key.to_string(),
            })
        } else if let Some(path) = url.strip_prefix("file://") {
            ensure!(
                path.starts_with('/'),
                "`{url}`: file:// urls need an absolute path"
            );
            Ok(Self::File(PathBuf::from(path)))
        } else {
            bail!("`{url}`: unsupported scheme; use https://, s3:// or file://")
        }
    }

    fn is_glob(&self) -> bool {
        match self {
            Self::Http(_) => false,
            Self::S3 { key, .. } => key.contains('*'),
            Self::File(path) => path.to_string_lossy().contains('*'),
        }
    }
}

/// Expands a URL that may contain a glob into concrete URLs, sorted.
pub fn expand(url: &str) -> anyhow::Result<Vec<String>> {
    let parsed = RemoteUrl::parse(url)?;
    if !parsed.is_glob() {
        return Ok(vec![url.to_string()]);
    }
    let mut urls = match parsed {
        RemoteUrl::Http(_) => unreachable!("http urls are never globs"),
        RemoteUrl::File(path) => {
            let pattern = path.to_str().context("file:// path is not valid UTF-8")?;
            glob::glob(pattern)?
                .map(|entry| Ok(format!("file://{}", entry?.display())))
                .collect::<anyhow::Result<Vec<_>>>()?
        }
        RemoteUrl::S3 { bucket, key } => {
            let prefix = key.split('*').next().unwrap_or_default();
            let prefix = prefix.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
            let pattern = Pattern::new(&key)?;
            let store = s3(&bucket)?;
            let listed = runtime()?.block_on(async {
                let prefix = (!prefix.is_empty()).then(|| ObjectPath::from(prefix));
                store
                    .list(prefix.as_ref())
                    .map(|meta| meta.map(|m| m.location.to_string()))
                    .collect::<Vec<_>>()
                    .await
            });
            listed
                .into_iter()
                .filter_map(|location| match location {
                    Ok(location) if pattern.matches(&location) => {
                        Some(Ok(format!("s3://{bucket}/{location}")))
                    }
                    Ok(_) => None,
                    Err(error) => Some(Err(anyhow::Error::from(error))),
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        }
    };
    urls.sort();
    if urls.is_empty() {
        bail!("no files match `{url}`");
    }
    Ok(urls)
}

/// Where remote files are kept locally.
#[derive(Debug, Clone)]
pub struct RawData {
    /// The cache for `reference` sources.
    pub cache_dir: PathBuf,
    /// The store root, which holds `copy` sources under `raw/`. Without one, `copy` files stay in
    /// the cache until a run with a store copies them in.
    pub store: Option<PathBuf>,
}

impl RawData {
    /// The cache directory from `--cache-dir`, else `$VX_LAB_CACHE`, else `~/.cache/vx-lab`.
    pub fn new(cache_dir: Option<PathBuf>, store: Option<PathBuf>) -> Self {
        let cache_dir = cache_dir
            .or_else(|| std::env::var_os(CACHE_ENV).map(PathBuf::from))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/vx-lab")))
            .unwrap_or_else(|| std::env::temp_dir().join("vx-lab-cache"));
        Self { cache_dir, store }
    }

    fn cache_path(&self, sha256: &str) -> PathBuf {
        sharded(&self.cache_dir, sha256)
    }

    fn store_path(&self, sha256: &str) -> Option<PathBuf> {
        self.store
            .as_ref()
            .map(|store| sharded(&store.join("raw"), sha256))
    }

    /// Fetches `url` when its content hash is not known yet (at plan time), verifying it against
    /// `expected` if given. Returns the local path and the hash.
    pub fn fetch_new(
        &self,
        url: &str,
        expected: Option<&str>,
        storage: Storage,
    ) -> anyhow::Result<(PathBuf, String)> {
        if let Some(expected) = expected
            && let Some(path) = self.existing(expected, storage)?
        {
            return Ok((path, expected.to_string()));
        }
        let (temp, sha256) = download(url, &self.cache_dir)?;
        if let Some(expected) = expected
            && expected != sha256
        {
            fs::remove_file(&temp).ok();
            bail!("`{url}` has sha256 {sha256}, but the spec pins {expected}");
        }
        let path = self.place(&temp, &sha256, storage)?;
        Ok((path, sha256))
    }

    /// The local path of a file whose hash the plan recorded, fetching it from `url` if no local
    /// copy exists.
    pub fn locate(&self, url: &str, sha256: &str, storage: Storage) -> anyhow::Result<PathBuf> {
        match self.existing(sha256, storage)? {
            Some(path) => Ok(path),
            None => Ok(self.fetch_new(url, Some(sha256), storage)?.0),
        }
    }

    /// A local copy of `sha256`, moved into the store first for `copy` sources.
    fn existing(&self, sha256: &str, storage: Storage) -> anyhow::Result<Option<PathBuf>> {
        if storage == Storage::Copy
            && let Some(path) = self.store_path(sha256)
            && path.exists()
        {
            return Ok(Some(path));
        }
        let cached = self.cache_path(sha256);
        if !cached.exists() {
            return Ok(None);
        }
        if storage == Storage::Copy && self.store.is_some() {
            return self.place(&cached, sha256, storage).map(Some);
        }
        Ok(Some(cached))
    }

    /// Moves a downloaded file to where `storage` keeps it.
    fn place(&self, from: &Path, sha256: &str, storage: Storage) -> anyhow::Result<PathBuf> {
        let target = match (storage, self.store_path(sha256)) {
            (Storage::Copy, Some(path)) => path,
            _ => self.cache_path(sha256),
        };
        if target == from {
            return Ok(target);
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        if fs::rename(from, &target).is_err() {
            // Across filesystems: copy to a temp name beside the target, then rename.
            let partial = target.with_extension("partial");
            fs::copy(from, &partial)
                .with_context(|| format!("copying {} to {}", from.display(), partial.display()))?;
            fs::rename(&partial, &target)?;
            fs::remove_file(from).ok();
        }
        Ok(target)
    }
}

fn sharded(root: &Path, sha256: &str) -> PathBuf {
    root.join(sha256.get(..2).unwrap_or("00")).join(sha256)
}

/// Downloads `url` into a temp file under `dir`, hashing as it streams.
fn download(url: &str, dir: &Path) -> anyhow::Result<(PathBuf, String)> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let temp = tempfile::Builder::new()
        .prefix(".fetch-")
        .tempfile_in(dir)?
        .into_temp_path()
        .keep()?;
    let mut writer = HashingWriter::new(BufWriter::new(File::create(&temp)?));
    let result = (|| -> anyhow::Result<()> {
        match RemoteUrl::parse(url)? {
            RemoteUrl::Http(url) => {
                let mut response = reqwest::blocking::get(&url)
                    .and_then(|r| r.error_for_status())
                    .with_context(|| format!("fetching {url}"))?;
                response.copy_to(&mut writer)?;
            }
            RemoteUrl::File(path) => {
                let mut file =
                    File::open(&path).with_context(|| format!("opening {}", path.display()))?;
                std::io::copy(&mut file, &mut writer)?;
            }
            RemoteUrl::S3 { bucket, key } => {
                let store = s3(&bucket)?;
                runtime()?.block_on(async {
                    let mut stream = store
                        .get(&ObjectPath::from(key.as_str()))
                        .await?
                        .into_stream();
                    while let Some(bytes) = stream.next().await {
                        writer.write_all(&bytes?)?;
                    }
                    anyhow::Ok(())
                })?;
            }
        }
        writer.flush()?;
        Ok(())
    })();
    if let Err(error) = result {
        fs::remove_file(&temp).ok();
        return Err(error.context(format!("downloading {url}")));
    }
    Ok((temp, writer.finish()))
}

fn s3(bucket: &str) -> anyhow::Result<impl ObjectStore> {
    AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .build()
        .with_context(|| format!("configuring S3 for bucket `{bucket}` from the environment"))
}

fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finish(self) -> String {
        self.hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
