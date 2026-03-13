// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::manifest::Manifest;

/// Abstraction over local filesystem and S3 fixture stores.
pub trait FixtureStore: Send + Sync {
    /// Read a file's bytes. Returns None if not found.
    fn read(&self, key: &str) -> VortexResult<Option<Vec<u8>>>;

    /// Write bytes to a key in the store.
    fn write(&self, key: &str, data: &[u8]) -> VortexResult<()>;

    /// Copy a local file into the store.
    fn write_from_path(&self, key: &str, local_path: &Path) -> VortexResult<()>;

    /// Discover versions from the store (via versions.json or directory listing).
    fn list_versions(&self) -> VortexResult<Vec<String>>;

    /// Read and parse versions.json. Returns empty vec if not found.
    fn read_versions_json(&self) -> VortexResult<Vec<String>>;

    /// Write versions.json (with locking for S3).
    fn write_versions_json(&self, versions: &[String]) -> VortexResult<()>;

    /// Display name for user-facing output.
    fn display_name(&self) -> String;

    /// Fetch and parse a version's manifest.
    fn fetch_manifest(&self, version: &str) -> VortexResult<Manifest> {
        let key = format!("v{version}/manifest.json");
        let data = self
            .read(&key)?
            .ok_or_else(|| vortex_err!("manifest not found for v{version}"))?;
        serde_json::from_slice(&data)
            .map_err(|e| vortex_err!("failed to parse manifest for v{version}: {e}"))
    }

    /// Fetch a fixture file's bytes.
    fn fetch_fixture(&self, version: &str, name: &str) -> VortexResult<Vec<u8>> {
        let key = format!("v{version}/{name}");
        self.read(&key)?
            .ok_or_else(|| vortex_err!("fixture {name} not found for v{version}"))
    }
}

/// Local filesystem store.
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl FixtureStore for LocalStore {
    fn read(&self, key: &str) -> VortexResult<Option<Vec<u8>>> {
        let path = self.root.join(key);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read(&path)
            .map_err(|e| vortex_err!("failed to read {}: {e}", path.display()))?;
        Ok(Some(data))
    }

    fn write(&self, key: &str, data: &[u8]) -> VortexResult<()> {
        let path = self.root.join(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| vortex_err!("failed to create dir {}: {e}", parent.display()))?;
        }
        std::fs::write(&path, data)
            .map_err(|e| vortex_err!("failed to write {}: {e}", path.display()))
    }

    fn write_from_path(&self, key: &str, local_path: &Path) -> VortexResult<()> {
        let dest = self.root.join(key);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| vortex_err!("failed to create dir {}: {e}", parent.display()))?;
        }
        std::fs::copy(local_path, &dest).map_err(|e| {
            vortex_err!(
                "failed to copy {} -> {}: {e}",
                local_path.display(),
                dest.display()
            )
        })?;
        Ok(())
    }

    fn list_versions(&self) -> VortexResult<Vec<String>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut versions = Vec::new();
        for entry in std::fs::read_dir(&self.root)
            .map_err(|e| vortex_err!("failed to read dir {}: {e}", self.root.display()))?
        {
            let entry = entry.map_err(|e| vortex_err!("failed to read dir entry: {e}"))?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(version) = name.strip_prefix('v')
                && entry.path().join("manifest.json").exists()
            {
                versions.push(version.to_string());
            }
        }
        versions.sort_by_key(|a| version_sort_key(a));
        Ok(versions)
    }

    fn read_versions_json(&self) -> VortexResult<Vec<String>> {
        match self.read("versions.json")? {
            Some(data) => serde_json::from_slice(&data)
                .map_err(|e| vortex_err!("failed to parse versions.json: {e}")),
            None => Ok(Vec::new()),
        }
    }

    fn write_versions_json(&self, versions: &[String]) -> VortexResult<()> {
        let json = serde_json::to_string_pretty(versions)
            .map_err(|e| vortex_err!("failed to serialize versions.json: {e}"))?;
        self.write("versions.json", format!("{json}\n").as_bytes())
    }

    fn display_name(&self) -> String {
        self.root.display().to_string()
    }
}

/// S3 fixture store. Reads via public HTTPS, writes via `aws` CLI.
pub struct S3Store {
    bucket: String,
    https_base: String,
}

impl S3Store {
    pub fn new(bucket: String) -> Self {
        let https_base = format!("https://{bucket}.s3.amazonaws.com");
        Self { bucket, https_base }
    }
}

impl FixtureStore for S3Store {
    fn read(&self, key: &str) -> VortexResult<Option<Vec<u8>>> {
        let url = format!("{}/{key}", self.https_base);
        http_get_bytes_optional(&url)
    }

    fn write(&self, key: &str, data: &[u8]) -> VortexResult<()> {
        // Write data to a temp file, then upload via aws CLI.
        let tmp = tempfile::NamedTempFile::new()
            .map_err(|e| vortex_err!("failed to create temp file: {e}"))?;
        std::fs::write(tmp.path(), data)
            .map_err(|e| vortex_err!("failed to write temp file: {e}"))?;
        self.write_from_path(key, tmp.path())
    }

    fn write_from_path(&self, key: &str, local_path: &Path) -> VortexResult<()> {
        let result = Command::new("aws")
            .args([
                "s3",
                "cp",
                &local_path.display().to_string(),
                &format!("s3://{}/{key}", self.bucket),
            ])
            .status()
            .map_err(|e| vortex_err!("failed to run aws s3 cp: {e}"))?;
        if !result.success() {
            vortex_bail!("aws s3 cp failed for key {key}");
        }
        Ok(())
    }

    fn list_versions(&self) -> VortexResult<Vec<String>> {
        self.read_versions_json()
    }

    fn read_versions_json(&self) -> VortexResult<Vec<String>> {
        match self.read("versions.json")? {
            Some(data) => serde_json::from_slice(&data)
                .map_err(|e| vortex_err!("failed to parse versions.json: {e}")),
            None => Ok(Vec::new()),
        }
    }

    fn write_versions_json(&self, versions: &[String]) -> VortexResult<()> {
        let json = serde_json::to_string_pretty(versions)
            .map_err(|e| vortex_err!("failed to serialize versions.json: {e}"))?;
        let tmp = tempfile::NamedTempFile::new()
            .map_err(|e| vortex_err!("failed to create temp file: {e}"))?;
        std::fs::write(tmp.path(), format!("{json}\n"))
            .map_err(|e| vortex_err!("failed to write temp file: {e}"))?;

        // Optimistic locking with ETag-based compare-and-swap.
        let max_retries = 5;
        for attempt in 1..=max_retries {
            let etag = head_etag(&self.bucket, "versions.json");
            if put_object_with_etag(&self.bucket, "versions.json", tmp.path(), etag.as_deref())? {
                eprintln!("  versions.json uploaded.");
                return Ok(());
            }
            if attempt == max_retries {
                break;
            }
            let delay = Duration::from_secs(u64::min(1 << attempt, 30));
            eprintln!(
                "  versions.json upload failed (attempt {attempt}/{max_retries}), retrying in {}s...",
                delay.as_secs()
            );
            thread::sleep(delay);
        }
        vortex_bail!("versions.json upload failed after {max_retries} attempts");
    }

    fn display_name(&self) -> String {
        format!("s3://{}", self.bucket)
    }
}

/// Parse a `--store` argument into a boxed `FixtureStore`.
pub fn parse_store(spec: &str) -> VortexResult<Box<dyn FixtureStore>> {
    if let Some(bucket) = spec.strip_prefix("s3://") {
        Ok(Box::new(S3Store::new(bucket.to_string())))
    } else {
        Ok(Box::new(LocalStore::new(PathBuf::from(spec))))
    }
}

/// Default store spec.
pub const DEFAULT_STORE: &str = "s3://vortex-compat-fixtures";

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn http_get_bytes_optional(url: &str) -> VortexResult<Option<Vec<u8>>> {
    let response = reqwest::blocking::get(url)
        .map_err(|e| vortex_err!("HTTP request failed for {url}: {e}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Ok(None);
    }
    if !response.status().is_success() {
        vortex_bail!("HTTP {} fetching {url}", response.status());
    }
    let bytes = response
        .bytes()
        .map_err(|e| vortex_err!("failed to read response body from {url}: {e}"))?;
    Ok(Some(bytes.to_vec()))
}

// ---------------------------------------------------------------------------
// S3 helpers (aws CLI)
// ---------------------------------------------------------------------------

fn head_etag(bucket: &str, key: &str) -> Option<String> {
    let result = Command::new("aws")
        .args([
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ETag",
            "--output",
            "text",
        ])
        .output()
        .ok()?;
    if !result.status.success() {
        return None;
    }
    let etag = String::from_utf8_lossy(&result.stdout).trim().to_string();
    if etag.is_empty() || etag == "null" {
        None
    } else {
        Some(etag)
    }
}

fn put_object_with_etag(
    bucket: &str,
    key: &str,
    body: &Path,
    if_match: Option<&str>,
) -> VortexResult<bool> {
    let mut cmd = Command::new("aws");
    cmd.args([
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        &body.display().to_string(),
    ]);
    if let Some(etag) = if_match {
        cmd.args(["--if-match", etag]);
    }
    let result = cmd
        .output()
        .map_err(|e| vortex_err!("failed to run aws s3api put-object: {e}"))?;
    Ok(result.status.success())
}

fn version_sort_key(v: &str) -> Vec<u64> {
    v.split('.').filter_map(|s| s.parse().ok()).collect()
}
