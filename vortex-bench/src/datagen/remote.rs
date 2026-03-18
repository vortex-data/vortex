// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Remote storage operations for the dataset repository.
//!
//! Handles push, pull, checkout, delete, gc, and verify against a remote
//! that can be S3, GCS, or a local filesystem directory — anything the
//! `object_store` crate supports.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use object_store::ObjectStore;
use object_store::PutMode;
use object_store::PutOptions;
use object_store::PutPayload;
use object_store::path::Path as ObjPath;
use sha2::Digest;
use sha2::Sha256;
use tracing::info;
use tracing::warn;

use super::catalog::Catalog;
use super::catalog::DatasetEntry;
use super::dataset::DatasetDescriptor;
use super::manifest::Manifest;
use super::manifest::hash_file;

/// Resolve a URL or local path to an ObjectStore + base path.
pub fn resolve_store(url: &str) -> Result<(Arc<dyn ObjectStore>, ObjPath)> {
    // Try as a URL first.
    if let Ok(parsed) = url::Url::parse(url) {
        if parsed.scheme() == "file" {
            let fs_path = parsed
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("invalid file URL: {url}"))?;
            std::fs::create_dir_all(&fs_path)?;
            let store = object_store::local::LocalFileSystem::new_with_prefix(&fs_path)?;
            return Ok((Arc::new(store), ObjPath::default()));
        }

        // S3, GCS, etc.
        let normalized_env = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));
        let (store, path) = object_store::parse_url_opts(&parsed, normalized_env)
            .map_err(|e| anyhow::anyhow!("resolving object store for {url}: {e}"))?;
        return Ok((Arc::new(store), path));
    }

    // Treat as a local filesystem path.
    let fs_path = Path::new(url);
    std::fs::create_dir_all(fs_path)?;
    let store = object_store::local::LocalFileSystem::new_with_prefix(fs_path)?;
    Ok((Arc::new(store), ObjPath::default()))
}

/// Read catalog.json from remote. Returns empty catalog if not found.
pub async fn read_catalog(store: &dyn ObjectStore, base: &ObjPath) -> Result<Catalog> {
    let path = catalog_path(base);
    match store.get(&path).await {
        Ok(result) => {
            let bytes = result.bytes().await?;
            Catalog::from_json(&bytes)
        }
        Err(object_store::Error::NotFound { .. }) => Ok(Catalog::new()),
        Err(e) => Err(e.into()),
    }
}

/// Write catalog.json to remote. Uses conditional put when possible (CAS).
pub async fn write_catalog(
    store: &dyn ObjectStore,
    base: &ObjPath,
    catalog: &Catalog,
) -> Result<()> {
    let path = catalog_path(base);
    let bytes = catalog.to_json()?;
    let payload = PutPayload::from_bytes(bytes.into());

    // Try conditional put (CAS) first — falls back to unconditional if not supported.
    match store
        .put_opts(
            &path,
            payload.clone(),
            PutOptions {
                mode: PutMode::Overwrite,
                ..Default::default()
            },
        )
        .await
    {
        Ok(_) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Read a manifest from remote.
pub async fn read_manifest(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_path: &str,
) -> Result<Manifest> {
    let path = obj_path(base, &format!("{dataset_path}manifest.json"));
    let result = store.get(&path).await?;
    let bytes = result.bytes().await?;
    Manifest::from_json(&bytes)
}

/// Read a dataset descriptor from remote.
pub async fn read_dataset_descriptor(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_path: &str,
) -> Result<DatasetDescriptor> {
    let path = obj_path(base, &format!("{dataset_path}dataset.yaml"));
    let result = store.get(&path).await?;
    let bytes = result.bytes().await?;
    DatasetDescriptor::from_yaml(&bytes)
}

/// Check if a dataset already exists in the remote catalog.
///
/// Returns the existing entry if found. Use this before [`push`] to prompt
/// the user for confirmation.
pub async fn check_existing(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_name: &str,
) -> Result<Option<DatasetEntry>> {
    let catalog = read_catalog(store, base).await?;
    Ok(catalog.find(dataset_name).cloned())
}

/// Acquire an upload lock for a dataset.
///
/// Creates `{dataset_name}.uploading` at the repo root using `PutMode::Create`
/// (CAS — fails if the file already exists). This prevents two concurrent
/// pushes from racing on the same dataset name.
async fn acquire_upload_lock(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_name: &str,
) -> Result<ObjPath> {
    let lock_path = obj_path(base, &format!("{dataset_name}.uploading"));
    let payload = PutPayload::from_bytes(
        format!(
            "locked at {}",
            chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        )
        .into(),
    );

    match store
        .put_opts(
            &lock_path,
            payload,
            PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            },
        )
        .await
    {
        Ok(_) => {
            info!(lock = %lock_path, "acquired upload lock");
            Ok(lock_path)
        }
        Err(object_store::Error::AlreadyExists { path, .. }) => {
            bail!(
                "another upload is in progress for '{}' (lock file: {}). \
                 If this is stale, delete the lock file manually and retry.",
                dataset_name,
                path
            );
        }
        Err(object_store::Error::NotSupported { .. }) => {
            // Backend doesn't support conditional put — fall back to overwrite.
            warn!("object store does not support conditional put; upload lock is best-effort");
            store
                .put(
                    &lock_path,
                    PutPayload::from_bytes(b"locked (best-effort)"[..].into()),
                )
                .await?;
            Ok(lock_path)
        }
        Err(e) => Err(e.into()),
    }
}

/// Release the upload lock (delete the lock file).
async fn release_upload_lock(store: &dyn ObjectStore, lock_path: &ObjPath) {
    match store.delete(lock_path).await {
        Ok(()) => info!(lock = %lock_path, "released upload lock"),
        Err(e) => warn!(lock = %lock_path, error = %e, "failed to release upload lock"),
    }
}

/// Push a local dataset directory to remote.
///
/// 1. Validates dataset descriptor.
/// 2. Acquires upload lock (`{name}.uploading`, CAS).
/// 3. Scans and hashes data files, builds manifest.
/// 4. Uploads each file to `{name}-{rand}/`.
/// 5. Uploads dataset.yaml and manifest.json.
/// 6. Updates catalog.json.
/// 7. Releases upload lock.
///
/// If `force` is false and the dataset already exists in the catalog, returns
/// an error. The CLI should call [`check_existing`] first to prompt the user.
pub async fn push(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_dir: &Path,
    force: bool,
) -> Result<()> {
    let descriptor = DatasetDescriptor::from_file(dataset_dir.join("dataset.yaml"))?;
    let problems = descriptor.validate();
    if !problems.is_empty() {
        bail!(
            "dataset.yaml validation failed:\n{}",
            problems.join("\n  - ")
        );
    }

    // Check for existing dataset if not forced.
    if let (false, Some(existing)) = (force, check_existing(store, base, &descriptor.name).await?) {
        bail!(
            "dataset '{}' already exists at '{}'. Use --force to overwrite.",
            descriptor.name,
            existing.path
        );
    }

    let data_dir = dataset_dir.join("data");
    if !data_dir.exists() {
        bail!("data/ directory not found in {}", dataset_dir.display());
    }

    // Acquire upload lock.
    let lock_path = acquire_upload_lock(store, base, &descriptor.name).await?;

    // From here on, ensure we release the lock even on error.
    let result = push_inner(store, base, &descriptor, &data_dir).await;

    // Always release the lock.
    release_upload_lock(store, &lock_path).await;

    result
}

/// Inner push logic, separated so the lock can be released on any exit path.
async fn push_inner(
    store: &dyn ObjectStore,
    base: &ObjPath,
    descriptor: &DatasetDescriptor,
    data_dir: &Path,
) -> Result<()> {
    // Build manifest by scanning data/.
    let manifest = build_manifest_from_dir(&descriptor.name, data_dir)?;
    info!(
        name = descriptor.name,
        files = manifest.total_files(),
        total_bytes = manifest.total_size_bytes(),
        "built manifest"
    );

    // Generate unique path for this upload.
    let rand_suffix = &uuid::Uuid::new_v4().to_string()[..6];
    let dataset_path = format!("{}-{}/", descriptor.name, rand_suffix);

    // Upload data files.
    for (_format, _table, file) in manifest.iter_files() {
        let local_path = data_dir.join(&file.path);
        let remote_path = obj_path(base, &format!("{}{}", dataset_path, file.path));

        info!(path = %file.path, size = file.size_bytes, "uploading");
        let bytes = tokio::fs::read(&local_path)
            .await
            .with_context(|| format!("reading {}", local_path.display()))?;
        store
            .put(&remote_path, PutPayload::from_bytes(bytes.into()))
            .await
            .with_context(|| format!("uploading {}", file.path))?;
    }

    // Upload dataset.yaml.
    let descriptor_bytes = descriptor.to_yaml_bytes()?;
    store
        .put(
            &obj_path(base, &format!("{dataset_path}dataset.yaml")),
            PutPayload::from_bytes(descriptor_bytes.into()),
        )
        .await?;

    // Upload manifest.json.
    let manifest_bytes = manifest.to_json()?;
    let manifest_hash = sha256_hex(&manifest_bytes);
    store
        .put(
            &obj_path(base, &format!("{dataset_path}manifest.json")),
            PutPayload::from_bytes(manifest_bytes.into()),
        )
        .await?;

    // Update catalog.
    let mut catalog = read_catalog(store, base).await?;
    let old = catalog.upsert(DatasetEntry {
        name: descriptor.name.clone(),
        path: dataset_path.clone(),
        manifest_hash,
    });
    if let Some(old) = &old {
        info!(
            old_path = old.path,
            new_path = dataset_path,
            "replacing existing dataset"
        );
    }
    write_catalog(store, base, &catalog).await?;

    info!(name = descriptor.name, path = dataset_path, "push complete");
    Ok(())
}

/// Pull catalog + all manifests + dataset descriptors from remote to local mirror.
pub async fn pull(store: &dyn ObjectStore, base: &ObjPath, local_root: &Path) -> Result<()> {
    std::fs::create_dir_all(local_root)?;

    // Fetch catalog.
    let catalog = read_catalog(store, base).await?;
    let catalog_bytes = catalog.to_json()?;
    std::fs::write(local_root.join("catalog.json"), &catalog_bytes)?;
    info!(datasets = catalog.datasets.len(), "pulled catalog");

    // Fetch each dataset's manifest and descriptor.
    for entry in &catalog.datasets {
        let dataset_dir = local_root.join(&entry.path);
        std::fs::create_dir_all(&dataset_dir)?;

        // Check if we already have this manifest version.
        let manifest_path = dataset_dir.join("manifest.json");
        if manifest_path.exists() {
            let existing_bytes = std::fs::read(&manifest_path)?;
            let existing_hash = sha256_hex(&existing_bytes);
            if existing_hash == entry.manifest_hash {
                info!(name = entry.name, "manifest up to date, skipping");
                continue;
            }
        }

        // Fetch manifest.
        match read_manifest(store, base, &entry.path).await {
            Ok(manifest) => {
                let bytes = manifest.to_json()?;
                std::fs::write(&manifest_path, &bytes)?;
                info!(name = entry.name, "pulled manifest");
            }
            Err(e) => {
                warn!(name = entry.name, error = %e, "failed to pull manifest");
            }
        }

        // Fetch descriptor.
        match read_dataset_descriptor(store, base, &entry.path).await {
            Ok(desc) => {
                desc.write_to_file(dataset_dir.join("dataset.yaml"))?;
                info!(name = entry.name, "pulled dataset.yaml");
            }
            Err(e) => {
                warn!(name = entry.name, error = %e, "failed to pull dataset.yaml");
            }
        }
    }

    Ok(())
}

/// Checkout data files for a dataset from remote to local mirror.
/// Skips files that already exist locally with matching hash.
pub async fn checkout(
    store: &dyn ObjectStore,
    base: &ObjPath,
    local_root: &Path,
    dataset_name: &str,
) -> Result<()> {
    // Read catalog to find the dataset path.
    let catalog_path = local_root.join("catalog.json");
    if !catalog_path.exists() {
        bail!("no local catalog found — run `pull` first");
    }
    let catalog_bytes = std::fs::read(&catalog_path)?;
    let catalog = Catalog::from_json(&catalog_bytes)?;

    let entry = catalog
        .find(dataset_name)
        .ok_or_else(|| anyhow::anyhow!("dataset '{}' not found in catalog", dataset_name))?;

    // Read manifest.
    let dataset_dir = local_root.join(&entry.path);
    let manifest_path = dataset_dir.join("manifest.json");
    if !manifest_path.exists() {
        bail!(
            "no local manifest for '{}' — run `pull` first",
            dataset_name
        );
    }
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest = Manifest::from_json(&manifest_bytes)?;

    // Download each file, skipping if hash matches.
    let data_dir = dataset_dir.join("data");
    for (_format, _table, file) in manifest.iter_files() {
        let local_path = data_dir.join(&file.path);
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Skip if already exists with correct hash.
        if local_path.exists() {
            let (existing_hash, _) = hash_file(&local_path)?;
            if existing_hash == file.sha256 {
                info!(path = %file.path, "already cached, skipping");
                continue;
            }
        }

        let remote_path = obj_path(base, &format!("{}{}", entry.path, file.path));
        info!(path = %file.path, size = file.size_bytes, "downloading");
        let result = store.get(&remote_path).await?;
        let bytes = result.bytes().await?;

        // Verify hash.
        let actual_hash = sha256_hex(&bytes);
        if actual_hash != file.sha256 {
            bail!(
                "hash mismatch for {}: expected {}, got {}",
                file.path,
                file.sha256,
                actual_hash
            );
        }

        tokio::fs::write(&local_path, &bytes).await?;
    }

    info!(name = dataset_name, "checkout complete");
    Ok(())
}

/// Delete a dataset from the catalog. Optionally removes S3 files.
pub async fn delete(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_name: &str,
    delete_files: bool,
) -> Result<()> {
    let mut catalog = read_catalog(store, base).await?;
    let entry = catalog
        .remove(dataset_name)
        .ok_or_else(|| anyhow::anyhow!("dataset '{}' not found in catalog", dataset_name))?;

    if delete_files {
        info!(path = entry.path, "deleting remote files");
        let prefix = obj_path(base, &entry.path);
        let mut list = store.list(Some(&prefix));
        use futures::StreamExt;
        while let Some(meta) = list.next().await {
            let meta = meta?;
            store.delete(&meta.location).await?;
        }
    }

    write_catalog(store, base, &catalog).await?;
    info!(name = dataset_name, "deleted from catalog");
    Ok(())
}

/// Garbage collect: remove S3 directories not referenced by the catalog.
pub async fn gc(store: &dyn ObjectStore, base: &ObjPath) -> Result<Vec<String>> {
    let catalog = read_catalog(store, base).await?;
    let referenced: vortex::utils::aliases::hash_set::HashSet<&str> =
        catalog.datasets.iter().map(|d| d.path.as_str()).collect();

    // List all top-level "directories" in the store.
    let list_result = store.list_with_delimiter(Some(base)).await?;
    let mut removed = Vec::new();

    for prefix in &list_result.common_prefixes {
        let dir_name = prefix.as_ref().trim_end_matches('/');
        let dir_with_slash = format!("{}/", dir_name);

        // Skip catalog.json itself.
        if dir_name == "catalog.json" {
            continue;
        }

        if !referenced.contains(dir_with_slash.as_str()) {
            info!(path = %prefix, "garbage collecting orphaned directory");
            let mut list = store.list(Some(prefix));
            use futures::StreamExt;
            while let Some(meta) = list.next().await {
                let meta = meta?;
                store.delete(&meta.location).await?;
            }
            removed.push(dir_with_slash);
        }
    }

    Ok(removed)
}

/// Verify a dataset: check manifest hash in catalog, check file hashes.
pub async fn verify(
    store: &dyn ObjectStore,
    base: &ObjPath,
    dataset_name: &str,
) -> Result<Vec<String>> {
    let catalog = read_catalog(store, base).await?;
    let entry = catalog
        .find(dataset_name)
        .ok_or_else(|| anyhow::anyhow!("dataset '{}' not found in catalog", dataset_name))?;

    let mut problems = Vec::new();

    // Check manifest hash.
    let manifest_path = obj_path(base, &format!("{}manifest.json", entry.path));
    let manifest_bytes = store.get(&manifest_path).await?.bytes().await?;
    let actual_manifest_hash = sha256_hex(&manifest_bytes);
    if actual_manifest_hash != entry.manifest_hash {
        problems.push(format!(
            "manifest hash mismatch: catalog says {}, actual is {}",
            entry.manifest_hash, actual_manifest_hash
        ));
    }

    let manifest = Manifest::from_json(&manifest_bytes)?;

    // Check file hashes.
    for (_format, _table, file) in manifest.iter_files() {
        let remote_path = obj_path(base, &format!("{}{}", entry.path, file.path));
        match store.get(&remote_path).await {
            Ok(result) => {
                let bytes = result.bytes().await?;
                let actual_hash = sha256_hex(&bytes);
                if actual_hash != file.sha256 {
                    problems.push(format!(
                        "{}: hash mismatch (expected {}, got {})",
                        file.path, file.sha256, actual_hash
                    ));
                }
                if bytes.len() as u64 != file.size_bytes {
                    problems.push(format!(
                        "{}: size mismatch (expected {}, got {})",
                        file.path,
                        file.size_bytes,
                        bytes.len()
                    ));
                }
            }
            Err(object_store::Error::NotFound { .. }) => {
                problems.push(format!("{}: file not found in remote", file.path));
            }
            Err(e) => {
                problems.push(format!("{}: error reading: {}", file.path, e));
            }
        }
    }

    Ok(problems)
}

/// Build a manifest by scanning a `data/` directory.
/// Expects layout: `data/{format}/{table}/{files}`.
pub fn build_manifest_from_dir(name: &str, data_dir: &Path) -> Result<Manifest> {
    let mut manifest = Manifest::new(name);

    for format_entry in std::fs::read_dir(data_dir)? {
        let format_entry = format_entry?;
        if !format_entry.file_type()?.is_dir() {
            continue;
        }
        let format_name = format_entry.file_name().to_string_lossy().to_string();

        for table_entry in std::fs::read_dir(format_entry.path())? {
            let table_entry = table_entry?;
            if !table_entry.file_type()?.is_dir() {
                continue;
            }
            let table_name = table_entry.file_name().to_string_lossy().to_string();

            for file_entry in std::fs::read_dir(table_entry.path())? {
                let file_entry = file_entry?;
                if !file_entry.file_type()?.is_file() {
                    continue;
                }

                let rel_path = format!(
                    "{}/{}/{}",
                    format_name,
                    table_name,
                    file_entry.file_name().to_string_lossy()
                );

                let (sha256, size_bytes) = hash_file(file_entry.path())?;

                manifest.add_file(
                    &format_name,
                    &table_name,
                    super::manifest::FileEntry {
                        path: rel_path,
                        sha256,
                        size_bytes,
                    },
                );
            }
        }
    }

    Ok(manifest)
}

// -- helpers --

fn catalog_path(base: &ObjPath) -> ObjPath {
    if base.as_ref().is_empty() {
        ObjPath::from("catalog.json")
    } else {
        ObjPath::from(format!("{}/catalog.json", base))
    }
}

fn obj_path(base: &ObjPath, suffix: &str) -> ObjPath {
    if base.as_ref().is_empty() {
        ObjPath::from(suffix)
    } else {
        ObjPath::from(format!("{}/{}", base, suffix))
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    hash.iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}
