// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjPath;
use tempfile::TempDir;

use super::catalog::Catalog;
use super::catalog::DatasetEntry;
use super::dataset::DatasetDescriptor;
use super::dataset::Source;
use super::local;
use super::manifest::Manifest;
use super::manifest::hash_file;
use super::remote;

/// Helper: create a local ObjectStore rooted at a temp dir.
fn local_store(dir: &std::path::Path) -> (Arc<dyn ObjectStore>, ObjPath) {
    std::fs::create_dir_all(dir).unwrap();
    let store = LocalFileSystem::new_with_prefix(dir).unwrap();
    (Arc::new(store), ObjPath::default())
}

/// Helper: create a dataset directory with real data files.
fn create_test_dataset(
    parent: &std::path::Path,
    name: &str,
    files: &[(&str, &[u8])],
) -> std::path::PathBuf {
    let dataset_dir = parent.join(name);
    std::fs::create_dir_all(dataset_dir.join("data")).unwrap();

    // Write descriptor.
    let desc = DatasetDescriptor {
        name: name.to_string(),
        description: format!("Test dataset: {name}"),
        author: "Test User <test@example.com>".to_string(),
        tags: vec!["test".to_string()],
        source: Some(Source {
            kind: "generator".to_string(),
            description: "test data".to_string(),
            command: None,
            parent: None,
            url: None,
        }),
        extra: Default::default(),
    };
    desc.write_to_file(dataset_dir.join("dataset.yaml"))
        .unwrap();

    // Write data files.
    for (path, content) in files {
        let full_path = dataset_dir.join("data").join(path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full_path, content).unwrap();
    }

    dataset_dir
}

// -- Catalog tests --

#[test]
fn test_catalog_roundtrip() -> anyhow::Result<()> {
    let mut catalog = Catalog::new();
    catalog.upsert(DatasetEntry {
        name: "test-dataset".to_string(),
        path: "test-dataset-abc123/".to_string(),
        manifest_hash: "deadbeef".to_string(),
    });

    let json = catalog.to_json()?;
    let parsed = Catalog::from_json(&json)?;
    assert_eq!(parsed.datasets.len(), 1);
    assert_eq!(parsed.datasets[0].name, "test-dataset");
    assert_eq!(parsed.datasets[0].manifest_hash, "deadbeef");
    Ok(())
}

#[test]
fn test_catalog_upsert_replaces() -> anyhow::Result<()> {
    let mut catalog = Catalog::new();
    catalog.upsert(DatasetEntry {
        name: "ds".to_string(),
        path: "ds-aaa/".to_string(),
        manifest_hash: "hash1".to_string(),
    });
    let old = catalog.upsert(DatasetEntry {
        name: "ds".to_string(),
        path: "ds-bbb/".to_string(),
        manifest_hash: "hash2".to_string(),
    });

    assert!(old.is_some());
    assert_eq!(old.unwrap().path, "ds-aaa/");
    assert_eq!(catalog.datasets.len(), 1);
    assert_eq!(catalog.datasets[0].path, "ds-bbb/");
    Ok(())
}

#[test]
fn test_catalog_remove() -> anyhow::Result<()> {
    let mut catalog = Catalog::new();
    catalog.upsert(DatasetEntry {
        name: "ds".to_string(),
        path: "ds-aaa/".to_string(),
        manifest_hash: "hash1".to_string(),
    });

    let removed = catalog.remove("ds");
    assert!(removed.is_some());
    assert!(catalog.datasets.is_empty());

    let not_found = catalog.remove("nonexistent");
    assert!(not_found.is_none());
    Ok(())
}

// -- Manifest tests --

#[test]
fn test_manifest_roundtrip() -> anyhow::Result<()> {
    let mut manifest = Manifest::new("test");
    manifest.add_file(
        "parquet",
        "lineitem",
        super::manifest::FileEntry {
            path: "parquet/lineitem/data.parquet".to_string(),
            sha256: "abc123".to_string(),
            size_bytes: 1024,
        },
    );

    let json = manifest.to_json()?;
    let parsed = Manifest::from_json(&json)?;
    assert_eq!(parsed.name, "test");
    assert_eq!(parsed.total_files(), 1);
    assert_eq!(parsed.total_size_bytes(), 1024);
    Ok(())
}

#[test]
fn test_manifest_content_hash_deterministic() -> anyhow::Result<()> {
    let mut m1 = Manifest::new("test");
    m1.add_file(
        "parquet",
        "t1",
        super::manifest::FileEntry {
            path: "p".to_string(),
            sha256: "h".to_string(),
            size_bytes: 1,
        },
    );
    let mut m2 = m1.clone();

    assert_eq!(m1.content_hash()?, m2.content_hash()?);

    m2.add_file(
        "vortex",
        "t1",
        super::manifest::FileEntry {
            path: "v".to_string(),
            sha256: "h2".to_string(),
            size_bytes: 2,
        },
    );
    assert_ne!(m1.content_hash()?, m2.content_hash()?);
    Ok(())
}

#[test]
fn test_hash_file() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("test.bin");
    std::fs::write(&path, b"hello world")?;

    let (hash, size) = hash_file(&path)?;
    assert_eq!(size, 11);
    assert!(!hash.is_empty());

    // Same content should produce same hash.
    let path2 = dir.path().join("test2.bin");
    std::fs::write(&path2, b"hello world")?;
    let (hash2, _) = hash_file(&path2)?;
    assert_eq!(hash, hash2);
    Ok(())
}

// -- Dataset descriptor tests --

#[test]
fn test_descriptor_yaml_roundtrip() -> anyhow::Result<()> {
    let desc = DatasetDescriptor {
        name: "my-dataset".to_string(),
        description: "A test".to_string(),
        author: "Test <test@test.com>".to_string(),
        tags: vec!["a".to_string(), "b".to_string()],
        source: Some(Source {
            kind: "generator".to_string(),
            description: "generated".to_string(),
            command: Some("gen --x".to_string()),
            parent: None,
            url: None,
        }),
        extra: Default::default(),
    };

    let yaml = desc.to_yaml_bytes()?;
    let parsed = DatasetDescriptor::from_yaml(&yaml)?;
    assert_eq!(parsed.name, "my-dataset");
    assert_eq!(parsed.tags, vec!["a", "b"]);
    assert_eq!(
        parsed.source.as_ref().unwrap().command.as_deref(),
        Some("gen --x")
    );
    Ok(())
}

#[test]
fn test_descriptor_extra_fields_preserved() -> anyhow::Result<()> {
    let yaml = r#"
name: test
description: test
author: test
tags: []
custom_field: hello
nested:
  a: 1
  b: 2
"#;
    let desc = DatasetDescriptor::from_yaml(yaml.as_bytes())?;
    assert!(desc.extra.contains_key("custom_field"));
    assert!(desc.extra.contains_key("nested"));

    // Round-trip preserves extra fields.
    let bytes = desc.to_yaml_bytes()?;
    let parsed = DatasetDescriptor::from_yaml(&bytes)?;
    assert!(parsed.extra.contains_key("custom_field"));
    Ok(())
}

#[test]
fn test_descriptor_validation() -> anyhow::Result<()> {
    let desc = DatasetDescriptor {
        name: String::new(),
        description: String::new(),
        author: String::new(),
        tags: vec![],
        source: Some(Source {
            kind: "derived".to_string(),
            description: String::new(),
            command: None,
            parent: None,
            url: None,
        }),
        extra: Default::default(),
    };

    let problems = desc.validate();
    assert!(problems.iter().any(|p| p.contains("name is empty")));
    assert!(problems.iter().any(|p| p.contains("description is empty")));
    assert!(problems.iter().any(|p| p.contains("author is empty")));
    assert!(
        problems
            .iter()
            .any(|p| p.contains("source.description is empty"))
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("source.parent is required"))
    );
    Ok(())
}

// -- Local operations tests --

#[test]
fn test_init_creates_structure() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let dataset_dir = dir.path().join("my-dataset");
    local::init(&dataset_dir, "my-dataset")?;

    assert!(dataset_dir.join("dataset.yaml").exists());
    assert!(dataset_dir.join("data").is_dir());

    let desc = DatasetDescriptor::from_file(dataset_dir.join("dataset.yaml"))?;
    assert_eq!(desc.name, "my-dataset");
    Ok(())
}

#[test]
fn test_init_fails_if_exists() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let dataset_dir = dir.path().join("my-dataset");
    std::fs::create_dir_all(&dataset_dir)?;

    let result = local::init(&dataset_dir, "my-dataset");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_manifest_generation() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let dataset_dir = create_test_dataset(
        dir.path(),
        "test-ds",
        &[
            ("parquet/events/events.parquet", b"fake parquet data"),
            ("vortex/events/events.vortex", b"fake vortex data"),
        ],
    );

    let manifest = local::manifest(&dataset_dir)?;
    assert_eq!(manifest.total_files(), 2);
    assert!(manifest.formats.contains_key("parquet"));
    assert!(manifest.formats.contains_key("vortex"));
    assert!(dataset_dir.join("manifest.json").exists());
    Ok(())
}

#[test]
fn test_validate_pass() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let dataset_dir = create_test_dataset(
        dir.path(),
        "test-ds",
        &[("parquet/events/events.parquet", b"data")],
    );

    let problems = local::validate(&dataset_dir)?;
    assert!(problems.is_empty(), "unexpected problems: {problems:?}");
    Ok(())
}

#[test]
fn test_validate_empty_data() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let dataset_dir = dir.path().join("empty-ds");
    local::init(&dataset_dir, "empty-ds")?;

    // Fill in required fields so only data emptiness is caught.
    let desc = DatasetDescriptor {
        name: "empty-ds".to_string(),
        description: "test".to_string(),
        author: "test".to_string(),
        tags: vec![],
        source: Some(Source {
            kind: "generator".to_string(),
            description: "test".to_string(),
            command: None,
            parent: None,
            url: None,
        }),
        extra: Default::default(),
    };
    desc.write_to_file(dataset_dir.join("dataset.yaml"))?;

    let problems = local::validate(&dataset_dir)?;
    assert!(
        problems.iter().any(|p| p.contains("no files")),
        "expected 'no files' problem, got: {problems:?}"
    );
    Ok(())
}

// -- End-to-end remote tests using local filesystem --

#[tokio::test]
async fn test_push_pull_checkout_e2e() -> anyhow::Result<()> {
    let work = TempDir::new()?;

    // Create two "remotes" — just local directories.
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    // Create a dataset locally.
    let dataset_dir = create_test_dataset(
        work.path(),
        "my-bench",
        &[
            ("parquet/events/events_000.parquet", b"parquet content 000"),
            ("parquet/events/events_001.parquet", b"parquet content 001"),
            ("vortex/events/events.vortex", b"vortex content"),
        ],
    );

    // Push.
    remote::push(store.as_ref(), &base, &dataset_dir, true).await?;

    // Verify catalog was created.
    let catalog = remote::read_catalog(store.as_ref(), &base).await?;
    assert_eq!(catalog.datasets.len(), 1);
    assert_eq!(catalog.datasets[0].name, "my-bench");
    assert!(!catalog.datasets[0].manifest_hash.is_empty());

    // Pull to a local mirror.
    let mirror = work.path().join("mirror");
    remote::pull(store.as_ref(), &base, &mirror).await?;
    assert!(mirror.join("catalog.json").exists());

    // Find the dataset dir in mirror.
    let entry = catalog.find("my-bench").unwrap();
    let mirror_dataset = mirror.join(&entry.path);
    assert!(mirror_dataset.join("manifest.json").exists());
    assert!(mirror_dataset.join("dataset.yaml").exists());

    // Checkout data files.
    remote::checkout(store.as_ref(), &base, &mirror, "my-bench").await?;
    let data_dir = mirror_dataset.join("data");
    assert!(data_dir.join("parquet/events/events_000.parquet").exists());
    assert!(data_dir.join("parquet/events/events_001.parquet").exists());
    assert!(data_dir.join("vortex/events/events.vortex").exists());

    // Verify file contents.
    let content = std::fs::read(data_dir.join("parquet/events/events_000.parquet"))?;
    assert_eq!(&content, b"parquet content 000");

    Ok(())
}

#[tokio::test]
async fn test_push_replaces_existing() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    // Push version 1.
    let ds1 = create_test_dataset(
        work.path(),
        "my-bench",
        &[("parquet/t1/a.parquet", b"version 1")],
    );
    remote::push(store.as_ref(), &base, &ds1, true).await?;

    let catalog1 = remote::read_catalog(store.as_ref(), &base).await?;
    let path1 = catalog1.datasets[0].path.clone();

    // Push version 2 (same name, different content).
    let ds2_dir = work.path().join("ds2");
    let ds2 = create_test_dataset(
        &ds2_dir,
        "my-bench",
        &[("parquet/t1/a.parquet", b"version 2")],
    );
    remote::push(store.as_ref(), &base, &ds2, true).await?;

    let catalog2 = remote::read_catalog(store.as_ref(), &base).await?;
    assert_eq!(catalog2.datasets.len(), 1);
    // Path should have changed (different random suffix).
    assert_ne!(catalog2.datasets[0].path, path1);

    Ok(())
}

#[tokio::test]
async fn test_delete() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(
        work.path(),
        "deleteme",
        &[("parquet/t1/a.parquet", b"data")],
    );
    remote::push(store.as_ref(), &base, &ds, true).await?;

    // Delete without purge.
    remote::delete(store.as_ref(), &base, "deleteme", false).await?;
    let catalog = remote::read_catalog(store.as_ref(), &base).await?;
    assert!(catalog.datasets.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_delete_with_purge() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(work.path(), "purgeme", &[("parquet/t1/a.parquet", b"data")]);
    remote::push(store.as_ref(), &base, &ds, true).await?;

    let catalog = remote::read_catalog(store.as_ref(), &base).await?;
    let dataset_path = catalog.datasets[0].path.clone();

    remote::delete(store.as_ref(), &base, "purgeme", true).await?;

    // Data files should be gone.
    let list_result = store
        .list_with_delimiter(Some(&ObjPath::from(dataset_path)))
        .await?;
    assert!(list_result.objects.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_verify_passes() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(
        work.path(),
        "verify-me",
        &[("parquet/t1/a.parquet", b"data")],
    );
    remote::push(store.as_ref(), &base, &ds, true).await?;

    let problems = remote::verify(store.as_ref(), &base, "verify-me").await?;
    assert!(problems.is_empty(), "unexpected problems: {problems:?}");
    Ok(())
}

#[tokio::test]
async fn test_gc_removes_orphans() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    // Push a dataset.
    let ds = create_test_dataset(work.path(), "keeper", &[("parquet/t1/a.parquet", b"keep")]);
    remote::push(store.as_ref(), &base, &ds, true).await?;

    // Manually create an orphaned directory.
    let orphan_path = ObjPath::from("orphan-xyz123/manifest.json");
    store
        .put(
            &orphan_path,
            object_store::PutPayload::from_bytes(b"orphan"[..].into()),
        )
        .await?;

    let removed = remote::gc(store.as_ref(), &base).await?;
    assert!(
        removed.iter().any(|p| p.contains("orphan")),
        "expected orphan to be removed, got: {removed:?}"
    );

    // Keeper should still be in catalog.
    let catalog = remote::read_catalog(store.as_ref(), &base).await?;
    assert_eq!(catalog.datasets.len(), 1);
    assert_eq!(catalog.datasets[0].name, "keeper");
    Ok(())
}

#[tokio::test]
async fn test_checkout_skips_cached_files() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(
        work.path(),
        "cache-test",
        &[("parquet/t1/a.parquet", b"cached data")],
    );
    remote::push(store.as_ref(), &base, &ds, true).await?;

    let mirror = work.path().join("mirror");
    remote::pull(store.as_ref(), &base, &mirror).await?;

    // First checkout downloads.
    remote::checkout(store.as_ref(), &base, &mirror, "cache-test").await?;

    // Second checkout should skip (file already exists with correct hash).
    // This just verifies it doesn't error — the skip is logged.
    remote::checkout(store.as_ref(), &base, &mirror, "cache-test").await?;

    Ok(())
}

#[tokio::test]
async fn test_two_remotes_independent() -> anyhow::Result<()> {
    let work = TempDir::new()?;

    // Two independent remotes.
    let remote_a = work.path().join("remote-a");
    let remote_b = work.path().join("remote-b");
    let (store_a, base_a) = local_store(&remote_a);
    let (store_b, base_b) = local_store(&remote_b);

    // Push different datasets to each.
    let ds1 = create_test_dataset(
        work.path(),
        "dataset-a",
        &[("parquet/t1/a.parquet", b"data a")],
    );
    let ds2_parent = work.path().join("ds2p");
    let ds2 = create_test_dataset(
        &ds2_parent,
        "dataset-b",
        &[("parquet/t1/b.parquet", b"data b")],
    );

    remote::push(store_a.as_ref(), &base_a, &ds1, true).await?;
    remote::push(store_b.as_ref(), &base_b, &ds2, true).await?;

    // Each remote has only its own dataset.
    let cat_a = remote::read_catalog(store_a.as_ref(), &base_a).await?;
    let cat_b = remote::read_catalog(store_b.as_ref(), &base_b).await?;
    assert_eq!(cat_a.datasets.len(), 1);
    assert_eq!(cat_a.datasets[0].name, "dataset-a");
    assert_eq!(cat_b.datasets.len(), 1);
    assert_eq!(cat_b.datasets[0].name, "dataset-b");

    Ok(())
}

// -- Upload lock tests --

#[tokio::test]
async fn test_push_without_force_fails_if_exists() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(
        work.path(),
        "locked-ds",
        &[("parquet/t1/a.parquet", b"data")],
    );

    // First push succeeds (no existing dataset).
    remote::push(store.as_ref(), &base, &ds, false).await?;

    // Second push without force fails.
    let result = remote::push(store.as_ref(), &base, &ds, false).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("already exists"),
        "expected 'already exists' error, got: {err_msg}"
    );

    Ok(())
}

#[tokio::test]
async fn test_push_with_force_overwrites() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(work.path(), "force-ds", &[("parquet/t1/a.parquet", b"v1")]);
    remote::push(store.as_ref(), &base, &ds, true).await?;

    // Overwrite with force.
    std::fs::write(ds.join("data/parquet/t1/a.parquet"), b"v2")?;
    remote::push(store.as_ref(), &base, &ds, true).await?;

    let catalog = remote::read_catalog(store.as_ref(), &base).await?;
    assert_eq!(catalog.datasets.len(), 1);
    assert_eq!(catalog.datasets[0].name, "force-ds");
    Ok(())
}

#[tokio::test]
async fn test_upload_lock_prevents_concurrent_push() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    // Manually create a lock file to simulate a concurrent upload.
    let lock_path = ObjPath::from("concurrent-ds.uploading");
    store
        .put(
            &lock_path,
            object_store::PutPayload::from_bytes(b"locked"[..].into()),
        )
        .await?;

    let ds = create_test_dataset(
        work.path(),
        "concurrent-ds",
        &[("parquet/t1/a.parquet", b"data")],
    );

    // Push should fail because lock exists.
    let result = remote::push(store.as_ref(), &base, &ds, true).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("another upload") || err_msg.contains("uploading"),
        "expected lock error, got: {err_msg}"
    );

    // Clean up lock.
    store.delete(&lock_path).await?;

    // Now push should succeed.
    remote::push(store.as_ref(), &base, &ds, true).await?;
    Ok(())
}

#[tokio::test]
async fn test_upload_lock_released_after_push() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    let ds = create_test_dataset(
        work.path(),
        "lock-cleanup",
        &[("parquet/t1/a.parquet", b"data")],
    );
    remote::push(store.as_ref(), &base, &ds, true).await?;

    // Lock file should be gone after successful push.
    let lock_path = ObjPath::from("lock-cleanup.uploading");
    let result = store.get(&lock_path).await;
    assert!(
        matches!(result, Err(object_store::Error::NotFound { .. })),
        "expected lock file to be deleted after push"
    );

    Ok(())
}

#[tokio::test]
async fn test_check_existing() -> anyhow::Result<()> {
    let work = TempDir::new()?;
    let remote_dir = work.path().join("remote");
    let (store, base) = local_store(&remote_dir);

    // No dataset yet.
    let existing = remote::check_existing(store.as_ref(), &base, "nonexistent").await?;
    assert!(existing.is_none());

    // Push a dataset.
    let ds = create_test_dataset(
        work.path(),
        "exists-test",
        &[("parquet/t1/a.parquet", b"data")],
    );
    remote::push(store.as_ref(), &base, &ds, true).await?;

    // Now it exists.
    let existing = remote::check_existing(store.as_ref(), &base, "exists-test").await?;
    assert!(existing.is_some());
    assert_eq!(existing.unwrap().name, "exists-test");

    Ok(())
}
