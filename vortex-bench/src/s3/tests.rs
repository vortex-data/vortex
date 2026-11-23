// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::TryStreamExt;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;

use super::Profile;
use super::Server;

async fn endpoint() -> anyhow::Result<(Server, Arc<dyn ObjectStore>)> {
    let server = Server::start(
        None,
        Profile::zero_delay(),
        "test-bucket".to_string(),
        "127.0.0.1:0".parse()?,
        1024 * 1024,
    )
    .await?;
    let store = AmazonS3Builder::new()
        .with_bucket_name("test-bucket")
        .with_region("us-east-1")
        .with_access_key_id("benchmark")
        .with_secret_access_key("benchmark")
        .with_endpoint(server.endpoint())
        .with_allow_http(true)
        .build()?;
    Ok((server, Arc::new(store)))
}

#[tokio::test]
async fn s3_client_reads_ranges_suffixes_metadata_and_preconditions() -> anyhow::Result<()> {
    let (server, store) = endpoint().await?;
    let path = Path::from("directory/object.bin");
    store.put(&path, b"0123456789".to_vec().into()).await?;
    assert_eq!(store.head(&path).await?.size, 10);
    assert_eq!(store.get_range(&path, 2..6).await?.as_ref(), b"2345");
    let suffix = store
        .get_opts(
            &path,
            GetOptions {
                range: Some(GetRange::Suffix(3)),
                ..GetOptions::default()
            },
        )
        .await?;
    assert_eq!(suffix.bytes().await?.as_ref(), b"789");
    let meta = store.head(&path).await?;
    let result = store
        .get_opts(
            &path,
            GetOptions {
                if_match: meta.e_tag.clone(),
                ..GetOptions::default()
            },
        )
        .await?;
    assert_eq!(result.bytes().await?.as_ref(), b"0123456789");
    assert!(
        store
            .get_opts(
                &path,
                GetOptions {
                    if_match: Some("different".to_string()),
                    ..GetOptions::default()
                }
            )
            .await
            .is_err()
    );
    assert_eq!(server.resident_bytes().await, 10);
    Ok(())
}

#[tokio::test]
async fn listing_handles_prefixes_delimiters_and_pagination() -> anyhow::Result<()> {
    let (_server, store) = endpoint().await?;
    for name in ["top/file", "top/nested/part", "other/file"] {
        store.put(&Path::from(name), vec![1].into()).await?;
    }
    let list = store.list_with_delimiter(Some(&Path::from("top"))).await?;
    assert_eq!(list.objects.len(), 1);
    assert_eq!(list.common_prefixes, vec![Path::from("top/nested")]);
    for index in 0..1001 {
        store
            .put(&Path::from(format!("many/{index:04}")), vec![1].into())
            .await?;
    }
    let objects = store
        .list(Some(&Path::from("many")))
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(objects.len(), 1001);
    Ok(())
}

#[tokio::test]
async fn overwrite_delete_and_resident_budget_are_consistent() -> anyhow::Result<()> {
    let (server, store) = endpoint().await?;
    let path = Path::from("object");
    store.put(&path, vec![1; 16].into()).await?;
    store.put(&path, vec![2; 32].into()).await?;
    assert_eq!(server.resident_bytes().await, 32);
    assert_eq!(store.get(&path).await?.bytes().await?.as_ref(), &[2; 32]);
    store.delete(&path).await?;
    assert_eq!(server.resident_bytes().await, 0);
    assert!(store.head(&path).await.is_err());
    assert!(
        store
            .put(&path, vec![0; 1024 * 1024 + 1].into())
            .await
            .is_err()
    );
    assert_eq!(server.resident_bytes().await, 0);
    Ok(())
}

#[tokio::test]
async fn a_snapshot_preserves_the_local_key_layout() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    std::fs::create_dir(directory.path().join("nested"))?;
    std::fs::write(directory.path().join("nested/data"), b"snapshot")?;
    let server = Server::start(
        Some(directory.path()),
        Profile::zero_delay(),
        "test-bucket".to_string(),
        "127.0.0.1:0".parse()?,
        1024,
    )
    .await?;
    let client = reqwest::get(format!("{}/test-bucket/nested/data", server.endpoint())).await?;
    assert_eq!(client.bytes().await?.as_ref(), b"snapshot");
    Ok(())
}
