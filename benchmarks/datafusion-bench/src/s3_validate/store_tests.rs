// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::*;

async fn model(cap: usize) -> anyhow::Result<(Store, ObjectMeta)> {
    let source = Arc::new(object_store::memory::InMemory::new());
    let key = Path::from("table/data");
    source
        .put(&key, Bytes::from_static(b"abcdefgh").into())
        .await?;
    let meta = source.head(&key).await?;
    let snapshot = snapshot(source.as_ref(), &Path::from("table")).await?;
    let model = Model {
        ttfb_ms: 10.0,
        request_mib_per_sec: 100.0,
        max_inflight: cap,
    };
    Ok((Store::new(source, Some(snapshot), Some(model)), meta))
}

#[tokio::test]
async fn range_metadata_head_and_cancellation() -> anyhow::Result<()> {
    let (store, meta) = model(1).await?;
    let options = GetOptions::new()
        .with_range(Some(2..5))
        .with_if_match(meta.e_tag.clone());
    let result = store.get_opts(&meta.location, options).await?;
    assert_eq!(result.meta, meta);
    assert_eq!(result.bytes().await?, Bytes::from_static(b"cde"));
    assert_eq!(store.head(&meta.location).await?, meta);
    let response = store.get(&meta.location).await?;
    // Dropping an unconsumed response must release the semaphore and in-flight guard.
    drop(response);
    let (counts, _) = store.take()?;
    assert_eq!(counts.get, 2);
    assert_eq!(counts.head, 1);
    assert_eq!(counts.bytes, 3);
    assert_eq!(counts.peak_inflight, 1);
    Ok(())
}

#[tokio::test]
async fn async_requests_overlap_and_obey_cap() -> anyhow::Result<()> {
    for (cap, expected) in [(0, 8), (2, 2)] {
        let (store, meta) = model(cap).await?;
        futures::future::try_join_all(
            (0..8).map(|_| async { store.get(&meta.location).await?.bytes().await }),
        )
        .await?;
        let (counts, _) = store.take()?;
        assert_eq!(counts.get, 8);
        assert_eq!(counts.bytes, 64);
        assert_eq!(counts.peak_inflight, expected);
    }
    Ok(())
}
