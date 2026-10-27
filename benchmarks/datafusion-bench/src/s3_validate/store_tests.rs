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
        latency_cdf_ms: Vec::new(),
        latency_bins: Vec::new(),
        seed: 42,
    };
    Ok((Store::new(source, Some(snapshot), Some(model), true), meta))
}

#[test]
fn latency_draws_are_repeatable_and_independent_of_request_order() {
    let model = Model {
        ttfb_ms: 10.0,
        request_mib_per_sec: 100.0,
        max_inflight: 0,
        latency_cdf_ms: (0..1000).map(f64::from).collect(),
        latency_bins: Vec::new(),
        seed: 42,
    };
    let draw = |key: usize, sample| {
        model.delay_ms("GET", &format!("table/{key}"), "0..100", sample, 0, 100)
    };
    let forward = (0..1000).map(|key| draw(key, 1)).collect::<Vec<_>>();
    let reverse = (0..1000)
        .rev()
        .map(|key| draw(key, 1))
        .rev()
        .collect::<Vec<_>>();
    assert_eq!(forward, reverse);
    assert_ne!(
        forward,
        (0..1000).map(|key| draw(key, 2)).collect::<Vec<_>>()
    );
    let mean = forward.iter().sum::<f64>() / forward.len() as f64;
    assert!((450.0..550.0).contains(&mean));
    assert_eq!(model.delay_ms("HEAD", "table", "", 1, 0, 0), 10.0);
}

#[tokio::test]
async fn range_size_selects_the_learned_latency_class() -> anyhow::Result<()> {
    let (mut store, meta) = model(0).await?;
    let model = Arc::make_mut(store.model.as_mut().vortex_expect("model store"));
    model.latency_bins = vec![
        LatencyBin {
            max_bytes: 3,
            latency_cdf_ms: vec![2.0],
        },
        LatencyBin {
            max_bytes: u64::MAX,
            latency_cdf_ms: vec![20.0],
        },
    ];
    store.get_range(&meta.location, 0..3).await?;
    store.get_range(&meta.location, 0..8).await?;
    let (counts, requests) = store.take()?;
    assert_eq!(counts.get, 2);
    assert_eq!(counts.bytes, 11);
    assert_eq!(requests[0].model_delay_ms, Some(2.0));
    assert_eq!(requests[1].model_delay_ms, Some(20.0));
    assert!(requests[0].start_ms <= requests[1].start_ms);
    assert!(requests.iter().all(|request| {
        request
            .response_ms
            .is_some_and(|response| response <= request.duration_ms)
    }));
    Ok(())
}

#[tokio::test]
async fn run_specific_prefixes_do_not_change_seeded_delays() -> anyhow::Result<()> {
    let mut delays = Vec::new();
    for prefix in ["run-one/table", "run-two/table"] {
        let source = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from(format!("{prefix}/data"));
        source
            .put(&path, Bytes::from_static(b"abcdefgh").into())
            .await?;
        let objects = snapshot(source.as_ref(), &Path::from(prefix)).await?;
        let store = Store::new(
            source,
            Some(objects),
            Some(Model {
                ttfb_ms: 1.0,
                request_mib_per_sec: 100.0,
                max_inflight: 0,
                latency_cdf_ms: vec![1.0, 3.0, 7.0],
                latency_bins: Vec::new(),
                seed: 42,
            }),
            true,
        );
        store.get(&path).await?.bytes().await?;
        delays.push(store.take()?.1[0].model_delay_ms);
    }
    assert_eq!(delays[0], delays[1]);
    Ok(())
}

#[tokio::test]
async fn empirical_delays_preserve_request_metrics() -> anyhow::Result<()> {
    let (mut store, meta) = model(0).await?;
    if let Some(model) = store.model.as_mut() {
        Arc::make_mut(model).latency_cdf_ms = vec![2.0, 5.0, 10.0, 20.0];
    }
    futures::future::try_join_all(
        (0..8).map(|_| async { store.get(&meta.location).await?.bytes().await }),
    )
    .await?;
    let (counts, requests) = store.take()?;
    assert_eq!(counts.get, 8);
    assert_eq!(counts.bytes, 64);
    assert_eq!(counts.peak_inflight, 8);
    assert!(
        requests
            .iter()
            .all(|r| r.model_delay_ms.is_some_and(|delay| r.duration_ms >= delay))
    );
    Ok(())
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

#[tokio::test]
async fn duplicate_tail_reads_share_one_request_per_sample() -> anyhow::Result<()> {
    let (store, meta) = model(0).await?;
    let values =
        futures::future::try_join_all((0..8).map(|_| store.get_range(&meta.location, 0..8)))
            .await?;
    assert!(values.iter().all(|bytes| bytes.as_ref() == b"abcdefgh"));
    let (counts, _) = store.take()?;
    assert_eq!(counts.get, 1);
    assert_eq!(counts.get_calls, 8);
    assert_eq!(counts.tail_cache_hits, 7);
    assert_eq!(counts.bytes, 8);
    store.reset()?;
    store.get_range(&meta.location, 0..8).await?;
    assert_eq!(store.take()?.0.get, 1);
    Ok(())
}

#[tokio::test]
async fn real_backend_delegates_even_with_a_snapshot() -> anyhow::Result<()> {
    let (model, meta) = model(0).await?;
    model
        .source
        .put(&meta.location, Bytes::from_static(b"ijklmnop").into())
        .await?;
    let real = Store::new(
        Arc::clone(&model.source),
        model.snapshot.clone(),
        None,
        true,
    );
    assert_eq!(
        real.get(&meta.location).await?.bytes().await?.as_ref(),
        b"ijklmnop"
    );
    assert_eq!(
        model.get(&meta.location).await?.bytes().await?.as_ref(),
        b"abcdefgh"
    );
    Ok(())
}
