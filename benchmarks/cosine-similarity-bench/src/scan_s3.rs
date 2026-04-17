//! S3 scan: concurrent ranged `GetObject` requests, compute kernel run on each
//! returned buffer as it arrives.
//!
//! Concurrency is controlled by a fixed in-flight window. We use
//! `buffered_unordered` over a `FuturesUnordered`-like stream of range fetches
//! so that compute can overlap with the next network wait and no single slow
//! range stalls the pipeline.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use aws_sdk_s3::Client;
use futures::StreamExt;
use futures::stream;

use crate::kernel::DotKernel;
use crate::kernel::ScanSink;
use crate::kernel::scan_block;
use crate::metrics::CpuSampler;
use crate::metrics::IterationResult;

pub struct S3ScanConfig {
    pub bucket: String,
    pub key: String,
    pub dim: usize,
    pub query: Vec<f32>,
    pub concurrency: usize,
    pub range_bytes: u64,
    pub kernel: DotKernel,
}

pub async fn run_once(client: &Client, cfg: &S3ScanConfig) -> Result<IterationResult> {
    anyhow::ensure!(cfg.dim > 0, "dim must be positive");
    anyhow::ensure!(cfg.concurrency > 0, "concurrency must be positive");
    let bytes_per_vec = (cfg.dim * std::mem::size_of::<f32>()) as u64;
    anyhow::ensure!(cfg.range_bytes > 0, "range_bytes must be positive");
    anyhow::ensure!(
        cfg.range_bytes.is_multiple_of(bytes_per_vec),
        "range_bytes ({}) must be a multiple of vector size ({})",
        cfg.range_bytes,
        bytes_per_vec
    );
    anyhow::ensure!(cfg.query.len() == cfg.dim, "query dim mismatch");

    // HEAD object to determine size.
    let head = client
        .head_object()
        .bucket(&cfg.bucket)
        .key(&cfg.key)
        .send()
        .await
        .with_context(|| format!("HEAD s3://{}/{}", cfg.bucket, cfg.key))?;
    let total_bytes = head.content_length().unwrap_or(0) as u64;
    anyhow::ensure!(
        total_bytes > 0,
        "empty object s3://{}/{}",
        cfg.bucket,
        cfg.key
    );
    anyhow::ensure!(
        total_bytes.is_multiple_of(bytes_per_vec),
        "object size {} not a multiple of vector size {}",
        total_bytes,
        bytes_per_vec
    );

    let mut ranges = Vec::<(u64, u64)>::new();
    let mut off = 0;
    while off < total_bytes {
        let len = cfg.range_bytes.min(total_bytes - off);
        ranges.push((off, len));
        off += len;
    }

    let query = Arc::new(cfg.query.clone());
    let sink = Arc::new(Mutex::new(ScanSink::new()));
    let latencies = Arc::new(Mutex::new(Vec::<u64>::with_capacity(ranges.len())));
    let kernel = cfg.kernel;
    let dim = cfg.dim;

    let cpu = CpuSampler::new();
    let t0 = Instant::now();

    let bucket = cfg.bucket.clone();
    let key = cfg.key.clone();
    let sink_for_map = Arc::clone(&sink);
    let latencies_for_map = Arc::clone(&latencies);
    stream::iter(ranges.into_iter().map(move |(offset, len)| {
        let client = client.clone();
        let bucket = bucket.clone();
        let key = key.clone();
        let query = Arc::clone(&query);
        let sink = Arc::clone(&sink_for_map);
        let latencies = Arc::clone(&latencies_for_map);
        async move {
            let range = format!("bytes={}-{}", offset, offset + len - 1);
            let req_start = Instant::now();
            let resp = client
                .get_object()
                .bucket(&bucket)
                .key(&key)
                .range(range)
                .send()
                .await
                .with_context(|| format!("GetObject offset={} len={}", offset, len))?;
            let body = resp
                .body
                .collect()
                .await
                .context("collecting range body")?
                .into_bytes();
            anyhow::ensure!(
                body.len() as u64 == len,
                "short read: wanted {} got {}",
                len,
                body.len()
            );
            latencies
                .lock()
                .unwrap()
                .push(req_start.elapsed().as_micros() as u64);

            // Compute on the returned buffer. AWS SDK delivers aligned heap
            // bytes (Bytes backing). As with scan_local, f32 alignment is
            // guaranteed by slice-to-bytes reinterpretation from a Vec<u8>
            // allocation - AWS SDK's body is a `Bytes` whose internal buffer
            // is heap allocated and 8-byte aligned, so `as *const f32` is ok.
            //
            // Copy out to an owned Vec to decouple from the `Bytes` lifetime
            // and to ensure alignment. The copy is negligible next to the
            // network wait.
            let mut owned = Vec::<f32>::with_capacity((len / 4) as usize);
            // SAFETY: we copy len bytes of f32 from body.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body.as_ptr() as *const f32,
                    owned.as_mut_ptr(),
                    (len / 4) as usize,
                );
                owned.set_len((len / 4) as usize);
            }

            let mut local = ScanSink::new();
            scan_block(kernel, &query, &owned, dim, &mut local);
            sink.lock().unwrap().merge(&local);
            Ok::<_, anyhow::Error>(())
        }
    }))
    .buffer_unordered(cfg.concurrency)
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    let elapsed = t0.elapsed();
    let cpu_percent = cpu.finish();
    let combined = *sink.lock().unwrap();
    let all_latencies = std::mem::take(&mut *latencies.lock().unwrap());

    std::hint::black_box(combined.sum);
    std::hint::black_box(combined.max);

    Ok(IterationResult {
        elapsed,
        bytes: total_bytes,
        vectors: total_bytes / bytes_per_vec,
        chunk_latencies_us: all_latencies,
        cpu_percent,
        sink_sum: combined.sum,
        sink_max: combined.max,
    })
}
