// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark for pushing `length(str)` into the scan projection.
//!
//! Demonstrates the cost of materializing a wide string column purely to compute its length,
//! versus pushing the `octet_len` expression into the scan so only an integer column is produced.
//!
//! Run with:
//! ```text
//! cargo test -p vortex-file --release --test len_pushdown_bench -- --ignored --nocapture
//! ```

#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::get_item;
use vortex_array::expr::octet_len;
use vortex_array::expr::root;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session
});

/// A [`VortexReadAt`] that counts the number of bytes read from the underlying buffer.
struct ReadStats {
    bytes_read: AtomicU64,
    num_requests: AtomicU64,
    max_request: AtomicU64,
}

struct CountingReadAt {
    buffer: ByteBuffer,
    stats: Arc<ReadStats>,
}

impl CountingReadAt {
    fn new(buffer: ByteBuffer) -> (Arc<Self>, Arc<ReadStats>) {
        let stats = Arc::new(ReadStats {
            bytes_read: AtomicU64::new(0),
            num_requests: AtomicU64::new(0),
            max_request: AtomicU64::new(0),
        });
        let this = Arc::new(Self {
            buffer,
            stats: Arc::clone(&stats),
        });
        (this, stats)
    }
}

impl VortexReadAt for CountingReadAt {
    fn concurrency(&self) -> usize {
        16
    }

    fn size(&self) -> BoxFuture<'static, vortex_error::VortexResult<u64>> {
        let len = self.buffer.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, vortex_error::VortexResult<BufferHandle>> {
        self.stats
            .bytes_read
            .fetch_add(length as u64, Ordering::Relaxed);
        self.stats.num_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .max_request
            .fetch_max(length as u64, Ordering::Relaxed);
        let buffer = self.buffer.clone();
        async move {
            let start = offset as usize;
            let end = start + length;
            Ok(BufferHandle::new_host(
                buffer.slice_unaligned(start..end).aligned(alignment),
            ))
        }
        .boxed()
    }
}

/// Generate a struct array with a narrow `id` column, a wide `url` string column, and a
/// precomputed `url_len` column. The `url_len` column models a "length sidecar": the lengths
/// stored separately from (and independently readable of) the string bytes.
fn make_data(num_rows: usize) -> vortex_array::ArrayRef {
    let ids = PrimitiveArray::from_iter((0..num_rows as i64).map(|i| i % 1000)).into_array();

    // Synthetic URLs of varying length, similar in spirit to the ClickBench `URL` column.
    let url_strings: Vec<String> = (0..num_rows)
        .map(|i| {
            let path_repeats = (i % 16) + 1;
            format!(
                "https://example.com/path/{}/{}?q={}",
                i % 4096,
                "segment/".repeat(path_repeats),
                i
            )
        })
        .collect();

    let url_lens =
        PrimitiveArray::from_iter(url_strings.iter().map(|s| s.len() as u64)).into_array();
    let urls = VarBinViewArray::from_iter_str(url_strings.iter()).into_array();

    StructArray::new(
        FieldNames::from(["id", "url", "url_len"]),
        vec![ids, urls, url_lens],
        num_rows,
        Validity::NonNullable,
    )
    .into_array()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "benchmark, run manually with --ignored --nocapture"]
async fn bench_len_pushdown() {
    let num_rows = 4_000_000;
    let data = make_data(num_rows);

    // Compress the columns the way the real write path would (BtrBlocks).
    let strategy = Arc::new(CompressingStrategy::new(
        FlatLayoutStrategy::default(),
        BtrBlocksCompressor::default(),
    ));

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write");
    let file_bytes = ByteBuffer::from(bytes);
    println!("\n=== len pushdown benchmark ===");
    println!("rows: {num_rows}, file size: {} MiB", file_bytes.len() >> 20);

    // ---- Baseline: read the full URL column, then compute length in-process. ----
    let (baseline_sum, baseline_bytes, baseline_reqs, baseline_max, baseline_time) = {
        let (reader, stats) = CountingReadAt::new(file_bytes.clone());
        let vxf = SESSION
            .open_options()
            .open(reader)
            .await
            .expect("open baseline");
        let start = Instant::now();
        let result = vxf
            .scan()
            .expect("scan")
            .with_projection(get_item("url", root()))
            .into_array_stream()
            .expect("stream")
            .read_all()
            .await
            .expect("read_all");
        // Compute the total length the way an engine would, after materializing the strings.
        let lengths = result
            .apply(&octet_len(root()))
            .expect("len")
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .expect("primitive");
        let sum: u64 = lengths.as_slice::<u64>().iter().sum();
        (
            sum,
            stats.bytes_read.load(Ordering::Relaxed),
            stats.num_requests.load(Ordering::Relaxed),
            stats.max_request.load(Ordering::Relaxed),
            start.elapsed(),
        )
    };

    // ---- Pushdown: compute octet_len(url) inside the scan, materialize only u64. ----
    let (pushdown_sum, pushdown_bytes, pushdown_reqs, pushdown_max, pushdown_time) = {
        let (reader, stats) = CountingReadAt::new(file_bytes.clone());
        let vxf = SESSION
            .open_options()
            .open(reader)
            .await
            .expect("open pushdown");
        let start = Instant::now();
        let result = vxf
            .scan()
            .expect("scan")
            .with_projection(octet_len(get_item("url", root())))
            .into_array_stream()
            .expect("stream")
            .read_all()
            .await
            .expect("read_all");
        let lengths = result
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .expect("primitive");
        let sum: u64 = lengths.as_slice::<u64>().iter().sum();
        (
            sum,
            stats.bytes_read.load(Ordering::Relaxed),
            stats.num_requests.load(Ordering::Relaxed),
            stats.max_request.load(Ordering::Relaxed),
            start.elapsed(),
        )
    };

    // ---- Sidecar: project a separately-stored precomputed length column. ----
    // This is the I/O ceiling: what `len(url)` could read if lengths were stored in their own
    // segment, independent of the string bytes.
    let (sidecar_sum, sidecar_bytes, sidecar_reqs, sidecar_max, sidecar_time) = {
        let (reader, stats) = CountingReadAt::new(file_bytes.clone());
        let vxf = SESSION
            .open_options()
            .open(reader)
            .await
            .expect("open sidecar");
        let start = Instant::now();
        let result = vxf
            .scan()
            .expect("scan")
            .with_projection(get_item("url_len", root()))
            .into_array_stream()
            .expect("stream")
            .read_all()
            .await
            .expect("read_all");
        let lengths = result
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .expect("primitive");
        let sum: u64 = lengths.as_slice::<u64>().iter().sum();
        (
            sum,
            stats.bytes_read.load(Ordering::Relaxed),
            stats.num_requests.load(Ordering::Relaxed),
            stats.max_request.load(Ordering::Relaxed),
            start.elapsed(),
        )
    };

    assert_eq!(
        baseline_sum, pushdown_sum,
        "both paths must produce the same total length"
    );
    assert_eq!(
        baseline_sum, sidecar_sum,
        "sidecar must produce the same total length"
    );

    let report = |label: &str, bytes: u64, reqs: u64, max: u64, time: std::time::Duration| {
        println!(
            "{label:<28} time: {:>9.2?}   read: {:>7.2} MiB in {:>4} reqs (max {:.2} MiB)",
            time,
            bytes as f64 / (1024.0 * 1024.0),
            reqs,
            max as f64 / (1024.0 * 1024.0),
        );
    };
    println!();
    report(
        "baseline (read url, len)",
        baseline_bytes,
        baseline_reqs,
        baseline_max,
        baseline_time,
    );
    report(
        "pushdown octet_len(url)",
        pushdown_bytes,
        pushdown_reqs,
        pushdown_max,
        pushdown_time,
    );
    report(
        "sidecar (read url_len)",
        sidecar_bytes,
        sidecar_reqs,
        sidecar_max,
        sidecar_time,
    );
    // NOTE: the in-memory segment source coalesces the whole file into a single bulk read (see
    // the `2 reqs / max ~= file size` above), so the byte counter reflects file size, not
    // per-projection I/O. The meaningful signal here is scan/decode time.
    println!(
        "\npushdown octet_len(url) vs baseline:   {:.2}x   (function pushdown alone)",
        baseline_time.as_secs_f64() / pushdown_time.as_secs_f64(),
    );
    println!(
        "sidecar (separate len column) vs base: {:.2}x   (length-separable storage ceiling)\n",
        baseline_time.as_secs_f64() / sidecar_time.as_secs_f64(),
    );
}
