// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex's vector index against Lance's, over the same table.
//!
//! A plain criterion benchmark, not a CodSpeed one: it compares two systems on one machine, so its
//! numbers only mean something side by side. The default table is small enough that building both
//! sides' indexes and measuring every query takes a few seconds; set `INDEX_BENCH_SCALE=large` for
//! 400k rows of 128-dimensional vectors, where the index work outweighs each query's fixed cost.
//!
//! | Query | Vortex | Lance |
//! |---|---|---|
//! | Cosine top-10 | IVF with int8 codes, then a take of the hits' ids | IVF_SQ |
//!
//! The query is timed three ways:
//!
//! - `vortex` and `lance` return ids, so both read the data as well as the index. Vortex keeps
//!   its index handle open between queries, and Lance keeps its opened index in its session's
//!   index cache, so these two are the like-for-like comparison.
//! - `vortex_reopen` opens the index file again for every query, the cost of a cold query.
//!   Lance still looks its cached index up per query, but does not reload it.
//! - `cosine_search` times the index search alone, returning row positions and scores without
//!   reading the data.
//!
//! Both sides read from memory. Before timing, the benchmark prints both sides' recall@10 against
//! an exact cosine search over every vector. The `vbench_sql` group runs VBench Q1 and Q4
//! in DataFusion on both indexed tables. It includes SQL planning, native ANN retrieval,
//! fetching candidate rows, and final cosine scoring and ordering. Queries rotate over the same
//! 21 vectors on each side. Set `VBENCH_VORTEX_NPROBES` and `VBENCH_LANCE_NPROBES`
//! independently to compare latency at similar measured recall (both default to 8).
//!
//! ```text
//! cargo bench -p lance-bench --bench index_vs_lance
//! INDEX_BENCH_SCALE=large cargo bench -p lance-bench --bench index_vs_lance
//! ```

#![expect(clippy::unwrap_used, reason = "benchmark setup fails loudly")]
#![expect(clippy::cast_possible_truncation, reason = "small benchmark constants")]

#[path = "../workloads/fixture.rs"]
mod fixture;
mod vbench;

use std::sync::Arc;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use fixture::Data;
use fixture::K;
use fixture::Scale;
use fixture::build_ivf;
use fixture::exact_cosine;
use fixture::lance_dataset;
use fixture::price;
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::scanner::Scanner;
use lance::deps::arrow_array as la;
use lance::deps::arrow_array::Array as _;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability::NonNullable;
use vortex::array::dtype::PType;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::expr::col;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex_index::Hit;
use vortex_index::IndexFile;
use vortex_index::rows::collect;
use vortex_index_ivf::IvfIndex;
use vortex_index_ivf::VectorSearch;

const NPROBE: usize = 8;
/// The Vortex side: a plain data file and a standalone IVF index file.
struct VortexSide {
    data: VortexFile,
    ivf: IndexFile<IvfIndex>,
    /// The opened index file, for the variant that opens an index handle per query.
    ivf_file: VortexFile,
}

fn open(session: &VortexSession, bytes: ByteBufferMut) -> VortexResult<VortexFile> {
    session
        .open_options()
        .include_metadata()
        .open_buffer(bytes.freeze())
}

impl VortexSide {
    async fn build(session: &VortexSession, data: &Data, scale: Scale) -> VortexResult<Self> {
        let mut bytes = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut bytes, data.vortex_array()?.to_array_stream())
            .await?;
        let file = open(session, bytes)?;

        let ivf = build_ivf(session, &file, scale).await?;
        let mut ivf_bytes = ByteBufferMut::empty();
        ivf.write(&mut ivf_bytes).await?;

        let ivf_file = open(session, ivf_bytes)?;
        Ok(Self {
            data: file,
            ivf: IndexFile::open(IvfIndex, &ivf_file)?,
            ivf_file,
        })
    }

    async fn nearest(&self, query: &[f32]) -> VortexResult<Vec<u64>> {
        let hits = self.nearest_search(query).await?;
        self.take_ids(hits).await
    }

    async fn nearest_reopen(&self, query: &[f32]) -> VortexResult<Vec<u64>> {
        let index = IndexFile::open(IvfIndex, &self.ivf_file)?;
        let hits = index.rows().search_vectors(query, K, NPROBE).await?;
        self.take_ids(hits).await
    }

    async fn nearest_search(&self, query: &[f32]) -> VortexResult<Vec<Hit>> {
        self.ivf.rows().search_vectors(query, K, NPROBE).await
    }

    /// The ids of the hits' rows, in row order.
    async fn take_ids(&self, hits: Vec<Hit>) -> VortexResult<Vec<u64>> {
        let mut rows: Vec<u64> = hits.iter().map(|hit| hit.row).collect();
        rows.sort_unstable();
        rows.dedup();
        let rows = StrictSortedBuffer::try_new(Buffer::from(rows))?;
        let session = self.data.session();
        let chunks = collect(
            ScanBuilder::new(session.clone(), self.data.scan_file())
                .with_row_indices(rows)
                .with_projection(col("id").bind(self.data.dtype())?),
        )
        .await?;
        self.u64s(chunks)
    }

    fn u64s(&self, chunks: Vec<ArrayRef>) -> VortexResult<Vec<u64>> {
        let mut ctx = self.data.session().create_execution_ctx();
        let ids = ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, NonNullable))?
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        Ok(ids.as_slice::<u64>().to_vec())
    }
}

/// The Lance side: one dataset with an IVF_SQ index.
async fn lance_ids(scanner: &mut Scanner) -> lance::Result<Vec<u64>> {
    let batches: Vec<la::RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<la::UInt64Array>()
            .unwrap()
            .clone();
        ids.extend(column.values().iter().copied());
    }
    ids.sort_unstable();
    Ok(ids)
}

fn nearest_scanner(dataset: &Dataset, query: &[f32]) -> lance::Result<Scanner> {
    let mut scanner = dataset.scan();
    scanner
        .nearest("emb", &la::Float32Array::from(query.to_vec()), K)?
        .nprobes(NPROBE);
    Ok(scanner)
}

async fn lance_nearest(dataset: &Dataset, query: &[f32]) -> lance::Result<Vec<u64>> {
    let mut scanner = nearest_scanner(dataset, query)?;
    scanner.project(&["id"])?;
    lance_ids(&mut scanner).await
}

/// Runs a search that projects only row ids and scores, so Lance reads no data columns, and
/// returns how many rows it found.
async fn lance_search(mut scanner: Scanner) -> lance::Result<usize> {
    scanner.with_row_id().project::<&str>(&[])?;
    let batches: Vec<la::RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    Ok(batches.iter().map(la::RecordBatch::num_rows).sum())
}

/// How many of `found` are in `exact`.
fn overlap(found: &[u64], exact: &[u64]) -> usize {
    found.iter().filter(|id| exact.contains(id)).count()
}

fn rows(hits: &[Hit]) -> Vec<u64> {
    hits.iter().map(|hit| hit.row).collect()
}

/// Prints each side's mean recall@10 against exact cosine search, over the benchmark's query
/// and the additional fixture queries.
fn print_recall(runtime: &Runtime, data: &Data, vortex: &VortexSide, lance: &Dataset) {
    let recall = |found: usize, queries: usize| found as f64 / (K * queries) as f64;

    let queries = data.vector_queries();
    let (mut vortex_found, mut lance_found) = (0, 0);
    for query in &queries {
        let exact = exact_cosine(data, query, None);
        let hits = runtime.block_on(vortex.nearest_search(query)).unwrap();
        vortex_found += overlap(&rows(&hits), &exact);
        let ids = runtime.block_on(lance_nearest(lance, query)).unwrap();
        lance_found += overlap(&ids, &exact);
    }
    eprintln!(
        "cosine recall@{K} over {} queries: Vortex {:.3}, Lance {:.3}",
        queries.len(),
        recall(vortex_found, queries.len()),
        recall(lance_found, queries.len()),
    );
}

fn index_vs_lance(c: &mut Criterion) {
    let scale = if std::env::var("INDEX_BENCH_SCALE").as_deref() == Ok("large") {
        Scale::LARGE
    } else {
        Scale::SMALL
    };
    let runtime = Runtime::new().unwrap();
    let data = Data::generate(scale);
    let session = VortexSession::default().with_tokio();
    vortex_index::initialize(&session).unwrap();
    vortex_index_ivf::register(&session);
    let vortex = runtime
        .block_on(VortexSide::build(&session, &data, scale))
        .unwrap();
    // Both sides read from memory, so neither pays for file IO the other skips.
    let lance = runtime
        .block_on(lance_dataset("memory://index-vs-lance", &data))
        .unwrap();

    // The two sides answer the same question before either is timed: the query sits next to
    // row 0.
    let vortex_near = runtime.block_on(vortex.nearest(&data.query)).unwrap();
    let lance_near = runtime
        .block_on(lance_nearest(&lance, &data.query))
        .unwrap();
    assert!(vortex_near.contains(&0), "Vortex's nearest rows miss row 0");
    assert!(lance_near.contains(&0), "Lance's nearest rows miss row 0");
    assert_eq!(
        runtime
            .block_on(vortex.nearest_reopen(&data.query))
            .unwrap(),
        vortex_near
    );
    assert_eq!(
        runtime
            .block_on(lance_search(nearest_scanner(&lance, &data.query).unwrap()))
            .unwrap(),
        K
    );
    print_recall(&runtime, &data, &vortex, &lance);

    let label = scale.label;
    let mut group = c.benchmark_group("cosine_top10");
    group.bench_function(BenchmarkId::new("vortex", label), |b| {
        b.iter(|| runtime.block_on(vortex.nearest(&data.query)).unwrap())
    });
    group.bench_function(BenchmarkId::new("vortex_reopen", label), |b| {
        b.iter(|| {
            runtime
                .block_on(vortex.nearest_reopen(&data.query))
                .unwrap()
        })
    });
    group.bench_function(BenchmarkId::new("lance", label), |b| {
        b.iter(|| {
            runtime
                .block_on(lance_nearest(&lance, &data.query))
                .unwrap()
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cosine_search");
    group.bench_function(BenchmarkId::new("vortex", label), |b| {
        b.iter(|| {
            runtime
                .block_on(vortex.nearest_search(&data.query))
                .unwrap()
        })
    });
    group.bench_function(BenchmarkId::new("lance", label), |b| {
        b.iter(|| {
            runtime
                .block_on(lance_search(nearest_scanner(&lance, &data.query).unwrap()))
                .unwrap()
        })
    });
    group.finish();
    vbench::benchmark(c, &runtime, &data, Arc::new(vortex), Arc::new(lance));
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500));
    targets = index_vs_lance
}
criterion_main!(benches);
