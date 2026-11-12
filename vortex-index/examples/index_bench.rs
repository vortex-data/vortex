// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::print_stdout,
    reason = "the benchmark reports results on stdout"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "benchmark bookkeeping over small counts"
)]

//! Build and query the `vortex-index` kinds over real data, for comparison with other systems.
//!
//! ```text
//! index_bench build <trigram|fts|ivf> <input.parquet> <column> <outdir> [lists]
//! index_bench query <trigram|fts|ivf> <outdir> <queries> [--repeat N] [--only I] [--scan]
//!                   [--index-only] [--top-k-only] [--e2e] [--nprobe N] [--k K]
//! ```
//!
//! `build` writes `<outdir>/data.vortex` holding the column and a standalone index file
//! `<outdir>/<kind>.vxi` over it, and prints sizes and timings as JSON. `DATA_ROW_BLOCK=<rows>`
//! writes the data file in blocks of that many rows instead of the default 8,192. `query` prints one JSON
//! line per query with its result, the median warm latency, and the exact bytes Vortex read from
//! each file on a fresh open. `--scan` answers trigram queries without the index, and `--index-only`
//! probes the trigram index without reading data. `--top-k-only` runs only the BM25 ranking of a
//! full-text query, skipping its match count. `--e2e` also reads the text of the top-ranked rows
//! from the data file, as a search that shows its results would.
//!
//! `SIM_LATENCY_MS=<ms>` serves every read as an object store would: each request waits that long
//! for its first byte and then streams at `SIM_MBPS` (default 100) MB/s, and reads coalesce with the
//! object-store policy. The first, fresh-open run of each query is then timed as `cold_ms`.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use futures::future::BoxFuture;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::InvertedIndex;
use vortex_index::IvfIndex;
use vortex_index::IvfOptions;
use vortex_index::TrigramIndex;
use vortex_index::TrigramOptions;
use vortex_index::rows::collect;
use vortex_index::text_match;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::std_file::FileReadAt;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

/// Microseconds since the first call, for `TRACE_READ_TIMES`.
fn trace_clock() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e6
}

/// A file source that counts the bytes it is asked for.
struct CountingReadAt {
    inner: Arc<dyn VortexReadAt>,
    bytes: Arc<AtomicU64>,
    requests: Arc<AtomicU64>,
    sim: Option<SimStore>,
    name: String,
}

/// The latency and bandwidth of a simulated object store request.
#[derive(Clone, Copy)]
struct SimStore {
    latency_ms: f64,
    mbps: f64,
}

impl SimStore {
    fn from_env() -> Option<Self> {
        let latency_ms = std::env::var("SIM_LATENCY_MS").ok()?.parse().ok()?;
        let mbps = std::env::var("SIM_MBPS")
            .ok()
            .and_then(|mbps| mbps.parse().ok())
            .unwrap_or(100.0);
        Some(Self { latency_ms, mbps })
    }

    #[expect(clippy::cast_precision_loss, reason = "a simulated delay")]
    fn delay(&self, length: usize) -> std::time::Duration {
        std::time::Duration::from_secs_f64(
            self.latency_ms / 1000.0 + length as f64 / (self.mbps * 1e6),
        )
    }
}

impl VortexReadAt for CountingReadAt {
    /// `NO_COALESCE=1` turns off merging of nearby reads, so the count is the bytes the scan
    /// actually needs rather than what the local-file read policy fetches around them.
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        if std::env::var_os("NO_COALESCE").is_some() {
            return None;
        }
        if self.sim.is_some() {
            return Some(CoalesceConfig::object_storage());
        }
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.inner.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.bytes.fetch_add(length as u64, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
        let mut read = self.inner.read_at(offset, length, alignment);
        if let Some(sim) = self.sim {
            read = Box::pin(async move {
                let (result, ()) = futures::join!(read, tokio::time::sleep(sim.delay(length)));
                result
            });
        }
        if std::env::var_os("TRACE_READS").is_some() {
            eprintln!("read {} {offset} {length}", self.name);
        }
        if std::env::var_os("TRACE_READ_TIMES").is_some() {
            let name = self.name.clone();
            let start = trace_clock();
            let thread = format!("{:?}", std::thread::current().id());
            return Box::pin(async move {
                let result = read.await;
                eprintln!(
                    "readtime {name} {offset} {length} start_us={start:.1} end_us={:.1} thread={thread}",
                    trace_clock()
                );
                result
            });
        }
        read
    }
}

async fn open(
    session: &VortexSession,
    path: &Path,
    metadata: bool,
) -> VortexResult<(VortexFile, Counts)> {
    let counts = Counts::default();
    let source = Arc::new(CountingReadAt {
        inner: Arc::new(FileReadAt::open(path, session.handle())?),
        bytes: Arc::clone(&counts.bytes),
        requests: Arc::clone(&counts.requests),
        sim: SimStore::from_env(),
        name: path.display().to_string(),
    });
    let file = session
        .open_options()
        .with_include_metadata(metadata)
        .open(source)
        .await?;
    Ok((file, counts))
}

/// The bytes and requests a file was asked for.
#[derive(Clone, Default)]
struct Counts {
    bytes: Arc<AtomicU64>,
    requests: Arc<AtomicU64>,
}

impl Counts {
    fn take(&self) -> (u64, u64) {
        (
            self.bytes.load(Ordering::Relaxed),
            self.requests.load(Ordering::Relaxed),
        )
    }
}

fn file_len(path: &Path) -> VortexResult<u64> {
    Ok(std::fs::metadata(path)?.len())
}

fn kind_file(dir: &Path, kind: &str) -> PathBuf {
    dir.join(format!("{kind}.vxi"))
}

/// Read a single-column parquet file into Vortex arrays.
fn read_parquet(session: &VortexSession, path: &str) -> VortexResult<(DType, Vec<ArrayRef>)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)
        .map_err(|e| vortex_err!("{e}"))?
        .with_batch_size(64 * 1024)
        .build()
        .map_err(|e| vortex_err!("{e}"))?;
    let schema = arrow_array::RecordBatchReader::schema(&reader);
    let mut chunks = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| vortex_err!("{e}"))?;
        chunks.push(session.arrow().from_arrow_record_batch(batch, &schema)?);
    }
    let dtype = chunks
        .first()
        .map(|c| c.dtype().clone())
        .ok_or_else(|| vortex_err!("empty input"))?;
    Ok((dtype, chunks))
}

async fn build(session: &VortexSession, args: &[String]) -> VortexResult<()> {
    let [kind, input, column, outdir, rest @ ..] = args else {
        vortex_bail!("usage: build <kind> <input.parquet> <column> <outdir> [lists]");
    };
    let outdir = PathBuf::from(outdir);
    std::fs::create_dir_all(&outdir)?;
    let data_path = outdir.join("data.vortex");

    let (dtype, chunks) = read_parquet(session, input)?;
    let start = Instant::now();
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let mut options = session.write_options();
    if let Some(rows) = std::env::var("DATA_ROW_BLOCK")
        .ok()
        .and_then(|rows| rows.parse().ok())
    {
        options = options.with_strategy(
            WriteStrategyBuilder::from_session(session)
                .with_row_block_size(rows)
                .build(),
        );
    }
    options
        .write(
            &mut tokio::fs::File::create(&data_path).await?,
            array.to_array_stream(),
        )
        .await?;
    let write_secs = start.elapsed().as_secs_f64();

    let (data, _) = open(session, &data_path, false).await?;
    let source_dtype = data
        .dtype()
        .as_struct_fields()
        .field(column.as_str())
        .ok_or_else(|| vortex_err!("no column {column}"))?;
    let index_path = kind_file(&outdir, kind);
    let start = Instant::now();
    match kind.as_str() {
        "trigram" => {
            let mut b = IndexFileBuilder::new(
                TrigramIndex,
                TrigramOptions {
                    block_len: std::env::var("BLOCK_LEN")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                },
                column.as_str(),
                source_dtype,
                session,
            )?;
            b.add_file("data.vortex", &data).await?;
            b.write(&mut tokio::fs::File::create(&index_path).await?)
                .await?;
        }
        "fts" => {
            let mut b =
                IndexFileBuilder::new(InvertedIndex, (), column.as_str(), source_dtype, session)?;
            b.add_file("data.vortex", &data).await?;
            b.write(&mut tokio::fs::File::create(&index_path).await?)
                .await?;
        }
        "ivf" => {
            let lists = rest
                .first()
                .map(|l| l.parse())
                .transpose()
                .map_err(|e| vortex_err!("{e}"))?;
            let mut b = IndexFileBuilder::new(
                IvfIndex,
                IvfOptions {
                    lists: lists.unwrap_or(0),
                    dim: 0,
                },
                column.as_str(),
                source_dtype,
                session,
            )?;
            b.add_file("data.vortex", &data).await?;
            b.write(&mut tokio::fs::File::create(&index_path).await?)
                .await?;
        }
        other => vortex_bail!("unknown kind {other}"),
    }
    let build_secs = start.elapsed().as_secs_f64();
    let (index, _) = open(session, &index_path, false).await?;
    let chunk_rows = |f: &VortexFile| -> VortexResult<Vec<u64>> {
        Ok(f.splits()?.iter().map(|r| r.end - r.start).collect())
    };
    if std::env::var_os("SHOW_LAYOUT").is_some() {
        eprintln!(
            "{}",
            index
                .footer()
                .layout()
                .display_tree_with_segments(index.segment_source())
                .await?
        );
    }
    let data_chunks = chunk_rows(&data)?;
    let index_chunks = chunk_rows(&index)?;
    println!(
        "{}",
        serde_json::json!({
            "kind": kind,
            "rows": data.row_count(),
            "data_bytes": file_len(&data_path)?,
            "index_bytes": file_len(&index_path)?,
            "data_write_secs": write_secs,
            "index_build_secs": build_secs,
            "data_chunks": data_chunks.len(),
            "data_chunk_rows_max": data_chunks.iter().max(),
            "index_rows": index.row_count(),
            "index_chunks": index_chunks.len(),
        })
    );
    Ok(())
}

struct QueryArgs {
    repeat: usize,
    only: Option<usize>,
    scan: bool,
    index_only: bool,
    top_k_only: bool,
    e2e: bool,
    nprobe: usize,
    k: usize,
}

fn parse_flags(rest: &[String]) -> VortexResult<QueryArgs> {
    let mut q = QueryArgs {
        repeat: 5,
        only: None,
        scan: false,
        index_only: false,
        top_k_only: false,
        e2e: false,
        nprobe: 8,
        k: 10,
    };
    let mut it = rest.iter();
    while let Some(flag) = it.next() {
        let mut value = || -> VortexResult<usize> {
            it.next()
                .ok_or_else(|| vortex_err!("{flag} needs a value"))?
                .parse()
                .map_err(|e| vortex_err!("{e}"))
        };
        match flag.as_str() {
            "--repeat" => q.repeat = value()?,
            "--only" => q.only = Some(value()?),
            "--nprobe" => q.nprobe = value()?,
            "--k" => q.k = value()?,
            "--scan" => q.scan = true,
            "--index-only" => q.index_only = true,
            "--top-k-only" => q.top_k_only = true,
            "--e2e" => q.e2e = true,
            other => vortex_bail!("unknown flag {other}"),
        }
    }
    Ok(q)
}

/// Count rows of `data` passing `filter` within `selection`.
async fn count(data: &VortexFile, filter: &Expression, selection: Selection) -> VortexResult<u64> {
    let chunks = collect(
        ScanBuilder::new(data.session().clone(), data.scan_file())
            .with_some_filter(Some(filter.bind(data.dtype())?))
            .with_selection(selection),
    )
    .await?;
    Ok(chunks.iter().map(|c| c.len() as u64).sum())
}

/// An opened index file of the benchmarked kind.
enum Opened {
    Trigram(IndexFile<TrigramIndex>),
    Fts(IndexFile<InvertedIndex>),
    Ivf(IndexFile<IvfIndex>),
}

impl Opened {
    fn open(kind: &str, file: &VortexFile) -> VortexResult<Self> {
        Ok(match kind {
            "trigram" => Self::Trigram(IndexFile::open(TrigramIndex, file)?),
            "fts" => Self::Fts(IndexFile::open(InvertedIndex, file)?),
            "ivf" => Self::Ivf(IndexFile::open(IvfIndex, file)?),
            other => vortex_bail!("unknown index kind {other}"),
        })
    }
}

/// One query's answer, as JSON.
type Answer = serde_json::Value;

/// Probe the trigram index alone: the rows its candidate blocks cover, with no data read.
async fn run_trigram_index(index: &IndexFile<TrigramIndex>, pattern: &str) -> VortexResult<Answer> {
    let candidates: u64 = match index.locate_files(&like(root(), lit(pattern))).await? {
        Some((_, selections)) => selections
            .iter()
            .map(|s| s.selection.row_count(u64::MAX))
            .sum(),
        None => u64::MAX,
    };
    Ok(serde_json::json!({ "candidate_rows": candidates }))
}

async fn run_trigram(
    data: &VortexFile,
    index: Option<&IndexFile<TrigramIndex>>,
    pattern: &str,
    column: &str,
) -> VortexResult<Answer> {
    let filter = like(col(column), lit(pattern));
    let rows = match index {
        None => count(data, &filter, Selection::All).await?,
        Some(index) => match index.locate_files(&like(root(), lit(pattern))).await? {
            None => count(data, &filter, Selection::All).await?,
            Some((_, selections)) => {
                let mut rows = 0;
                for s in selections {
                    rows += count(data, &filter, s.selection).await?;
                }
                rows
            }
        },
    };
    Ok(serde_json::json!({ "rows": rows }))
}

/// Count the rows holding every query token, unless `top_k_only`, then rank the top `k` by BM25.
/// With `fetch`, also read the text of the top `k` rows from the data file.
async fn run_fts(
    data: &VortexFile,
    index: &IndexFile<InvertedIndex>,
    query: &str,
    k: usize,
    top_k_only: bool,
    fetch: bool,
) -> VortexResult<Answer> {
    let (hits, matches) = if top_k_only {
        (index.rows().search_bm25(query, k).await?, 0)
    } else if std::env::var_os("LOCATE_COUNT").is_some() {
        // The count through the generic index planner, as a table scan's filter would get it.
        let mut matches = 0;
        if let Some((_, selections)) = index.locate_files(&text_match(root(), query)).await? {
            matches = selections
                .iter()
                .map(|s| s.selection.row_count(u64::MAX))
                .sum::<u64>();
        }
        (index.rows().search_bm25(query, k).await?, matches)
    } else {
        index.rows().search_bm25_counted(query, k).await?
    };
    let mut fetched_bytes = 0u64;
    if fetch && !hits.is_empty() {
        let mut rows: Vec<u64> = index.hits_by_file(&hits)?.iter().map(|h| h.row).collect();
        rows.sort_unstable();
        let column = index.column().to_string();
        let texts = collect(
            ScanBuilder::new(data.session().clone(), data.scan_file())
                .with_projection(col(column.as_str()).bind(data.dtype())?)
                .with_selection(Selection::IncludeByIndex(StrictSortedBuffer::try_new(
                    rows.into_iter().collect(),
                )?)),
        )
        .await?;
        fetched_bytes = texts.iter().map(|t| t.nbytes()).sum();
    }
    Ok(serde_json::json!({
        "fetched_bytes": fetched_bytes,
        "matches": matches,
        "ids": hits.iter().map(|h| h.row).collect::<Vec<_>>(),
    }))
}

async fn run_ivf(
    index: &IndexFile<IvfIndex>,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> VortexResult<Answer> {
    let hits = index.rows().search_vectors(query, k, nprobe).await?;
    Ok(serde_json::json!({ "ids": hits.iter().map(|h| h.row).collect::<Vec<_>>() }))
}

/// Queries: a JSON array of strings, or raw little-endian f32 vectors of the index's dimension.
fn load_text_queries(path: &str) -> VortexResult<Vec<String>> {
    serde_json::from_slice(&std::fs::read(path)?).map_err(|e| vortex_err!("{e}"))
}

fn load_vector_queries(path: &str, dim: usize) -> VortexResult<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let floats: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(floats.chunks_exact(dim).map(<[f32]>::to_vec).collect())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

async fn query(session: &VortexSession, args: &[String]) -> VortexResult<()> {
    let [kind, outdir, queries, rest @ ..] = args else {
        vortex_bail!("usage: query <kind> <outdir> <queries> [flags]");
    };
    let flags = parse_flags(rest)?;
    let outdir = PathBuf::from(outdir);
    let data_path = outdir.join("data.vortex");
    let index_path = kind_file(&outdir, kind);

    let texts = match kind.as_str() {
        "trigram" | "fts" => load_text_queries(queries)?,
        _ => Vec::new(),
    };
    let vectors = if kind == "ivf" {
        let (index, _) = open(session, &index_path, true).await?;
        let dim = IndexFile::open(IvfIndex, &index)?.rows().options().dim as usize;
        load_vector_queries(queries, dim)?
    } else {
        Vec::new()
    };
    let count_queries = texts.len().max(vectors.len());

    // Files kept open across warm runs, as a long-lived reader would.
    let (warm_data, _) = open(session, &data_path, false).await?;
    let (warm_index, _) = open(session, &index_path, true).await?;
    // Opened once, as a long-lived reader would, so caches it fills survive across warm runs.
    let warm = Opened::open(kind, &warm_index)?;

    for qi in 0..count_queries {
        if flags.only.is_some_and(|only| only != qi) {
            continue;
        }
        let mut times = Vec::new();
        let mut answer = serde_json::Value::Null;
        let mut bytes = ((0, 0), (0, 0));
        let mut cold_ms = 0.0;
        // The first run opens fresh files and counts their bytes; later runs measure warm latency
        // on files that stay open.
        for run in 0..=flags.repeat {
            let fresh = if run == 0 {
                Some((
                    open(session, &data_path, false).await?,
                    open(session, &index_path, true).await?,
                ))
            } else {
                None
            };
            let fresh_index = match &fresh {
                Some((_, (index, _))) => Some(Opened::open(kind, index)?),
                None => None,
            };
            let (data, index) = match (&fresh, &fresh_index) {
                (Some(((data, _), _)), Some(index)) => (data, index),
                _ => (&warm_data, &warm),
            };
            let start = Instant::now();
            if std::env::var_os("TRACE_READ_TIMES").is_some() {
                eprintln!("query_start us={:.1}", trace_clock());
            }
            answer = match index {
                Opened::Trigram(index) => {
                    if flags.index_only {
                        run_trigram_index(index, &texts[qi]).await?
                    } else {
                        let column = index.column().to_string();
                        run_trigram(data, (!flags.scan).then_some(index), &texts[qi], &column)
                            .await?
                    }
                }
                Opened::Fts(index) => {
                    run_fts(
                        data,
                        index,
                        &texts[qi],
                        flags.k,
                        flags.top_k_only,
                        flags.e2e,
                    )
                    .await?
                }
                Opened::Ivf(index) => run_ivf(index, &vectors[qi], flags.k, flags.nprobe).await?,
            };
            match &fresh {
                Some(((_, data_counts), (_, index_counts))) => {
                    cold_ms = start.elapsed().as_secs_f64() * 1000.0;
                    bytes = (data_counts.take(), index_counts.take());
                }
                None => {
                    times.push(start.elapsed().as_secs_f64() * 1000.0);
                    if std::env::var_os("TRACE_READ_TIMES").is_some() {
                        eprintln!("query_end us={:.1}", trace_clock());
                    }
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "q": qi,
                "answer": answer,
                "warm_ms": if times.is_empty() { 0.0 } else { median(times) },
                "cold_ms": cold_ms,
                "data_bytes_read": bytes.0.0,
                "data_requests": bytes.0.1,
                "index_bytes_read": bytes.1.0,
                "index_requests": bytes.1.1,
            })
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    let session = VortexSession::default().with_tokio();
    vortex_index::initialize(&session)?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => build(&session, &args[1..]).await,
        Some("query") => query(&session, &args[1..]).await,
        _ => vortex_bail!("usage: index_bench <build|query> ..."),
    }
}
