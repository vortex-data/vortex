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
//!                   [--nprobe N] [--k K]
//! ```
//!
//! `build` writes `<outdir>/data.vortex` holding the column and a standalone index file
//! `<outdir>/<kind>.vxi` over it, and prints sizes and timings as JSON. `query` prints one JSON
//! line per query with its result, the median warm latency, and the exact bytes Vortex read from
//! each file on a fresh open. `--scan` answers trigram queries without the index.

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
use vortex_file::v2::collect;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::InvertedIndex;
use vortex_index::IvfIndex;
use vortex_index::IvfOptions;
use vortex_index::TrigramIndex;
use vortex_index::TrigramOptions;
use vortex_index::text_match;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::std_file::FileReadAt;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

/// A file source that counts the bytes it is asked for.
struct CountingReadAt {
    inner: Arc<dyn VortexReadAt>,
    bytes: Arc<AtomicU64>,
    name: String,
}

impl VortexReadAt for CountingReadAt {
    /// `NO_COALESCE=1` turns off merging of nearby reads, so the count is the bytes the scan
    /// actually needs rather than what the local-file read policy fetches around them.
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        if std::env::var_os("NO_COALESCE").is_some() {
            return None;
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
        if std::env::var_os("TRACE_READS").is_some() {
            eprintln!("read {} {offset} {length}", self.name);
        }
        self.inner.read_at(offset, length, alignment)
    }
}

async fn open(
    session: &VortexSession,
    path: &Path,
    metadata: bool,
) -> VortexResult<(VortexFile, Arc<AtomicU64>)> {
    let bytes = Arc::new(AtomicU64::new(0));
    let source = Arc::new(CountingReadAt {
        inner: Arc::new(FileReadAt::open(path, session.handle())?),
        bytes: Arc::clone(&bytes),
        name: path.display().to_string(),
    });
    let file = session
        .open_options()
        .with_include_metadata(metadata)
        .open(source)
        .await?;
    Ok((file, bytes))
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
    session
        .write_options()
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
    nprobe: usize,
    k: usize,
}

fn parse_flags(rest: &[String]) -> VortexResult<QueryArgs> {
    let mut q = QueryArgs {
        repeat: 5,
        only: None,
        scan: false,
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

/// One query's answer, as JSON.
type Answer = serde_json::Value;

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

async fn run_fts(index: &IndexFile<InvertedIndex>, query: &str, k: usize) -> VortexResult<Answer> {
    let matches = match index.locate_files(&text_match(root(), query)).await? {
        Some((_, selections)) => selections
            .iter()
            .map(|s| s.selection.row_count(u64::MAX))
            .sum(),
        None => 0,
    };
    let hits = index.rows().search_bm25(query, k).await?;
    Ok(serde_json::json!({
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

    for qi in 0..count_queries {
        if flags.only.is_some_and(|only| only != qi) {
            continue;
        }
        let mut times = Vec::new();
        let mut answer = serde_json::Value::Null;
        let mut bytes = (0, 0);
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
            let (data, index_file) = match &fresh {
                Some(((data, _), (index, _))) => (data, index),
                None => (&warm_data, &warm_index),
            };
            let start = Instant::now();
            answer = match kind.as_str() {
                "trigram" => {
                    let index = IndexFile::open(TrigramIndex, index_file)?;
                    let column = index.column().to_string();
                    run_trigram(data, (!flags.scan).then_some(&index), &texts[qi], &column).await?
                }
                "fts" => {
                    let index = IndexFile::open(InvertedIndex, index_file)?;
                    run_fts(&index, &texts[qi], flags.k).await?
                }
                _ => {
                    let index = IndexFile::open(IvfIndex, index_file)?;
                    run_ivf(&index, &vectors[qi], flags.k, flags.nprobe).await?
                }
            };
            match &fresh {
                Some(((_, data_bytes), (_, index_bytes))) => {
                    bytes = (
                        data_bytes.load(Ordering::Relaxed),
                        index_bytes.load(Ordering::Relaxed),
                    );
                }
                None => times.push(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "q": qi,
                "answer": answer,
                "warm_ms": if times.is_empty() { 0.0 } else { median(times) },
                "data_bytes_read": bytes.0,
                "index_bytes_read": bytes.1,
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
