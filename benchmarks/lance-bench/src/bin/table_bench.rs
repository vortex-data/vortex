// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Multi-file table scans: a Vortex table (`vortex-table`, a manifest over Vortex files, run by a
//! plan driver) against a Lance dataset holding the same rows in the same number of files.
//!
//! Each system answers each workload with its own native API rather than SQL:
//!
//! | id   | workload                                                       |
//! |------|----------------------------------------------------------------|
//! | `a1` | full scan of one column, summed                                |
//! | `a2` | narrow time range: most files pruned on statistics             |
//! | `a3` | low-cardinality equality filter over every file                |
//! | `a4` | selective filter that no statistics can prune, rows counted    |
//! | `d1` | data loader: batches of uniformly random rows (id, embedding)  |
//! | `d2` | data loader: random contiguous windows of rows                 |
//! | `v1` | exact vector top-10 by dot product, no index                   |
//! | `t1` | substring search for a rare token, no index, rows counted      |
//!
//! With `--indexes`, both systems also build indexes once and answer:
//!
//! | id   | workload                                                                 |
//! |------|--------------------------------------------------------------------------|
//! | `v2` | approximate cosine top-10 through an IVF index (Lance IVF_SQ, Vortex IVF) |
//! | `t2` | BM25 top-10 for a two-word query through an inverted index               |
//! | `t3` | every row holding a term, through an inverted index, rows counted        |
//!
//! Vortex reads its indexes from standalone index files bound to the manifest
//! (`vortex_table::index`), Lance from its dataset's own indexes. Approximate answers are
//! reported as recall against the exact answer rather than compared.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use clap::Parser;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance::dataset::WriteParams;
use lance::dataset::scanner::Scanner;
use lance::deps::arrow_array as la;
use lance::deps::arrow_array::Array as _;
use lance::deps::arrow_schema as ls;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault as _;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::expr::Expression;
use vortex::array::expr::and;
use vortex::array::expr::col;
use vortex::array::expr::eq;
use vortex::array::expr::gt_eq;
use vortex::array::expr::like;
use vortex::array::expr::lit;
use vortex::array::expr::lt;
use vortex::array::expr::root;
use vortex::array::expr::select;
use vortex::array::validity::Validity;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::planning::drive::MapFn;
use vortex::scan::planning::drive::ParallelDriver;
use vortex::scan::planning::drive::PlanDriver;
use vortex::scan::planning::drive::PlanRoot;
use vortex::scan::planning::drive::ScopedBatch;
use vortex::scan::planning::drive::SequentialDriver;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_table::FileCache;
use vortex_table::LocalStorage;
use vortex_table::Manifest;
use vortex_table::ManifestBuilder;
use vortex_table::TableScan;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const VOCABULARY: u32 = 20_000;
const WORDS_PER_DOC: usize = 12;
const CATEGORIES: u32 = 50;

#[derive(Parser)]
struct Args {
    /// Rows in the table.
    #[arg(long, default_value_t = 1_000_000)]
    rows: u64,
    /// Data files the rows are split over, in both systems.
    #[arg(long, default_value_t = 16)]
    files: u64,
    /// Embedding dimensions.
    #[arg(long, default_value_t = 64)]
    dims: usize,
    /// Timed runs of each workload; the median is reported.
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    /// Where both tables are written.
    #[arg(long, default_value = "data/table-bench")]
    dir: PathBuf,
    /// Workers for the Vortex parallel driver; one per core by default.
    #[arg(long)]
    workers: Option<usize>,
    /// Run the Vortex table with the sequential driver instead.
    #[arg(long)]
    sequential: bool,
    /// Only these workloads, such as `a1,d1`.
    #[arg(short, long, value_delimiter = ',')]
    workloads: Option<Vec<String>>,
    /// Build indexes on both systems and run the indexed workloads.
    #[arg(long)]
    indexes: bool,
    /// IVF lists (Vortex) and partitions (Lance).
    #[arg(long, default_value_t = 256)]
    ivf_lists: usize,
    /// IVF lists searched per query.
    #[arg(long, default_value_t = 16)]
    nprobe: usize,
    /// Approximate top-k candidates per result that both systems re-score by the stored vectors;
    /// zero ranks by the index's quantized vectors alone.
    #[arg(long, default_value_t = 4)]
    refine: usize,
    /// Distinct query vectors the vector search rotates through.
    #[arg(long, default_value_t = 200)]
    queries: usize,
    /// Timed passes over the query vectors, after one untimed warm-up pass.
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Fresh opens timed for the cold first query.
    #[arg(long, default_value_t = 5)]
    cold_opens: usize,
    /// Inline footers into the Vortex manifest.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    inline_footers: bool,
}

/// The rows of one data file, generated identically for both systems.
struct FileRows {
    id: Vec<u64>,
    ts: Vec<i64>,
    category: Vec<u32>,
    price: Vec<f64>,
    text: Vec<String>,
    emb: Vec<f32>,
}

fn word(rng: &mut StdRng) -> u32 {
    // Skewed towards low ids, so a few words are common and most are rare.
    let u: f64 = rng.random();
    let rank = (u * u * u * f64::from(VOCABULARY)).floor();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "in 0..VOCABULARY"
    )]
    let rank = rank as u32;
    rank.min(VOCABULARY - 1)
}

fn generate(file: u64, rows_per_file: u64, dims: usize) -> FileRows {
    let mut rng = StdRng::seed_from_u64(file);
    let ids: Vec<u64> = (file * rows_per_file..(file + 1) * rows_per_file).collect();
    let n = ids.len();
    FileRows {
        ts: ids
            .iter()
            .map(|&id| i64::try_from(id).unwrap_or(i64::MAX) * 10 + rng.random_range(0..10))
            .collect(),
        category: (0..n).map(|_| rng.random_range(0..CATEGORIES)).collect(),
        price: (0..n).map(|_| rng.random_range(1.0..1000.0)).collect(),
        text: (0..n)
            .map(|_| {
                (0..WORDS_PER_DOC)
                    .map(|_| format!("q{}q", word(&mut rng)))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect(),
        emb: (0..n * dims)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect(),
        id: ids,
    }
}

fn vortex_array(rows: &FileRows, dims: usize) -> anyhow::Result<ArrayRef> {
    let emb = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter(rows.emb.iter().copied()).into_array(),
        u32::try_from(dims)?,
        Validity::NonNullable,
        rows.id.len(),
    )?;
    Ok(StructArray::from_fields(&[
        (
            "id",
            PrimitiveArray::from_iter(rows.id.iter().copied()).into_array(),
        ),
        (
            "ts",
            PrimitiveArray::from_iter(rows.ts.iter().copied()).into_array(),
        ),
        (
            "category",
            PrimitiveArray::from_iter(rows.category.iter().copied()).into_array(),
        ),
        (
            "price",
            PrimitiveArray::from_iter(rows.price.iter().copied()).into_array(),
        ),
        ("text", VarBinArray::from(rows.text.clone()).into_array()),
        ("emb", emb.into_array()),
    ])?
    .into_array())
}

fn lance_batch(rows: &FileRows, dims: usize) -> anyhow::Result<la::RecordBatch> {
    let item = Arc::new(ls::Field::new("item", ls::DataType::Float32, true));
    let emb = la::FixedSizeListArray::try_new(
        Arc::clone(&item),
        i32::try_from(dims)?,
        Arc::new(la::Float32Array::from(rows.emb.clone())),
        None,
    )?;
    let schema = Arc::new(ls::Schema::new(vec![
        ls::Field::new("id", ls::DataType::UInt64, false),
        ls::Field::new("ts", ls::DataType::Int64, false),
        ls::Field::new("category", ls::DataType::UInt32, false),
        ls::Field::new("price", ls::DataType::Float64, false),
        ls::Field::new("text", ls::DataType::Utf8, false),
        ls::Field::new(
            "emb",
            ls::DataType::FixedSizeList(item, i32::try_from(dims)?),
            false,
        ),
    ]));
    Ok(la::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(la::UInt64Array::from(rows.id.clone())),
            Arc::new(la::Int64Array::from(rows.ts.clone())),
            Arc::new(la::UInt32Array::from(rows.category.clone())),
            Arc::new(la::Float64Array::from(rows.price.clone())),
            Arc::new(la::StringArray::from(rows.text.clone())),
            Arc::new(emb),
        ],
    )?)
}

/// Writes both tables under `dir` unless they exist already.
async fn prepare(args: &Args, session: &VortexSession) -> anyhow::Result<()> {
    let marker = args.dir.join(format!(
        "ready-{}-{}-{}-{}",
        args.rows, args.files, args.dims, args.inline_footers
    ));
    if marker.exists() {
        return Ok(());
    }
    if args.dir.exists() {
        fs::remove_dir_all(&args.dir)?;
    }
    let vortex_dir = args.dir.join("vortex");
    fs::create_dir_all(vortex_dir.join("data"))?;
    let rows_per_file = args.rows.div_ceil(args.files);
    let mut builder =
        ManifestBuilder::new(session.clone()).with_inline_footers(args.inline_footers);
    let mut lance: Option<Dataset> = None;
    let lance_uri = args.dir.join("lance").to_string_lossy().to_string();
    for file in 0..args.files {
        let rows = generate(file, rows_per_file, args.dims);
        let mut buf = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut buf, vortex_array(&rows, args.dims)?.to_array_stream())
            .await?;
        let bytes: ByteBuffer = buf.freeze();
        let path = format!("data/part-{file:05}.vortex");
        fs::write(vortex_dir.join(&path), bytes.as_slice())?;
        builder.add(path, &bytes).await?;

        let batch = lance_batch(&rows, args.dims)?;
        let reader = la::RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        let params = WriteParams {
            max_rows_per_file: usize::MAX,
            mode: if lance.is_some() {
                lance::dataset::WriteMode::Append
            } else {
                lance::dataset::WriteMode::Create
            },
            ..Default::default()
        };
        lance = Some(Dataset::write(reader, lance_uri.as_str(), Some(params)).await?);
    }
    let mut manifest = ByteBufferMut::empty();
    builder.finish()?.write(session, &mut manifest).await?;
    fs::write(
        vortex_dir.join("manifest.vortex"),
        manifest.freeze().as_slice(),
    )?;
    fs::write(marker, b"")?;
    Ok(())
}

/// What a workload returns, compared across systems to show both did the same work.
#[derive(Debug, PartialEq)]
enum Answer {
    Rows(u64),
    /// Rows, and their sum rounded to a whole number, so that both systems agree despite summing
    /// in different orders.
    Sum(u64, i64),
    TopK(Vec<u64>),
}

#[expect(clippy::cast_possible_truncation, reason = "sums of prices fit an i64")]
fn rounded(sum: f64) -> i64 {
    sum.round() as i64
}

#[derive(Clone)]
struct VortexTable {
    session: VortexSession,
    storage: Arc<LocalStorage>,
    manifest: Arc<Manifest>,
    cache: Arc<FileCache>,
    driver: Driver,
    /// Yields batches by file, then row, for lookups whose output must line up with their input.
    ordered: Driver,
}

/// The Vortex plan driver a run uses.
#[derive(Clone)]
enum Driver {
    Sequential(SequentialDriver),
    Parallel(ParallelDriver),
}

impl Driver {
    fn as_dyn(&self) -> &dyn PlanDriver {
        match self {
            Self::Sequential(driver) => driver,
            Self::Parallel(driver) => driver,
        }
    }

    /// Runs `root`, applying `map_fn` to each batch where it is produced.
    fn execute_with<A: Send + 'static>(
        &self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        match self {
            Self::Sequential(driver) => driver.execute_with(root, map_fn),
            Self::Parallel(driver) => driver.execute_with(root, map_fn),
        }
    }
}

impl VortexTable {
    fn scan(&self) -> TableScan {
        TableScan::new(
            self.session.clone(),
            Arc::clone(&self.storage) as _,
            Arc::clone(&self.manifest),
        )
        .with_file_cache(Arc::clone(&self.cache))
    }

    async fn collect(&self, scan: TableScan) -> anyhow::Result<Vec<ScopedBatch>> {
        Ok(scan.execute(self.driver.as_dyn()).try_collect().await?)
    }

    fn column(&self, batch: ArrayRef, name: &str) -> anyhow::Result<PrimitiveArray> {
        let mut ctx = self.session.create_execution_ctx();
        Ok(batch
            .execute::<StructArray>(&mut ctx)?
            .unmasked_field_by_name(name)?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?)
    }

    async fn sum_price(&self, scan: TableScan) -> anyhow::Result<Answer> {
        let mut rows = 0;
        let mut sum = 0.0;
        for batch in self
            .collect(scan.with_projection(select(["price"], root())))
            .await?
        {
            let price = self.column(batch.batch, "price")?;
            rows += price.len() as u64;
            sum += price.as_slice::<f64>().iter().sum::<f64>();
        }
        Ok(Answer::Sum(rows, rounded(sum)))
    }

    async fn count(&self, scan: TableScan) -> anyhow::Result<Answer> {
        let batches = self
            .collect(scan.with_projection(select(["id"], root())))
            .await?;
        Ok(Answer::Rows(
            batches.iter().map(|b| b.batch.len() as u64).sum(),
        ))
    }

    /// Takes table rows by sorted global index, reading only the files that hold them.
    async fn take(&self, rows: &[u64]) -> anyhow::Result<Answer> {
        let mut starts = Vec::new();
        let mut start = 0;
        for entry in self.manifest.entries() {
            starts.push(start);
            start += entry.row_count;
        }
        let mut by_file: HashMap<u64, Vec<u64>> = HashMap::default();
        for &row in rows {
            let file = starts.partition_point(|&s| s <= row) - 1;
            by_file
                .entry(file as u64)
                .or_default()
                .push(row - starts[file]);
        }
        let selections = by_file
            .into_iter()
            .map(|(file, rows)| {
                Ok((
                    file,
                    Selection::IncludeByIndex(StrictSortedBuffer::try_new(rows.into())?),
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let scan = self
            .scan()
            .with_projection(select(["id", "emb"], root()))
            .with_file_selections(selections);
        let batches = self.collect(scan).await?;
        Ok(Answer::Rows(
            batches.iter().map(|b| b.batch.len() as u64).sum(),
        ))
    }

    async fn top_k(&self, query: &[f32], k: usize) -> anyhow::Result<Answer> {
        let scan = self.scan().with_projection(select(["id", "emb"], root()));
        let query: Arc<[f32]> = query.into();
        let session = self.session.clone();
        // Each batch is scored, and cut to its best k, on the worker that decoded it.
        let score: MapFn<Vec<(f32, u64)>> = Arc::new(move |batch| {
            let mut ctx = session.create_execution_ctx();
            let array = batch.execute::<StructArray>(&mut ctx)?;
            let ids = array
                .unmasked_field_by_name("id")?
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            let emb = array
                .unmasked_field_by_name("emb")?
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let values = emb.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
            let mut best: Vec<(f32, u64)> = values
                .as_slice::<f32>()
                .chunks_exact(query.len())
                .zip(ids.as_slice::<u64>())
                .map(|(vector, &id)| (dot(vector, &query), id))
                .collect();
            keep_best(&mut best, k);
            Ok(best)
        });
        let mut best: Vec<(f32, u64)> = Vec::new();
        let mut batches = self.driver.execute_with(scan.plan(), score);
        while let Some(batch) = batches.try_next().await? {
            best.extend(batch.batch);
            keep_best(&mut best, k);
        }
        Ok(Answer::TopK(best.into_iter().map(|(_, id)| id).collect()))
    }
}

/// A dot product over eight lanes, which the compiler turns into SIMD.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (a_chunks, a_rest) = a.as_chunks::<8>();
    let (b_chunks, b_rest) = b.as_chunks::<8>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    let rest: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + rest
}

fn keep_best(best: &mut Vec<(f32, u64)>, k: usize) {
    let order = |a: &(f32, u64), b: &(f32, u64)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if best.len() > k {
        best.select_nth_unstable_by(k, order);
        best.truncate(k);
    }
    best.sort_by(order);
}

async fn lance_sum(scanner: &mut Scanner) -> anyhow::Result<Answer> {
    scanner.project(&["price"])?;
    let mut stream = scanner.try_into_stream().await?;
    let (mut rows, mut sum) = (0u64, 0.0);
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let price = batch
            .column(0)
            .as_any()
            .downcast_ref::<la::Float64Array>()
            .context("price is f64")?;
        rows += price.len() as u64;
        sum += price.values().iter().sum::<f64>();
    }
    Ok(Answer::Sum(rows, rounded(sum)))
}

async fn lance_count(scanner: &mut Scanner) -> anyhow::Result<Answer> {
    scanner.project(&["id"])?;
    let batches: Vec<la::RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    Ok(Answer::Rows(
        batches.iter().map(|b| b.num_rows() as u64).sum(),
    ))
}

async fn lance_top_k(dataset: &Dataset, query: &[f32], k: usize) -> anyhow::Result<Answer> {
    let mut scanner = dataset.scan();
    scanner.project(&["id"])?;
    scanner.nearest("emb", &la::Float32Array::from(query.to_vec()), k)?;
    scanner.distance_metric(lance_linalg::distance::MetricType::Dot);
    let batches: Vec<la::RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .context("id")?
            .as_any()
            .downcast_ref::<la::UInt64Array>()
            .context("id is u64")?
            .clone();
        ids.extend(column.values().iter().copied());
    }
    Ok(Answer::TopK(ids))
}

/// Inputs drawn once per run, shared by both systems.
struct Inputs {
    random_rows: Vec<Vec<u64>>,
    windows: Vec<Vec<u64>>,
    query: Vec<f32>,
    /// Distinct query vectors for the indexed vector search.
    queries: Vec<Vec<f32>>,
    ts_range: (i64, i64),
    category: u32,
    rare_token: String,
    /// A term in roughly one row in a thousand.
    term: String,
    /// Two mid-frequency terms.
    bm25_query: String,
}

fn inputs(args: &Args) -> Inputs {
    let mut rng = StdRng::seed_from_u64(42);
    let batches = 8;
    let batch_rows = 256;
    let random_rows = (0..batches)
        .map(|_| {
            let mut rows: Vec<u64> = (0..batch_rows)
                .map(|_| rng.random_range(0..args.rows))
                .collect();
            rows.sort_unstable();
            rows.dedup();
            rows
        })
        .collect();
    let window = 1024.min(args.rows);
    let windows = (0..batches)
        .map(|_| {
            let start = rng.random_range(0..=args.rows - window);
            (start..start + window).collect()
        })
        .collect();
    let width = i64::try_from(args.rows / 100).unwrap_or(1).max(1) * 10;
    let start = rng.random_range(0..i64::try_from(args.rows).unwrap_or(1) * 10 - width);
    Inputs {
        random_rows,
        windows,
        query: (0..args.dims)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect(),
        queries: {
            let mut rng = StdRng::seed_from_u64(7);
            (0..args.queries)
                .map(|_| {
                    (0..args.dims)
                        .map(|_| rng.random_range(-1.0f32..1.0))
                        .collect()
                })
                .collect()
        },
        ts_range: (start, start + width),
        category: 7,
        rare_token: format!("q{}q", VOCABULARY - 3),
        term: format!("q{}q", VOCABULARY / 2),
        bm25_query: format!("q{}q q{}q", VOCABULARY / 4, VOCABULARY / 3),
    }
}

async fn time<F, Fut>(iterations: usize, mut f: F) -> anyhow::Result<(Duration, Answer)>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<Answer>>,
{
    let answer = f().await?;
    let mut times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        f().await?;
        times.push(start.elapsed());
    }
    times.sort();
    Ok((times[times.len() / 2], answer))
}

fn vortex_filter(workload: &str, inputs: &Inputs) -> Option<Expression> {
    let (lo, hi) = inputs.ts_range;
    match workload {
        "a2" => Some(and(gt_eq(col("ts"), lit(lo)), lt(col("ts"), lit(hi)))),
        "a3" => Some(eq(col("category"), lit(inputs.category))),
        "a4" => Some(gt_eq(col("price"), lit(990.0f64))),
        "t1" => Some(like(col("text"), lit(format!("%{}%", inputs.rare_token)))),
        _ => None,
    }
}

fn lance_filter(workload: &str, inputs: &Inputs) -> Option<String> {
    let (lo, hi) = inputs.ts_range;
    match workload {
        "a2" => Some(format!("ts >= {lo} AND ts < {hi}")),
        "a3" => Some(format!("category = {}", inputs.category)),
        "a4" => Some("price >= 990.0".to_string()),
        "t1" => Some(format!("text LIKE '%{}%'", inputs.rare_token)),
        _ => None,
    }
}

async fn run_vortex(
    table: &VortexTable,
    workload: &str,
    inputs: &Inputs,
) -> anyhow::Result<Answer> {
    let mut scan = table.scan();
    if let Some(filter) = vortex_filter(workload, inputs) {
        scan = scan.with_filter(filter);
    }
    match workload {
        "a1" | "a2" | "a3" => table.sum_price(scan).await,
        "a4" | "t1" => table.count(scan).await,
        "d1" | "d2" => {
            let batches = if workload == "d1" {
                &inputs.random_rows
            } else {
                &inputs.windows
            };
            let mut rows = 0;
            for batch in batches {
                let Answer::Rows(n) = table.take(batch).await? else {
                    unreachable!()
                };
                rows += n;
            }
            Ok(Answer::Rows(rows))
        }
        "v1" => table.top_k(&inputs.query, 10).await,
        other => anyhow::bail!("unknown workload {other}"),
    }
}

async fn run_lance(dataset: &Dataset, workload: &str, inputs: &Inputs) -> anyhow::Result<Answer> {
    let mut scanner = dataset.scan();
    if let Some(filter) = lance_filter(workload, inputs) {
        scanner.filter(&filter)?;
    }
    match workload {
        "a1" | "a2" | "a3" => lance_sum(&mut scanner).await,
        "a4" | "t1" => lance_count(&mut scanner).await,
        "d1" | "d2" => {
            let batches = if workload == "d1" {
                &inputs.random_rows
            } else {
                &inputs.windows
            };
            let projection = ProjectionRequest::from_columns(["id", "emb"], dataset.schema());
            let mut rows = 0;
            for batch in batches {
                rows += dataset.take(batch, projection.clone()).await?.num_rows() as u64;
            }
            Ok(Answer::Rows(rows))
        }
        "v1" => lance_top_k(dataset, &inputs.query, 10).await,
        other => anyhow::bail!("unknown workload {other}"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let session = VortexSession::default().with_tokio();
    prepare(&args, &session).await?;

    let vortex_dir = args.dir.join("vortex");
    let manifest_bytes = ByteBuffer::from(fs::read(vortex_dir.join("manifest.vortex"))?);
    let manifest = Arc::new(Manifest::read(&session, Arc::new(manifest_bytes)).await?);
    let parallel = || {
        let mut driver = ParallelDriver::new(session.handle());
        if let Some(workers) = args.workers {
            driver = driver.with_workers(workers);
        }
        driver
    };
    let (driver, ordered) = if args.sequential {
        (
            Driver::Sequential(SequentialDriver::new()),
            Driver::Sequential(SequentialDriver::new()),
        )
    } else {
        (
            Driver::Parallel(parallel()),
            Driver::Parallel(parallel().with_ordered(true)),
        )
    };
    let table = VortexTable {
        storage: Arc::new(LocalStorage::new(&vortex_dir, session.handle())),
        session,
        manifest,
        cache: Arc::new(FileCache::new()),
        driver,
        ordered,
    };
    let mut dataset = Dataset::open(&args.dir.join("lance").to_string_lossy()).await?;
    let indexes = if args.indexes {
        let indexes = indexed::vortex_indexes(&args, &table).await?;
        indexed::lance_indexes(&args, &mut dataset).await?;
        Some(indexes)
    } else {
        None
    };

    let inputs = inputs(&args);
    let all = ["a1", "a2", "a3", "a4", "d1", "d2", "v1", "t1"];
    println!(
        "{} rows, {} files, {} dims; median of {} runs\n",
        args.rows, args.files, args.dims, args.iterations
    );
    println!("| workload | vortex (ms) | lance (ms) | vortex / lance | answers agree |");
    println!("|---|---:|---:|---:|---|");
    for workload in all {
        if let Some(only) = &args.workloads
            && !only.iter().any(|w| w == workload)
        {
            continue;
        }
        let (vortex_time, vortex_answer) =
            time(args.iterations, || run_vortex(&table, workload, &inputs)).await?;
        let (lance_time, lance_answer) =
            time(args.iterations, || run_lance(&dataset, workload, &inputs)).await?;
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        println!(
            "| {workload} | {:.1} | {:.1} | {:.2} | {} |",
            ms(vortex_time),
            ms(lance_time),
            ms(vortex_time) / ms(lance_time),
            if vortex_answer == lance_answer {
                "yes".to_string()
            } else {
                format!("no: {vortex_answer:?} vs {lance_answer:?}")
            },
        );
    }
    if let Some(indexes) = &indexes {
        indexed::run(&args, &table, indexes, &dataset, &inputs).await?;
    }
    Ok(())
}

#[path = "table_bench/indexed.rs"]
mod indexed;
