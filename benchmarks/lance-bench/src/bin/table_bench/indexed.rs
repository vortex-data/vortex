// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The indexed workloads: IVF vector top-k, BM25 top-k, and term matches.

use std::fs;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::scanner::Scanner;
use lance::deps::arrow_array as la;
use lance::deps::arrow_array::Array as _;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::InvertedIndexParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::array::dtype::PType;
use vortex::array::expr::root;
use vortex::array::expr::select;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::layout::layouts::indexed::IndexVTable;
use vortex::scan::planning::drive::MapFn;
use vortex::scan::planning::drive::ScopedBatch;
use vortex_index::Hit;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::IndexStep;
use vortex_index::InvertedIndex;
use vortex_index::IvfIndex;
use vortex_index::IvfOptions;
use vortex_index::text_match;
use vortex_table::FileCache;
use vortex_table::TableScan;
use vortex_table::index::TableIndex;
use vortex_table::index::index_table;
use vortex_table::index::open_index_file;

use super::Answer;
use super::Args;
use super::Inputs;
use super::VortexTable;
use super::time;

/// The Vortex indexes, bound to the table.
pub(super) struct Indexes {
    ivf: Arc<TableIndex<IvfIndex>>,
    text: Arc<TableIndex<InvertedIndex>>,
}

/// Builds the Vortex index files once, next to the table, and binds them to its manifest.
pub(super) async fn vortex_indexes(args: &Args, table: &VortexTable) -> anyhow::Result<Indexes> {
    let dir = args
        .dir
        .join("vortex")
        .join(format!("indexes-{}", args.ivf_lists));
    fs::create_dir_all(&dir)?;
    let session = &table.session;
    let ivf_path = dir.join("emb.ivf.vortex");
    if !ivf_path.exists() {
        let start = Instant::now();
        let dims = u32::try_from(args.dims)?;
        let mut builder = IndexFileBuilder::new(
            IvfIndex,
            IvfOptions {
                lists: u32::try_from(args.ivf_lists)?,
                dim: 0,
            },
            "emb",
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                dims,
                Nullability::NonNullable,
            ),
            session,
        )?;
        index_table(
            &mut builder,
            &table.manifest,
            table.storage.as_ref(),
            session,
        )
        .await?;
        let mut bytes = ByteBufferMut::empty();
        builder.write(&mut bytes).await?;
        fs::write(&ivf_path, bytes.freeze().as_slice())?;
        println!(
            "vortex IVF index built in {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }
    let text_path = dir.join("text.inverted.vortex");
    if !text_path.exists() {
        let start = Instant::now();
        let mut builder = IndexFileBuilder::new(
            InvertedIndex,
            (),
            "text",
            DType::Utf8(Nullability::NonNullable),
            session,
        )?;
        index_table(
            &mut builder,
            &table.manifest,
            table.storage.as_ref(),
            session,
        )
        .await?;
        let mut bytes = ByteBufferMut::empty();
        builder.write(&mut bytes).await?;
        fs::write(&text_path, bytes.freeze().as_slice())?;
        println!(
            "vortex inverted index built in {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }
    let open = |path: std::path::PathBuf| async move {
        let bytes = ByteBuffer::from(fs::read(path)?);
        anyhow::Ok(open_index_file(session, Arc::new(bytes)).await?)
    };
    let ivf = IndexFile::open(IvfIndex, &open(ivf_path).await?)?;
    let text = IndexFile::open(InvertedIndex, &open(text_path).await?)?;
    Ok(Indexes {
        ivf: Arc::new(TableIndex::bind(ivf, &table.manifest)),
        text: Arc::new(TableIndex::bind(text, &table.manifest)),
    })
}

/// Opens the Vortex IVF index file afresh, as a cold query would, and binds it to the table.
async fn open_vortex_ivf(
    args: &Args,
    table: &VortexTable,
) -> anyhow::Result<Arc<TableIndex<IvfIndex>>> {
    let path = args
        .dir
        .join("vortex")
        .join(format!("indexes-{}", args.ivf_lists))
        .join("emb.ivf.vortex");
    let bytes = ByteBuffer::from(fs::read(path)?);
    let file = open_index_file(&table.session, Arc::new(bytes)).await?;
    Ok(Arc::new(TableIndex::bind(
        IndexFile::open(IvfIndex, &file)?,
        &table.manifest,
    )))
}

/// Builds Lance's indexes once: IVF_SQ over the embeddings, inverted over the text.
pub(super) async fn lance_indexes(args: &Args, dataset: &mut Dataset) -> anyhow::Result<()> {
    let existing = dataset.load_indices().await?;
    let indexed = |column: &str| {
        let field = dataset.schema().field(column).map(|f| f.id);
        existing.iter().any(|index| Some(index.fields[0]) == field)
    };
    let (has_emb, has_text) = (indexed("emb"), indexed("text"));
    if !has_emb {
        let start = Instant::now();
        let params = VectorIndexParams::with_ivf_sq_params(
            MetricType::Cosine,
            IvfBuildParams::new(args.ivf_lists),
            SQBuildParams::default(),
        );
        dataset
            .create_index(&["emb"], IndexType::Vector, None, &params, true)
            .await?;
        println!(
            "lance IVF_SQ index built in {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }
    if !has_text {
        let start = Instant::now();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await?;
        println!(
            "lance inverted index built in {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

impl VortexTable {
    /// The ids of `batches`, in their order.
    fn ids(&self, batches: Vec<ScopedBatch>) -> anyhow::Result<Vec<u64>> {
        let mut ids = Vec::new();
        for batch in batches {
            ids.extend_from_slice(self.column(batch.batch, "id")?.as_slice::<u64>());
        }
        Ok(ids)
    }

    fn id_scan(&self) -> TableScan {
        self.scan().with_projection(select(["id"], root()))
    }

    /// The ids of a ranked search's hits, best first, from one plan that runs the search and
    /// then reads the hits' rows.
    async fn ranked_ids<V: IndexVTable>(
        &self,
        index: &Arc<TableIndex<V>>,
        search: IndexStep<Vec<Hit>>,
    ) -> anyhow::Result<Vec<u64>> {
        let (plan, ranked) = index.plan_ranked(search, self.id_scan())?;
        // Ordered, so ids line up with the hits of the search.
        let batches = self.ordered.as_dyn().execute(plan).try_collect().await?;
        let ids = self.ids(batches)?;
        let ranked = ranked.get().context("the plan ran the search")?;
        anyhow::ensure!(ids.len() == ranked.hits.len(), "a hit's row is missing");
        Ok(ranked.best_first().into_iter().map(|i| ids[i]).collect())
    }

    /// The best `k` of a ranked search's candidates, re-scored by their stored vectors, from one
    /// plan that runs the search and then reads the candidates' vectors.
    async fn refined(
        &self,
        index: &Arc<TableIndex<IvfIndex>>,
        search: IndexStep<Vec<Hit>>,
        query: &[f32],
        k: usize,
    ) -> anyhow::Result<Vec<u64>> {
        let scan = self.scan().with_projection(select(["id", "emb"], root()));
        let (plan, _) = index.plan_ranked(search, scan)?;
        let query: Arc<[f32]> = query.into();
        let session = self.session.clone();
        // Each batch of candidates is decoded and re-scored on the worker that read it.
        let score: MapFn<Vec<(f32, u64)>> = Arc::new(move |batch| {
            let mut best = cosine_scores(batch, &query, &mut session.create_execution_ctx())?;
            super::keep_best(&mut best, k);
            Ok(best)
        });
        let mut best: Vec<(f32, u64)> = Vec::new();
        let mut batches = self.driver.execute_with(plan, score);
        while let Some(batch) = batches.try_next().await? {
            best.extend(batch.batch);
            super::keep_best(&mut best, k);
        }
        Ok(best.into_iter().map(|(_, id)| id).collect())
    }

    /// The exact cosine top-k of each of `queries`, by one full scan, to measure recall against.
    async fn exact_cosine_many(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> anyhow::Result<Vec<Vec<u64>>> {
        let scan = self.scan().with_projection(select(["id", "emb"], root()));
        let batches = self.collect(scan).await?;
        let mut best: Vec<Vec<(f32, u64)>> = vec![Vec::new(); queries.len()];
        let mut ctx = self.session.create_execution_ctx();
        for batch in batches {
            for (values, ids) in vectors(batch.batch, &mut ctx)? {
                for (query, best) in queries.iter().zip(&mut best) {
                    best.extend(cosine(
                        values.as_slice::<f32>(),
                        ids.as_slice::<u64>(),
                        query,
                    ));
                    super::keep_best(best, k);
                }
            }
        }
        Ok(best
            .into_iter()
            .map(|best| best.into_iter().map(|(_, id)| id).collect())
            .collect())
    }
}

/// The cosine similarity to `query` of each row of a batch of `id` and `emb`, with its id.
fn cosine_scores(
    batch: ArrayRef,
    query: &[f32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<(f32, u64)>> {
    let mut scores = Vec::with_capacity(batch.len());
    for (values, ids) in vectors(batch, ctx)? {
        scores.extend(cosine(
            values.as_slice::<f32>(),
            ids.as_slice::<u64>(),
            query,
        ));
    }
    Ok(scores)
}

/// The flat vectors and ids of a batch of `id` and `emb`, one pair per chunk, so the chunks are
/// read as they are rather than copied into one array first.
fn vectors(
    batch: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<(PrimitiveArray, PrimitiveArray)>> {
    if let Some(chunked) = batch.as_opt::<Chunked>() {
        let mut parts = Vec::new();
        for chunk in chunked.iter_chunks() {
            parts.extend(vectors(chunk.clone(), ctx)?);
        }
        return Ok(parts);
    }
    let array = batch.execute::<StructArray>(ctx)?;
    let ids = array
        .unmasked_field_by_name("id")?
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let emb = array
        .unmasked_field_by_name("emb")?
        .clone()
        .execute::<FixedSizeListArray>(ctx)?;
    let values = emb.elements().clone().execute::<PrimitiveArray>(ctx)?;
    Ok(vec![(values, ids)])
}

/// The cosine similarity to `query` of each of `values`' vectors, with its id.
fn cosine<'a>(
    values: &'a [f32],
    ids: &'a [u64],
    query: &'a [f32],
) -> impl Iterator<Item = (f32, u64)> + 'a {
    let norm = |v: &[f32]| super::dot(v, v).sqrt().max(f32::MIN_POSITIVE);
    let q_norm = norm(query);
    values
        .chunks_exact(query.len())
        .zip(ids)
        .map(move |(v, &id)| (super::dot(v, query) / (norm(v) * q_norm), id))
}

/// The approximate top-`k` search over every query vector, timed the way ANN benchmarks time it:
/// distinct queries in rotation after an untimed warm-up pass, median and p99 latency, and recall
/// averaged over the queries. Vortex runs with and without its data-segment cache, since Lance
/// keeps no data pages of its own, and both systems are also timed cold: opening the table and
/// its index afresh, then running one query.
async fn vector_search(
    args: &Args,
    table: &VortexTable,
    indexes: &Indexes,
    dataset: &Dataset,
    inputs: &Inputs,
    k: usize,
) -> anyhow::Result<()> {
    let queries = &inputs.queries;
    let exact = table.exact_cosine_many(queries, k).await?;
    let candidates = k * args.refine.max(1);
    let vortex = |table: VortexTable, ivf: Arc<TableIndex<IvfIndex>>, query: Vec<f32>| async move {
        let search = ivf
            .index()
            .rows()
            .search_vectors_steps(&query, candidates, args.nprobe)?;
        if args.refine == 0 {
            table.ranked_ids(&ivf, search).await
        } else {
            table.refined(&ivf, search, &query, k).await
        }
    };
    let lance = |dataset: Dataset, query: Vec<f32>| async move {
        let mut scanner = dataset.scan();
        scanner.project(&["id"])?;
        scanner.nearest("emb", &la::Float32Array::from(query), k)?;
        scanner.distance_metric(MetricType::Cosine);
        scanner.nprobes(args.nprobe);
        if args.refine > 0 {
            scanner.refine(u32::try_from(args.refine)?);
        }
        lance_ids(&mut scanner).await
    };

    let uncached = VortexTable {
        cache: Arc::new(FileCache::retaining(0)),
        ..table.clone()
    };
    let cached = latencies(args, queries, &exact, |q| {
        vortex(table.clone(), Arc::clone(&indexes.ivf), q)
    })
    .await?;
    let no_data_cache = latencies(args, queries, &exact, |q| {
        vortex(uncached.clone(), Arc::clone(&indexes.ivf), q)
    })
    .await?;
    let lance_warm = latencies(args, queries, &exact, |q| lance(dataset.clone(), q)).await?;

    // Cold: a fresh file cache and a freshly opened index, or a freshly opened dataset, timed
    // through its first query. The OS page cache stays warm for both.
    let mut vortex_cold = Vec::new();
    let mut lance_cold = Vec::new();
    for query in queries.iter().take(args.cold_opens) {
        let start = Instant::now();
        let fresh = VortexTable {
            cache: Arc::new(FileCache::new()),
            ..table.clone()
        };
        let ivf = open_vortex_ivf(args, &fresh).await?;
        vortex(fresh, ivf, query.clone()).await?;
        vortex_cold.push(start.elapsed());

        let start = Instant::now();
        let fresh = Dataset::open(&args.dir.join("lance").to_string_lossy()).await?;
        lance(fresh, query.clone()).await?;
        lance_cold.push(start.elapsed());
    }

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    println!(
        "\nv2: top-{k} of {} distinct queries, nprobe {}, refine {}; {} timed rounds after a \
         warm-up round; cold is open + first query, median of {}",
        queries.len(),
        args.nprobe,
        args.refine,
        args.rounds,
        vortex_cold.len()
    );
    println!(
        "\n| v2 | vortex p50 | vortex p99 | lance p50 | lance p99 | p50 vortex / lance | recall@{k} vortex | lance |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for (name, vortex) in [
        ("warm, data cache", &cached),
        ("warm, no data cache", &no_data_cache),
    ] {
        println!(
            "| {name} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} | {:.3} |",
            ms(vortex.p50),
            ms(vortex.p99),
            ms(lance_warm.p50),
            ms(lance_warm.p99),
            ms(vortex.p50) / ms(lance_warm.p50),
            vortex.recall,
            lance_warm.recall,
        );
    }
    vortex_cold.sort();
    lance_cold.sort();
    let (vc, lc) = (
        vortex_cold[vortex_cold.len() / 2],
        lance_cold[lance_cold.len() / 2],
    );
    println!(
        "| cold (open + first query) | {:.2} | | {:.2} | | {:.2} | | |",
        ms(vc),
        ms(lc),
        ms(vc) / ms(lc)
    );
    Ok(())
}

/// The latency percentiles and mean recall of a search over every query.
struct Latencies {
    p50: Duration,
    p99: Duration,
    recall: f64,
}

/// Runs `search` once over every query untimed, for the warm-up and recall, then times
/// `args.rounds` passes over them.
async fn latencies<F, Fut>(
    args: &Args,
    queries: &[Vec<f32>],
    exact: &[Vec<u64>],
    search: F,
) -> anyhow::Result<Latencies>
where
    F: Fn(Vec<f32>) -> Fut,
    Fut: Future<Output = anyhow::Result<Vec<u64>>>,
{
    let mut recall_sum = 0.0;
    for (query, exact) in queries.iter().zip(exact) {
        recall_sum += recall(&search(query.clone()).await?, exact);
    }
    let mut times = Vec::with_capacity(args.rounds * queries.len());
    for _ in 0..args.rounds {
        for query in queries {
            let start = Instant::now();
            search(query.clone()).await?;
            times.push(start.elapsed());
        }
    }
    times.sort();
    // The nearest-rank percentile, in integers.
    let at = |percent: usize| times[(times.len() - 1) * percent / 100];
    Ok(Latencies {
        p50: at(50),
        p99: at(99),
        recall: recall_sum / queries.len().max(1) as f64,
    })
}

async fn lance_ids(scanner: &mut Scanner) -> anyhow::Result<Vec<u64>> {
    let batches: Vec<la::RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .ok_or_else(|| anyhow::anyhow!("no id column"))?
            .as_any()
            .downcast_ref::<la::UInt64Array>()
            .ok_or_else(|| anyhow::anyhow!("id is not u64"))?
            .clone();
        ids.extend(column.values().iter().copied());
    }
    Ok(ids)
}

fn recall(found: &[u64], exact: &[u64]) -> f64 {
    let hits = found.iter().filter(|id| exact.contains(id)).count();
    hits as f64 / exact.len().max(1) as f64
}

/// Runs the indexed workloads and prints their rows of the results table.
pub(super) async fn run(
    args: &Args,
    table: &VortexTable,
    indexes: &Indexes,
    dataset: &Dataset,
    inputs: &Inputs,
) -> anyhow::Result<()> {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let row = |workload: &str, vortex: Duration, lance: Duration, note: String| {
        println!(
            "| {workload} | {:.1} | {:.1} | {:.2} | {note} |",
            ms(vortex),
            ms(lance),
            ms(vortex) / ms(lance)
        );
    };
    let selected = |w: &str| {
        args.workloads
            .as_ref()
            .is_none_or(|only| only.iter().any(|o| o == w))
    };
    let k = 10;

    if selected("v2") {
        vector_search(args, table, indexes, dataset, inputs, k).await?;
    }

    if selected("t2") {
        let (vortex_time, vortex) = time(args.iterations, || async {
            let search = indexes
                .text
                .index()
                .rows()
                .search_bm25_steps(&inputs.bm25_query, k)?;
            Ok(Answer::TopK(table.ranked_ids(&indexes.text, search).await?))
        })
        .await?;
        let (lance_time, lance) = time(args.iterations, || async {
            let mut scanner = dataset.scan();
            scanner.project(&["id"])?;
            scanner.full_text_search(
                FullTextSearchQuery::new(inputs.bm25_query.clone()).limit(Some(k as i64)),
            )?;
            Ok(Answer::TopK(lance_ids(&mut scanner).await?))
        })
        .await?;
        let (Answer::TopK(vortex), Answer::TopK(lance)) = (vortex, lance) else {
            anyhow::bail!("top-k answers");
        };
        row(
            "t2",
            vortex_time,
            lance_time,
            format!("top-{k} overlap {:.1}", recall(&vortex, &lance)),
        );
    }

    if selected("t3") {
        let (vortex_time, vortex) = time(args.iterations, || async {
            let plan = indexes
                .text
                .plan_matching(&text_match(root(), inputs.term.as_str()), table.id_scan())?;
            let batches: Vec<ScopedBatch> =
                table.driver.as_dyn().execute(plan).try_collect().await?;
            Ok(Answer::Rows(
                batches.iter().map(|batch| batch.batch.len() as u64).sum(),
            ))
        })
        .await?;
        let (lance_time, lance) = time(args.iterations, || async {
            let mut scanner = dataset.scan();
            scanner.project(&["id"])?;
            scanner.full_text_search(FullTextSearchQuery::new(inputs.term.clone()))?;
            Ok(Answer::Rows(lance_ids(&mut scanner).await?.len() as u64))
        })
        .await?;
        row(
            "t3",
            vortex_time,
            lance_time,
            if vortex == lance {
                "yes".to_string()
            } else {
                format!("no: {vortex:?} vs {lance:?}")
            },
        );
    }
    Ok(())
}
