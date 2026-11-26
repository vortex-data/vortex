// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex's index kinds against Lance's, over the same table.
//!
//! A plain criterion benchmark, not a CodSpeed one: it compares two systems on one machine, so its
//! numbers only mean something side by side. The default table is small enough that building both
//! sides' indexes and measuring every query takes a few seconds; set `INDEX_BENCH_SCALE=large` for
//! 400k rows of 128-dimensional vectors, where the index work outweighs each query's fixed cost.
//!
//! | Query | Vortex | Lance |
//! |---|---|---|
//! | Substring filter | trigram index, then a scan v2 of the kept rows | NGram index on `contains` |
//! | BM25 top-10 | inverted index, then a take of the hits' ids | inverted index |
//! | Cosine top-10 | IVF with int8 codes, then a take of the hits' ids | IVF_SQ |
//!
//! Each top-10 query is timed three ways:
//!
//! - `vortex` and `lance` return ids, so both read the data as well as the index. Vortex keeps
//!   its index handle open between queries, and Lance keeps its opened indexes in its session's
//!   index cache, so these two are the like-for-like comparison.
//! - `vortex_reopen` opens the index file again for every query, the cost of a cold query.
//!   Lance still looks its cached index up per query, but does not reload it.
//! - The `_search` groups time the index search alone, returning row positions and scores
//!   without reading the data.
//!
//! Both sides read from memory. Before timing, the benchmark prints recall@10 against an exact
//! search for both sides: cosine over every vector, and BM25 scored over every row, counting a
//! hit tied with the exact 10th score as a match.
//!
//! ```text
//! cargo bench -p lance-bench --bench index_vs_lance
//! INDEX_BENCH_SCALE=large cargo bench -p lance-bench --bench index_vs_lance
//! ```

#![expect(clippy::unwrap_used, reason = "benchmark setup fails loudly")]
#![expect(clippy::cast_possible_truncation, reason = "small benchmark constants")]

use std::sync::Arc;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::scanner::Scanner;
use lance::deps::arrow_array as la;
use lance::deps::arrow_array::Array as _;
use lance::deps::arrow_schema as ls;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::scalar::BuiltinIndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::ScalarIndexParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability::NonNullable;
use vortex::array::dtype::PType;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::col;
use vortex::expr::like;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex_index::Hit;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::rows::collect;
use vortex_index_inverted::Bm25Search;
use vortex_index_inverted::InvertedIndex;
use vortex_index_ivf::IvfIndex;
use vortex_index_ivf::IvfOptions;
use vortex_index_ivf::VectorSearch;
use vortex_index_trigram::TrigramIndex;

const WORDS_PER_ROW: usize = 12;
const VOCABULARY: u64 = 2_000;
const NPROBE: usize = 8;
const K: usize = 10;
/// Every this many rows holds the rare word, so a substring filter keeps a few dozen rows.
const NEEDLE_EVERY: usize = 499;
const NEEDLE: &str = "needle";
const BM25_QUERY: &str = "w7 w42 needle";
/// How many random queries, besides the benchmark's own, the recall check runs.
const RECALL_QUERIES: usize = 20;

/// The table's size and the IVF's list count.
#[derive(Clone, Copy)]
struct Scale {
    label: &'static str,
    rows: usize,
    dim: usize,
    lists: u32,
}

impl Scale {
    fn from_env() -> Self {
        match std::env::var("INDEX_BENCH_SCALE").as_deref() {
            Ok("large") => Self {
                label: "400k",
                rows: 400_000,
                dim: 128,
                lists: 256,
            },
            _ => Self {
                label: "20k",
                rows: 20_000,
                dim: 32,
                lists: 64,
            },
        }
    }
}

/// A small deterministic generator, so both sides get the same data.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

/// The table: `id`, a skewed bag of words, and an embedding near one of a few topics.
struct Data {
    scale: Scale,
    texts: Vec<String>,
    vectors: Vec<f32>,
    query: Vec<f32>,
}

impl Data {
    fn generate(scale: Scale) -> Self {
        let Scale { rows, dim, .. } = scale;
        let mut rng = Lcg(42);
        let topics: Vec<f32> = (0..16 * dim).map(|_| rng.unit() - 0.5).collect();
        let mut vectors = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let topic = (rng.next() % 16) as usize;
            vectors.extend(
                topics[topic * dim..(topic + 1) * dim]
                    .iter()
                    .map(|t| t + (rng.unit() - 0.5) * 0.2),
            );
        }
        // The texts draw from their own generator, so the vectors match the vector-only runs.
        let mut rng = Lcg(7);
        let texts = (0..rows)
            .map(|row| {
                // Squaring skews word frequencies towards low ids, as in natural text.
                let mut words: Vec<String> = (0..WORDS_PER_ROW)
                    .map(|_| {
                        let u = rng.next() % VOCABULARY;
                        format!("w{}", u * u / VOCABULARY)
                    })
                    .collect();
                if row.is_multiple_of(NEEDLE_EVERY) {
                    words.push(NEEDLE.to_string());
                }
                words.join(" ")
            })
            .collect();
        let query = vectors[..dim].iter().map(|v| v + 0.01).collect();
        Self {
            scale,
            texts,
            vectors,
            query,
        }
    }

    fn vortex_array(&self) -> VortexResult<ArrayRef> {
        let ids: PrimitiveArray = (0..self.scale.rows as u64).collect();
        let elements = PrimitiveArray::new(Buffer::copy_from(&self.vectors), Validity::NonNullable);
        let vectors = FixedSizeListArray::new(
            elements.into_array(),
            self.scale.dim as u32,
            Validity::NonNullable,
            self.scale.rows,
        );
        Ok(StructArray::from_fields(&[
            ("id", ids.into_array()),
            (
                "text",
                VarBinViewArray::from_iter_str(self.texts.iter().map(String::as_str)).into_array(),
            ),
            ("emb", vectors.into_array()),
        ])?
        .into_array())
    }

    fn lance_batch(&self) -> la::RecordBatch {
        let item = Arc::new(ls::Field::new("item", ls::DataType::Float32, false));
        let schema = Arc::new(ls::Schema::new(vec![
            ls::Field::new("id", ls::DataType::UInt64, false),
            ls::Field::new("text", ls::DataType::Utf8, false),
            ls::Field::new(
                "emb",
                ls::DataType::FixedSizeList(Arc::clone(&item), self.scale.dim as i32),
                false,
            ),
        ]));
        let vectors = la::FixedSizeListArray::new(
            item,
            self.scale.dim as i32,
            Arc::new(la::Float32Array::from(self.vectors.clone())),
            None,
        );
        la::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(la::UInt64Array::from_iter_values(0..self.scale.rows as u64)),
                Arc::new(la::StringArray::from_iter_values(&self.texts)),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    /// The benchmark's cosine query, then [`RECALL_QUERIES`] more near random rows.
    fn vector_queries(&self) -> Vec<Vec<f32>> {
        let dim = self.scale.dim;
        let mut rng = Lcg(7);
        let mut queries = vec![self.query.clone()];
        for _ in 0..RECALL_QUERIES {
            let row = (rng.next() as usize) % self.scale.rows;
            queries.push(
                self.vectors[row * dim..(row + 1) * dim]
                    .iter()
                    .map(|v| v + (rng.unit() - 0.5) * 0.2)
                    .collect(),
            );
        }
        queries
    }

    /// The benchmark's BM25 query, then [`RECALL_QUERIES`] more of two words from a random row
    /// and one random word.
    fn text_queries(&self) -> Vec<String> {
        let mut rng = Lcg(11);
        let mut queries = vec![BM25_QUERY.to_string()];
        for _ in 0..RECALL_QUERIES {
            let words: Vec<&str> = self.texts[(rng.next() as usize) % self.scale.rows]
                .split(' ')
                .collect();
            let first = words[(rng.next() as usize) % words.len()];
            let second = words[(rng.next() as usize) % words.len()];
            queries.push(format!("{first} {second} w{}", rng.next() % 300));
        }
        queries
    }
}

/// Exact cosine top-[`K`] rows, best first, ties by row.
fn exact_cosine(data: &Data, query: &[f32]) -> Vec<u64> {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let query_norm = norm(query);
    let mut scored: Vec<(u64, f32)> = data
        .vectors
        .chunks_exact(data.scale.dim)
        .enumerate()
        .map(|(row, v)| {
            let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
            (row as u64, dot / (norm(v) * query_norm))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored[..K].iter().map(|(row, _)| *row).collect()
}

/// Every row's term counts and length, for scoring BM25 exactly.
struct Corpus {
    docs: Vec<Vec<String>>,
    average_len: f32,
}

impl Corpus {
    fn new(texts: &[String]) -> Self {
        let docs: Vec<Vec<String>> = texts.iter().map(|t| tokens(t)).collect();
        let average_len = docs.iter().map(Vec::len).sum::<usize>() as f32 / docs.len() as f32;
        Self { docs, average_len }
    }

    /// The rows that score at least the exact [`K`]th BM25 score, with k1 1.2, b 0.75 and
    /// IDF `ln(1 + (N - n + 0.5) / (n + 0.5))`, as both libraries score.
    fn top(&self, query: &str) -> Vec<u64> {
        let rows = self.docs.len() as f32;
        let mut terms = tokens(query);
        terms.sort();
        terms.dedup();
        let idfs: Vec<f32> = terms
            .iter()
            .map(|term| {
                let df = self.docs.iter().filter(|d| d.contains(term)).count() as f32;
                ((rows - df + 0.5) / (df + 0.5)).ln_1p()
            })
            .collect();
        let mut scored: Vec<(u64, f32)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(row, doc)| {
                let norm = 1.2 * (0.25 + 0.75 * doc.len() as f32 / self.average_len);
                let score = terms
                    .iter()
                    .zip(&idfs)
                    .map(|(term, idf)| {
                        let tf = doc.iter().filter(|w| *w == term).count() as f32;
                        idf * 2.2 * tf / (tf + norm)
                    })
                    .sum::<f32>();
                (row as u64, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let kth = scored[K - 1].1;
        scored
            .iter()
            .take_while(|(_, score)| *score >= kth - 1e-4)
            .map(|(row, _)| *row)
            .collect()
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The Vortex side: a plain data file and one standalone index file per kind.
struct VortexSide {
    data: VortexFile,
    trigram: IndexFile<TrigramIndex>,
    inverted: IndexFile<InvertedIndex>,
    ivf: IndexFile<IvfIndex>,
    /// The opened index files, for the variants that open an index handle per query.
    inverted_file: VortexFile,
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

        let mut trigram = IndexFileBuilder::new(
            TrigramIndex,
            Default::default(),
            "text",
            DType::Utf8(NonNullable),
            session,
        )?;
        let mut inverted =
            IndexFileBuilder::new(InvertedIndex, (), "text", DType::Utf8(NonNullable), session)?;
        let vector_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, NonNullable)),
            scale.dim as u32,
            NonNullable,
        );
        let mut ivf = IndexFileBuilder::new(
            IvfIndex,
            IvfOptions {
                lists: scale.lists,
                ..Default::default()
            },
            "emb",
            vector_dtype,
            session,
        )?;
        trigram.add_file("data.vortex", &file).await?;
        inverted.add_file("data.vortex", &file).await?;
        ivf.add_file("data.vortex", &file).await?;
        let mut trigram_bytes = ByteBufferMut::empty();
        trigram.write(&mut trigram_bytes).await?;
        let mut inverted_bytes = ByteBufferMut::empty();
        inverted.write(&mut inverted_bytes).await?;
        let mut ivf_bytes = ByteBufferMut::empty();
        ivf.write(&mut ivf_bytes).await?;

        let inverted_file = open(session, inverted_bytes)?;
        let ivf_file = open(session, ivf_bytes)?;
        Ok(Self {
            data: file,
            trigram: IndexFile::open(TrigramIndex, &open(session, trigram_bytes)?)?,
            inverted: IndexFile::open(InvertedIndex, &inverted_file)?,
            ivf: IndexFile::open(IvfIndex, &ivf_file)?,
            inverted_file,
            ivf_file,
        })
    }

    /// The ids of the rows holding the needle: the trigram index keeps the blocks that may, and
    /// the filter rechecks them.
    async fn contains(&self) -> VortexResult<Vec<u64>> {
        let filter = like(col("text"), lit(format!("%{NEEDLE}%")));
        let selection = match self
            .trigram
            .locate_files(&like(root(), lit(format!("%{NEEDLE}%"))))
            .await?
        {
            Some((_, mut found)) => match found.pop() {
                Some(found) => found.selection,
                None => return Ok(Vec::new()),
            },
            None => Selection::All,
        };
        self.ids(Some(filter), selection).await
    }

    async fn bm25(&self) -> VortexResult<Vec<u64>> {
        let hits = self.bm25_search(BM25_QUERY).await?;
        self.take_ids(hits).await
    }

    async fn bm25_reopen(&self) -> VortexResult<Vec<u64>> {
        let index = IndexFile::open(InvertedIndex, &self.inverted_file)?;
        let hits = index.rows().search_bm25(BM25_QUERY, K).await?;
        self.take_ids(hits).await
    }

    async fn bm25_search(&self, query: &str) -> VortexResult<Vec<Hit>> {
        self.inverted.rows().search_bm25(query, K).await
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

    async fn ids(
        &self,
        filter: Option<Expression>,
        selection: Selection,
    ) -> VortexResult<Vec<u64>> {
        let dtype = self.data.dtype();
        let chunks = collect(
            ScanBuilder::new(self.data.session().clone(), self.data.scan_file())
                .with_some_filter(filter.map(|f| f.bind(dtype)).transpose()?)
                .with_selection(selection)
                .with_projection(col("id").bind(dtype)?),
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

/// The Lance side: one dataset with an NGram, an inverted and an IVF_SQ index.
async fn lance_dataset(uri: &str, data: &Data) -> lance::Result<Dataset> {
    let batch = data.lance_batch();
    let reader = la::RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, uri, Some(WriteParams::default())).await?;
    dataset
        .create_index(
            &["text"],
            IndexType::NGram,
            Some("text_ngram".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
            true,
        )
        .await?;
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_fts".to_string()),
            &InvertedIndexParams::default(),
            true,
        )
        .await?;
    let params = VectorIndexParams::with_ivf_sq_params(
        MetricType::Cosine,
        IvfBuildParams::new(data.scale.lists as usize),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["emb"], IndexType::Vector, None, &params, true)
        .await?;
    Ok(dataset)
}

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

async fn lance_contains(dataset: &Dataset) -> lance::Result<Vec<u64>> {
    let mut scanner = dataset.scan();
    scanner
        .filter(&format!("contains(text, '{NEEDLE}')"))?
        .project(&["id"])?;
    lance_ids(&mut scanner).await
}

fn bm25_scanner(dataset: &Dataset, query: &str) -> lance::Result<Scanner> {
    let mut scanner = dataset.scan();
    scanner.full_text_search(FullTextSearchQuery::new(query.to_string()).limit(Some(K as i64)))?;
    Ok(scanner)
}

fn nearest_scanner(dataset: &Dataset, query: &[f32]) -> lance::Result<Scanner> {
    let mut scanner = dataset.scan();
    scanner
        .nearest("emb", &la::Float32Array::from(query.to_vec()), K)?
        .nprobes(NPROBE);
    Ok(scanner)
}

async fn lance_bm25(dataset: &Dataset, query: &str) -> lance::Result<Vec<u64>> {
    let mut scanner = bm25_scanner(dataset, query)?;
    scanner.project(&["id"])?;
    lance_ids(&mut scanner).await
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

/// Prints each side's mean recall@10 against exact search, over the benchmark's query and
/// [`RECALL_QUERIES`] more of each kind.
fn print_recall(runtime: &Runtime, data: &Data, vortex: &VortexSide, lance: &Dataset) {
    let recall = |found: usize, queries: usize| found as f64 / (K * queries) as f64;

    let queries = data.vector_queries();
    let (mut vortex_found, mut lance_found) = (0, 0);
    for query in &queries {
        let exact = exact_cosine(data, query);
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

    let corpus = Corpus::new(&data.texts);
    let queries = data.text_queries();
    let (mut vortex_found, mut lance_found) = (0, 0);
    for query in &queries {
        let exact = corpus.top(query);
        let hits = runtime.block_on(vortex.bm25_search(query)).unwrap();
        vortex_found += overlap(&rows(&hits), &exact);
        let ids = runtime.block_on(lance_bm25(lance, query)).unwrap();
        lance_found += overlap(&ids, &exact);
    }
    eprintln!(
        "BM25 recall@{K} over {} queries, ties counted: Vortex {:.3}, Lance {:.3}",
        queries.len(),
        recall(vortex_found, queries.len()),
        recall(lance_found, queries.len()),
    );
}

fn index_vs_lance(c: &mut Criterion) {
    let scale = Scale::from_env();
    let runtime = Runtime::new().unwrap();
    let data = Data::generate(scale);
    let session = VortexSession::default().with_tokio();
    vortex_index::initialize(&session).unwrap();
    vortex_index_trigram::register(&session);
    vortex_index_inverted::register(&session);
    vortex_index_ivf::register(&session);
    let vortex = runtime
        .block_on(VortexSide::build(&session, &data, scale))
        .unwrap();
    // Both sides read from memory, so neither pays for file IO the other skips.
    let lance = runtime
        .block_on(lance_dataset("memory://index-vs-lance", &data))
        .unwrap();

    // The two sides answer the same questions before either is timed.
    let expected: Vec<u64> = (0..scale.rows as u64)
        .filter(|row| (*row as usize).is_multiple_of(NEEDLE_EVERY))
        .collect();
    assert_eq!(runtime.block_on(vortex.contains()).unwrap(), expected);
    assert_eq!(runtime.block_on(lance_contains(&lance)).unwrap(), expected);
    let vortex_top = runtime.block_on(vortex.bm25()).unwrap();
    let lance_top = runtime.block_on(lance_bm25(&lance, BM25_QUERY)).unwrap();
    assert_eq!(vortex_top.len(), K);
    assert_eq!(lance_top.len(), K);
    let vortex_near = runtime.block_on(vortex.nearest(&data.query)).unwrap();
    let lance_near = runtime
        .block_on(lance_nearest(&lance, &data.query))
        .unwrap();
    // Ties at the 10th BM25 score let the two sides pick different rows, so only check that
    // every hit holds a query term. The cosine query sits next to row 0.
    let terms: Vec<&str> = BM25_QUERY.split(' ').collect();
    for row in vortex_top.iter().chain(&lance_top) {
        let text = &data.texts[*row as usize];
        assert!(
            text.split(' ').any(|word| terms.contains(&word)),
            "BM25 hit {row} holds no query term: {text}"
        );
    }
    assert!(vortex_near.contains(&0), "Vortex's nearest rows miss row 0");
    assert!(lance_near.contains(&0), "Lance's nearest rows miss row 0");
    assert_eq!(runtime.block_on(vortex.bm25_reopen()).unwrap(), vortex_top);
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
    assert_eq!(
        runtime
            .block_on(lance_search(bm25_scanner(&lance, BM25_QUERY).unwrap()))
            .unwrap(),
        K
    );
    print_recall(&runtime, &data, &vortex, &lance);

    let label = scale.label;
    let mut group = c.benchmark_group("contains");
    group.bench_function(BenchmarkId::new("vortex", label), |b| {
        b.iter(|| runtime.block_on(vortex.contains()).unwrap())
    });
    group.bench_function(BenchmarkId::new("lance", label), |b| {
        b.iter(|| runtime.block_on(lance_contains(&lance)).unwrap())
    });
    group.finish();

    let mut group = c.benchmark_group("bm25_top10");
    group.bench_function(BenchmarkId::new("vortex", label), |b| {
        b.iter(|| runtime.block_on(vortex.bm25()).unwrap())
    });
    group.bench_function(BenchmarkId::new("vortex_reopen", label), |b| {
        b.iter(|| runtime.block_on(vortex.bm25_reopen()).unwrap())
    });
    group.bench_function(BenchmarkId::new("lance", label), |b| {
        b.iter(|| runtime.block_on(lance_bm25(&lance, BM25_QUERY)).unwrap())
    });
    group.finish();

    let mut group = c.benchmark_group("bm25_search");
    group.bench_function(BenchmarkId::new("vortex", label), |b| {
        b.iter(|| runtime.block_on(vortex.bm25_search(BM25_QUERY)).unwrap())
    });
    group.bench_function(BenchmarkId::new("lance", label), |b| {
        b.iter(|| {
            runtime
                .block_on(lance_search(bm25_scanner(&lance, BM25_QUERY).unwrap()))
                .unwrap()
        })
    });
    group.finish();

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
