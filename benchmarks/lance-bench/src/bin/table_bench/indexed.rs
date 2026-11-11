// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The indexed workloads: IVF vector top-k, BM25 top-k, and term matches.

use std::fs;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

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
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::array::dtype::PType;
use vortex::array::expr::root;
use vortex::array::expr::select;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::planning::drive::ParallelDriver;
use vortex::scan::planning::drive::ScopedBatch;
use vortex::scan::selection::Selection;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::InvertedIndex;
use vortex_index::IvfIndex;
use vortex_index::IvfOptions;
use vortex_index::text_match;
use vortex_table::index::RankedSelections;
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
    ivf: TableIndex<IvfIndex>,
    text: TableIndex<InvertedIndex>,
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
        ivf: TableIndex::bind(ivf, &table.manifest),
        text: TableIndex::bind(text, &table.manifest),
    })
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
    /// The ids of the rows `selections` picks, by file and row.
    async fn ids_of(&self, selections: HashMap<u64, Selection>) -> anyhow::Result<Vec<u64>> {
        // Ordered, so ids line up with the hits of a ranked search.
        let driver = ParallelDriver::new(self.session.handle()).with_ordered(true);
        let batches: Vec<ScopedBatch> = self
            .scan()
            .with_projection(select(["id"], root()))
            .with_file_selections(selections)
            .execute(&driver)
            .try_collect()
            .await?;
        let mut ids = Vec::new();
        for batch in batches {
            ids.extend_from_slice(self.column(batch.batch, "id")?.as_slice::<u64>());
        }
        Ok(ids)
    }

    /// The ids of a ranked search's hits, best first.
    async fn ranked_ids(&self, ranked: RankedSelections) -> anyhow::Result<Vec<u64>> {
        let order = ranked.best_first();
        let ids = self.ids_of(ranked.selections).await?;
        anyhow::ensure!(ids.len() == ranked.hits.len(), "a hit's row is missing");
        Ok(order.into_iter().map(|i| ids[i]).collect())
    }

    /// The exact cosine top-k, by a full scan, to measure recall against.
    async fn exact_cosine(&self, query: &[f32], k: usize) -> anyhow::Result<Vec<u64>> {
        let norm = |v: &[f32]| {
            v.iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(f32::MIN_POSITIVE)
        };
        let q_norm = norm(query);
        let mut best: Vec<(f32, u64)> = Vec::new();
        let mut ctx = self.session.create_execution_ctx();
        let scan = self.scan().with_projection(select(["id", "emb"], root()));
        for batch in self.collect(scan).await? {
            let array = batch.batch.execute::<StructArray>(&mut ctx)?;
            let ids = array
                .unmasked_field_by_name("id")?
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            let emb = array
                .unmasked_field_by_name("emb")?
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let values = emb.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
            for (row, v) in values.as_slice::<f32>().chunks(query.len()).enumerate() {
                let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                best.push((dot / (norm(v) * q_norm), ids.as_slice::<u64>()[row]));
            }
            super::keep_best(&mut best, k);
        }
        Ok(best.into_iter().map(|(_, id)| id).collect())
    }
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
        let exact = table.exact_cosine(&inputs.query, k).await?;
        let (vortex_time, vortex) = time(args.iterations, || async {
            let hits = indexes
                .ivf
                .index()
                .rows()
                .search_vectors(&inputs.query, k, args.nprobe)
                .await?;
            Ok(Answer::TopK(
                table.ranked_ids(indexes.ivf.ranked(&hits)?).await?,
            ))
        })
        .await?;
        let (lance_time, lance) = time(args.iterations, || async {
            let mut scanner = dataset.scan();
            scanner.project(&["id"])?;
            scanner.nearest("emb", &la::Float32Array::from(inputs.query.clone()), k)?;
            scanner.distance_metric(MetricType::Cosine);
            scanner.nprobes(args.nprobe);
            Ok(Answer::TopK(lance_ids(&mut scanner).await?))
        })
        .await?;
        let (Answer::TopK(vortex), Answer::TopK(lance)) = (vortex, lance) else {
            anyhow::bail!("top-k answers");
        };
        row(
            "v2",
            vortex_time,
            lance_time,
            format!(
                "recall@{k}: vortex {:.1}, lance {:.1}",
                recall(&vortex, &exact),
                recall(&lance, &exact)
            ),
        );
    }

    if selected("t2") {
        let (vortex_time, vortex) = time(args.iterations, || async {
            let hits = indexes
                .text
                .index()
                .rows()
                .search_bm25(&inputs.bm25_query, k)
                .await?;
            Ok(Answer::TopK(
                table.ranked_ids(indexes.text.ranked(&hits)?).await?,
            ))
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
            let found = indexes
                .text
                .locate(&text_match(root(), inputs.term.as_str()))
                .await?
                .ok_or_else(|| anyhow::anyhow!("the inverted index claims text_match"))?;
            Ok(Answer::Rows(
                table.ids_of(found.selections).await?.len() as u64
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
