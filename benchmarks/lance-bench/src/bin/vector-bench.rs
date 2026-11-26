// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Warm-index, end-to-end DataFusion benchmark for cosine top-10 and filtered cosine top-10.

#[path = "../../workloads/fixture.rs"]
mod fixture;

use std::num::NonZeroU32;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use clap::Parser;
use clap::ValueEnum;
use datafusion::arrow::array::Array;
use datafusion::arrow::array::AsArray;
use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::array::types::UInt64Type;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::displayable;
use datafusion::prelude::SessionContext;
use fixture::Data;
use fixture::K;
use fixture::Scale;
use fixture::build_ivf;
use fixture::exact_cosine;
use fixture::lance_dataset;
use fixture::price;
use lance_bench::vbench::register_lance;
use lance_bench::vbench::register_topk;
use lance_bench::vbench::register_vortex;
use serde_json::json;
use tokio::fs::File;
use vortex::VortexSessionDefault;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_bench::utils::GIT_COMMIT_ID;
use vortex_bench::v3::ENV_TRIPLE;
use vortex_bench::v3::QueryMeasurementRecord;
use vortex_bench::v3::V3Record;
use vortex_bench::v3::write_jsonl_to_path;
use vortex_index::IndexFile;
use vortex_index_ivf::IvfIndex;

#[derive(Clone, Copy, ValueEnum)]
enum Engine {
    Vortex,
    Lance,
}

impl Engine {
    fn name(self) -> &'static str {
        match self {
            Self::Vortex => "vortex",
            Self::Lance => "lance",
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Query {
    Q1,
    Q4,
}

impl Query {
    fn number(self) -> u32 {
        match self {
            Self::Q1 => 1,
            Self::Q4 => 4,
        }
    }
    fn floor(self) -> Option<i64> {
        match self {
            Self::Q1 => None,
            Self::Q4 => Some(90),
        }
    }
}

#[derive(Parser)]
#[command(
    about = "Measure Vortex and Lance indexed SQL retrieval over the same seeded recipe table"
)]
struct Args {
    /// Use the 400k x 128 fixture (default: 20k x 32).
    #[arg(long)]
    large: bool,
    /// Restrict the run to one engine or query.
    #[arg(long, value_enum)]
    engine: Option<Engine>,
    #[arg(long, value_enum)]
    query: Option<Query>,
    /// Complete passes over the 21 query vectors, after one warmup pass.
    #[arg(long, default_value = "5")]
    rounds: NonZeroU32,
    /// IVF partitions to probe on each engine.
    #[arg(long, default_value = "8")]
    nprobes: NonZeroU32,
    /// Parent directory for disposable, file-backed datasets and indexes.
    #[arg(long, default_value = ".")]
    work_dir: PathBuf,
    /// Save quality, sample counts, and timing percentiles as JSON.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Write timing samples in the same JSONL format as the SQL benchmark suites.
    #[arg(long)]
    ingest_jsonl: Option<PathBuf>,
}

async fn setup(
    ctx: &SessionContext,
    engine: Engine,
    dir: &Path,
    data: &Data,
    probes: usize,
) -> Result<()> {
    match engine {
        Engine::Vortex => {
            let session = VortexSession::default().with_tokio();
            vortex_index::initialize(&session)?;
            vortex_index_ivf::register(&session);
            let path = dir.join("data.vortex");
            let index_path = dir.join("ivf.vortex");
            session
                .write_options()
                .write(
                    File::create(&path).await?,
                    data.vortex_array()?.to_array_stream(),
                )
                .await?;
            let file = session
                .open_options()
                .include_metadata()
                .open_path(&path)
                .await?;
            build_ivf(&session, &file, data.scale)
                .await?
                .write(File::create(&index_path).await?)
                .await?;
            let index_file = session
                .open_options()
                .include_metadata()
                .open_path(index_path)
                .await?;
            register_vortex(
                ctx,
                "vortex",
                file,
                &IndexFile::open(IvfIndex, &index_file)?,
                probes,
            )?;
        }
        Engine::Lance => {
            let path = dir.join("recipes.lance");
            let dataset = lance_dataset(
                path.to_str().context("expected a UTF-8 dataset path")?,
                data,
            )
            .await?;
            register_lance(ctx, "lance", Arc::new(dataset), probes)?;
        }
    }
    register_topk(ctx, engine.name(), &format!("{}_knn", engine.name())).await?;
    Ok(())
}

async fn execute(
    ctx: &SessionContext,
    sql: &str,
    check: Option<Engine>,
) -> Result<Vec<(u64, f64)>> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    if let Some(engine) = check {
        let plan = displayable(plan.as_ref()).indent(true).to_string();
        let index = match engine {
            Engine::Vortex => "VortexIvfExec",
            Engine::Lance => "ANNIvf",
        };
        ensure!(
            plan.contains(index),
            "expected {index} in the physical plan:\n{plan}"
        );
    }
    let mut hits = Vec::with_capacity(K);
    for batch in collect(plan, ctx.task_ctx()).await? {
        let ids = batch.column(0).as_primitive::<UInt64Type>();
        let distances = batch.column(1).as_primitive::<Float64Type>();
        ensure!(
            ids.null_count() + distances.null_count() == 0,
            "unexpected null search result"
        );
        hits.extend(
            ids.values()
                .iter()
                .copied()
                .zip(distances.values().iter().copied()),
        );
    }
    Ok(hits)
}

struct Case {
    engine: Engine,
    sql: Vec<String>,
    samples: Vec<u64>,
    matches: usize,
    returned: usize,
}

impl Case {
    fn new(engine: Engine, vectors: &[Vec<f32>], query: Query) -> Result<Self> {
        let name = engine.name();
        let template = match query {
            Query::Q1 => include_str!("../../queries/vbench/indexed/q1.sql"),
            Query::Q4 => include_str!("../../queries/vbench/indexed/q4.sql"),
        }
        .replace("recipe_knn", &format!("{name}_knn"));
        let sql = vectors
            .iter()
            .map(|v| {
                let vector = serde_json::to_string(v)?;
                Ok(template.replace("$vector", &vector))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            engine,
            sql,
            samples: vec![],
            matches: 0,
            returned: 0,
        })
    }

    async fn run(
        &mut self,
        ctx: &SessionContext,
        i: usize,
        exact: &[u64],
        query: Query,
        rows: usize,
        warmup: bool,
    ) -> Result<()> {
        let start = Instant::now();
        let check = (warmup && i == 0).then_some(self.engine);
        let hits = execute(ctx, &self.sql[i], check).await?;
        let elapsed = u64::try_from(start.elapsed().as_nanos())?;
        ensure!(
            hits.len() <= K && hits.windows(2).all(|p| p[0].1 <= p[1].1),
            "invalid top-k ordering or length"
        );
        let mut distinct = Vec::with_capacity(K);
        for &(id, distance) in &hits {
            let row = usize::try_from(id)?;
            ensure!(
                row < rows
                    && distance.is_finite()
                    && !distinct.contains(&id)
                    && query.floor().is_none_or(|floor| price(row) >= floor),
                "invalid or duplicate row {id}"
            );
            distinct.push(id);
        }
        if !warmup {
            self.samples.push(elapsed);
            self.matches += exact.iter().filter(|id| distinct.contains(id)).count();
            self.returned += hits.len();
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let scale = if args.large {
        Scale::LARGE
    } else {
        Scale::SMALL
    };
    let probes = args.nprobes.get();
    ensure!(
        probes <= scale.lists,
        "nprobes exceeds the {} IVF lists",
        scale.lists
    );
    let engines = args
        .engine
        .map_or_else(|| vec![Engine::Vortex, Engine::Lance], |e| vec![e]);
    let queries = args
        .query
        .map_or_else(|| vec![Query::Q1, Query::Q4], |q| vec![q]);
    std::fs::create_dir_all(&args.work_dir)?;
    let dir = tempfile::tempdir_in(args.work_dir.canonicalize()?)?;
    let ctx = SessionContext::new();
    let data = Data::generate(scale);
    for &engine in &engines {
        setup(&ctx, engine, dir.path(), &data, probes as usize).await?;
    }
    let vectors = data.vector_queries();
    let mut records = Vec::new();
    let mut results = Vec::new();
    for query in queries {
        let exact: Vec<_> = vectors
            .iter()
            .map(|q| exact_cosine(&data, q, query.floor()))
            .collect();
        let mut cases = engines
            .iter()
            .map(|e| Case::new(*e, &vectors, query))
            .collect::<Result<Vec<_>>>()?;
        for pass in 0..=args.rounds.get() {
            for (i, expected) in exact.iter().enumerate() {
                let count = cases.len();
                for offset in 0..count {
                    let case = &mut cases[(offset + i + pass as usize) % count];
                    case.run(&ctx, i, expected, query, scale.rows, pass == 0)
                        .await?;
                }
            }
        }
        for case in cases {
            let mut sorted = case.samples.clone();
            sorted.sort_unstable();
            let p50 = sorted[sorted.len() / 2];
            let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1];
            let possible = exact.iter().map(Vec::len).sum::<usize>() * args.rounds.get() as usize;
            ensure!(
                case.matches > 0,
                "{} Q{} returned zero correct neighbors across the workload",
                case.engine.name(),
                query.number()
            );
            let recall = case.matches as f64 / possible as f64;
            let name = case.engine.name();
            println!(
                "Q{} {name}: p50 {:.1} µs, p95 {:.1} µs, recall@{K} {recall:.3} ({} queries)",
                query.number(),
                p50 as f64 / 1000.,
                p95 as f64 / 1000.,
                sorted.len()
            );
            results.push(json!({
                "query": query.number(),
                "format": name,
                "p50_ns": p50,
                "p95_ns": p95,
                "recall_at_10": recall,
                "returned": case.returned,
                "correct": case.matches,
                "expected": possible,
                "samples": sorted.len(),
            }));
            records.push(V3Record::QueryMeasurement(QueryMeasurementRecord {
                commit_sha: GIT_COMMIT_ID.clone(),
                dataset: "vbench".into(),
                dataset_variant: Some(format!(
                    "synthetic-{}-d{}-ivf{}-p{probes}",
                    scale.label, scale.dim, scale.lists
                )),
                scale_factor: None,
                query_idx: query.number(),
                storage: "nvme".into(),
                engine: "datafusion".into(),
                format: name.into(),
                value_ns: p50,
                all_runtimes_ns: case.samples,
                peak_physical: None,
                peak_virtual: None,
                physical_delta: None,
                virtual_delta: None,
                env_triple: Some(ENV_TRIPLE.clone()),
            }));
        }
    }
    if let Some(path) = args.report {
        let report = json!({
            "commit_sha": *GIT_COMMIT_ID,
            "rows": scale.rows,
            "dimensions": scale.dim,
            "lists": scale.lists,
            "nprobes": probes,
            "rounds": args.rounds.get(),
            "results": results,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    if let Some(path) = args.ingest_jsonl {
        write_jsonl_to_path(&path, &records)?;
    }
    Ok(())
}
