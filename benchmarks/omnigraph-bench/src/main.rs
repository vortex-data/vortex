// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare Omnigraph's Lance-backed graph storage against Vortex.
//!
//! Omnigraph (<https://github.com/ModernRelay/omnigraph>) stores every node type and edge type of
//! a graph as its own Lance dataset (`<graph>/nodes/<id>`, `<graph>/edges/<id>`). This tool:
//!
//! 1. `convert`: reads the current version of each of those datasets in a graph that Omnigraph
//!    wrote and writes the same rows as a Vortex file with Vortex's default write options. The
//!    Lance baseline is Omnigraph's own data files, untouched. For reference it also writes a
//!    Parquet file (zstd level 3) and a Lance copy with zstd forced on every column. It reports
//!    on-disk sizes.
//! 2. `query`: runs Omnigraph's benchmark queries, translated to SQL, in DataFusion over
//!    Omnigraph's Lance tables, the zstd Lance copy and the Vortex files, and reports runtimes. All
//!    formats must return identical results before the tool reports timings.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use futures::TryStreamExt;
use lance::datafusion::LanceTableProvider;
use lance::dataset::Dataset as LanceDataset;
use lance::dataset::WriteParams;
use lance::deps::arrow_array::RecordBatch as LanceRecordBatch;
use lance::deps::arrow_array::RecordBatchIterator;
use lance::deps::arrow_schema::Schema;
use lance::deps::arrow_schema::SchemaRef;
use lance::deps::datafusion as df54;
use lance_file::version::LanceFileVersion;
use mimalloc::MiMalloc;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use serde::Serialize;
use url::Url;
use vortex_bench::CompactionStrategy;
use vortex_bench::Format;
use vortex_bench::conversions::convert_parquet_file_to_vortex;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(about = "Omnigraph on Lance vs Vortex: file size and query runtime")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Export the node and edge tables of an Omnigraph graph to Lance, Parquet and Vortex.
    Convert {
        /// An Omnigraph table to export, as `<sql name>=<dataset directory>`, for example
        /// `person=graph.omni/nodes/0000000000000001-0000000000000008`. Omnigraph keeps each node
        /// type's Lance dataset under `nodes/` and each edge type's under `edges/`.
        #[arg(long = "table", required = true, value_parser = parse_table)]
        tables: Vec<(String, PathBuf)>,
        /// Output directory; gets `lance/`, `parquet/` and `vortex/` subdirectories.
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the graph query suite over the converted Lance and Vortex tables.
    Query {
        /// The `--out` directory of `convert`.
        #[arg(long)]
        data: PathBuf,
        /// Timed runs per query and format, after one warm-up run.
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        /// Write the results as JSON to this file.
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn parse_table(arg: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = arg
        .split_once('=')
        .ok_or_else(|| format!("expected <name>=<path>, got {arg}"))?;
    let path = PathBuf::from(path);
    if !path.join("_versions").is_dir() {
        return Err(format!("{} is not a Lance dataset", path.display()));
    }
    Ok((name.to_string(), path))
}

/// Omnigraph's own benchmark queries (`benchmarks/workloads/e2e_query_*.gqt` and
/// `e2e_traversal_*.gqt` for the `Person`/`Knows` schema), translated from GQ to SQL. Each entry
/// is `(Omnigraph scenario, GQ query, SQL)`. A `$a knows $b` edge pattern joins the edge table on
/// its `__src`/`__dst` node keys and requires both endpoints to exist, as Omnigraph does.
///
/// Workloads that need a Lance vector or full-text index (`nearest`, `rrf`, `search`) have no
/// Vortex equivalent and are left out, except `destination_search`, which runs as a substring
/// filter without an index on both formats. Full-graph `hop2`/`hop3` return `count(*)` over the
/// same join instead of every path, which would not fit in memory at this scale.
fn queries() -> Vec<(&'static str, &'static str, String)> {
    let person = |alias: &str| format!("JOIN person {alias} ON {alias}.name");
    vec![
        (
            "count_all",
            "match { $p: Person } return { count($p) as n }",
            "SELECT count(*) AS n FROM person".into(),
        ),
        (
            "scan",
            "match { $p: Person } return { $p.name }",
            "SELECT name FROM person".into(),
        ),
        (
            "wide_scan",
            "match { $p: Person } return { $p }",
            "SELECT * FROM person".into(),
        ),
        (
            "lookup",
            "match { $p: Person { name: \"p500\" } } return { $p.name, $p.age }",
            "SELECT name, age FROM person WHERE name = 'p500'".into(),
        ),
        (
            "filter",
            "match { $p: Person $p.age >= 90 } return { $p.name, $p.age }",
            "SELECT name, age FROM person WHERE age >= 90".into(),
        ),
        (
            "top_people",
            "return { $p.name, $p.age } order { $p.age desc, $p.name asc } limit 20",
            "SELECT name, age FROM person ORDER BY age DESC, name ASC LIMIT 20".into(),
        ),
        (
            "grouped",
            "return { $p.team, sum($p.age), avg($p.age), count($p) }",
            "SELECT team, sum(age) AS total, round(avg(age), 6) AS mean, count(*) AS n \
             FROM person GROUP BY team"
                .into(),
        ),
        (
            "friends",
            "match { $p: Person $p knows $q } return { $p.name, $q.name }",
            format!(
                "SELECT p.name AS p, q.name AS q FROM knows k {} = k.__src {} = k.__dst",
                person("p"),
                person("q")
            ),
        ),
        (
            "destination_projection",
            "match { $p: Person $p knows $q } return { $q.name }",
            format!(
                "SELECT q.name FROM knows k {} = k.__src {} = k.__dst",
                person("p"),
                person("q")
            ),
        ),
        (
            "filtered_friends",
            "match { $p: Person $p.age >= 90 $p knows $q } return { $p.name, $q.name }",
            format!(
                "SELECT p.name AS p, q.name AS q FROM knows k {} = k.__src {} = k.__dst \
                 WHERE p.age >= 90",
                person("p"),
                person("q")
            ),
        ),
        (
            "grouped_fanout",
            "match { $p: Person $p knows $q } return { $p.team, count($q) as n }",
            format!(
                "SELECT p.team, count(q.name) AS n FROM knows k {} = k.__src {} = k.__dst \
                 GROUP BY p.team",
                person("p"),
                person("q")
            ),
        ),
        (
            "no_friends",
            "match { $p: Person not { $p knows $q } } return { $p.name }",
            "SELECT p.name FROM person p WHERE NOT EXISTS (SELECT 1 FROM knows k \
             JOIN person q ON q.name = k.__dst WHERE k.__src = p.name)"
                .into(),
        ),
        (
            "destination_search",
            "match { $p: Person $p knows $q search($q.bio, \"vector search\") } return { $q.name }",
            format!(
                "SELECT q.name FROM knows k {} = k.__src {} = k.__dst \
                 WHERE q.bio LIKE '%vector search%'",
                person("p"),
                person("q")
            ),
        ),
        (
            "hop1",
            "$a knows $b return { $a.name, $b.name }",
            format!(
                "SELECT a.name AS a, b.name AS b FROM knows k1 {} = k1.__src {} = k1.__dst",
                person("a"),
                person("b")
            ),
        ),
        (
            "hop2",
            "$a knows $b $b knows $c return { $a.name, $c.name } (as count)",
            format!(
                "SELECT count(*) AS paths FROM knows k1 {} = k1.__src {} = k1.__dst \
                 JOIN knows k2 ON k2.__src = b.name {} = k2.__dst",
                person("a"),
                person("b"),
                person("c")
            ),
        ),
        (
            "hop3",
            "$a knows $b $b knows $c $c knows $d return { $a.name, $d.name } (as count)",
            format!(
                "SELECT count(*) AS paths FROM knows k1 {} = k1.__src {} = k1.__dst \
                 JOIN knows k2 ON k2.__src = b.name {} = k2.__dst \
                 JOIN knows k3 ON k3.__src = c.name {} = k3.__dst",
                person("a"),
                person("b"),
                person("c"),
                person("d")
            ),
        ),
        (
            "selective",
            "match { $a: Person { name: \"p42\" } $a knows{1,2} $b } return { $b.name }",
            format!(
                "SELECT b.name FROM knows k1 {} = k1.__dst WHERE k1.__src = 'p42' \
                 UNION ALL \
                 SELECT c.name FROM knows k1 {} = k1.__dst JOIN knows k2 ON k2.__src = b.name \
                 {} = k2.__dst WHERE k1.__src = 'p42'",
                person("b"),
                person("b"),
                person("c")
            ),
        ),
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Args::parse().command {
        Command::Convert { tables, out } => convert(tables, &out).await,
        Command::Query {
            data,
            iterations,
            json,
        } => query(&data, iterations, json.as_deref()).await,
    }
}

/// Open the version of an Omnigraph table that its graph commit points at.
///
/// Omnigraph commits each table version as a Lance *detached* version (`_versions/d<N>.manifest`)
/// and records it in the graph's own `__manifest`, so the dataset's regular latest version is the
/// empty one `init` created. After `omnigraph cleanup --keep 1` exactly one detached version
/// remains: the current one.
async fn open_omnigraph_table(path: &Path) -> anyhow::Result<LanceDataset> {
    let uri = path.to_str().context("non UTF-8 path")?;
    let mut detached = Vec::new();
    for entry in fs::read_dir(path.join("_versions"))? {
        let file = entry?.file_name();
        let file = file.to_string_lossy();
        if let Some(version) = file
            .strip_prefix('d')
            .and_then(|f| f.strip_suffix(".manifest"))
        {
            detached.push(version.parse::<u64>()?);
        }
    }
    let dataset = LanceDataset::open(uri)
        .await
        .with_context(|| format!("opening {uri}"))?;
    match detached.as_slice() {
        [] => Ok(dataset),
        [version] => Ok(dataset.checkout_version(*version).await?),
        _ => bail!(
            "{uri} has {} detached versions; run `omnigraph cleanup --keep 1` first",
            detached.len()
        ),
    }
}

fn dir_size(path: &Path) -> anyhow::Result<u64> {
    if path.is_file() {
        return Ok(fs::metadata(path)?.len());
    }
    let mut total = 0;
    for entry in fs::read_dir(path)? {
        total += dir_size(&entry?.path())?;
    }
    Ok(total)
}

/// Bytes of the Lance data files (`data/*.lance`) of a dataset, excluding manifests and indexes.
fn lance_data_size(path: &Path) -> anyhow::Result<u64> {
    let data = path.join("data");
    if data.is_dir() {
        dir_size(&data)
    } else {
        Ok(0)
    }
}

async fn write_lance(
    path: &Path,
    batches: &[LanceRecordBatch],
    schema: SchemaRef,
) -> anyhow::Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    let reader = RecordBatchIterator::new(batches.to_vec().into_iter().map(Ok), schema);
    LanceDataset::write(
        reader,
        path.to_str().context("non UTF-8 path")?,
        Some(WriteParams::with_storage_version(LanceFileVersion::V2_2)),
    )
    .await?;
    Ok(())
}

/// `schema` with Lance's zstd compression requested on every column.
fn zstd_schema(schema: &Schema) -> SchemaRef {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let mut metadata = field.metadata().clone();
            metadata.insert("lance-encoding:compression".into(), "zstd".into());
            metadata.insert("lance-encoding:compression-level".into(), "3".into());
            Arc::new(field.as_ref().clone().with_metadata(metadata))
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[derive(Serialize)]
struct SizeRow {
    table: String,
    rows: usize,
    omnigraph_dir_bytes: u64,
    lance_bytes: u64,
    lance_zstd_bytes: u64,
    parquet_zstd3_bytes: u64,
    vortex_bytes: u64,
}

async fn convert(tables: Vec<(String, PathBuf)>, out: &Path) -> anyhow::Result<()> {
    for sub in ["lance_zstd", "parquet", "vortex"] {
        fs::create_dir_all(out.join(sub))?;
    }

    let mut rows = Vec::new();
    let mut lance_tables = BTreeMap::new();
    for (table, path) in &tables {
        let name = table.clone();
        lance_tables.insert(name.clone(), fs::canonicalize(path)?);
        let dataset = open_omnigraph_table(path).await?;
        let batches: Vec<LanceRecordBatch> = dataset
            .scan()
            .try_into_stream()
            .await?
            .try_collect()
            .await?;
        let schema = match batches.first() {
            Some(batch) => batch.schema(),
            None => Arc::new(dataset.schema().into()),
        };
        let num_rows = batches.iter().map(|b| b.num_rows()).sum();
        println!(
            "{table}: {num_rows} rows, schema: {}",
            schema
                .fields()
                .iter()
                .map(|f| format!("{}: {}", f.name(), f.data_type()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Secondary comparison: the same rows rewritten as Lance with zstd forced on every column.
        let zstd_schema = zstd_schema(&schema);
        let zstd_batches = batches
            .iter()
            .map(|b| b.clone().with_schema(Arc::clone(&zstd_schema)))
            .collect::<Result<Vec<_>, _>>()?;
        let lance_zstd_path = out.join("lance_zstd").join(format!("{name}.lance"));
        write_lance(&lance_zstd_path, &zstd_batches, zstd_schema).await?;

        let parquet_path = out.join("parquet").join(format!("{name}.parquet"));
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        let mut writer = ArrowWriter::try_new(
            fs::File::create(&parquet_path)?,
            Arc::clone(&schema),
            Some(props),
        )?;
        for batch in &batches {
            writer.write(batch)?;
        }
        writer.close()?;

        let vortex_path = out.join("vortex").join(format!("{name}.vortex"));
        convert_parquet_file_to_vortex(&parquet_path, &vortex_path, CompactionStrategy::Default)
            .await?;

        rows.push(SizeRow {
            table: table.clone(),
            rows: num_rows,
            omnigraph_dir_bytes: dir_size(path)?,
            lance_bytes: lance_data_size(path)?,
            lance_zstd_bytes: lance_data_size(&lance_zstd_path)?,
            parquet_zstd3_bytes: dir_size(&parquet_path)?,
            vortex_bytes: dir_size(&vortex_path)?,
        });
    }

    println!();
    println!(
        "| table | rows | Omnigraph table dir (data + manifests + indexes) | Omnigraph Lance data files (as written) | Lance + forced zstd | Parquet zstd(3) | Vortex (default) | Vortex vs Omnigraph Lance | Vortex vs Lance+zstd |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    let mut totals = [0u64; 5];
    let print_row = |table: &str, rows: String, t: [u64; 5]| {
        println!(
            "| {table} | {rows} | {:.2} MiB | {:.2} MiB | {:.2} MiB | {:.2} MiB | {:.2} MiB | {:.2}x smaller | {:.2}x smaller |",
            mib(t[0]),
            mib(t[1]),
            mib(t[2]),
            mib(t[3]),
            mib(t[4]),
            t[1] as f64 / t[4] as f64,
            t[2] as f64 / t[4] as f64,
        );
    };
    for r in &rows {
        let t = [
            r.omnigraph_dir_bytes,
            r.lance_bytes,
            r.lance_zstd_bytes,
            r.parquet_zstd3_bytes,
            r.vortex_bytes,
        ];
        for (total, v) in totals.iter_mut().zip(t) {
            *total += v;
        }
        print_row(&r.table, r.rows.to_string(), t);
    }
    print_row("**total**", String::new(), totals);
    fs::write(out.join("sizes.json"), serde_json::to_string_pretty(&rows)?)?;
    fs::write(
        out.join("lance_tables.json"),
        serde_json::to_string_pretty(&lance_tables)?,
    )?;
    Ok(())
}

fn table_files(dir: &Path, ext: &str) -> anyhow::Result<Vec<(String, PathBuf)>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == ext) {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .context("bad file name")?
                .to_string();
            files.push((stem, path));
        }
    }
    files.sort();
    Ok(files)
}

async fn lance_context(
    tables: Vec<(String, PathBuf)>,
) -> anyhow::Result<df54::prelude::SessionContext> {
    let ctx = df54::prelude::SessionContext::new();
    for (name, path) in tables {
        let dataset = open_omnigraph_table(&path).await?;
        ctx.register_table(
            name.as_str(),
            Arc::new(LanceTableProvider::new(Arc::new(dataset), false, false)),
        )?;
    }
    Ok(ctx)
}

async fn vortex_context(dir: &Path) -> anyhow::Result<datafusion::prelude::SessionContext> {
    let ctx = datafusion_bench::get_session_context();
    let base = Url::from_directory_path(fs::canonicalize(dir)?)
        .map_err(|()| anyhow::anyhow!("bad vortex directory"))?;
    datafusion_bench::make_object_store(&ctx, &base)?;
    let format = datafusion_bench::format_to_df_format(Format::OnDiskVortex, false)?;
    for (name, _) in table_files(dir, "vortex")? {
        let url = ListingTableUrl::parse(base.join(&format!("{name}.vortex"))?)?;
        let options = ListingOptions::new(Arc::clone(&format)).with_file_extension(".vortex");
        let config = ListingTableConfig::new(url)
            .with_listing_options(options)
            .infer_schema(&ctx.state())
            .await?;
        ctx.register_table(name.as_str(), Arc::new(ListingTable::try_new(config)?))?;
    }
    Ok(ctx)
}

/// What a query returned, comparable across the two DataFusion (and Arrow) versions: the row
/// count and, for results small enough to render, the rows as sorted text lines.
#[derive(PartialEq, Eq, Debug)]
struct Answer {
    rows: usize,
    lines: Option<Vec<String>>,
}

const RENDER_ROWS: usize = 20_000;

fn answer(rows: usize, render: impl FnOnce() -> anyhow::Result<String>) -> anyhow::Result<Answer> {
    let lines = if rows <= RENDER_ROWS {
        let mut lines: Vec<String> = render()?.lines().map(str::to_string).collect();
        lines.sort();
        Some(lines)
    } else {
        None
    };
    Ok(Answer { rows, lines })
}

enum Engine {
    Lance(df54::prelude::SessionContext),
    Vortex(datafusion::prelude::SessionContext),
}

impl Engine {
    async fn run(&self, sql: &str) -> anyhow::Result<Answer> {
        match self {
            Engine::Lance(ctx) => {
                let batches = ctx.sql(sql).await?.collect().await?;
                let rows = batches.iter().map(|b| b.num_rows()).sum();
                answer(rows, || {
                    Ok(df54::arrow::util::pretty::pretty_format_batches(&batches)?.to_string())
                })
            }
            Engine::Vortex(ctx) => {
                let batches = ctx.sql(sql).await?.collect().await?;
                let rows = batches.iter().map(|b| b.num_rows()).sum();
                answer(rows, || {
                    Ok(
                        datafusion::arrow::util::pretty::pretty_format_batches(&batches)?
                            .to_string(),
                    )
                })
            }
        }
    }

    /// Time only query execution and result collection, not the comparison rendering.
    async fn time(&self, sql: &str) -> anyhow::Result<Duration> {
        let start = Instant::now();
        match self {
            Engine::Lance(ctx) => {
                ctx.sql(sql).await?.collect().await?;
            }
            Engine::Vortex(ctx) => {
                ctx.sql(sql).await?.collect().await?;
            }
        }
        Ok(start.elapsed())
    }
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[derive(Serialize)]
struct QueryRow {
    query: String,
    gq: String,
    sql: String,
    result_rows: usize,
    /// Median milliseconds per format, in `FORMATS` order.
    median_ms: Vec<f64>,
    min_ms: Vec<f64>,
}

const FORMATS: [&str; 3] = ["omnigraph_lance", "lance_zstd", "vortex"];

async fn query(data: &Path, iterations: usize, json: Option<&Path>) -> anyhow::Result<()> {
    let omnigraph_tables: BTreeMap<String, PathBuf> =
        serde_json::from_str(&fs::read_to_string(data.join("lance_tables.json"))?)?;
    let omnigraph_tables = omnigraph_tables.into_iter().collect();
    let engines = [
        // The tables exactly as Omnigraph wrote them, opened at their current version.
        Engine::Lance(lance_context(omnigraph_tables).await?),
        Engine::Lance(lance_context(table_files(&data.join("lance_zstd"), "lance")?).await?),
        Engine::Vortex(vortex_context(&data.join("vortex")).await?),
    ];

    let mut results = Vec::new();
    for (name, gq, sql) in queries() {
        // Warm-up, and check every format returns the same answer.
        let mut answers = Vec::new();
        for (format, engine) in FORMATS.iter().zip(&engines) {
            answers.push(
                engine
                    .run(&sql)
                    .await
                    .with_context(|| format!("{format} {name}"))?,
            );
        }
        for (format, answer) in FORMATS.iter().zip(&answers).skip(1) {
            if answer != &answers[0] {
                bail!(
                    "{name}: {format} returned a different result from lance\n{answer:?}\n{:?}",
                    answers[0]
                );
            }
        }

        // Interleave the formats so drift in machine state affects all of them equally.
        let mut times = vec![Vec::with_capacity(iterations); engines.len()];
        for _ in 0..iterations {
            for (engine, times) in engines.iter().zip(&mut times) {
                times.push(engine.time(&sql).await?);
            }
        }
        let row = QueryRow {
            query: name.to_string(),
            gq: gq.to_string(),
            sql: sql.clone(),
            result_rows: answers[0].rows,
            median_ms: times.iter_mut().map(|t| ms(median(t))).collect(),
            min_ms: times.iter().map(|t| ms(t[0])).collect(),
        };
        eprintln!(
            "{name} ({} rows): omnigraph lance {:.2} ms, lance+zstd {:.2} ms, vortex {:.2} ms",
            row.result_rows, row.median_ms[0], row.median_ms[1], row.median_ms[2]
        );
        results.push(row);
    }

    println!();
    println!(
        "| Omnigraph query | rows | Omnigraph Lance (ms) | Lance+zstd (ms) | Vortex default (ms) | Vortex speedup vs Omnigraph Lance | vs Lance+zstd |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for r in &results {
        let m = &r.median_ms;
        println!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2}x | {:.2}x |",
            r.query,
            r.result_rows,
            m[0],
            m[1],
            m[2],
            m[0] / m[2],
            m[1] / m[2]
        );
    }
    let geo = |i: usize| {
        (results
            .iter()
            .map(|r| (r.median_ms[i] / r.median_ms[2]).ln())
            .sum::<f64>()
            / results.len() as f64)
            .exp()
    };
    println!(
        "| **geomean speedup** | | | | | {:.2}x | {:.2}x |",
        geo(0),
        geo(1)
    );
    if let Some(path) = json {
        fs::write(path, serde_json::to_string_pretty(&results)?)?;
    }
    Ok(())
}
