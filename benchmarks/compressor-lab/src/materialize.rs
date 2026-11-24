// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Turning a store's facts and observations into training tables.
//!
//! The output keeps the CSV layout `benchmarks/compressor-feasibility/train.py` reads:
//! `rows-<tag>.csv` (one row per chunk and candidate) and `features-<tag>.csv` (one row per
//! chunk), plus `dataset.json` recording exactly what went in. Timings are pooled over every run
//! on one machine; other machines are never mixed in.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use anyhow::Context;
use serde::Serialize;

use crate::chunks::CHUNKS_VERSION;
use crate::plan::CANDIDATES_VERSION;
use crate::plan::FEATURES_VERSION;
use crate::plan::PlannedRun;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::store::Store;
use crate::store::machine_dir;
use crate::table::ChunkRow;
use crate::table::EncodingRow;
use crate::table::FeatureCostRow;
use crate::table::FeatureRow;
use crate::table::MeasureRow;
use crate::table::SearchRow;
use crate::table::SiteFactRow;
use crate::table::TimingRow;

/// A map from a key to the samples pooled for it.
type Samples = BTreeMap<(String, String), Vec<u64>>;

/// What a materialized dataset contains.
#[derive(Debug, Serialize)]
pub struct DatasetManifest {
    /// The plan it was built from.
    pub plan_id: String,
    /// The code the facts were produced with.
    pub git_commit: String,
    /// The Vortex version.
    pub vx_version: String,
    /// The machine whose timings were used, if any.
    pub machine_id: Option<String>,
    /// Chunks with all facts present.
    pub chunks: u64,
    /// Chunks skipped because a fact was missing.
    pub incomplete: u64,
    /// Chunks with at least one timing run.
    pub timed_chunks: u64,
    /// The largest number of timing runs pooled for one chunk.
    pub max_runs: u64,
    /// Each source's cross-validation group. Sources in one group are held out together.
    pub groups: BTreeMap<String, String>,
}

/// What one timing run recorded for one candidates task, besides candidate timings.
#[derive(Default)]
struct RunRows {
    measures: Vec<MeasureRow>,
    costs: Vec<FeatureCostRow>,
    production: bool,
    reference_ns: Option<u64>,
    compress_ns: Vec<u64>,
    replay_ns: Vec<u64>,
}

fn median(values: &mut [u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

fn variant(name: &str) -> &str {
    match name {
        "production" => "default",
        "sizemodel" => "model/runend+sparse",
        other => other,
    }
}

fn csv(text: &str) -> String {
    text.replace([',', '\n'], ";")
}

/// Writes the dataset for `plan` from `store` into `out`.
pub fn materialize(
    plan: &PlannedRun,
    store: &Store,
    out: &Path,
    machine: Option<&str>,
    tag: &str,
) -> anyhow::Result<DatasetManifest> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let machine = match machine {
        Some(m) => Some(m.to_string()),
        None => {
            let machines = store.timing_machines()?;
            if machines.len() > 1 {
                anyhow::bail!(
                    "timings exist for several machines ({}); pass --machine to pick one, since \
                     timings from different machines are never mixed",
                    machines.join(", ")
                );
            }
            machines.into_iter().next()
        }
    };

    let chunks: BTreeMap<String, ChunkRow> = store
        .read_table::<ChunkRow>(CHUNKS_VERSION)?
        .into_iter()
        .map(|r| (r.task_key.clone(), r))
        .collect();
    // Every done features task has an entry, even one with no whole-chunk features (non-int
    // chunks only have site features).
    let mut features: BTreeMap<String, BTreeMap<String, Option<f64>>> = store
        .ledger(TaskKind::Features)?
        .into_keys()
        .map(|k| (k.to_string(), BTreeMap::new()))
        .collect();
    let mut site_features: BTreeMap<String, Vec<FeatureRow>> = BTreeMap::new();
    for row in store.read_table::<FeatureRow>(FEATURES_VERSION)? {
        if row.site_hash.is_some() {
            site_features
                .entry(row.task_key.clone())
                .or_default()
                .push(row);
        } else if row.site == "root" && row.window.is_none() {
            features
                .entry(row.task_key)
                .or_default()
                .insert(row.feature, row.value);
        }
    }
    let mut encodings: BTreeMap<String, Vec<EncodingRow>> = BTreeMap::new();
    for row in store.read_table::<EncodingRow>(CANDIDATES_VERSION)? {
        encodings.entry(row.task_key.clone()).or_default().push(row);
    }
    let mut site_rows: BTreeMap<String, Vec<SiteFactRow>> = BTreeMap::new();
    for row in store.read_table::<SiteFactRow>(CANDIDATES_VERSION)? {
        site_rows.entry(row.task_key.clone()).or_default().push(row);
    }
    let search_rows: BTreeMap<String, SearchRow> = store
        .read_table::<SearchRow>(CANDIDATES_VERSION)?
        .into_iter()
        .map(|r| (r.task_key.clone(), r))
        .collect();

    // Plan replays by (candidates task, encoding); the sampling compressor by (candidates task,
    // candidate); decode by (chunk, encoding). Samples whose replay did not reproduce the
    // recorded encoding are left out.
    let mut compress = Samples::new();
    let mut adaptive = Samples::new();
    let mut decode = Samples::new();
    let mut runs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // By candidates task, then run: what step 5's measures, feature costs and production
    // timings recorded.
    let mut per_run: BTreeMap<String, BTreeMap<String, RunRows>> = BTreeMap::new();
    if let Some(m) = &machine {
        for row in store.read_rows::<MeasureRow>(&machine_dir::<MeasureRow>(m))? {
            per_run
                .entry(row.task_key.clone())
                .or_default()
                .entry(row.run_id.clone())
                .or_default()
                .measures
                .push(row);
        }
        for row in store.read_rows::<FeatureCostRow>(&machine_dir::<FeatureCostRow>(m))? {
            per_run
                .entry(row.task_key.clone())
                .or_default()
                .entry(row.run_id.clone())
                .or_default()
                .costs
                .push(row);
        }
        for row in store.read_rows::<TimingRow>(&crate::store::timings_dir(m))? {
            if matches!(
                row.op.as_str(),
                "production_compress" | "production_replay" | "reference"
            ) {
                let run = per_run
                    .entry(row.task_key.clone())
                    .or_default()
                    .entry(row.run_id.clone())
                    .or_default();
                run.production = true;
                match row.op.as_str() {
                    "production_compress" => run.compress_ns.push(row.ns),
                    "production_replay" => run.replay_ns.push(row.ns),
                    _ => run.reference_ns = Some(row.ns),
                }
                continue;
            }
            if !row.replay_matched {
                continue;
            }
            runs.entry(row.task_key.clone())
                .or_default()
                .insert(row.run_id.clone());
            let (map, key) = match (row.op.as_str(), row.candidate) {
                ("compress", _) => (&mut compress, (row.task_key, row.encoded_hash)),
                ("adaptive_compress", Some(candidate)) => {
                    (&mut adaptive, (row.task_key, candidate))
                }
                ("decode", _) => (&mut decode, (row.chunk_key, row.encoded_hash)),
                _ => continue,
            };
            map.entry(key).or_default().push(row.ns);
        }
    }

    let mut rows = BufWriter::new(File::create(out.join(format!("rows-{tag}.csv")))?);
    writeln!(
        rows,
        "tag,source,column,chunk,ptype,len,variant,ok,canonical_bytes,bytes,nbytes,root,tree,compress_ns,decode_ns_median,decode_ns_min,error,plan,adaptive_compress_ns,predicted_nbytes,predicted_decode_ns"
    )?;
    let mut sites = BufWriter::new(File::create(out.join(format!("sites-{tag}.csv")))?);
    writeln!(
        sites,
        "tag,source,column,chunk,path,scheme,won,nbytes,decode_ns,best_nbytes,best_decode_ns"
    )?;
    let mut searches = BufWriter::new(File::create(out.join(format!("search-{tag}.csv")))?);
    writeln!(
        searches,
        "tag,source,column,chunk,sites,compressions,memo_hits,capped_sites,search_ms"
    )?;
    let mut feats = BufWriter::new(File::create(out.join(format!("features-{tag}.csv")))?);
    writeln!(feats, "source,column,chunk,{}", crate::features::header())?;

    let mut site_feats =
        BufWriter::new(File::create(out.join(format!("site-features-{tag}.csv")))?);
    writeln!(
        site_feats,
        "source,column,chunk,site_hash,path,applied,family,tier,name,value"
    )?;
    let mut costs = BufWriter::new(File::create(out.join(format!("feature-costs-{tag}.csv")))?);
    writeln!(costs, "source,column,chunk,site_hash,run_id,group,tier,ns")?;
    let mut measures = BufWriter::new(File::create(out.join(format!("measures-{tag}.csv")))?);
    writeln!(
        measures,
        "tag,source,column,chunk,variant,run_id,op,correct,batch,samples,converged,median_ns,min_ns,ci,peak_bytes,allocations,error"
    )?;
    let mut production = BufWriter::new(File::create(out.join(format!("production-{tag}.csv")))?);
    writeln!(
        production,
        "tag,source,column,chunk,run_id,reference_ns,compress_ns_median,replay_ns_median"
    )?;

    let mut manifest = DatasetManifest {
        plan_id: plan.plan.plan_id.clone(),
        git_commit: plan.plan.identity.git_commit.clone(),
        vx_version: plan.plan.identity.vx_version.clone(),
        machine_id: machine,
        chunks: 0,
        incomplete: 0,
        timed_chunks: 0,
        max_runs: 0,
        groups: plan
            .plan
            .spec
            .sources
            .iter()
            .map(|s| (s.name().to_string(), s.group().to_string()))
            .collect(),
    };

    // Each candidates task names its chunk; the matching features task shares the chunk key.
    let features_by_chunk: BTreeMap<_, _> = plan
        .tasks
        .iter()
        .filter_map(|t| match &t.input {
            TaskInput::Features { chunk, .. } => Some((chunk.clone(), t.key.clone())),
            _ => None,
        })
        .collect();

    for task in plan.tasks.iter().filter(|t| t.kind == TaskKind::Candidates) {
        let TaskInput::Candidates { chunk, .. } = &task.input else {
            continue;
        };
        let feature_values = features_by_chunk
            .get(chunk)
            .and_then(|k| features.get(k.as_str()));
        let (Some(chunk_row), Some(feature_values), Some(candidates)) = (
            chunks.get(chunk.as_str()),
            feature_values,
            encodings.get(task.key.as_str()),
        ) else {
            manifest.incomplete += 1;
            continue;
        };
        let run_count = runs.get(task.key.as_str()).map_or(0, |r| r.len() as u64);
        manifest.chunks += 1;
        if run_count > 0 {
            manifest.timed_chunks += 1;
        }
        manifest.max_runs = manifest.max_runs.max(run_count);

        let source = csv(&chunk_row.source);
        let column = csv(&chunk_row.column);
        let id = chunk.as_str();
        writeln!(
            feats,
            "{source},{column},{id},{}",
            crate::features::NAMES
                .iter()
                .map(|n| feature_values
                    .get(*n)
                    .copied()
                    .flatten()
                    .map(|v| v.to_string())
                    .unwrap_or_default())
                .collect::<Vec<_>>()
                .join(",")
        )?;

        for row in site_rows.get(task.key.as_str()).into_iter().flatten() {
            writeln!(
                sites,
                "{tag},{source},{column},{id},{},{},{},{},{},{},{}",
                csv(&row.path),
                csv(&row.scheme),
                u8::from(row.won),
                row.nbytes,
                row.decode_ns,
                row.best_nbytes,
                row.best_decode_ns
            )?;
        }
        if let Some(search) = search_rows.get(task.key.as_str()) {
            writeln!(
                searches,
                "{tag},{source},{column},{id},{},{},{},{},{}",
                search.sites,
                search.compressions,
                search.memo_hits,
                search.capped_sites,
                search.search_ms
            )?;
        }
        let features_key = features_by_chunk.get(chunk).map(ToString::to_string);
        for row in features_key
            .and_then(|k| site_features.get(&k))
            .into_iter()
            .flatten()
        {
            writeln!(
                site_feats,
                "{source},{column},{id},{},{},{},{},{},{},{}",
                row.site_hash.as_deref().unwrap_or_default(),
                csv(&row.site),
                csv(row.applied.as_deref().unwrap_or_default()),
                row.family.as_deref().unwrap_or_default(),
                row.tier.map(|t| t.to_string()).unwrap_or_default(),
                csv(&row.feature),
                row.value.map(|v| v.to_string()).unwrap_or_default()
            )?;
        }
        for (run_id, run) in per_run.get_mut(task.key.as_str()).into_iter().flatten() {
            for c in &run.costs {
                writeln!(
                    costs,
                    "{source},{column},{id},{},{run_id},{},{},{}",
                    c.site_hash,
                    csv(&c.group),
                    c.tier,
                    c.ns
                )?;
            }
            for m in &run.measures {
                let mut sorted = m.samples_ns.clone();
                sorted.sort_by(f64::total_cmp);
                let show = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or_default();
                let opt = |v: Option<u64>| v.map(|v| v.to_string()).unwrap_or_default();
                writeln!(
                    measures,
                    "{tag},{source},{column},{id},{},{run_id},{},{},{},{},{},{},{},{},{},{},{}",
                    variant(&m.candidate),
                    m.op,
                    u8::from(m.correct),
                    m.batch,
                    sorted.len(),
                    u8::from(m.converged),
                    show(sorted.get(sorted.len() / 2).copied()),
                    show(sorted.first().copied()),
                    if sorted.len() >= 3 {
                        format!("{:.4}", crate::measure::median_ci(&sorted))
                    } else {
                        String::new()
                    },
                    opt(m.peak_bytes),
                    opt(m.allocations),
                    csv(m.error.as_deref().unwrap_or_default()),
                )?;
            }
            if run.production {
                let show = |v: Option<u64>| v.map(|v| v.to_string()).unwrap_or_default();
                writeln!(
                    production,
                    "{tag},{source},{column},{id},{run_id},{},{},{}",
                    show(run.reference_ns),
                    show(median(&mut run.compress_ns)),
                    show(median(&mut run.replay_ns)),
                )?;
            }
        }

        let canonical = chunk_row.canonical_bytes;
        let len = chunk_row.len;
        let ptype = &chunk_row.ptype;
        let show = |v: Option<u64>| v.map(|v| v.to_string()).unwrap_or_default();
        let pooled = |map: &Samples, a: &str, b: &str| {
            map.get(&(a.to_string(), b.to_string()))
                .cloned()
                .unwrap_or_default()
        };
        for result in candidates {
            let prefix = format!(
                "{tag},{source},{column},{id},{ptype},{len},{}",
                variant(&result.candidate)
            );
            match (result.ok, &result.encoded_hash) {
                (true, Some(hash)) => {
                    let mut compress = pooled(&compress, task.key.as_str(), hash);
                    let mut decode = pooled(&decode, id, hash);
                    let mut adaptive = pooled(&adaptive, task.key.as_str(), &result.candidate);
                    let decode_min = decode.iter().copied().min();
                    writeln!(
                        rows,
                        "{prefix},1,{canonical},{},{},{},{},{},{},{},,{},{},{},{}",
                        show(result.bytes),
                        show(result.nbytes),
                        csv(result.root.as_deref().unwrap_or_default()),
                        csv(result.tree.as_deref().unwrap_or_default()),
                        show(median(&mut compress)),
                        show(median(&mut decode)),
                        show(decode_min),
                        csv(result.plan.as_deref().unwrap_or_default()),
                        show(median(&mut adaptive)),
                        show(result.predicted_nbytes),
                        show(result.predicted_decode_ns),
                    )?;
                }
                _ => writeln!(
                    rows,
                    "{prefix},0,{canonical},,,,,,,,{},,,,",
                    csv(result.error.as_deref().unwrap_or_default())
                )?,
            }
        }
    }
    rows.flush()?;
    feats.flush()?;
    sites.flush()?;
    searches.flush()?;
    site_feats.flush()?;
    costs.flush()?;
    measures.flush()?;
    production.flush()?;
    fs::write(
        out.join("dataset.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .context("writing dataset.json")?;
    Ok(manifest)
}
