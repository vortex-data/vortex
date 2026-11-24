// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Turning a store's facts and observations into training tables.
//!
//! The output keeps the CSV layout `benchmarks/compressor-feasibility/train.py` reads:
//! `rows-<tag>.csv` (one row per chunk and candidate) and `features-<tag>.csv` (one row per
//! chunk), plus `dataset.json` recording exactly what went in. Timings are pooled over every run
//! on one machine; other machines are never mixed in.

use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use anyhow::Context;
use serde::Serialize;

use crate::plan::PlannedRun;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::runner::CandidatesFact;
use crate::runner::ChunkFact;
use crate::runner::FeaturesFact;
use crate::runner::TimeObservation;
use crate::store::Store;

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
            let machines = store.machines("time");
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

    let mut rows = BufWriter::new(File::create(out.join(format!("rows-{tag}.csv")))?);
    writeln!(
        rows,
        "tag,source,column,chunk,ptype,len,variant,ok,canonical_bytes,bytes,nbytes,root,tree,compress_ns,decode_ns_median,decode_ns_min,error"
    )?;
    let mut feats = BufWriter::new(File::create(out.join(format!("features-{tag}.csv")))?);
    writeln!(feats, "source,column,chunk,{}", crate::features::header())?;

    let mut sites = BufWriter::new(File::create(out.join(format!("site-features-{tag}.csv")))?);
    writeln!(
        sites,
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
        machine_id: machine.clone(),
        chunks: 0,
        incomplete: 0,
        timed_chunks: 0,
        max_runs: 0,
    };

    // Each candidates task names its chunk; the matching features task shares the chunk key.
    let features_by_chunk: std::collections::BTreeMap<_, _> = plan
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
        let Some(features_key) = features_by_chunk.get(chunk) else {
            continue;
        };
        if !(store.is_done(chunk) && store.is_done(features_key) && store.is_done(&task.key)) {
            manifest.incomplete += 1;
            continue;
        }
        let chunk_fact: ChunkFact = store.fact(TaskKind::Chunk, chunk)?;
        let features: FeaturesFact = store.fact(TaskKind::Features, features_key)?;
        let candidates: CandidatesFact = store.fact(TaskKind::Candidates, &task.key)?;
        let observations: Vec<TimeObservation> = match &machine {
            Some(m) => store.observations("time", m, &task.key)?,
            None => Vec::new(),
        };
        manifest.chunks += 1;
        if !observations.is_empty() {
            manifest.timed_chunks += 1;
        }
        manifest.max_runs = manifest.max_runs.max(observations.len() as u64);

        let source = csv(chunk_fact.input.source());
        let column = csv(chunk_fact.input.column());
        let id = chunk.as_str();
        writeln!(
            feats,
            "{source},{column},{id},{}",
            features
                .values
                .iter()
                .map(|v| v.map(|v| v.to_string()).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(",")
        )?;

        for row in &features.sites {
            let path = csv(&row.site.path.join(" > "));
            let applied = csv(row.applied.as_deref().unwrap_or_default());
            for f in &row.features.features {
                writeln!(
                    sites,
                    "{source},{column},{id},{},{path},{applied},{},{},{},{}",
                    row.hash,
                    row.features.family,
                    f.tier,
                    csv(&f.name),
                    f.value.map(|v| v.to_string()).unwrap_or_default()
                )?;
            }
        }
        for o in &observations {
            for site in &o.feature_costs {
                for c in &site.costs {
                    writeln!(
                        costs,
                        "{source},{column},{id},{},{},{},{},{}",
                        site.hash,
                        o.run_id,
                        csv(&c.group),
                        c.tier,
                        c.ns
                    )?;
                }
            }
            for m in &o.measures {
                for op in &m.ops {
                    let mut sorted = op.samples_ns.clone();
                    sorted.sort_by(f64::total_cmp);
                    let show = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or_default();
                    writeln!(
                        measures,
                        "{tag},{source},{column},{id},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        variant(&m.name),
                        o.run_id,
                        op.op,
                        u8::from(op.correct),
                        op.batch,
                        sorted.len(),
                        u8::from(op.converged),
                        show(sorted.get(sorted.len() / 2).copied()),
                        show(sorted.first().copied()),
                        if sorted.len() >= 3 {
                            format!("{:.4}", crate::measure::median_ci(&sorted))
                        } else {
                            String::new()
                        },
                        op.memory
                            .map(|m| m.peak_bytes.to_string())
                            .unwrap_or_default(),
                        op.memory
                            .map(|m| m.allocations.to_string())
                            .unwrap_or_default(),
                        csv(op.error.as_deref().unwrap_or_default()),
                    )?;
                }
            }
            let show = |v: Option<u64>| v.map(|v| v.to_string()).unwrap_or_default();
            let mut p = o.production.clone().unwrap_or_default();
            writeln!(
                production,
                "{tag},{source},{column},{id},{},{},{},{}",
                o.run_id,
                show(o.reference_ns),
                show(median(&mut p.compress_ns)),
                show(median(&mut p.replay_ns)),
            )?;
        }

        let canonical = chunk_fact.blob.bytes;
        let len = chunk_fact.blob.len;
        let ptype = &chunk_fact.blob.ptype;
        for result in &candidates.results {
            let prefix = format!(
                "{tag},{source},{column},{id},{ptype},{len},{}",
                variant(&result.name)
            );
            match &result.blob {
                Some(blob) => {
                    let mut compress: Vec<u64> = observations
                        .iter()
                        .flat_map(|o| o.timings.iter().filter(|t| t.name == result.name))
                        .flat_map(|t| t.compress_ns.iter().copied())
                        .collect();
                    let mut decode: Vec<u64> = observations
                        .iter()
                        .flat_map(|o| o.timings.iter().filter(|t| t.name == result.name))
                        .flat_map(|t| t.decode_ns.iter().copied())
                        .collect();
                    let decode_min = decode.iter().copied().min();
                    let show = |v: Option<u64>| v.map(|v| v.to_string()).unwrap_or_default();
                    writeln!(
                        rows,
                        "{prefix},1,{canonical},{},{},{},{},{},{},{},",
                        blob.bytes,
                        show(result.nbytes),
                        csv(result.root.as_deref().unwrap_or_default()),
                        csv(result.tree.as_deref().unwrap_or_default()),
                        show(median(&mut compress)),
                        show(median(&mut decode)),
                        show(decode_min),
                    )?;
                }
                None => writeln!(
                    rows,
                    "{prefix},0,{canonical},,,,,,,,{}",
                    csv(result.error.as_deref().unwrap_or_default())
                )?,
            }
        }
    }
    rows.flush()?;
    feats.flush()?;
    sites.flush()?;
    costs.flush()?;
    measures.flush()?;
    production.flush()?;
    fs::write(
        out.join("dataset.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}
