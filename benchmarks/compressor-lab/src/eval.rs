// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluating the model-driven compressor against production on a plan's chunks.
//!
//! Every chunk is compressed by production and by the model-driven compressor at each target
//! bandwidth, with real bytes, compression and decode times measured in the same process. The
//! model for a chunk is `<models>/<source>.json`, trained without that chunk's source, so every
//! number is on data the model has not seen; `all.json` is used only when no held-out model
//! exists, and the output says so.

use std::collections::BTreeMap;
use std::fs::File;
use std::hint::black_box;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::blob;
use crate::model;
use crate::model::Model;
use crate::plan::PlannedRun;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::runner::ChunkFact;
use crate::shard::Shard;
use crate::store::Store;

/// How to evaluate.
#[derive(Debug, Clone)]
pub struct EvalOptions {
    /// Directory of models exported by `train.py --export`.
    pub models: PathBuf,
    /// Target bandwidths, as (label, bytes per second).
    pub bandwidths: Vec<(String, f64)>,
    /// Reads per write; compression time is divided by this. Use a huge value to ignore it.
    pub reads: f64,
    /// Minimum predicted saving before a proposal is compressed and verified.
    pub gate: f64,
    /// Candidates the model may not use.
    pub exclude: Vec<String>,
    /// How many of the model's proposals to compress and verify; `usize::MAX` tries all.
    pub top_k: Vec<usize>,
    /// Fixed trial sets, e.g. `["for", "for+sparse"]`: compress with production and each listed
    /// candidate, keep the lowest measured cost. No model involved.
    pub trial_sets: Vec<String>,
    /// Repetitions for compression timing.
    pub compress_reps: u32,
    /// Repetitions for decode timing.
    pub decode_reps: u32,
    /// Only chunks in this shard.
    pub shard: Option<Shard>,
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn time_decode(array: &ArrayRef, reps: u32, ctx: &mut ExecutionCtx) -> anyhow::Result<u128> {
    black_box(array.clone().execute::<Canonical>(ctx)?);
    let mut times = Vec::with_capacity(reps as usize);
    for _ in 0..reps.max(1) {
        let start = Instant::now();
        black_box(array.clone().execute::<Canonical>(ctx)?);
        times.push(start.elapsed().as_nanos());
    }
    Ok(median(times))
}

/// Runs the evaluation and writes one CSV row per chunk and variant.
pub fn evaluate(
    plan: &PlannedRun,
    store: &Store,
    options: &EvalOptions,
    out: &Path,
) -> anyhow::Result<u64> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();
    if crate::wrap::SESSION.get().is_none() && crate::wrap::SESSION.set(session.clone()).is_err() {
        anyhow::bail!("session already set");
    }

    // The model names candidates without the `forced/` prefix.
    let names: Vec<String> = plan
        .plan
        .candidates
        .iter()
        .filter(|n| {
            !options
                .exclude
                .iter()
                .any(|e| e == n.trim_start_matches("forced/"))
        })
        .cloned()
        .collect();
    let built = crate::candidates::build(&session, &names)?;
    let candidates: BTreeMap<String, &BtrBlocksCompressor> = built
        .iter()
        .map(|(name, c)| (name.trim_start_matches("forced/").to_string(), c))
        .collect();
    let production = candidates
        .get("production")
        .context("the plan has no production candidate")?;

    let mut models: BTreeMap<String, (String, Model)> = BTreeMap::new();
    let mut csv =
        BufWriter::new(File::create(out).with_context(|| format!("creating {}", out.display()))?);
    writeln!(
        csv,
        "source,column,chunk,variant,model,proposed,tried,kept,canonical_bytes,bytes,compress_ns,decode_ns"
    )?;

    let mut evaluated = 0;
    for task in plan.tasks.iter().filter(|t| t.kind == TaskKind::Candidates) {
        if options.shard.is_some_and(|s| !s.contains(task.shard_key())) {
            continue;
        }
        let TaskInput::Candidates { chunk, .. } = &task.input else {
            continue;
        };
        if !store.is_done(chunk) {
            continue;
        }
        let fact: ChunkFact = store.fact(TaskKind::Chunk, chunk)?;
        let array = blob::deserialize(&fact.blob, store.blob(&fact.blob.hash)?, &session)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();
        let source = fact.input.source().to_string();
        if !models.contains_key(&source) {
            let own = options.models.join(format!("{source}.json"));
            let (which, path) = if own.exists() {
                ("held-out", own)
            } else {
                ("all", options.models.join("all.json"))
            };
            models.insert(source.clone(), (which.to_string(), Model::load(&path)?));
        }
        let (which, model) = &models[&source];
        let prefix = format!("{},{},{}", source, fact.input.column(), chunk);
        let canonical = fact.blob.bytes;

        let mut compress = Vec::new();
        let mut encoded = None;
        for _ in 0..options.compress_reps.max(1) {
            let start = Instant::now();
            encoded = Some(production.compress(&array, &mut ctx)?);
            compress.push(start.elapsed().as_nanos());
        }
        let encoded = encoded.context("no repetitions")?;
        writeln!(
            csv,
            "{prefix},production,{which},,,,{canonical},{},{},{}",
            blob::serialized_size(&encoded, &session)?,
            median(compress),
            time_decode(&encoded, options.decode_reps, &mut ctx)?
        )?;

        for ((label, bandwidth), top_k) in options
            .bandwidths
            .iter()
            .flat_map(|b| options.top_k.iter().map(move |k| (b, *k)))
        {
            let k_label = if top_k == usize::MAX {
                "all".to_string()
            } else {
                top_k.to_string()
            };
            let mut compress = Vec::new();
            let mut outcome = None;
            for _ in 0..options.compress_reps.max(1) {
                let start = Instant::now();
                outcome = Some(model::compress(
                    model,
                    &candidates,
                    *bandwidth,
                    options.reads,
                    options.gate,
                    top_k,
                    &array,
                    &session,
                    &mut ctx,
                )?);
                compress.push(start.elapsed().as_nanos());
            }
            let outcome = outcome.context("no repetitions")?;
            writeln!(
                csv,
                "{prefix},model@{label}@k{k_label},{which},{},{},{},{canonical},{},{},{}",
                outcome.proposed,
                u8::from(outcome.tried),
                outcome.kept,
                blob::serialized_size(&outcome.array, &session)?,
                median(compress),
                time_decode(&outcome.array, options.decode_reps, &mut ctx)?
            )?;
        }
        for (label, bandwidth) in &options.bandwidths {
            for set in &options.trial_sets {
                let names: Vec<&str> = set.split('+').collect();
                let mut compress = Vec::new();
                let mut chosen = None;
                for _ in 0..options.compress_reps.max(1) {
                    let start = Instant::now();
                    let mut best = production.compress(&array, &mut ctx)?;
                    let mut best_name = "production";
                    let mut best_cost =
                        model::measured_cost(&best, *bandwidth, &session, &mut ctx)?;
                    for name in &names {
                        let compressor = candidates
                            .get(*name)
                            .with_context(|| format!("unknown trial candidate `{name}`"))?;
                        let Ok(Ok(encoded)) =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                compressor.compress(&array, &mut ctx)
                            }))
                        else {
                            continue;
                        };
                        let cost = model::measured_cost(&encoded, *bandwidth, &session, &mut ctx)?;
                        if cost < best_cost {
                            best = encoded;
                            best_name = name;
                            best_cost = cost;
                        }
                    }
                    compress.push(start.elapsed().as_nanos());
                    chosen = Some((best, best_name.to_string()));
                }
                let (encoded, kept) = chosen.context("no repetitions")?;
                writeln!(
                    csv,
                    "{prefix},trial@{label}@{set},-,{set},1,{kept},{canonical},{},{},{}",
                    blob::serialized_size(&encoded, &session)?,
                    median(compress),
                    time_decode(&encoded, options.decode_reps, &mut ctx)?
                )?;
            }
        }
        evaluated += 1;
    }
    csv.flush()?;
    Ok(evaluated)
}
