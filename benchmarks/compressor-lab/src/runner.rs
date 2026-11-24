// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executing a plan's tasks against a store.
//!
//! Fact tasks (`chunk`, `features`, `candidates`) are skipped when their done marker exists, so a
//! run can be interrupted and resumed, split across shards, and spread over machines. With
//! `verify`, done tasks are re-run and their output compared, which detects nondeterminism.
//!
//! The `time` stage writes observations: each run appends new samples for its machine, so timings
//! accumulate across runs and stay separate per machine.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::bail;
use serde::Deserialize;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::blob;
use crate::blob::BlobRef;
use crate::features;
use crate::key::TaskKey;
use crate::machine::MachineInfo;
use crate::machine::TimingEnv;
use crate::measure;
use crate::measure::CandidateMeasures;
use crate::measure::TimingConfig;
use crate::measure::Workload;
use crate::plan::PlannedRun;
use crate::plan::Task;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::shard::Shard;
use crate::site_features;
use crate::site_features::SiteRow;
use crate::source::ChunkInput;
use crate::store::Store;

/// A chunk task's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkFact {
    /// Where the rows came from.
    pub input: ChunkInput,
    /// The canonical chunk.
    pub blob: BlobRef,
}

/// A features task's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesFact {
    /// The feature version.
    pub version: u32,
    /// Feature names, in order.
    pub names: Vec<String>,
    /// Feature values; `None` where undefined (e.g. all-null chunks).
    pub values: Vec<Option<f64>>,
    /// Tiered features of every distinct site the production compressor visited, root first.
    #[serde(default)]
    pub sites: Vec<SiteRow>,
}

/// One candidate's result on one chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateResult {
    /// The candidate name.
    pub name: String,
    /// The error, if the candidate failed (e.g. a forced scheme that cannot encode the chunk).
    pub error: Option<String>,
    /// The encoding.
    pub blob: Option<BlobRef>,
    /// In-memory buffer bytes of the encoding.
    pub nbytes: Option<u64>,
    /// The root encoding id.
    pub root: Option<String>,
    /// The encoding tree.
    pub tree: Option<String>,
}

/// A candidates task's output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatesFact {
    /// One result per candidate, in plan order.
    pub results: Vec<CandidateResult>,
}

/// One candidate's timings in one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateTiming {
    /// The candidate name.
    pub name: String,
    /// Compression times, ns.
    pub compress_ns: Vec<u64>,
    /// Decode-to-canonical times, ns.
    pub decode_ns: Vec<u64>,
    /// Whether re-compressing reproduced the stored encoding byte for byte.
    pub deterministic: bool,
}

/// The production compressor timed on one chunk: with its own selection, and replaying the plan
/// it chose. The difference is what selection (sampling and estimation) costs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProductionTiming {
    /// Full production compression times, ns.
    pub compress_ns: Vec<u64>,
    /// Times to replay production's own plan without estimating, ns.
    pub replay_ns: Vec<u64>,
}

/// A time observation: one run of one candidates task on one machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeObservation {
    /// The run id.
    pub run_id: String,
    /// The machine.
    pub machine_id: String,
    /// The code the timings were taken with.
    pub git_commit: String,
    /// Per-candidate timings.
    pub timings: Vec<CandidateTiming>,
    /// Decode, pushdown, memory and correctness measures per candidate, the canonical array
    /// included as `canonical`.
    #[serde(default)]
    pub measures: Vec<CandidateMeasures>,
    /// The reference kernel's time on this run, ns (see [`measure::reference_ns`]).
    #[serde(default)]
    pub reference_ns: Option<u64>,
    /// Production compression with and without selection.
    #[serde(default)]
    pub production: Option<ProductionTiming>,
    /// The timing conditions of this run.
    #[serde(default)]
    pub env: Option<TimingEnv>,
    /// What computing each site's feature groups cost on this run, by site hash.
    #[serde(default)]
    pub feature_costs: Vec<SiteCosts>,
}

/// The feature-group costs of one site, measured in the `time` stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteCosts {
    /// The site's content hash, as in [`SiteRow::hash`].
    pub hash: String,
    /// Cost per group.
    pub costs: Vec<site_features::GroupCost>,
}

/// What to run.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Stages to run, in order: any of `chunk`, `features`, `candidates`, `time`.
    pub stages: Vec<String>,
    /// Only tasks in this shard.
    pub shard: Option<Shard>,
    /// Re-run done fact tasks and check they reproduce their output.
    pub verify: bool,
    /// Where Parquet paths in the plan are resolved from.
    pub data_root: PathBuf,
    /// Timing repetitions (from the plan's measurement settings).
    pub decode_reps: u32,
    /// Compression timing repetitions.
    pub compress_reps: u32,
    /// How the `time` stage measures decode and pushdown operations.
    pub timing: TimingConfig,
}

/// Counts of what a stage did.
#[derive(Debug, Default, Clone)]
pub struct StageReport {
    /// Tasks executed.
    pub ran: u64,
    /// Tasks already done and skipped.
    pub skipped: u64,
    /// Tasks waiting on a dependency that isn't done.
    pub blocked: u64,
    /// Done tasks whose re-run produced different output.
    pub mismatched: u64,
}

/// Runs plans against a store.
pub struct Runner<'a> {
    plan: &'a PlannedRun,
    store: &'a Store,
    session: VortexSession,
    candidates: BTreeMap<String, BtrBlocksCompressor>,
}

impl<'a> Runner<'a> {
    /// Prepares a runner. The plan's candidates are built once.
    pub fn new(plan: &'a PlannedRun, store: &'a Store) -> anyhow::Result<Self> {
        let session = VortexSession::default();
        if crate::wrap::SESSION.get().is_none()
            && crate::wrap::SESSION.set(session.clone()).is_err()
        {
            bail!("session already set");
        }
        let candidates = crate::candidates::build(&session, &plan.plan.candidates)?;
        Ok(Self {
            plan,
            store,
            session,
            candidates,
        })
    }

    fn tasks(&self, kind: TaskKind, shard: Option<Shard>) -> impl Iterator<Item = &Task> {
        self.plan
            .tasks
            .iter()
            .filter(move |t| t.kind == kind && shard.is_none_or(|s| s.contains(t.shard_key())))
    }

    /// Runs the requested stages.
    pub fn run(&self, options: &RunOptions) -> anyhow::Result<BTreeMap<String, StageReport>> {
        let mut reports = BTreeMap::new();
        for stage in &options.stages {
            let report = match stage.as_str() {
                "chunk" => self.run_facts(TaskKind::Chunk, options)?,
                "features" => self.run_facts(TaskKind::Features, options)?,
                "candidates" => self.run_facts(TaskKind::Candidates, options)?,
                "time" => self.run_time(options)?,
                other => {
                    bail!("unknown stage `{other}`; stages are chunk, features, candidates, time")
                }
            };
            eprintln!(
                "{stage}: ran {}, already done {}, blocked {}, mismatched {}",
                report.ran, report.skipped, report.blocked, report.mismatched
            );
            reports.insert(stage.clone(), report);
        }
        Ok(reports)
    }

    fn run_facts(&self, kind: TaskKind, options: &RunOptions) -> anyhow::Result<StageReport> {
        let mut report = StageReport::default();
        let mut ctx = self.session.create_execution_ctx();
        for task in self.tasks(kind, options.shard) {
            let previous = self.store.ledger(&task.key)?;
            if previous.is_some() && !options.verify {
                report.skipped += 1;
                continue;
            }
            if !task.deps.iter().all(|d| self.store.is_done(d)) {
                report.blocked += 1;
                continue;
            }
            let start = Instant::now();
            let hash = match &task.input {
                TaskInput::Chunk(input) => {
                    let fact = self.chunk(input, &options.data_root, &mut ctx)?;
                    self.store
                        .put_fact(kind, &task.key, &fact, start.elapsed().as_millis())?
                }
                TaskInput::Features { chunk, version } => {
                    let fact = self.features(chunk, *version, &mut ctx)?;
                    self.store
                        .put_fact(kind, &task.key, &fact, start.elapsed().as_millis())?
                }
                TaskInput::Candidates {
                    chunk, candidates, ..
                } => {
                    let fact = self.candidates(chunk, candidates, &mut ctx)?;
                    self.store
                        .put_fact(kind, &task.key, &fact, start.elapsed().as_millis())?
                }
            };
            report.ran += 1;
            if let Some(previous) = previous
                && previous.output_hash != hash
            {
                report.mismatched += 1;
                eprintln!(
                    "nondeterminism: {} {} produced different output",
                    kind.as_str(),
                    task.key
                );
            }
        }
        Ok(report)
    }

    fn load_chunk(
        &self,
        chunk: &TaskKey,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<(ChunkFact, ArrayRef)> {
        let fact: ChunkFact = self.store.fact(TaskKind::Chunk, chunk)?;
        let array =
            blob::deserialize(&fact.blob, self.store.blob(&fact.blob.hash)?, &self.session)?;
        let array = array.execute::<Canonical>(ctx)?.into_array();
        Ok((fact, array))
    }

    fn chunk(
        &self,
        input: &ChunkInput,
        data_root: &std::path::Path,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<ChunkFact> {
        let arrow = crate::load::arrow(input, data_root)?;
        let array = crate::load::to_vortex(&arrow, ctx)?;
        let (blob, bytes) = blob::serialize(&array, &self.session)?;
        self.store.put_blob(&blob.hash, &bytes)?;
        Ok(ChunkFact {
            input: input.clone(),
            blob,
        })
    }

    fn features(
        &self,
        chunk: &TaskKey,
        version: u32,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<FeaturesFact> {
        if version != crate::plan::FEATURES_VERSION {
            bail!(
                "this binary computes features version {}, the plan asks for {version}",
                crate::plan::FEATURES_VERSION
            );
        }
        let (_, array) = self.load_chunk(chunk, ctx)?;
        let (names, values) = if site_features::family(array.dtype()) == "int" {
            let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
            let base = features::from_primitive(&primitive, ctx)?;
            (
                features::NAMES.iter().map(|n| n.to_string()).collect(),
                base.iter().map(|v| v.is_finite().then_some(*v)).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(FeaturesFact {
            version,
            names,
            values,
            // Costs are timings, so they are measured in the `time` stage; facts stay deterministic.
            sites: site_features::production_sites(&array, &self.session, ctx)?
                .into_iter()
                .map(|mut row| {
                    row.features.costs.clear();
                    row
                })
                .collect(),
        })
    }

    fn candidates(
        &self,
        chunk: &TaskKey,
        names: &[String],
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<CandidatesFact> {
        let (_, array) = self.load_chunk(chunk, ctx)?;
        let mut results = Vec::with_capacity(names.len());
        for name in names {
            let compressor = self
                .candidates
                .get(name)
                .with_context(|| format!("candidate `{name}` was not built"))?;
            let outcome =
                std::panic::catch_unwind(AssertUnwindSafe(|| compressor.compress(&array, ctx)));
            let result = match outcome {
                Ok(Ok(encoded)) => {
                    let (blob, bytes) = blob::serialize(&encoded, &self.session)?;
                    self.store.put_blob(&blob.hash, &bytes)?;
                    CandidateResult {
                        name: name.clone(),
                        error: None,
                        nbytes: Some(encoded.nbytes()),
                        root: Some(encoded.encoding_id().to_string()),
                        tree: Some(tree(&encoded)),
                        blob: Some(blob),
                    }
                }
                Ok(Err(error)) => failed(name, error.to_string()),
                Err(_) => failed(name, "panic".to_string()),
            };
            results.push(result);
        }
        Ok(CandidatesFact { results })
    }

    fn run_time(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let machine = MachineInfo::current()?;
        self.store.put_machine(&machine)?;
        let env = TimingEnv::current();
        if !env.warnings().is_empty() {
            eprintln!("timing conditions: {}", env.warnings().join("; "));
        }
        let run_id = format!(
            "{}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            std::process::id()
        );
        let mut report = StageReport::default();
        let mut ctx = self.session.create_execution_ctx();
        for task in self.tasks(TaskKind::Candidates, options.shard) {
            if !self.store.is_done(&task.key) {
                report.blocked += 1;
                continue;
            }
            let TaskInput::Candidates { chunk, .. } = &task.input else {
                continue;
            };
            let (_, array) = self.load_chunk(chunk, &mut ctx)?;
            let fact: CandidatesFact = self.store.fact(TaskKind::Candidates, &task.key)?;
            let mut timings = Vec::new();
            let mut encoded_candidates: Vec<(String, ArrayRef)> = Vec::new();
            for result in &fact.results {
                let Some(stored) = &result.blob else {
                    continue;
                };
                let compressor = &self.candidates[&result.name];
                let mut compress_ns = Vec::with_capacity(options.compress_reps as usize);
                let mut deterministic = true;
                for _ in 0..options.compress_reps {
                    let start = Instant::now();
                    let encoded = compressor.compress(&array, &mut ctx)?;
                    compress_ns.push(u64::try_from(start.elapsed().as_nanos())?);
                    deterministic &=
                        blob::serialize(&encoded, &self.session)?.0.hash == stored.hash;
                }
                let encoded =
                    blob::deserialize(stored, self.store.blob(&stored.hash)?, &self.session)?;
                black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
                let mut decode_ns = Vec::with_capacity(options.decode_reps as usize);
                for _ in 0..options.decode_reps {
                    let start = Instant::now();
                    black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
                    decode_ns.push(u64::try_from(start.elapsed().as_nanos())?);
                }
                if !deterministic {
                    eprintln!(
                        "nondeterminism: {} re-compressed differently on {}",
                        result.name, task.key
                    );
                }
                encoded_candidates.push((result.name.clone(), encoded));
                timings.push(CandidateTiming {
                    name: result.name.clone(),
                    compress_ns,
                    decode_ns,
                    deterministic,
                });
            }
            let workload = Workload::standard(&array, options.timing.seed, &mut ctx)?;
            let measures = measure::measure(
                &array,
                &encoded_candidates,
                &workload,
                &options.timing,
                &mut ctx,
            )?;
            let production = self.time_production(&array, options.compress_reps, &mut ctx)?;
            let feature_costs = site_features::production_sites(&array, &self.session, &mut ctx)?
                .into_iter()
                .map(|row| SiteCosts {
                    hash: row.hash,
                    costs: row.features.costs,
                })
                .collect();
            self.store.put_observation(
                "time",
                &machine.machine_id,
                &task.key,
                &run_id,
                &TimeObservation {
                    run_id: run_id.clone(),
                    machine_id: machine.machine_id.clone(),
                    git_commit: self.plan.plan.identity.git_commit.clone(),
                    timings,
                    measures,
                    reference_ns: Some(measure::reference_ns(array.nbytes() as usize)),
                    production: Some(production),
                    env: Some(env.clone()),
                    feature_costs,
                },
            )?;
            report.ran += 1;
        }
        Ok(report)
    }
}

impl Runner<'_> {
    /// Times production compression, interleaved with replaying the plan it chose.
    fn time_production(
        &self,
        array: &ArrayRef,
        reps: u32,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<ProductionTiming> {
        let production = BtrBlocksCompressor::from_session(&self.session);
        let (_, plan) = production.0.compress_recording_plan(array, ctx)?;
        let mut timing = ProductionTiming::default();
        for _ in 0..reps.max(1) {
            let start = Instant::now();
            black_box(production.compress(array, ctx)?);
            timing
                .compress_ns
                .push(u64::try_from(start.elapsed().as_nanos())?);
            let start = Instant::now();
            black_box(production.0.compress_with_plan(array, &plan, ctx)?);
            timing
                .replay_ns
                .push(u64::try_from(start.elapsed().as_nanos())?);
        }
        Ok(timing)
    }
}

fn failed(name: &str, error: String) -> CandidateResult {
    CandidateResult {
        name: name.to_string(),
        error: Some(error.lines().next().unwrap_or_default().to_string()),
        blob: None,
        nbytes: None,
        root: None,
        tree: None,
    }
}

fn tree(array: &ArrayRef) -> String {
    array
        .display_tree_encodings_only()
        .to_string()
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" / ")
}
