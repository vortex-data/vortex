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
use std::collections::BTreeSet;
use std::hint::black_box;
use std::panic::AssertUnwindSafe;
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
use crate::oracle::OracleParams;
use crate::oracle::SearchStats;
use crate::oracle::SiteRow;
use crate::plan::PlannedRun;
use crate::plan::Task;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::shard::Shard;
use crate::site_features;
use crate::site_features::SiteFeatureRow;
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
    pub sites: Vec<SiteFeatureRow>,
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
    /// The plan the encoding replays, as text (see [`vortex_compressor::plan::Plan`]).
    #[serde(default)]
    pub plan: Option<String>,
    /// For oracle front plans: the search's predicted buffer bytes.
    #[serde(default)]
    pub predicted_nbytes: Option<u64>,
    /// For oracle front plans: the search's predicted decode time, ns, from the shard machine.
    #[serde(default)]
    pub predicted_decode_ns: Option<u64>,
}

/// A candidates task's output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatesFact {
    /// One result per candidate, in plan order; the oracle adds one per front plan.
    pub results: Vec<CandidateResult>,
    /// The oracle's search effort, if the plan runs it.
    #[serde(default)]
    pub search: Option<SearchStats>,
    /// The oracle's per-site table: one row per (site, scheme) tried. Decode times come from the
    /// shard machine and are only for pruning and rough analysis.
    #[serde(default)]
    pub sites: Vec<SiteRow>,
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
    /// For `production` and `sizemodel`: compression times of the adaptive compressor, which
    /// samples and estimates, ns. `compress_ns` replays the plan, with no selection cost.
    #[serde(default)]
    pub adaptive_compress_ns: Vec<u64>,
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
    /// The site's content hash, as in [`SiteFeatureRow::hash`].
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
    /// Where the plan's raw data is found.
    pub data: crate::load::DataLocation,
    /// Timing repetitions (from the plan's measurement settings).
    pub decode_reps: u32,
    /// Compression timing repetitions.
    pub compress_reps: u32,
    /// Record a fact task that errors or panics in the stage report and go on, instead of
    /// stopping the run.
    pub keep_going: bool,
    /// Fact tasks not to run, e.g. ones that already failed too often. Tasks depending on them
    /// count as blocked.
    pub skip: BTreeSet<TaskKey>,
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
    /// Tasks that failed, with their error, when running with `keep_going`.
    pub failed: Vec<(TaskKey, String)>,
    /// Tasks left out because they were in `skip`.
    pub skipped_failed: u64,
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
        let adaptive: Vec<String> = plan
            .plan
            .candidates
            .iter()
            .filter(|n| matches!(n.as_str(), "production" | "sizemodel"))
            .cloned()
            .chain(std::iter::once("production".to_string()))
            .collect();
        let candidates = crate::candidates::build(&session, &adaptive)?;
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
                "{stage}: ran {}, already done {}, blocked {}, mismatched {}, failed {}",
                report.ran,
                report.skipped,
                report.blocked,
                report.mismatched,
                report.failed.len()
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
            if options.skip.contains(&task.key) {
                report.skipped_failed += 1;
                continue;
            }
            let start = Instant::now();
            let outcome = if options.keep_going {
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    self.run_fact(kind, task, options, &mut ctx, start)
                }))
                .unwrap_or_else(|panic| Err(anyhow::anyhow!("panic: {}", panic_message(&panic))))
            } else {
                self.run_fact(kind, task, options, &mut ctx, start)
            };
            let hash = match outcome {
                Ok(hash) => hash,
                Err(error) if options.keep_going => {
                    eprintln!("failed: {} {}: {error:#}", kind.as_str(), task.key);
                    report.failed.push((task.key.clone(), format!("{error:#}")));
                    continue;
                }
                Err(error) => return Err(error),
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

    fn run_fact(
        &self,
        kind: TaskKind,
        task: &Task,
        options: &RunOptions,
        ctx: &mut ExecutionCtx,
        start: Instant,
    ) -> anyhow::Result<String> {
        match &task.input {
            TaskInput::Chunk(input) => {
                let fact = self.chunk(input, &options.data, ctx)?;
                self.store
                    .put_fact(kind, &task.key, &fact, start.elapsed().as_millis())
            }
            TaskInput::Features { chunk, version } => {
                let fact = self.features(chunk, *version, ctx)?;
                self.store
                    .put_fact(kind, &task.key, &fact, start.elapsed().as_millis())
            }
            TaskInput::Candidates {
                chunk,
                candidates,
                search,
                ..
            } => {
                let params = OracleParams {
                    k: search.k as usize,
                    epsilon: search.epsilon,
                    max_compressions: search.max_compressions,
                    decode_reps: search.decode_reps,
                };
                let fact = self.candidates(chunk, candidates, params, ctx)?;
                self.store
                    .put_fact(kind, &task.key, &fact, start.elapsed().as_millis())
            }
        }
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
        data: &crate::load::DataLocation,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<ChunkFact> {
        let arrow = crate::load::arrow(input, data)?;
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
        params: OracleParams,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<CandidatesFact> {
        let (_, array) = self.load_chunk(chunk, ctx)?;
        let production = &self.candidates["production"];
        let mut results = Vec::with_capacity(names.len());
        let mut search = None;
        let mut sites = Vec::new();
        for name in names {
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                crate::candidates::produce(
                    name,
                    &array,
                    &self.candidates,
                    &self.session,
                    params,
                    ctx,
                )
            }));
            let produced = match outcome {
                Ok(Ok((produced, output))) => {
                    if let Some(output) = output {
                        search = Some(output.stats);
                        sites = output.sites;
                    }
                    produced
                }
                Ok(Err(error)) => {
                    results.push(failed(name, error.to_string()));
                    continue;
                }
                Err(_) => {
                    results.push(failed(name, "panic".to_string()));
                    continue;
                }
            };
            for produced in produced {
                let replayed = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    production.compress_with_plan(&array, &produced.plan, ctx)
                }));
                let result = match replayed {
                    Ok(Ok(encoded)) => {
                        let (blob, bytes) = blob::serialize(&encoded, &self.session)?;
                        self.store.put_blob(&blob.hash, &bytes)?;
                        CandidateResult {
                            name: produced.name,
                            error: None,
                            nbytes: Some(encoded.nbytes()),
                            root: Some(encoded.encoding_id().to_string()),
                            tree: Some(tree(&encoded)),
                            blob: Some(blob),
                            plan: Some(produced.plan.to_string()),
                            predicted_nbytes: produced.predicted.map(|p| p.0),
                            predicted_decode_ns: produced.predicted.map(|p| p.1),
                        }
                    }
                    Ok(Err(error)) => failed(&produced.name, error.to_string()),
                    Err(_) => failed(&produced.name, "panic".to_string()),
                };
                results.push(result);
            }
        }
        Ok(CandidatesFact {
            results,
            search,
            sites,
        })
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
            let mut measured: BTreeMap<String, (Vec<u64>, Vec<u64>, bool)> = BTreeMap::new();
            // The replayed encoding of each distinct plan output, by hash, for the measures.
            let mut replayed: BTreeMap<String, ArrayRef> = BTreeMap::new();
            let mut measured_names: Vec<(String, String)> = Vec::new();
            for result in &fact.results {
                let Some(stored) = &result.blob else {
                    continue;
                };
                // Identical encodings (the same plan reached by several candidates) are timed once
                // and share their samples, so the order candidates run in can't favor one name.
                let (compress_ns, decode_ns, deterministic) = if let Some(done) =
                    measured.get(&stored.hash)
                {
                    done.clone()
                } else {
                    let plan_text = result.plan.as_deref().with_context(|| {
                        format!("{} has no plan; re-run candidates", result.name)
                    })?;
                    let plan = crate::candidates::parse_plan(plan_text, &self.session)?;
                    let production = &self.candidates["production"];
                    let mut compress_ns = Vec::with_capacity(options.compress_reps as usize);
                    let mut deterministic = true;
                    for _ in 0..options.compress_reps {
                        let start = Instant::now();
                        let encoded = production.compress_with_plan(&array, &plan, &mut ctx)?;
                        compress_ns.push(u64::try_from(start.elapsed().as_nanos())?);
                        let same = blob::serialize(&encoded, &self.session)?.0.hash == stored.hash;
                        deterministic &= same;
                        if same {
                            replayed.insert(stored.hash.clone(), encoded);
                        }
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
                    let done = (compress_ns, decode_ns, deterministic);
                    measured.insert(stored.hash.clone(), done.clone());
                    done
                };
                let mut adaptive_compress_ns = Vec::new();
                if let Some(adaptive) = self.candidates.get(&result.name) {
                    for _ in 0..options.compress_reps {
                        let start = Instant::now();
                        black_box(adaptive.compress(&array, &mut ctx)?);
                        adaptive_compress_ns.push(u64::try_from(start.elapsed().as_nanos())?);
                    }
                }
                if !deterministic {
                    eprintln!(
                        "nondeterminism: {} re-compressed differently on {}",
                        result.name, task.key
                    );
                }
                measured_names.push((result.name.clone(), stored.hash.clone()));
                timings.push(CandidateTiming {
                    name: result.name.clone(),
                    compress_ns,
                    decode_ns,
                    deterministic,
                    adaptive_compress_ns,
                });
            }
            // Each distinct encoding is measured once, from its verified plan replay, and its
            // measures are shared by every candidate that produced it.
            let encoded_candidates: Vec<(String, ArrayRef)> = replayed
                .iter()
                .map(|(hash, array)| (hash.clone(), array.clone()))
                .collect();
            let workload = Workload::standard(&array, options.timing.seed, &mut ctx)?;
            let by_hash = measure::measure(
                &array,
                &encoded_candidates,
                &workload,
                &options.timing,
                &mut ctx,
            )?;
            let mut measures: Vec<CandidateMeasures> = by_hash
                .iter()
                .filter(|m| m.name == measure::CANONICAL)
                .cloned()
                .collect();
            for (name, hash) in &measured_names {
                if let Some(m) = by_hash.iter().find(|m| &m.name == hash) {
                    measures.push(CandidateMeasures {
                        name: name.clone(),
                        ops: m.ops.clone(),
                    });
                }
            }
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

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown".to_string())
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
        plan: None,
        predicted_nbytes: None,
        predicted_decode_ns: None,
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
