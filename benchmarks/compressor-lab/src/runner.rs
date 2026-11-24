// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executing a plan's tasks against a store.
//!
//! Fact tasks (`chunk`, `features`, `candidates`) are skipped when the ledger lists them, so a
//! run can be interrupted and resumed, split across shards, and spread over machines that all
//! write to one store. With `verify`, done tasks are re-run and their output compared, which
//! detects nondeterminism.
//!
//! The candidates stage keeps no encoded bytes: it records each candidate's plan and the hash of
//! the bytes the plan produces. The `time` stage, on any machine, replays each plan (the
//! compression it times), checks the hash, then times decoding the replayed encoding. Samples go
//! to that machine's own partition, tagged with its build, so any number of machines can time one
//! store.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::panic::AssertUnwindSafe;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::bail;
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
use crate::chunks::CHUNKS_VERSION;
use crate::chunks::ChunkLoader;
use crate::features;
use crate::key::TaskKey;
use crate::machine::MachineInfo;
use crate::machine::TimingEnv;
use crate::measure;
use crate::measure::TimingConfig;
use crate::measure::Workload;
use crate::oracle::OracleParams;
use crate::plan::CANDIDATES_VERSION;
use crate::plan::FEATURES_VERSION;
use crate::plan::PlannedRun;
use crate::plan::Task;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::segment::Batch;
use crate::segment::MachineWriter;
use crate::segment::Rows;
use crate::segment::SegmentWriter;
use crate::segment::TimingWriter;
use crate::shard::Shard;
use crate::site_features;
use crate::spec::Storage;
use crate::store::Store;
use crate::table::ChunkRow;
use crate::table::EncodingRow;
use crate::table::FeatureCostRow;
use crate::table::FeatureRow;
use crate::table::LedgerRow;
use crate::table::MeasureRow;
use crate::table::SearchRow;
use crate::table::SiteFactRow;
use crate::table::TimingRow;
use crate::table::TimingRunRow;

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
    storage: BTreeMap<String, Storage>,
    worker_id: String,
}

/// The rows a candidates task writes: encodings, plus the oracle's sites and search effort.
#[derive(Debug, Default, serde::Serialize)]
pub struct CandidateRows {
    /// One row per candidate, and per oracle front plan.
    pub encodings: Vec<EncodingRow>,
    /// The oracle's per-site table.
    pub sites: Vec<SiteFactRow>,
    /// The oracle's search effort.
    pub search: Vec<SearchRow>,
}

impl Batch for CandidateRows {
    fn append(&mut self, other: Self) {
        self.encodings.extend(other.encodings);
        self.sites.extend(other.sites);
        self.search.extend(other.search);
    }

    fn write(&self, store: &Store, segment: &str) -> anyhow::Result<()> {
        use crate::table::Table as _;
        store.put_segment(
            &EncodingRow::dir(CANDIDATES_VERSION),
            segment,
            &self.encodings,
        )?;
        if !self.sites.is_empty() {
            store.put_segment(&SiteFactRow::dir(CANDIDATES_VERSION), segment, &self.sites)?;
        }
        if !self.search.is_empty() {
            store.put_segment(&SearchRow::dir(CANDIDATES_VERSION), segment, &self.search)?;
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.encodings.clear();
        self.sites.clear();
        self.search.clear();
    }

    fn len(&self) -> usize {
        self.encodings.len() + self.sites.len() + self.search.len()
    }
}

/// The cost tier of each base feature: 0 free context, 1 compressor stats, 2 distinct counts,
/// 3 an extra pass.
fn tier(name: &str) -> u8 {
    match name {
        "ptype_bits" | "signed" | "len" => 0,
        "null_frac" | "bits_bp" | "bits_for" | "bits_zz" | "for_gain" | "avg_run" => 1,
        "distinct" | "distinct_frac" | "top1_frac" | "top10_cover" => 2,
        _ => 3,
    }
}

impl<'a> Runner<'a> {
    /// Prepares a runner. The plan's candidates are built once, and the plan is recorded in the
    /// store so GC knows it is live.
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
        let storage = plan
            .plan
            .spec
            .sources
            .iter()
            .map(|s| (s.name().to_string(), s.storage()))
            .collect();
        store.put_create(
            &format!("plans/{}/plan.json", plan.plan.plan_id),
            serde_json::to_vec_pretty(&plan.plan)?,
        )?;
        crate::gc::record_tasks(store, &plan.plan.plan_id, &plan.tasks)?;
        Ok(Self {
            plan,
            store,
            session,
            candidates,
            storage,
            worker_id: crate::segment::new_worker_id(),
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
                "chunk" => self.run_chunks(options)?,
                "features" => self.run_features(options)?,
                "candidates" => self.run_candidates(options)?,
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

    fn git_commit(&self) -> &str {
        &self.plan.plan.identity.git_commit
    }

    /// Runs one fact stage: `make` turns a task into rows, which a [`SegmentWriter`] flushes as
    /// segments followed by their ledger segment.
    fn run_facts<B: Batch>(
        &self,
        kind: TaskKind,
        options: &RunOptions,
        ready: impl Fn(&Task) -> bool,
        mut make: impl FnMut(&Task, &mut ExecutionCtx) -> anyhow::Result<B>,
    ) -> anyhow::Result<StageReport> {
        let done = self.store.ledger(kind)?;
        let mut writer: SegmentWriter<'_, B> =
            SegmentWriter::new(self.store, &self.worker_id, kind, self.git_commit());
        let mut report = StageReport::default();
        let mut ctx = self.session.create_execution_ctx();
        for task in self.tasks(kind, options.shard) {
            let previous: Option<&LedgerRow> = done.get(&task.key);
            if previous.is_some() && !options.verify {
                report.skipped += 1;
                continue;
            }
            if !ready(task) {
                report.blocked += 1;
                continue;
            }
            if options.skip.contains(&task.key) {
                report.skipped_failed += 1;
                continue;
            }
            let start = Instant::now();
            let outcome = if options.keep_going {
                std::panic::catch_unwind(AssertUnwindSafe(|| make(task, &mut ctx))).unwrap_or_else(
                    |panic| Err(anyhow::anyhow!("panic: {}", panic_message(&panic))),
                )
            } else {
                make(task, &mut ctx)
            };
            let rows = match outcome {
                Ok(rows) => rows,
                Err(error) if options.keep_going => {
                    eprintln!("failed: {} {}: {error:#}", kind.as_str(), task.key);
                    report.failed.push((task.key.clone(), format!("{error:#}")));
                    continue;
                }
                Err(error) => return Err(error),
            };
            let hash = writer.add(&task.key, rows, start.elapsed().as_millis())?;
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
        writer.flush()?;
        Ok(report)
    }

    fn run_chunks(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        self.run_facts::<Rows<ChunkRow>>(
            TaskKind::Chunk,
            options,
            |_| true,
            |task, ctx| {
                let TaskInput::Chunk(input) = &task.input else {
                    bail!("chunk task {} has a {:?} input", task.key, task.input);
                };
                let storage = self
                    .storage
                    .get(input.source())
                    .copied()
                    .unwrap_or(Storage::Local);
                let row = crate::chunks::make(
                    &task.key,
                    input,
                    storage,
                    &options.data,
                    &self.session,
                    ctx,
                )?;
                Ok(Rows::new(CHUNKS_VERSION, vec![row]))
            },
        )
    }

    fn run_features(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let loader = ChunkLoader::new(self.store, &options.data)?;
        self.run_facts::<Rows<FeatureRow>>(
            TaskKind::Features,
            options,
            |task| task.deps.iter().all(|d| loader.row(d).is_some()),
            |task, ctx| {
                let TaskInput::Features { chunk, version } = &task.input else {
                    bail!("features task {} has a {:?} input", task.key, task.input);
                };
                if *version != FEATURES_VERSION {
                    bail!(
                        "this binary computes features version {FEATURES_VERSION}, the plan asks \
                         for {version}"
                    );
                }
                let (_, array) = loader.load(chunk, ctx)?;
                let mut rows = Vec::new();
                if site_features::family(array.dtype()) == "int" {
                    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
                    let values = features::from_primitive(&primitive, ctx)?;
                    rows.extend(features::NAMES.iter().zip(values).map(|(name, value)| {
                        FeatureRow {
                            task_key: task.key.to_string(),
                            chunk_key: chunk.to_string(),
                            site: "root".to_string(),
                            site_depth: 0,
                            feature: (*name).to_string(),
                            tier: Some(tier(name)),
                            window: None,
                            value: value.is_finite().then_some(value),
                            site_hash: None,
                            applied: None,
                            family: None,
                            site_context: None,
                        }
                    }));
                }
                // Feature costs are timings, so the `time` stage measures them; facts stay
                // deterministic.
                for site in site_features::production_sites(&array, &self.session, ctx)? {
                    let context = serde_json::to_string(&site.site)?;
                    let path = site.site.path.join(" > ");
                    let depth = u8::try_from(site.site.path.len().saturating_sub(1))?;
                    for feature in site.features.features {
                        rows.push(FeatureRow {
                            task_key: task.key.to_string(),
                            chunk_key: chunk.to_string(),
                            site: path.clone(),
                            site_depth: depth,
                            feature: feature.name,
                            tier: Some(feature.tier),
                            window: None,
                            value: feature.value,
                            site_hash: Some(site.hash.clone()),
                            applied: site.applied.clone(),
                            family: Some(site.features.family.clone()),
                            site_context: Some(context.clone()),
                        });
                    }
                }
                Ok(Rows::new(FEATURES_VERSION, rows))
            },
        )
    }

    fn run_candidates(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let loader = ChunkLoader::new(self.store, &options.data)?;
        self.run_facts::<CandidateRows>(
            TaskKind::Candidates,
            options,
            |task| task.deps.iter().all(|d| loader.row(d).is_some()),
            |task, ctx| {
                let TaskInput::Candidates {
                    chunk,
                    candidates,
                    search,
                    ..
                } = &task.input
                else {
                    bail!("candidates task {} has a {:?} input", task.key, task.input);
                };
                let params = OracleParams {
                    k: search.k as usize,
                    epsilon: search.epsilon,
                    max_compressions: search.max_compressions,
                    decode_reps: search.decode_reps,
                };
                self.candidates(&task.key, chunk, candidates, params, &loader, ctx)
            },
        )
    }

    fn candidates(
        &self,
        task: &TaskKey,
        chunk: &TaskKey,
        names: &[String],
        params: OracleParams,
        loader: &ChunkLoader,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<CandidateRows> {
        let (row, array) = loader.load(chunk, ctx)?;
        let production = &self.candidates["production"];
        let mut out = CandidateRows::default();
        let base = |name: &str| EncodingRow {
            task_key: task.to_string(),
            chunk_key: chunk.to_string(),
            candidate: name.to_string(),
            ok: false,
            error: None,
            plan: None,
            plan_hash: None,
            encoded_hash: None,
            encoded_ids: Vec::new(),
            bytes: None,
            nbytes: None,
            root: None,
            tree: None,
            verified: false,
            predicted_nbytes: None,
            predicted_decode_ns: None,
        };
        let failed = |name: &str, error: &str| EncodingRow {
            error: Some(error.lines().next().unwrap_or_default().to_string()),
            ..base(name)
        };
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
                        out.search.push(SearchRow {
                            task_key: task.to_string(),
                            chunk_key: chunk.to_string(),
                            sites: output.stats.sites,
                            compressions: output.stats.compressions,
                            memo_hits: output.stats.memo_hits,
                            capped_sites: output.stats.capped_sites,
                            search_ms: output.stats.search_ms,
                        });
                        out.sites = output
                            .sites
                            .into_iter()
                            .map(|s| SiteFactRow {
                                task_key: task.to_string(),
                                chunk_key: chunk.to_string(),
                                path: s.path,
                                scheme: s.scheme,
                                won: s.won,
                                nbytes: s.nbytes,
                                decode_ns: s.decode_ns,
                                best_nbytes: s.best_nbytes,
                                best_decode_ns: s.best_decode_ns,
                            })
                            .collect();
                    }
                    produced
                }
                Ok(Err(error)) => {
                    out.encodings.push(failed(name, &error.to_string()));
                    continue;
                }
                Err(_) => {
                    out.encodings.push(failed(name, "panic"));
                    continue;
                }
            };
            for produced in produced {
                let replayed = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    production.compress_with_plan(&array, &produced.plan, ctx)
                }));
                let result = match replayed {
                    Ok(Ok(encoded)) => {
                        let (blob, _) = blob::serialize(&encoded, &self.session)?;
                        let decoded = encoded.clone().execute::<Canonical>(ctx)?.into_array();
                        let verified =
                            crate::chunks::logical_hash(&decoded, ctx)? == row.canonical_hash;
                        if !verified {
                            eprintln!(
                                "{} on chunk {chunk} does not decode to the chunk",
                                produced.name
                            );
                        }
                        let plan = produced.plan.to_string();
                        EncodingRow {
                            ok: true,
                            plan_hash: Some(crate::key::bytes_digest(plan.as_bytes())),
                            plan: Some(plan),
                            encoded_hash: Some(blob.hash),
                            encoded_ids: blob.ids,
                            bytes: Some(blob.bytes),
                            nbytes: Some(encoded.nbytes()),
                            root: Some(encoded.encoding_id().to_string()),
                            tree: Some(tree(&encoded)),
                            verified,
                            predicted_nbytes: produced.predicted.map(|p| p.0),
                            predicted_decode_ns: produced.predicted.map(|p| p.1),
                            ..base(&produced.name)
                        }
                    }
                    Ok(Err(error)) => failed(&produced.name, &error.to_string()),
                    Err(_) => failed(&produced.name, "panic"),
                };
                out.encodings.push(result);
            }
        }
        Ok(out)
    }

    fn run_time(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let machine = MachineInfo::current()?;
        let build_id = crate::machine::build_id()?;
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
        let mut runs =
            MachineWriter::<TimingRunRow>::new(self.store, &self.worker_id, &machine.machine_id);
        runs.extend([TimingRunRow {
            machine_id: machine.machine_id.clone(),
            build_id: build_id.clone(),
            run_id: run_id.clone(),
            git_commit: self.git_commit().to_string(),
            governor: env.governor.clone(),
            turbo: env.turbo,
            cpus_allowed: env.cpus_allowed.clone(),
            release: env.release,
            target_features: env.target_features.clone(),
            warnings: env.warnings(),
        }])?;
        runs.flush()?;
        let mut measures_out =
            MachineWriter::<MeasureRow>::new(self.store, &self.worker_id, &machine.machine_id);
        let mut costs_out =
            MachineWriter::<FeatureCostRow>::new(self.store, &self.worker_id, &machine.machine_id);
        let done = self.store.ledger(TaskKind::Candidates)?;
        let mut encodings: BTreeMap<String, Vec<EncodingRow>> = BTreeMap::new();
        for row in self.store.read_table::<EncodingRow>(CANDIDATES_VERSION)? {
            encodings.entry(row.task_key.clone()).or_default().push(row);
        }
        let loader = ChunkLoader::new(self.store, &options.data)?;
        let mut writer = TimingWriter::new(self.store, &self.worker_id, &machine.machine_id);
        let production = &self.candidates["production"];
        let mut report = StageReport::default();
        let mut ctx = self.session.create_execution_ctx();
        for task in self.tasks(TaskKind::Candidates, options.shard) {
            let (Some(_), Some(rows)) = (done.get(&task.key), encodings.get(task.key.as_str()))
            else {
                report.blocked += 1;
                continue;
            };
            let TaskInput::Candidates { chunk, .. } = &task.input else {
                continue;
            };
            let (_, array) = loader.load(chunk, &mut ctx)?;
            let sample = |candidate: Option<&str>, hash: &str, matched: bool, op: &str| {
                let base = TimingRow {
                    machine_id: machine.machine_id.clone(),
                    build_id: build_id.clone(),
                    run_id: run_id.clone(),
                    git_commit: self.git_commit().to_string(),
                    task_key: task.key.to_string(),
                    chunk_key: chunk.to_string(),
                    candidate: candidate.map(str::to_string),
                    encoded_hash: hash.to_string(),
                    replay_matched: matched,
                    op: op.to_string(),
                    rep: 0,
                    ns: 0,
                };
                move |rep: u32, ns: u64| TimingRow {
                    rep,
                    ns,
                    ..base.clone()
                }
            };
            let mut samples = Vec::new();
            // Identical encodings (the same plan reached by several candidates) are rebuilt and
            // timed once, keyed by their hash, so candidate order can't favor one name.
            let mut rebuilt: BTreeMap<String, (ArrayRef, bool)> = BTreeMap::new();
            for row in rows.iter().filter(|r| r.ok) {
                let (Some(hash), Some(plan_text)) = (&row.encoded_hash, &row.plan) else {
                    continue;
                };
                if !rebuilt.contains_key(hash) {
                    let plan = crate::candidates::parse_plan(plan_text, &self.session)?;
                    let mut times = Vec::with_capacity(options.compress_reps as usize);
                    let mut matched = true;
                    let mut last = None;
                    for _ in 0..options.compress_reps.max(1) {
                        let start = Instant::now();
                        let encoded = production.compress_with_plan(&array, &plan, &mut ctx)?;
                        times.push(u64::try_from(start.elapsed().as_nanos())?);
                        matched &= blob::serialize(&encoded, &self.session)?.0.hash == *hash;
                        last = Some(encoded);
                    }
                    if !matched {
                        eprintln!(
                            "nondeterminism: {} replayed differently on chunk {chunk}",
                            row.candidate
                        );
                    }
                    let make = sample(None, hash, matched, "compress");
                    samples.extend(times.into_iter().zip(0u32..).map(|(ns, rep)| make(rep, ns)));
                    if let Some(encoded) = last {
                        rebuilt.insert(hash.clone(), (encoded, matched));
                    }
                }
                if let Some(adaptive) = self.candidates.get(&row.candidate) {
                    let matched = rebuilt.get(hash).is_some_and(|r| r.1);
                    let make = sample(Some(&row.candidate), hash, matched, "adaptive_compress");
                    for rep in 0..options.compress_reps {
                        let start = Instant::now();
                        black_box(adaptive.compress(&array, &mut ctx)?);
                        samples.push(make(rep, u64::try_from(start.elapsed().as_nanos())?));
                    }
                }
            }
            for (hash, (encoded, matched)) in &rebuilt {
                black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
                let make = sample(None, hash, *matched, "decode");
                for rep in 0..options.decode_reps {
                    let start = Instant::now();
                    black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
                    samples.push(make(rep, u64::try_from(start.elapsed().as_nanos())?));
                }
            }

            // Production with its own selection, interleaved with replaying the plan it chose:
            // the difference is what selection costs.
            let (_, production_plan) = production.compress_recording_plan(&array, &mut ctx)?;
            let full = sample(Some("production"), "", true, "production_compress");
            let replay = sample(Some("production"), "", true, "production_replay");
            for rep in 0..options.compress_reps.max(1) {
                let start = Instant::now();
                black_box(production.compress(&array, &mut ctx)?);
                samples.push(full(rep, u64::try_from(start.elapsed().as_nanos())?));
                let start = Instant::now();
                black_box(production.compress_with_plan(&array, &production_plan, &mut ctx)?);
                samples.push(replay(rep, u64::try_from(start.elapsed().as_nanos())?));
            }
            let reference = measure::reference_ns(usize::try_from(array.nbytes())?);
            samples.push(sample(None, "", true, "reference")(0, reference));
            writer.extend(samples)?;

            // Each distinct encoding is measured once, from its verified replay, and its measures
            // are shared by every candidate that produced it.
            let verified: Vec<(String, ArrayRef)> = rebuilt
                .iter()
                .filter(|(_, (_, matched))| *matched)
                .map(|(hash, (encoded, _))| (hash.clone(), encoded.clone()))
                .collect();
            let workload = Workload::standard(&array, options.timing.seed, &mut ctx)?;
            let by_hash =
                measure::measure(&array, &verified, &workload, &options.timing, &mut ctx)?;
            let mut names: Vec<(String, Option<String>)> =
                vec![(measure::CANONICAL.to_string(), None)];
            names.extend(
                rows.iter()
                    .filter(|r| r.ok)
                    .filter_map(|r| Some((r.candidate.clone(), Some(r.encoded_hash.clone()?)))),
            );
            for (candidate, hash) in names {
                let key = hash.as_deref().unwrap_or(measure::CANONICAL);
                let Some(m) = by_hash.iter().find(|m| m.name == key) else {
                    continue;
                };
                measures_out.extend(m.ops.iter().map(|op| MeasureRow {
                    machine_id: machine.machine_id.clone(),
                    build_id: build_id.clone(),
                    run_id: run_id.clone(),
                    git_commit: self.git_commit().to_string(),
                    task_key: task.key.to_string(),
                    chunk_key: chunk.to_string(),
                    candidate: candidate.clone(),
                    encoded_hash: hash.clone(),
                    op: op.op.clone(),
                    correct: op.correct,
                    error: op.error.clone(),
                    batch: op.batch,
                    samples_ns: op.samples_ns.clone(),
                    converged: op.converged,
                    peak_bytes: op.memory.map(|m| m.peak_bytes),
                    allocations: op.memory.map(|m| m.allocations),
                }))?;
            }
            for site in site_features::production_sites(&array, &self.session, &mut ctx)? {
                costs_out.extend(site.features.costs.into_iter().map(|c| FeatureCostRow {
                    machine_id: machine.machine_id.clone(),
                    build_id: build_id.clone(),
                    run_id: run_id.clone(),
                    task_key: task.key.to_string(),
                    chunk_key: chunk.to_string(),
                    site_hash: site.hash.clone(),
                    group: c.group,
                    tier: c.tier,
                    ns: c.ns,
                }))?;
            }
            report.ran += 1;
        }
        writer.flush()?;
        measures_out.flush()?;
        costs_out.flush()?;
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

fn tree(array: &ArrayRef) -> String {
    array
        .display_tree_encodings_only()
        .to_string()
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" / ")
}
