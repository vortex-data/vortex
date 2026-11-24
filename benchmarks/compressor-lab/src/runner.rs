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
//! the bytes it produced. The `time` stage, on any machine, rebuilds each encoding by compressing
//! again (which is the compression it times), checks the hash, then times decoding. Samples go to
//! that machine's own partition, tagged with its build, so any number of machines can time one
//! store.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::bail;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::blob;
use crate::chunks::CHUNKS_VERSION;
use crate::chunks::ChunkLoader;
use crate::features;
use crate::machine::MachineInfo;
use crate::plan::CANDIDATES_VERSION;
use crate::plan::PlannedRun;
use crate::plan::Task;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::segment::SegmentWriter;
use crate::segment::TimingWriter;
use crate::shard::Shard;
use crate::spec::ChunkMode;
use crate::store::Store;
use crate::table::ChunkRow;
use crate::table::EncodingRow;
use crate::table::FeatureRow;
use crate::table::LedgerRow;
use crate::table::Table;
use crate::table::TimingRow;

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
    /// Where stored chunks of a remote store are cached.
    pub cache: Option<PathBuf>,
    /// Timing repetitions (from the plan's measurement settings).
    pub decode_reps: u32,
    /// Compression timing repetitions.
    pub compress_reps: u32,
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
    modes: BTreeMap<String, ChunkMode>,
    worker_id: String,
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
    /// Prepares a runner. The plan's candidates are built once, and the plan's header is recorded
    /// in the store so GC knows the plan is live.
    pub fn new(plan: &'a PlannedRun, store: &'a Store) -> anyhow::Result<Self> {
        let session = VortexSession::default();
        if crate::wrap::SESSION.get().is_none()
            && crate::wrap::SESSION.set(session.clone()).is_err()
        {
            bail!("session already set");
        }
        let candidates = crate::candidates::build(&session, &plan.plan.candidates)?;
        let modes = plan
            .plan
            .spec
            .sources
            .iter()
            .map(|s| (s.name().to_string(), s.chunk_mode()))
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
            modes,
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
            let start = Instant::now();
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
                "{stage}: ran {}, already done {}, blocked {}, mismatched {} in {:.1}s",
                report.ran,
                report.skipped,
                report.blocked,
                report.mismatched,
                start.elapsed().as_secs_f64()
            );
            reports.insert(stage.clone(), report);
        }
        Ok(reports)
    }

    fn git_commit(&self) -> &str {
        &self.plan.plan.identity.git_commit
    }

    /// Runs one fact stage: `make` turns a task into rows, which go to `T`'s table.
    fn run_facts<T: Table>(
        &self,
        kind: TaskKind,
        version: u32,
        options: &RunOptions,
        ready: impl Fn(&Task) -> bool,
        mut make: impl FnMut(&Task) -> anyhow::Result<Vec<T>>,
    ) -> anyhow::Result<StageReport> {
        let done = self.store.ledger(kind)?;
        let mut writer: SegmentWriter<'_, T> = SegmentWriter::new(
            self.store,
            &self.worker_id,
            kind,
            T::dir(version),
            self.git_commit(),
        );
        let mut report = StageReport::default();
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
            let start = Instant::now();
            let rows = make(task)?;
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
        let mut ctx = self.session.create_execution_ctx();
        self.run_facts::<ChunkRow>(
            TaskKind::Chunk,
            CHUNKS_VERSION,
            options,
            |_| true,
            |task| {
                let TaskInput::Chunk(input) = &task.input else {
                    bail!("chunk task {} has a {:?} input", task.key, task.input);
                };
                let mode = self
                    .modes
                    .get(input.source())
                    .copied()
                    .unwrap_or(ChunkMode::Copy);
                Ok(vec![crate::chunks::make(
                    self.store,
                    &task.key,
                    input,
                    mode,
                    &options.data_root,
                    &self.session,
                    &mut ctx,
                )?])
            },
        )
    }

    fn loader(&self, options: &RunOptions) -> anyhow::Result<ChunkLoader<'_>> {
        ChunkLoader::new(
            self.store,
            &self.session,
            &options.data_root,
            options.cache.clone(),
        )
    }

    fn run_features(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let loader = self.loader(options)?;
        let mut ctx = self.session.create_execution_ctx();
        self.run_facts::<FeatureRow>(
            TaskKind::Features,
            crate::plan::FEATURES_VERSION,
            options,
            |task| task.deps.iter().all(|d| loader.row(d).is_some()),
            |task| {
                let TaskInput::Features { chunk, version } = &task.input else {
                    bail!("features task {} has a {:?} input", task.key, task.input);
                };
                if *version != crate::plan::FEATURES_VERSION {
                    bail!(
                        "this binary computes features version {}, the plan asks for {version}",
                        crate::plan::FEATURES_VERSION
                    );
                }
                let (_, array) = loader.load(chunk, &mut ctx)?;
                let primitive = array.execute::<PrimitiveArray>(&mut ctx)?;
                let values = features::from_primitive(&primitive, &mut ctx)?;
                Ok(features::NAMES
                    .iter()
                    .zip(values)
                    .map(|(name, value)| FeatureRow {
                        task_key: task.key.to_string(),
                        chunk_key: chunk.to_string(),
                        site: "root".to_string(),
                        site_depth: 0,
                        feature: (*name).to_string(),
                        tier: Some(tier(name)),
                        window: None,
                        value: value.is_finite().then_some(value),
                    })
                    .collect())
            },
        )
    }

    fn run_candidates(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let loader = self.loader(options)?;
        let mut ctx = self.session.create_execution_ctx();
        self.run_facts::<EncodingRow>(
            TaskKind::Candidates,
            CANDIDATES_VERSION,
            options,
            |task| task.deps.iter().all(|d| loader.row(d).is_some()),
            |task| {
                let TaskInput::Candidates {
                    chunk, candidates, ..
                } = &task.input
                else {
                    bail!("candidates task {} has a {:?} input", task.key, task.input);
                };
                let (row, array) = loader.load(chunk, &mut ctx)?;
                let mut rows = Vec::with_capacity(candidates.len());
                for name in candidates {
                    let compressor = self
                        .candidates
                        .get(name)
                        .with_context(|| format!("candidate `{name}` was not built"))?;
                    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        compressor.0.compress_recording_plan(&array, &mut ctx)
                    }));
                    let mut result = EncodingRow {
                        task_key: task.key.to_string(),
                        chunk_key: chunk.to_string(),
                        candidate: name.clone(),
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
                    };
                    match outcome {
                        Ok(Ok((encoded, plan))) => {
                            let (blob, _) = blob::serialize(&encoded, &self.session)?;
                            let decoded = encoded.clone().execute::<Canonical>(&mut ctx)?;
                            let decoded =
                                crate::chunks::logical_hash(&decoded.into_array(), &mut ctx)?;
                            let plan = plan.to_string();
                            result.ok = true;
                            result.plan_hash = Some(crate::key::bytes_digest(plan.as_bytes()));
                            result.plan = Some(plan);
                            result.encoded_hash = Some(blob.hash);
                            result.encoded_ids = blob.ids;
                            result.bytes = Some(blob.bytes);
                            result.nbytes = Some(encoded.nbytes());
                            result.root = Some(encoded.encoding_id().to_string());
                            result.tree = Some(tree(&encoded));
                            result.verified = decoded == row.canonical_hash;
                            if !result.verified {
                                eprintln!(
                                    "candidate {name} on chunk {chunk} does not decode to the chunk"
                                );
                            }
                        }
                        Ok(Err(error)) => result.error = Some(first_line(&error.to_string())),
                        Err(_) => result.error = Some("panic".to_string()),
                    }
                    rows.push(result);
                }
                Ok(rows)
            },
        )
    }

    fn run_time(&self, options: &RunOptions) -> anyhow::Result<StageReport> {
        let machine = MachineInfo::current()?;
        let build_id = crate::machine::build_id()?;
        self.store.put_machine(&machine)?;
        let run_id = format!(
            "{}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            std::process::id()
        );
        let done = self.store.ledger(TaskKind::Candidates)?;
        let mut encodings: BTreeMap<String, Vec<EncodingRow>> = BTreeMap::new();
        for row in self.store.read_table::<EncodingRow>(CANDIDATES_VERSION)? {
            encodings.entry(row.task_key.clone()).or_default().push(row);
        }
        let loader = self.loader(options)?;
        let mut writer = TimingWriter::new(self.store, &self.worker_id, &machine.machine_id);
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
            // The rebuilt encoding of each distinct hash, and whether a replay reproduced it.
            let mut rebuilt: BTreeMap<String, (ArrayRef, bool)> = BTreeMap::new();
            for row in rows.iter().filter(|r| r.ok) {
                let Some(hash) = &row.encoded_hash else {
                    continue;
                };
                let compressor = self
                    .candidates
                    .get(&row.candidate)
                    .with_context(|| format!("candidate `{}` was not built", row.candidate))?;
                let mut times = Vec::with_capacity(options.compress_reps as usize);
                let mut matched = true;
                let mut last = None;
                for _ in 0..options.compress_reps.max(1) {
                    let start = Instant::now();
                    let encoded = compressor.compress(&array, &mut ctx)?;
                    times.push(u64::try_from(start.elapsed().as_nanos())?);
                    matched &= blob::serialize(&encoded, &self.session)?.0.hash == *hash;
                    last = Some(encoded);
                }
                if !matched {
                    eprintln!(
                        "nondeterminism: {} rebuilt differently on chunk {chunk}",
                        row.candidate
                    );
                }
                let make = sample(Some(&row.candidate), hash, matched, "compress");
                samples.extend(times.into_iter().zip(0u32..).map(|(ns, rep)| make(rep, ns)));
                if let Some(encoded) = last {
                    let entry = rebuilt.entry(hash.clone()).or_insert((encoded, matched));
                    entry.1 &= matched;
                }
            }
            let mut decoded = BTreeSet::new();
            for (hash, (encoded, matched)) in &rebuilt {
                if !decoded.insert(hash) {
                    continue;
                }
                black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
                let make = sample(None, hash, *matched, "decode");
                for rep in 0..options.decode_reps {
                    let start = Instant::now();
                    black_box(encoded.clone().execute::<Canonical>(&mut ctx)?);
                    samples.push(make(rep, u64::try_from(start.elapsed().as_nanos())?));
                }
            }
            writer.extend(samples)?;
            report.ran += 1;
        }
        writer.flush()?;
        Ok(report)
    }
}

fn first_line(error: &str) -> String {
    error.lines().next().unwrap_or_default().to_string()
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
