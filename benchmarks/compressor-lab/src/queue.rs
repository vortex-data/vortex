// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A work queue over a plan's shards, kept in the store, so any number of workers on any number
//! of machines can share one plan.
//!
//! A pushed plan is cut into many more shards than there are workers. Each worker loops: claim a
//! free shard with a lease, run its fact stages, mark it done, release the lease. A lease carries
//! an expiry that the worker keeps extending while it runs, so a shard whose worker died (a spot
//! machine reclaimed, a process killed) is claimed again once its lease expires. Fact tasks are
//! idempotent and skipped once done, so a reclaimed shard only does the work that is missing,
//! and the rare shard run twice at once only wastes time.
//!
//! ```text
//! <store>/queue/<plan id>/
//!   plan/plan.json, plan/tasks.jsonl       the pushed plan
//!   config.json                            shard count, lease length, retry limit
//!   leases/<shard>/<generation>.json       claims; a new claim adds the next generation
//!   done/<shard>.json                      a finished shard
//!   progress/<shard>.json                  per-stage counts, written as a shard runs
//!   failures/<shard>/<task key>.json       a failing task's attempts and last error
//!   workers/<worker id>.json               each worker's heartbeat
//! ```
//!
//! Every operation is a get, a put, a put-if-absent or a list, so the same layout works on an
//! object store. Claims never overwrite: a worker claims generation `g + 1` of a shard with a
//! put-if-absent, so of two workers racing for an expired lease only one wins.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::bail;
use parking_lot::Mutex;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::key::TaskKey;
use crate::machine::MachineInfo;
use crate::plan::PLAN_FILE;
use crate::plan::PlannedRun;
use crate::plan::TASKS_FILE;
use crate::plan::Task;
use crate::plan::TaskKind;
use crate::runner::RunOptions;
use crate::runner::Runner;
use crate::shard::Shard;
use crate::store::Store;

/// The fact stages a worker runs, in order. Timing runs separately, on a quiet machine.
pub const FACT_STAGES: [TaskKind; 3] = [TaskKind::Chunk, TaskKind::Features, TaskKind::Candidates];

/// How a pushed plan is split and retried. Fixed when the plan is first pushed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueConfig {
    /// The plan id.
    pub plan_id: String,
    /// How many shards the plan is cut into.
    pub shards: u64,
    /// How long a lease lasts without a heartbeat, in seconds.
    pub lease_secs: u64,
    /// How many times a failing task is tried before it is given up on.
    pub max_attempts: u32,
}

/// A claim on a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lease {
    /// The worker holding it.
    pub worker: String,
    /// Unix seconds when it was claimed.
    pub claimed_at: u64,
    /// Unix seconds after which another worker may claim the shard.
    pub expires_at: u64,
    /// Whether the holder gave it up.
    pub released: bool,
}

/// A finished shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardDone {
    /// The worker that finished it.
    pub worker: String,
    /// Unix seconds when that worker claimed it.
    pub started_at: u64,
    /// Unix seconds when it finished.
    pub finished_at: u64,
    /// Tasks given up on after `max_attempts`.
    pub failed: u64,
}

/// One stage's counts within one shard.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageCounts {
    /// Tasks of this stage in the shard.
    pub total: u64,
    /// Tasks done.
    pub done: u64,
    /// Tasks that failed and were given up on.
    pub failed: u64,
}

/// What a shard has done so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardProgress {
    /// The shard index.
    pub shard: u64,
    /// The worker running it.
    pub worker: String,
    /// `running` or `done`.
    pub state: String,
    /// Unix seconds when the worker claimed it.
    pub started_at: u64,
    /// Unix seconds of the last update.
    pub updated_at: u64,
    /// Counts per stage.
    pub stages: BTreeMap<String, StageCounts>,
}

/// A task that failed at least once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailure {
    /// The task kind.
    pub kind: String,
    /// The task key.
    pub key: TaskKey,
    /// How many times it failed.
    pub attempts: u32,
    /// The last error.
    pub error: String,
    /// The worker that last tried it.
    pub worker: String,
}

/// A worker's heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerBeat {
    /// The worker id.
    pub worker: String,
    /// The host name.
    pub host: String,
    /// The machine identity.
    pub machine_id: String,
    /// Unix seconds when the worker started.
    pub started_at: u64,
    /// Unix seconds of the last heartbeat.
    pub heartbeat_at: u64,
    /// The shard it is running, if any.
    pub shard: Option<u64>,
    /// `working`, `idle` or `exited`.
    pub state: String,
    /// Shards it finished.
    pub shards_done: u64,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

fn get_json<T: DeserializeOwned>(store: &Store, path: &str) -> anyhow::Result<Option<T>> {
    store
        .get_raw(path)?
        .map(|bytes| serde_json::from_slice(&bytes).with_context(|| format!("parsing {path}")))
        .transpose()
}

fn put_json<T: Serialize>(store: &Store, path: &str, value: &T) -> anyhow::Result<()> {
    store.put_raw(path, &serde_json::to_vec_pretty(value)?)
}

/// A plan's queue in a store.
#[derive(Debug, Clone)]
pub struct Queue {
    store: Store,
    config: QueueConfig,
}

impl Queue {
    fn dir(plan_id: &str) -> String {
        format!("queue/{plan_id}")
    }

    /// Pushes a plan into the store and fixes its shard count. Pushing the same plan again is a
    /// no-op; pushing it with different settings fails.
    pub fn push(store: &Store, run: &PlannedRun, config: QueueConfig) -> anyhow::Result<Self> {
        let dir = Self::dir(&run.plan.plan_id);
        let plan_dir = tempfile_dir(&run.plan.plan_id)?;
        run.write(&plan_dir, true)?;
        for name in [TASKS_FILE, PLAN_FILE] {
            let bytes = std::fs::read(plan_dir.join(name))?;
            match store.get_raw(&format!("{dir}/plan/{name}"))? {
                Some(existing) if existing != bytes => {
                    bail!(
                        "the store already holds a different {name} for plan {}",
                        run.plan.plan_id
                    )
                }
                Some(_) => {}
                None => store.put_raw(&format!("{dir}/plan/{name}"), &bytes)?,
            }
        }
        let path = format!("{dir}/config.json");
        if !store.put_raw_if_absent(&path, &serde_json::to_vec_pretty(&config)?)? {
            let existing: QueueConfig =
                get_json(store, &path)?.context("config vanished while pushing")?;
            if existing != config {
                bail!(
                    "plan {} was already pushed with {existing:?}; its shard count and limits \
                     cannot change once workers may have used them",
                    run.plan.plan_id
                );
            }
        }
        Ok(Self {
            store: store.clone(),
            config,
        })
    }

    /// Opens the queue of a pushed plan.
    pub fn open(store: &Store, plan_id: &str) -> anyhow::Result<Self> {
        let config = get_json(store, &format!("{}/config.json", Self::dir(plan_id)))?
            .with_context(|| format!("plan {plan_id} was not pushed to this store"))?;
        Ok(Self {
            store: store.clone(),
            config,
        })
    }

    /// Reads the pushed plan, caching it in a local directory.
    pub fn plan(&self) -> anyhow::Result<PlannedRun> {
        let dir = tempfile_dir(&self.config.plan_id)?;
        for name in [TASKS_FILE, PLAN_FILE] {
            let bytes = self
                .store
                .get_raw(&format!("{}/plan/{name}", self.base()))?
                .with_context(|| {
                    format!("plan {} has no {name} in the store", self.config.plan_id)
                })?;
            crate::store::write_atomically(&dir.join(name), &bytes)?;
        }
        PlannedRun::read(&dir)
    }

    /// The queue's settings.
    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    fn base(&self) -> String {
        Self::dir(&self.config.plan_id)
    }

    fn shard(&self, index: u64) -> anyhow::Result<Shard> {
        Shard::new(index, self.config.shards)
    }

    /// The shards marked done.
    pub fn done(&self) -> anyhow::Result<BTreeMap<u64, ShardDone>> {
        let mut done = BTreeMap::new();
        for name in self.store.list_raw(&format!("{}/done", self.base()))? {
            let Some(index) = name.strip_suffix(".json").and_then(|n| n.parse().ok()) else {
                continue;
            };
            if let Some(entry) = get_json(&self.store, &format!("{}/done/{name}", self.base()))? {
                done.insert(index, entry);
            }
        }
        Ok(done)
    }

    fn latest_lease(&self, index: u64) -> anyhow::Result<Option<(u64, Lease)>> {
        let dir = format!("{}/leases/{index}", self.base());
        let Some(name) = self.store.list_raw(&dir)?.into_iter().next_back() else {
            return Ok(None);
        };
        let generation = name
            .strip_suffix(".json")
            .and_then(|n| n.parse().ok())
            .with_context(|| format!("unexpected lease file {dir}/{name}"))?;
        Ok(get_json(&self.store, &format!("{dir}/{name}"))?.map(|lease| (generation, lease)))
    }

    fn lease_path(&self, index: u64, generation: u64) -> String {
        format!("{}/leases/{index}/{generation:010}.json", self.base())
    }

    /// Tries to claim a shard. Returns the lease generation on success.
    pub fn try_claim(&self, index: u64, worker: &str) -> anyhow::Result<Option<u64>> {
        let now = now();
        let generation = match self.latest_lease(index)? {
            None => 0,
            Some((generation, lease)) if lease.released || lease.expires_at <= now => {
                generation + 1
            }
            Some(_) => return Ok(None),
        };
        let lease = Lease {
            worker: worker.to_string(),
            claimed_at: now,
            expires_at: now + self.config.lease_secs,
            released: false,
        };
        let claimed = self.store.put_raw_if_absent(
            &self.lease_path(index, generation),
            &serde_json::to_vec_pretty(&lease)?,
        )?;
        Ok(claimed.then_some(generation))
    }

    /// Extends a lease. Returns false if another worker has since claimed the shard.
    pub fn renew(
        &self,
        index: u64,
        generation: u64,
        worker: &str,
        claimed_at: u64,
    ) -> anyhow::Result<bool> {
        if self
            .store
            .get_raw(&self.lease_path(index, generation + 1))?
            .is_some()
        {
            return Ok(false);
        }
        put_json(
            &self.store,
            &self.lease_path(index, generation),
            &Lease {
                worker: worker.to_string(),
                claimed_at,
                expires_at: now() + self.config.lease_secs,
                released: false,
            },
        )?;
        Ok(true)
    }

    /// Gives a lease up, so the shard can be claimed at once.
    pub fn release(
        &self,
        index: u64,
        generation: u64,
        worker: &str,
        claimed_at: u64,
    ) -> anyhow::Result<()> {
        put_json(
            &self.store,
            &self.lease_path(index, generation),
            &Lease {
                worker: worker.to_string(),
                claimed_at,
                expires_at: now(),
                released: true,
            },
        )
    }

    fn failures(&self, index: u64) -> anyhow::Result<BTreeMap<TaskKey, TaskFailure>> {
        let dir = format!("{}/failures/{index}", self.base());
        let mut failures = BTreeMap::new();
        for name in self.store.list_raw(&dir)? {
            if let Some(failure) = get_json::<TaskFailure>(&self.store, &format!("{dir}/{name}"))? {
                failures.insert(failure.key.clone(), failure);
            }
        }
        Ok(failures)
    }

    fn record_failure(
        &self,
        index: u64,
        kind: TaskKind,
        key: &TaskKey,
        error: &str,
        worker: &str,
    ) -> anyhow::Result<u32> {
        let path = format!("{}/failures/{index}/{key}.json", self.base());
        let attempts = get_json::<TaskFailure>(&self.store, &path)?.map_or(0, |f| f.attempts) + 1;
        put_json(
            &self.store,
            &path,
            &TaskFailure {
                kind: kind.as_str().to_string(),
                key: key.clone(),
                attempts,
                error: error.lines().next().unwrap_or_default().to_string(),
                worker: worker.to_string(),
            },
        )?;
        Ok(attempts)
    }

    /// Every failed task, over all shards.
    pub fn all_failures(&self) -> anyhow::Result<Vec<TaskFailure>> {
        let mut all = Vec::new();
        for index in 0..self.config.shards {
            all.extend(self.failures(index)?.into_values());
        }
        Ok(all)
    }

    /// Progress files, by shard.
    pub fn progress(&self) -> anyhow::Result<BTreeMap<u64, ShardProgress>> {
        let dir = format!("{}/progress", self.base());
        let mut progress = BTreeMap::new();
        for name in self.store.list_raw(&dir)? {
            if let Some(p) = get_json::<ShardProgress>(&self.store, &format!("{dir}/{name}"))? {
                progress.insert(p.shard, p);
            }
        }
        Ok(progress)
    }

    /// Worker heartbeats.
    pub fn workers(&self) -> anyhow::Result<Vec<WorkerBeat>> {
        let dir = format!("{}/workers", self.base());
        let mut workers = Vec::new();
        for name in self.store.list_raw(&dir)? {
            if let Some(w) = get_json::<WorkerBeat>(&self.store, &format!("{dir}/{name}"))? {
                workers.push(w);
            }
        }
        Ok(workers)
    }

    fn put_beat(&self, beat: &WorkerBeat) -> anyhow::Result<()> {
        put_json(
            &self.store,
            &format!("{}/workers/{}.json", self.base(), beat.worker),
            beat,
        )
    }
}

fn tempfile_dir(plan_id: &str) -> anyhow::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join("vx-lab-plans").join(plan_id);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// A worker id unique to this process: host, process id and start time.
pub fn worker_id() -> String {
    format!("{}-{}-{}", host_name(), std::process::id(), now())
}

/// This machine's host name.
pub fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| std::fs::read_to_string("/etc/hostname"))
        .map(|h| h.trim().to_string())
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_string())
        .replace(['/', ' '], "_")
}

/// How a worker runs.
#[derive(Debug, Clone)]
pub struct WorkOptions {
    /// This worker's id.
    pub worker: String,
    /// Where the plan's relative Parquet paths are resolved from.
    pub data_root: std::path::PathBuf,
    /// Keep polling for shards whose lease may expire, instead of exiting once nothing can be
    /// claimed.
    pub wait: bool,
    /// Seconds between polls with `wait`.
    pub poll_secs: u64,
    /// Stop after this many shards.
    pub max_shards: Option<u64>,
}

/// What a worker did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WorkSummary {
    /// Shards this worker finished.
    pub shards_done: u64,
    /// Tasks it ran.
    pub tasks_ran: u64,
    /// Tasks that failed at least once while it ran them.
    pub task_failures: u64,
    /// Whether every shard of the plan is done.
    pub plan_done: bool,
}

struct Current {
    index: u64,
    generation: u64,
    claimed_at: u64,
}

/// Claims and runs shards until none is left to claim.
pub fn work(
    queue: &Queue,
    run: &PlannedRun,
    runner: &Runner<'_>,
    options: &WorkOptions,
) -> anyhow::Result<WorkSummary> {
    let machine = MachineInfo::current()?;
    queue.store.put_machine(&machine)?;
    let shard_tasks = tasks_by_shard(run, queue.config.shards)?;
    let started_at = now();
    let current: Arc<Mutex<Option<Current>>> = Arc::new(Mutex::new(None));
    let beat = Arc::new(Mutex::new(WorkerBeat {
        worker: options.worker.clone(),
        host: host_name(),
        machine_id: machine.machine_id,
        started_at,
        heartbeat_at: started_at,
        shard: None,
        state: "idle".to_string(),
        shards_done: 0,
    }));
    queue.put_beat(&beat.lock().clone())?;

    let stop = Arc::new(AtomicBool::new(false));
    let lost = Arc::new(AtomicBool::new(false));
    let heartbeat = {
        let (queue, current, beat, stop, lost) = (
            queue.clone(),
            Arc::clone(&current),
            Arc::clone(&beat),
            Arc::clone(&stop),
            Arc::clone(&lost),
        );
        let worker = options.worker.clone();
        let every = Duration::from_secs((queue.config.lease_secs / 3).max(1));
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                park_for(every, &stop);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(c) = current.lock().as_ref() {
                    match queue.renew(c.index, c.generation, &worker, c.claimed_at) {
                        Ok(true) => {}
                        Ok(false) => {
                            eprintln!(
                                "lost the lease on shard {}; another worker claimed it",
                                c.index
                            );
                            lost.store(true, Ordering::Relaxed);
                        }
                        Err(e) => eprintln!("renewing the lease on shard {}: {e:#}", c.index),
                    }
                }
                let mut b = beat.lock();
                b.heartbeat_at = now();
                if let Err(e) = queue.put_beat(&b) {
                    eprintln!("writing heartbeat: {e:#}");
                }
            }
        })
    };

    let result = work_loop(queue, runner, options, &shard_tasks, &current, &beat, &lost);
    stop.store(true, Ordering::Relaxed);
    heartbeat.thread().unpark();
    heartbeat.join().ok();
    let mut b = beat.lock().clone();
    b.state = "exited".to_string();
    b.shard = None;
    b.heartbeat_at = now();
    queue.put_beat(&b)?;
    result
}

fn park_for(duration: Duration, stop: &AtomicBool) {
    let deadline = std::time::Instant::now() + duration;
    while !stop.load(Ordering::Relaxed) {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            break;
        }
        thread::park_timeout(left);
    }
}

#[allow(clippy::too_many_arguments)]
fn work_loop(
    queue: &Queue,
    runner: &Runner<'_>,
    options: &WorkOptions,
    shard_tasks: &[Vec<&Task>],
    current: &Mutex<Option<Current>>,
    beat: &Mutex<WorkerBeat>,
    lost: &AtomicBool,
) -> anyhow::Result<WorkSummary> {
    let shards = queue.config.shards;
    let mut summary = WorkSummary::default();
    // Start at a point derived from the worker id, so workers spread out instead of all racing
    // for shard 0.
    let offset = crate::key::bytes_digest(options.worker.as_bytes())
        .get(..12)
        .and_then(|h| u64::from_str_radix(h, 16).ok())
        .unwrap_or_default()
        % shards;
    loop {
        let done = queue.done()?;
        if done.len() as u64 == shards {
            summary.plan_done = true;
            return Ok(summary);
        }
        let mut claimed = None;
        for step in 0..shards {
            let index = (offset + step) % shards;
            if done.contains_key(&index) {
                continue;
            }
            if let Some(generation) = queue.try_claim(index, &options.worker)? {
                claimed = Some((index, generation));
                break;
            }
        }
        let Some((index, generation)) = claimed else {
            if !options.wait {
                return Ok(summary);
            }
            {
                let mut b = beat.lock();
                b.state = "idle".to_string();
                b.shard = None;
            }
            thread::sleep(Duration::from_secs(options.poll_secs));
            continue;
        };

        let claimed_at = now();
        *current.lock() = Some(Current {
            index,
            generation,
            claimed_at,
        });
        lost.store(false, Ordering::Relaxed);
        {
            let mut b = beat.lock();
            b.state = "working".to_string();
            b.shard = Some(index);
            b.heartbeat_at = now();
            queue.put_beat(&b)?;
        }
        eprintln!("shard {index}/{shards}: claimed (lease generation {generation})");
        let outcome = run_shard(
            queue,
            runner,
            options,
            index,
            &shard_tasks[index as usize],
            claimed_at,
        );
        *current.lock() = None;
        let (ran, failures, given_up) = match outcome {
            Ok(counts) => counts,
            Err(e) => {
                // Leave the lease to expire rather than release it, so a shard that keeps
                // erroring outside any task does not spin between workers.
                eprintln!("shard {index}: {e:#}");
                return Err(e);
            }
        };
        summary.tasks_ran += ran;
        summary.task_failures += failures;
        queue.store.put_raw(
            &format!("{}/done/{index}.json", queue.base()),
            &serde_json::to_vec_pretty(&ShardDone {
                worker: options.worker.clone(),
                started_at: claimed_at,
                finished_at: now(),
                failed: given_up,
            })?,
        )?;
        if !lost.load(Ordering::Relaxed) {
            queue.release(index, generation, &options.worker, claimed_at)?;
        }
        summary.shards_done += 1;
        beat.lock().shards_done = summary.shards_done;
        eprintln!("shard {index}/{shards}: done, {ran} tasks ran, {given_up} given up");
        if options.max_shards.is_some_and(|m| summary.shards_done >= m) {
            return Ok(summary);
        }
    }
}

/// Runs one shard's fact stages, retrying failed tasks up to the limit. Returns tasks run,
/// failures seen and tasks given up on.
fn run_shard(
    queue: &Queue,
    runner: &Runner<'_>,
    options: &WorkOptions,
    index: u64,
    tasks: &[&Task],
    started_at: u64,
) -> anyhow::Result<(u64, u64, u64)> {
    let shard = queue.shard(index)?;
    let max_attempts = queue.config.max_attempts;
    let mut ran = 0;
    let mut failures_seen = 0;
    let mut progress = ShardProgress {
        shard: index,
        worker: options.worker.clone(),
        state: "running".to_string(),
        started_at,
        updated_at: now(),
        stages: FACT_STAGES
            .iter()
            .map(|kind| {
                let total = tasks.iter().filter(|t| t.kind == *kind).count() as u64;
                (
                    kind.as_str().to_string(),
                    StageCounts {
                        total,
                        ..Default::default()
                    },
                )
            })
            .collect(),
    };
    let progress_path = format!("{}/progress/{index}.json", queue.base());
    put_json(&queue.store, &progress_path, &progress)?;

    loop {
        let failures = queue.failures(index)?;
        let skip: BTreeSet<TaskKey> = failures
            .values()
            .filter(|f| f.attempts >= max_attempts)
            .map(|f| f.key.clone())
            .collect();
        let mut retryable = false;
        for kind in FACT_STAGES {
            let reports = runner.run(&RunOptions {
                stages: vec![kind.as_str().to_string()],
                shard: Some(shard),
                verify: false,
                data_root: options.data_root.clone(),
                decode_reps: 0,
                compress_reps: 0,
                keep_going: true,
                skip: skip.clone(),
            })?;
            let report = &reports[kind.as_str()];
            ran += report.ran;
            for (key, error) in &report.failed {
                failures_seen += 1;
                let attempts = queue.record_failure(index, kind, key, error, &options.worker)?;
                retryable |= attempts < max_attempts;
            }
            let counts = progress
                .stages
                .get_mut(kind.as_str())
                .context("stage missing from progress")?;
            counts.done = report.ran + report.skipped;
            counts.failed = failures
                .values()
                .filter(|f| f.kind == kind.as_str() && f.attempts >= max_attempts)
                .count() as u64;
            progress.updated_at = now();
            put_json(&queue.store, &progress_path, &progress)?;
        }
        if !retryable {
            break;
        }
    }

    let given_up = queue
        .failures(index)?
        .values()
        .filter(|f| f.attempts >= max_attempts)
        .count() as u64;
    for kind in FACT_STAGES {
        if let Some(counts) = progress.stages.get_mut(kind.as_str()) {
            counts.failed = queue
                .failures(index)?
                .values()
                .filter(|f| f.kind == kind.as_str() && f.attempts >= max_attempts)
                .count() as u64;
        }
    }
    progress.state = "done".to_string();
    progress.updated_at = now();
    put_json(&queue.store, &progress_path, &progress)?;
    Ok((ran, failures_seen, given_up))
}

fn tasks_by_shard(run: &PlannedRun, shards: u64) -> anyhow::Result<Vec<Vec<&Task>>> {
    let mut by_shard: Vec<Vec<&Task>> = vec![Vec::new(); usize::try_from(shards)?];
    for task in &run.tasks {
        if !FACT_STAGES.contains(&task.kind) {
            continue;
        }
        // The same rule as `Shard::contains`.
        let index = task.shard_key().bucket() % shards;
        by_shard[usize::try_from(index)?].push(task);
    }
    Ok(by_shard)
}

/// A plan's progress, for people and for an agent deciding whether to add or stop machines.
#[derive(Debug, Clone, Serialize)]
pub struct Status {
    /// The plan id.
    pub plan_id: String,
    /// `done`, `running`, `stalled` (shards left but no live worker) or `pending` (not
    /// started).
    pub state: String,
    /// Unix seconds when this status was taken.
    pub at: u64,
    /// Shard counts.
    pub shards: ShardCounts,
    /// Task counts per fact stage, from the progress files.
    pub stages: BTreeMap<String, StageCounts>,
    /// Workers with a heartbeat within one lease length.
    pub live_workers: Vec<WorkerBeat>,
    /// Workers whose last heartbeat is older than one lease length and that did not exit.
    pub silent_workers: Vec<String>,
    /// Tasks given up on.
    pub failed_tasks: Vec<TaskFailure>,
    /// Mean seconds a finished shard took.
    pub mean_shard_secs: Option<f64>,
    /// Estimated seconds until every shard is done, at the current number of live workers.
    pub eta_secs: Option<f64>,
}

/// Shard counts.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ShardCounts {
    /// All shards.
    pub total: u64,
    /// Finished.
    pub done: u64,
    /// Being run by a live worker.
    pub running: u64,
    /// Started by a worker that has gone silent; claimable once the lease expires.
    pub stale: u64,
    /// Not started.
    pub pending: u64,
}

/// Reads a plan's progress from its queue.
pub fn status(queue: &Queue) -> anyhow::Result<Status> {
    let now = now();
    let lease = queue.config.lease_secs;
    let done = queue.done()?;
    let progress = queue.progress()?;
    let workers = queue.workers()?;
    let live: Vec<WorkerBeat> = workers
        .iter()
        .filter(|w| w.state != "exited" && w.heartbeat_at + lease > now)
        .cloned()
        .collect();
    let silent = workers
        .iter()
        .filter(|w| w.state != "exited" && w.heartbeat_at + lease <= now)
        .map(|w| w.worker.clone())
        .collect();
    let running_shards: BTreeSet<u64> = live.iter().filter_map(|w| w.shard).collect();

    let mut shards = ShardCounts {
        total: queue.config.shards,
        done: done.len() as u64,
        ..Default::default()
    };
    for index in 0..queue.config.shards {
        if done.contains_key(&index) {
            continue;
        }
        if running_shards.contains(&index) {
            shards.running += 1;
        } else if progress.contains_key(&index) {
            shards.stale += 1;
        } else {
            shards.pending += 1;
        }
    }

    let mut stages: BTreeMap<String, StageCounts> = BTreeMap::new();
    for p in progress.values() {
        for (name, counts) in &p.stages {
            let entry = stages.entry(name.clone()).or_default();
            entry.done += counts.done;
            entry.failed += counts.failed;
        }
    }
    let plan = queue.plan()?;
    for kind in FACT_STAGES {
        stages.entry(kind.as_str().to_string()).or_default().total =
            plan.tasks.iter().filter(|t| t.kind == kind).count() as u64;
    }

    let durations: Vec<f64> = done
        .values()
        .map(|d| d.finished_at.saturating_sub(d.started_at) as f64)
        .collect();
    let mean_shard_secs =
        (!durations.is_empty()).then(|| durations.iter().sum::<f64>() / durations.len() as f64);
    let left = shards.total - shards.done;
    let eta_secs = mean_shard_secs
        .filter(|_| !live.is_empty())
        .map(|mean| mean * left as f64 / live.len() as f64);

    let failed_tasks = queue
        .all_failures()?
        .into_iter()
        .filter(|f| f.attempts >= queue.config.max_attempts)
        .collect();

    let state = if shards.done == shards.total {
        "done"
    } else if !live.is_empty() {
        "running"
    } else if shards.done > 0 || shards.stale > 0 {
        "stalled"
    } else {
        "pending"
    };
    Ok(Status {
        plan_id: queue.config.plan_id.clone(),
        state: state.to_string(),
        at: now,
        shards,
        stages,
        live_workers: live,
        silent_workers: silent,
        failed_tasks,
        mean_shard_secs,
        eta_secs,
    })
}
