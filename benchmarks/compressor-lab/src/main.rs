// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vx-lab`: deterministic, checkpointed data generation for learning scheme selection.
//!
//! ```text
//! vx-lab plan create --spec lab.toml --out plan/        # deterministic task DAG
//! vx-lab plan show   --plan plan/ --store store/        # progress per stage
//! vx-lab plan tasks  --plan plan/ --kind candidates --shard 3/16 --pending --store store/
//! vx-lab run --plan plan/ --store store/ --stages chunk,features,candidates --shard 3/16
//! vx-lab run --plan plan/ --store store/ --stages time    # on a quiet machine; appends a run
//! vx-lab materialize --plan plan/ --store store/ --out dataset/
//! vx-lab machine                                         # this machine's identity
//!
//! # Many workers sharing one plan through the store's work queue:
//! vx-lab plan push --plan plan/ --store store/ --shards 256
//! vx-lab work   --store store/ --plan-id <id> --workers 16   # on each machine; exits when done
//! vx-lab status --store store/ --plan-id <id> [--json]
//! ```

use std::path::Path;
use std::path::PathBuf;

use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use compressor_lab::candidates;
use compressor_lab::fetch::RawData;
use compressor_lab::identity::CodeIdentity;
use compressor_lab::load::DataLocation;
use compressor_lab::machine::MachineInfo;
use compressor_lab::plan::PlannedRun;
use compressor_lab::plan::TaskKind;
use compressor_lab::plan::WriteOutcome;
use compressor_lab::plan::build_plan;
use compressor_lab::queue::Queue;
use compressor_lab::queue::QueueConfig;
use compressor_lab::queue::WorkOptions;
use compressor_lab::runner::RunOptions;
use compressor_lab::runner::Runner;
use compressor_lab::shard::Shard;
use compressor_lab::spec::PlanSpec;
use compressor_lab::store::Store;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;

#[derive(Parser, Debug)]
#[command(
    name = "vx-lab",
    about = "Data generation for learned compression scheme selection"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create or inspect plans.
    #[command(subcommand)]
    Plan(PlanCommand),
    /// Run pending tasks of a plan.
    Run {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: PathBuf,
        /// Stages to run in order: chunk, features, candidates, time.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "chunk,features,candidates"
        )]
        stages: Vec<String>,
        /// Only tasks in shard `i/n`.
        #[arg(long)]
        shard: Option<Shard>,
        /// Re-run done fact tasks and report any output that changed.
        #[arg(long)]
        verify: bool,
        /// Where the plan's relative Parquet paths are resolved from (the spec's directory).
        #[arg(long, default_value = ".")]
        data_root: PathBuf,
        /// Where `reference` sources' files are cached (default `$VX_LAB_CACHE`, else
        /// `~/.cache/vx-lab`). `copy` sources are kept in the store under `raw/`.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Run with a dirty checkout or a binary that does not match the plan's commit.
        #[arg(long)]
        allow_dirty: bool,
    },
    /// Build training tables from a plan's facts and one machine's timings.
    Materialize {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// The machine whose timings to use; required when several machines have timings.
        #[arg(long)]
        machine: Option<String>,
        #[arg(long, default_value = "stock")]
        tag: String,
    },
    /// Compare the model-driven compressor with production on a plan's chunks.
    Eval {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: PathBuf,
        /// Models from `train.py --export`: `<source>.json` held out per source.
        #[arg(long)]
        models: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Target bandwidths as `label=bytes_per_sec`.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "25MBps=2.5e7,100MBps=1e8,500MBps=5e8,2GBps=2e9,8GBps=8e9,32GBps=3.2e10"
        )]
        bandwidths: Vec<String>,
        /// Reads per write; compression time is divided by this. The default ignores it.
        #[arg(long, default_value_t = 1e12)]
        reads: f64,
        #[arg(long, default_value_t = 0.1)]
        gate: f64,
        /// Candidates the model may not use, e.g. `pco`.
        #[arg(long, value_delimiter = ',')]
        exclude: Vec<String>,
        /// How many proposals to compress and verify per chunk; `all` tries every candidate.
        #[arg(long, value_delimiter = ',', default_value = "1")]
        top_k: Vec<String>,
        /// Fixed trial sets to compare, e.g. `sizemodel,for,for+sparse` (no model involved).
        #[arg(long, value_delimiter = ',')]
        trial_sets: Vec<String>,
        #[arg(long)]
        shard: Option<Shard>,
    },
    /// Claim and run shards of a pushed plan until none is left. Run it on as many machines as
    /// you like.
    Work {
        #[arg(long)]
        store: PathBuf,
        /// The pushed plan's id.
        #[arg(long, conflicts_with = "plan", required_unless_present = "plan")]
        plan_id: Option<String>,
        /// A local plan directory, to take the plan id from. The plan must have been pushed.
        #[arg(long)]
        plan: Option<PathBuf>,
        /// Worker processes to start on this machine.
        #[arg(long, default_value_t = 1)]
        workers: usize,
        /// Keep polling for shards held by workers that may die, instead of exiting once
        /// nothing is claimable.
        #[arg(long)]
        wait: bool,
        #[arg(long, default_value_t = 30)]
        poll_secs: u64,
        /// Stop after this many shards per worker, e.g. to time a sample.
        #[arg(long)]
        max_shards: Option<u64>,
        #[arg(long, default_value = ".")]
        data_root: PathBuf,
        /// Where `reference` sources' files are cached (default `$VX_LAB_CACHE`, else
        /// `~/.cache/vx-lab`). `copy` sources are kept in the store under `raw/`.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        allow_dirty: bool,
    },
    /// Show a pushed plan's progress: shards, tasks per stage, workers, failures and time left.
    Status {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, conflicts_with = "plan", required_unless_present = "plan")]
        plan_id: Option<String>,
        #[arg(long)]
        plan: Option<PathBuf>,
        /// Print JSON, for scripts and agents.
        #[arg(long)]
        json: bool,
    },
    /// Print this machine's identity.
    Machine,
}

#[derive(Subcommand, Debug)]
enum PlanCommand {
    /// Expand a spec into a plan. Re-creating the same plan is a no-op.
    Create {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Plan with a dirty checkout; task keys are then marked dirty.
        #[arg(long)]
        allow_dirty: bool,
        /// Replace a different plan already in `out`.
        #[arg(long)]
        replace: bool,
        /// Where `reference` sources' files are cached (default `$VX_LAB_CACHE`, else
        /// `~/.cache/vx-lab`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// A store to keep `copy` sources' files in while planning. Without one they wait in the
        /// cache until the first run moves them into its store.
        #[arg(long)]
        store: Option<PathBuf>,
    },
    /// Show a plan's stages and, with a store, their progress.
    Show {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: Option<PathBuf>,
    },
    /// Copy a plan into a store's work queue and fix how it is sharded, so workers anywhere can
    /// run it with `vx-lab work`. Pushing the same plan again is a no-op.
    Push {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: PathBuf,
        /// Shards to cut the plan into; aim for several per worker. Defaults to one per eight
        /// chunks, at most 4,096.
        #[arg(long)]
        shards: Option<u64>,
        /// Seconds a worker's lease on a shard lasts without a heartbeat.
        #[arg(long, default_value_t = 600)]
        lease_secs: u64,
        /// Tries before a failing task is given up on.
        #[arg(long, default_value_t = 3)]
        max_attempts: u32,
    },
    /// List task keys, e.g. to hand shards to machines.
    Tasks {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        shard: Option<Shard>,
        /// Only tasks not yet done in `--store`.
        #[arg(long)]
        pending: bool,
        #[arg(long)]
        store: Option<PathBuf>,
    },
}

fn check_identity(run: &PlannedRun, allow_dirty: bool) -> anyhow::Result<()> {
    let identity = CodeIdentity::of_this_build(allow_dirty)?;
    if identity != run.plan.identity && !allow_dirty {
        bail!(
            "this binary is {} (commit {}{}), the plan was made with {} (commit {}); \
             facts are keyed by commit, so build the plan's commit or re-plan",
            identity.vx_version,
            identity.git_commit,
            if identity.dirty { ", dirty" } else { "" },
            run.plan.identity.vx_version,
            run.plan.identity.git_commit
        );
    }
    Ok(())
}

fn resolve_plan_id(plan_id: Option<String>, plan: Option<&Path>) -> anyhow::Result<String> {
    match (plan_id, plan) {
        (Some(id), _) => Ok(id),
        (None, Some(dir)) => Ok(PlannedRun::read(dir)?.plan.plan_id),
        (None, None) => bail!("pass --plan-id or --plan"),
    }
}

/// Re-runs this command as `count` single-worker processes and waits for all of them.
fn spawn_workers(count: usize) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let mut args: Vec<String> = Vec::new();
    let mut rest = std::env::args().skip(1);
    while let Some(arg) = rest.next() {
        if arg == "--workers" {
            rest.next();
        } else if !arg.starts_with("--workers=") {
            args.push(arg);
        }
    }
    let mut children = (0..count)
        .map(|_| {
            std::process::Command::new(&exe)
                .args(&args)
                .args(["--workers", "1"])
                .spawn()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut failed = 0;
    for child in &mut children {
        if !child.wait()?.success() {
            failed += 1;
        }
    }
    if failed > 0 {
        bail!("{failed} of {count} workers failed");
    }
    Ok(())
}

fn print_status(status: &compressor_lab::queue::Status) {
    let s = &status.shards;
    println!("plan {}: {}", status.plan_id, status.state);
    println!(
        "shards: {} done, {} running, {} stale, {} pending, of {}",
        s.done, s.running, s.stale, s.pending, s.total
    );
    println!(
        "\n{:12} {:>9} {:>9} {:>7}",
        "stage", "done", "total", "failed"
    );
    for (name, c) in &status.stages {
        println!("{name:12} {:>9} {:>9} {:>7}", c.done, c.total, c.failed);
    }
    println!("\nlive workers: {}", status.live_workers.len());
    for w in &status.live_workers {
        println!(
            "  {}  {}  shard {}  {} shards done",
            w.worker,
            w.state,
            w.shard.map_or_else(|| "-".to_string(), |i| i.to_string()),
            w.shards_done
        );
    }
    if !status.silent_workers.is_empty() {
        println!("silent workers: {}", status.silent_workers.join(", "));
    }
    if let Some(mean) = status.mean_shard_secs {
        print!("mean shard {mean:.1}s");
        match status.eta_secs {
            Some(eta) => println!(", about {:.0} minutes left", eta / 60.0),
            None => println!(),
        }
    }
    if !status.failed_tasks.is_empty() {
        println!("\n{} tasks given up on:", status.failed_tasks.len());
        for f in status.failed_tasks.iter().take(10) {
            println!("  {} {}: {}", f.kind, f.key, f.error);
        }
    }
}

/// A unit enum's serialized name, e.g. `sequential_per_machine`.
fn serde_name(value: &impl serde::Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn main() -> anyhow::Result<()> {
    std::panic::set_hook(Box::new(|_| {}));
    match Cli::parse().command {
        Command::Plan(PlanCommand::Create {
            spec,
            out,
            allow_dirty,
            replace,
            cache_dir,
            store,
        }) => {
            let identity = CodeIdentity::of_this_build(allow_dirty)?;
            let parsed = PlanSpec::from_toml_file(&spec)?;
            let spec_dir = spec.parent().map(PathBuf::from).unwrap_or_default();
            let session = VortexSession::default();
            let registered: Vec<String> = candidates::int_schemes(&session)
                .iter()
                .map(|s| s.scheme_name().to_string())
                .collect();
            let run = build_plan(
                &parsed,
                &spec_dir,
                &identity,
                &registered,
                &RawData::new(cache_dir, store),
            )?;
            let outcome = run.write(&out, replace)?;
            let verb = match outcome {
                WriteOutcome::Created => "created",
                WriteOutcome::Unchanged => "unchanged",
                WriteOutcome::Replaced => "replaced",
            };
            println!("{verb} plan {} in {}", run.plan.plan_id, out.display());
            for (kind, count) in &run.plan.task_counts {
                println!("  {kind}: {count} tasks");
            }
            if !run.plan.skipped.is_empty() {
                println!(
                    "  {} columns skipped (see plan.json)",
                    run.plan.skipped.len()
                );
            }
        }
        Command::Plan(PlanCommand::Show { plan, store }) => {
            let run = PlannedRun::read(&plan)?;
            let store = store.map(Store::open).transpose()?;
            println!(
                "plan {}  ({})  vx {} commit {}{}",
                run.plan.plan_id,
                run.plan.name,
                run.plan.identity.vx_version,
                run.plan.identity.git_commit,
                if run.plan.identity.dirty {
                    " (dirty)"
                } else {
                    ""
                }
            );
            println!("candidates: {}", run.plan.candidates.join(", "));
            println!(
                "\n{:>4}  {:12} {:>8} {:>8} {:>8}  parallelism / output",
                "wave", "stage", "tasks", "done", "pending"
            );
            for stage in &run.plan.stages {
                let tasks: Vec<_> = run
                    .tasks
                    .iter()
                    .filter(|t| t.kind.as_str() == stage.name)
                    .collect();
                let done = store
                    .as_ref()
                    .map(|s| tasks.iter().filter(|t| s.is_done(&t.key)).count());
                let (total, done, pending) = if tasks.is_empty() {
                    ("-".to_string(), "-".to_string(), "-".to_string())
                } else {
                    let done = done.map_or_else(|| "?".to_string(), |d| d.to_string());
                    let pending = store
                        .as_ref()
                        .map(|s| {
                            tasks
                                .iter()
                                .filter(|t| !s.is_done(&t.key))
                                .count()
                                .to_string()
                        })
                        .unwrap_or_else(|| "?".to_string());
                    (tasks.len().to_string(), done, pending)
                };
                println!(
                    "{:>4}  {:12} {:>8} {:>8} {:>8}  {} / {}",
                    stage.wave,
                    stage.name,
                    total,
                    done,
                    pending,
                    serde_name(&stage.parallelism),
                    serde_name(&stage.output)
                );
            }
            if let Some(store) = &store {
                for machine in store.machines("time") {
                    let timed = run
                        .tasks
                        .iter()
                        .filter(|t| t.kind == TaskKind::Candidates)
                        .map(|t| {
                            store
                                .observations::<serde_json::Value>("time", &machine, &t.key)
                                .map(|o| o.len())
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    println!(
                        "time on machine {machine}: {} of {} candidate tasks timed, {} runs total",
                        timed.iter().filter(|n| **n > 0).count(),
                        timed.len(),
                        timed.iter().sum::<usize>()
                    );
                }
            }
        }
        Command::Plan(PlanCommand::Tasks {
            plan,
            kind,
            shard,
            pending,
            store,
        }) => {
            let run = PlannedRun::read(&plan)?;
            let store = store.map(Store::open).transpose()?;
            if pending && store.is_none() {
                bail!("--pending needs --store");
            }
            for task in &run.tasks {
                if kind.as_deref().is_some_and(|k| k != task.kind.as_str())
                    || shard.is_some_and(|s| !s.contains(task.shard_key()))
                    || (pending && store.as_ref().is_some_and(|s| s.is_done(&task.key)))
                {
                    continue;
                }
                println!("{}\t{}", task.kind.as_str(), task.key);
            }
        }
        Command::Run {
            plan,
            store,
            stages,
            shard,
            verify,
            data_root,
            cache_dir,
            allow_dirty,
        } => {
            let run = PlannedRun::read(&plan)?;
            check_identity(&run, allow_dirty)?;
            let raw = RawData::new(cache_dir, Some(store.clone()));
            let store = Store::open(store)?;
            let runner = Runner::new(&run, &store)?;
            runner.run(&RunOptions {
                stages,
                shard,
                verify,
                data: DataLocation { data_root, raw },
                decode_reps: run.plan.spec.measure.reps,
                compress_reps: run.plan.spec.measure.compress_reps,
                keep_going: false,
                skip: Default::default(),
            })?;
        }
        Command::Materialize {
            plan,
            store,
            out,
            machine,
            tag,
        } => {
            let run = PlannedRun::read(&plan)?;
            let store = Store::open(store)?;
            let manifest = compressor_lab::materialize::materialize(
                &run,
                &store,
                &out,
                machine.as_deref(),
                &tag,
            )?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Eval {
            plan,
            store,
            models,
            out,
            bandwidths,
            reads,
            gate,
            exclude,
            top_k,
            trial_sets,
            shard,
        } => {
            let top_k = top_k
                .iter()
                .map(|k| {
                    if k == "all" {
                        Ok(usize::MAX)
                    } else {
                        k.parse::<usize>()
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let run = PlannedRun::read(&plan)?;
            let store = Store::open(store)?;
            let bandwidths = bandwidths
                .iter()
                .map(|b| {
                    let (label, value) = b.split_once('=').ok_or_else(|| {
                        anyhow::anyhow!("bandwidths look like label=bytes_per_sec")
                    })?;
                    Ok((label.to_string(), value.parse::<f64>()?))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let n = compressor_lab::eval::evaluate(
                &run,
                &store,
                &compressor_lab::eval::EvalOptions {
                    models,
                    bandwidths,
                    reads,
                    gate,
                    exclude,
                    top_k,
                    trial_sets,
                    compress_reps: run.plan.spec.measure.compress_reps,
                    decode_reps: run.plan.spec.measure.reps,
                    shard,
                },
                &out,
            )?;
            println!("evaluated {n} chunks into {}", out.display());
        }
        Command::Plan(PlanCommand::Push {
            plan,
            store,
            shards,
            lease_secs,
            max_attempts,
        }) => {
            let run = PlannedRun::read(&plan)?;
            let store = Store::open(store)?;
            let chunks = run
                .tasks
                .iter()
                .filter(|t| t.kind == TaskKind::Chunk)
                .count() as u64;
            let shards = shards.unwrap_or_else(|| chunks.div_ceil(8).clamp(1, 4096));
            let queue = Queue::push(
                &store,
                &run,
                QueueConfig {
                    plan_id: run.plan.plan_id.clone(),
                    shards,
                    lease_secs,
                    max_attempts,
                },
            )?;
            println!(
                "pushed plan {} to {}: {} shards over {chunks} chunks",
                queue.config().plan_id,
                store.root().display(),
                queue.config().shards
            );
        }
        Command::Work {
            store,
            plan_id,
            plan,
            workers,
            wait,
            poll_secs,
            max_shards,
            data_root,
            cache_dir,
            allow_dirty,
        } => {
            let plan_id = resolve_plan_id(plan_id, plan.as_deref())?;
            if workers > 1 {
                return spawn_workers(workers);
            }
            let raw = RawData::new(cache_dir, Some(store.clone()));
            let store = Store::open(store)?;
            let queue = Queue::open(&store, &plan_id)?;
            let run = queue.plan()?;
            check_identity(&run, allow_dirty)?;
            let runner = Runner::new(&run, &store)?;
            let worker = compressor_lab::queue::worker_id();
            eprintln!("worker {worker} on plan {plan_id}");
            let summary = compressor_lab::queue::work(
                &queue,
                &run,
                &runner,
                &WorkOptions {
                    worker,
                    data: DataLocation { data_root, raw },
                    wait,
                    poll_secs,
                    max_shards,
                },
            )?;
            println!("{}", serde_json::to_string(&summary)?);
        }
        Command::Status {
            store,
            plan_id,
            plan,
            json,
        } => {
            let plan_id = resolve_plan_id(plan_id, plan.as_deref())?;
            let store = Store::open(store)?;
            let status = compressor_lab::queue::status(&Queue::open(&store, &plan_id)?)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                print_status(&status);
            }
        }
        Command::Machine => println!(
            "{}",
            serde_json::to_string_pretty(&MachineInfo::current()?)?
        ),
    }
    Ok(())
}
