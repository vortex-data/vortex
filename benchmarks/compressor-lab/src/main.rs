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
//! vx-lab store pull --plan plan/ --store s3://bucket/lab --shard 0/4   # cache chunks locally
//! vx-lab store compact --store s3://bucket/lab            # merge small segments
//! vx-lab store gc --store s3://bucket/lab [--delete]      # drop chunks no live plan uses
//! vx-lab store retire --store s3://bucket/lab --plan-id <id>
//! ```
//!
//! `--store` is a local directory or a URL (`s3://`, `gs://`, `az://`, `hf://`, `file://`);
//! credentials come from the usual environment variables.

use std::path::PathBuf;

use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use compressor_lab::candidates;
use compressor_lab::identity::CodeIdentity;
use compressor_lab::machine::MachineInfo;
use compressor_lab::plan::PlannedRun;
use compressor_lab::plan::TaskKind;
use compressor_lab::plan::WriteOutcome;
use compressor_lab::plan::build_plan;
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
        /// Where stored chunks of a remote store are cached (default `~/.cache/vx-lab/chunks`).
        #[arg(long)]
        cache: Option<PathBuf>,
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
        /// Where the plan's relative Parquet paths are resolved from.
        #[arg(long, default_value = ".")]
        data_root: PathBuf,
    },
    /// Print this machine's identity.
    Machine,
    /// Maintain a store.
    #[command(subcommand)]
    Store(StoreCommand),
}

#[derive(Subcommand, Debug)]
enum StoreCommand {
    /// Copy a shard's stored chunks into the local cache, e.g. before timing.
    Pull {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        shard: Option<Shard>,
        #[arg(long)]
        cache: Option<PathBuf>,
    },
    /// Merge each table's segments into one. Safe while workers run.
    Compact {
        #[arg(long)]
        store: PathBuf,
        /// Report what would be merged without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// Delete stored chunks no live plan uses, and orphan segments. A dry run unless `--delete`.
    Gc {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        delete: bool,
        /// Run even though a worker heartbeat is recent.
        #[arg(long)]
        force: bool,
    },
    /// Mark a plan retired, so GC no longer keeps the chunks only it uses.
    Retire {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        plan_id: String,
    },
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
    },
    /// Show a plan's stages and, with a store, their progress.
    Show {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        store: Option<PathBuf>,
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

/// A unit enum's serialized name, e.g. `sequential_per_machine`.
fn serde_name(value: &impl serde::Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Every finished task key in the store, if there is a store.
fn done_keys(
    store: Option<&Store>,
) -> anyhow::Result<Option<std::collections::BTreeSet<compressor_lab::key::TaskKey>>> {
    let Some(store) = store else {
        return Ok(None);
    };
    let mut done = std::collections::BTreeSet::new();
    for kind in [TaskKind::Chunk, TaskKind::Features, TaskKind::Candidates] {
        done.extend(store.ledger(kind)?.into_keys());
    }
    Ok(Some(done))
}

fn main() -> anyhow::Result<()> {
    std::panic::set_hook(Box::new(|_| {}));
    match Cli::parse().command {
        Command::Plan(PlanCommand::Create {
            spec,
            out,
            allow_dirty,
            replace,
        }) => {
            let identity = CodeIdentity::of_this_build(allow_dirty)?;
            let parsed = PlanSpec::from_toml_file(&spec)?;
            let spec_dir = spec.parent().map(PathBuf::from).unwrap_or_default();
            let session = VortexSession::default();
            let registered: Vec<String> = candidates::int_schemes(&session)
                .iter()
                .map(|s| s.scheme_name().to_string())
                .collect();
            let run = build_plan(&parsed, &spec_dir, &identity, &registered)?;
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
            let done_set = done_keys(store.as_ref())?;
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
                let done = done_set
                    .as_ref()
                    .map(|d| tasks.iter().filter(|t| d.contains(&t.key)).count());
                let (total, done, pending) = if tasks.is_empty() {
                    ("-".to_string(), "-".to_string(), "-".to_string())
                } else {
                    let done = done.map_or_else(|| "?".to_string(), |d| d.to_string());
                    let pending = done_set
                        .as_ref()
                        .map(|d| {
                            tasks
                                .iter()
                                .filter(|t| !d.contains(&t.key))
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
                let candidates: Vec<_> = run
                    .tasks
                    .iter()
                    .filter(|t| t.kind == TaskKind::Candidates)
                    .map(|t| t.key.to_string())
                    .collect();
                for machine in store.timing_machines()? {
                    let samples: Vec<compressor_lab::table::TimingRow> =
                        store.read_rows(&compressor_lab::store::timings_dir(&machine))?;
                    let mut runs: std::collections::BTreeMap<
                        &str,
                        std::collections::BTreeSet<&str>,
                    > = Default::default();
                    for sample in &samples {
                        runs.entry(sample.task_key.as_str())
                            .or_default()
                            .insert(sample.run_id.as_str());
                    }
                    let timed = candidates
                        .iter()
                        .filter(|k| runs.contains_key(k.as_str()))
                        .count();
                    let total: usize = candidates
                        .iter()
                        .filter_map(|k| runs.get(k.as_str()))
                        .map(|r| r.len())
                        .sum();
                    println!(
                        "time on machine {machine}: {timed} of {} candidate tasks timed, {total} runs total",
                        candidates.len()
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
            let done = done_keys(store.as_ref())?;
            for task in &run.tasks {
                if kind.as_deref().is_some_and(|k| k != task.kind.as_str())
                    || shard.is_some_and(|s| !s.contains(task.shard_key()))
                    || (pending && done.as_ref().is_some_and(|d| d.contains(&task.key)))
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
            cache,
            allow_dirty,
        } => {
            let run = PlannedRun::read(&plan)?;
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
            let store = Store::open(store)?;
            let runner = Runner::new(&run, &store)?;
            runner.run(&RunOptions {
                stages,
                shard,
                verify,
                data_root,
                cache,
                decode_reps: run.plan.spec.measure.reps,
                compress_reps: run.plan.spec.measure.compress_reps,
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
            data_root,
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
                    data_root,
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
        Command::Machine => {
            println!(
                "{}",
                serde_json::to_string_pretty(&MachineInfo::current()?)?
            );
            println!("build_id: {}", compressor_lab::machine::build_id()?);
        }
        Command::Store(StoreCommand::Pull {
            plan,
            store,
            shard,
            cache,
        }) => {
            let run = PlannedRun::read(&plan)?;
            let store = Store::open(store)?;
            let session = VortexSession::default();
            let loader = compressor_lab::chunks::ChunkLoader::new(
                &store,
                &session,
                std::path::Path::new("."),
                cache,
            )?;
            let mut fetched = 0;
            for task in run.tasks.iter().filter(|t| t.kind == TaskKind::Chunk) {
                if shard.is_some_and(|s| !s.contains(task.shard_key())) {
                    continue;
                }
                fetched += u64::from(loader.prefetch(&task.key)?);
            }
            println!("fetched {fetched} chunks");
        }
        Command::Store(StoreCommand::Compact { store, dry_run }) => {
            let report = compressor_lab::gc::compact(&Store::open(store)?, dry_run)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Store(StoreCommand::Gc {
            store,
            delete,
            force,
        }) => {
            let report = compressor_lab::gc::collect(&Store::open(store)?, !delete, force)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Store(StoreCommand::Retire { store, plan_id }) => {
            let store = Store::open(store)?;
            if store.get(&format!("plans/{plan_id}/plan.json"))?.is_none() {
                bail!("no plan {plan_id} in {}", store.url());
            }
            store.put(&format!("plans/{plan_id}/RETIRED"), Vec::new())?;
            println!("retired plan {plan_id}");
        }
    }
    Ok(())
}
