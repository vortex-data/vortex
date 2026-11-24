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
//! ```

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

#[global_allocator]
static GLOBAL: compressor_lab::alloc::Counting = compressor_lab::alloc::Counting;

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
        /// Run with a dirty checkout or a binary that does not match the plan's commit.
        #[arg(long)]
        allow_dirty: bool,
        #[command(flatten)]
        timing: TimingArgs,
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
    /// Print this machine's identity and timing conditions.
    Machine,
    /// Site features and quality measures for every column of a TPC-H table, any dtype, without
    /// a plan. Writes one JSON line per column.
    Probe {
        /// The TPC-H table, e.g. `lineitem`.
        #[arg(long, default_value = "lineitem")]
        table: String,
        #[arg(long, default_value_t = 1.0)]
        scale_factor: f64,
        /// Rows probed from the start of each column.
        #[arg(long, default_value_t = 65536)]
        rows: usize,
        /// Where the JSON lines go.
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        timing: TimingArgs,
    },
}

/// How decode and pushdown operations are timed.
#[derive(clap::Args, Debug, Clone)]
struct TimingArgs {
    /// Samples per candidate and operation at most.
    #[arg(long, default_value_t = 31)]
    max_samples: usize,
    /// Stop once the 95% interval of the median is within this fraction of it.
    #[arg(long, default_value_t = 0.05)]
    target_ci: f64,
    /// The shortest sample, µs; shorter operations are batched.
    #[arg(long, default_value_t = 50)]
    min_sample_us: u64,
    /// Time budget per operation across candidates, ms.
    #[arg(long, default_value_t = 200)]
    op_budget_ms: u64,
}

impl TimingArgs {
    fn config(&self) -> compressor_lab::measure::TimingConfig {
        compressor_lab::measure::TimingConfig {
            max_samples: self.max_samples,
            target_ci: self.target_ci,
            min_sample_ns: self.min_sample_us * 1_000,
            budget_ns: self.op_budget_ms * 1_000_000,
            ..Default::default()
        }
    }
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
            allow_dirty,
            timing,
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
                decode_reps: run.plan.spec.measure.reps,
                compress_reps: run.plan.spec.measure.compress_reps,
                timing: timing.config(),
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
        Command::Machine => {
            println!(
                "{}",
                serde_json::to_string_pretty(&MachineInfo::current()?)?
            );
            let env = compressor_lab::machine::TimingEnv::current();
            println!("{}", serde_json::to_string_pretty(&env)?);
            for warning in env.warnings() {
                eprintln!("warning: {warning}");
            }
        }
        Command::Probe {
            table,
            scale_factor,
            rows,
            out,
            timing,
        } => {
            // Probing catches no panics, so report them.
            drop(std::panic::take_hook());
            let mut file = std::io::BufWriter::new(std::fs::File::create(&out)?);
            compressor_lab::probe::probe_tpch(
                &table,
                scale_factor,
                rows,
                &timing.config(),
                &mut file,
            )?;
            println!("wrote {}", out.display());
        }
    }
    Ok(())
}
