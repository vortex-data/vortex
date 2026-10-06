// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use vortex::VortexSessionDefault;
use vortex::session::VortexSession;

use crate::identity::CodeIdentity;
use crate::plan::PlannedRun;
use crate::plan::TaskKind;
use crate::plan::WriteOutcome;
use crate::plan::build_plan;
use crate::runner::RunOptions;
use crate::runner::Runner;
use crate::runner::TimeObservation;
use crate::shard::Shard;
use crate::spec::PlanSpec;
use crate::store::Store;

const SPEC: &str = r#"
name = "test"
chunk_rows = 4096
dtypes = ["int"]

[schemes]
include = ["vortex.int.for", "vortex.int.bitpacking", "vortex.int.runend", "vortex.int.sparse"]

[measure]
reps = 2
compress_reps = 1

[[sources]]
kind = "synthetic"
name = "runs"
generator = "runs"
ptype = "u32"
rows = 4096
seeds = [1, 2]
[sources.params]
mean_run = [2, 16]
bits = 12

[[sources]]
kind = "synthetic"
name = "sparse"
generator = "sparse"
ptype = "i64"
rows = 4096
seeds = [7]
[sources.params]
top_frac = 0.95
null_frac = 0.1
"#;

fn identity(commit: &str) -> CodeIdentity {
    CodeIdentity {
        vx_version: "0.0.0-test".to_string(),
        git_commit: commit.to_string(),
        dirty: false,
    }
}

fn registered() -> Vec<String> {
    crate::candidates::int_schemes(&VortexSession::default())
        .iter()
        .map(|s| s.scheme_name().to_string())
        .collect()
}

fn plan(spec: &str, commit: &str) -> anyhow::Result<PlannedRun> {
    build_plan(
        &PlanSpec::from_toml_str(spec)?,
        Path::new("."),
        &identity(commit),
        &registered(),
    )
}

#[test]
fn test_plan_is_deterministic() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let first = plan(SPEC, "abc")?;
    let second = plan(SPEC, "abc")?;
    assert_eq!(first, second);

    assert_eq!(first.write(dir.path(), false)?, WriteOutcome::Created);
    let bytes = fs::read(dir.path().join("plan.json"))?;
    assert_eq!(second.write(dir.path(), false)?, WriteOutcome::Unchanged);
    assert_eq!(fs::read(dir.path().join("plan.json"))?, bytes);
    assert_eq!(PlannedRun::read(dir.path())?, first);

    // 5 synthetic chunks, each with a chunk, features and candidates task.
    assert_eq!(first.tasks.len(), 15);
    assert_eq!(
        first.plan.candidates,
        vec![
            "production",
            "sizemodel",
            "forced/for",
            "forced/bitpacking",
            "forced/sparse",
            "forced/runend"
        ]
    );
    Ok(())
}

#[test]
fn test_keys_follow_inputs_and_commit() -> anyhow::Result<()> {
    let base = plan(SPEC, "abc")?;
    let other_commit = plan(SPEC, "def")?;
    let fewer = plan(&SPEC.replace(", \"vortex.int.sparse\"", ""), "abc")?;
    let keys = |p: &PlannedRun, k: TaskKind| -> BTreeSet<String> {
        p.tasks
            .iter()
            .filter(|t| t.kind == k)
            .map(|t| t.key.to_string())
            .collect()
    };
    // A new commit changes every key.
    assert!(keys(&base, TaskKind::Chunk).is_disjoint(&keys(&other_commit, TaskKind::Chunk)));
    // A different candidate set keeps chunk and feature work, and redoes only candidates.
    assert_eq!(keys(&base, TaskKind::Chunk), keys(&fewer, TaskKind::Chunk));
    assert_eq!(
        keys(&base, TaskKind::Features),
        keys(&fewer, TaskKind::Features)
    );
    assert!(keys(&base, TaskKind::Candidates).is_disjoint(&keys(&fewer, TaskKind::Candidates)));
    Ok(())
}

#[test]
fn test_shards_partition_tasks() -> anyhow::Result<()> {
    let run = plan(SPEC, "abc")?;
    let mut seen = BTreeSet::new();
    for i in 0..4 {
        let shard = Shard::new(i, 4)?;
        for task in run.tasks.iter().filter(|t| shard.contains(t.shard_key())) {
            assert!(seen.insert(task.key.clone()), "task in two shards");
            // A chunk's tasks share its shard, so dependencies never cross shards.
            assert!(task.deps.iter().all(|d| shard.contains(d)));
        }
    }
    assert_eq!(seen.len(), run.tasks.len());
    Ok(())
}

fn options(stages: &[&str], verify: bool) -> RunOptions {
    RunOptions {
        stages: stages.iter().map(|s| s.to_string()).collect(),
        shard: None,
        verify,
        data_root: ".".into(),
        decode_reps: 2,
        compress_reps: 1,
    }
}

#[test]
fn test_run_resume_verify_time_materialize() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let store = Store::open(dir.path().join("store"))?;
    let run = plan(SPEC, "abc")?;
    let runner = Runner::new(&run, &store)?;

    // Shard 0 of 2 first, then everything: done work is never repeated.
    let mut first = options(&["chunk", "features", "candidates"], false);
    first.shard = Some(Shard::new(0, 2)?);
    runner.run(&first)?;
    let rest = runner.run(&options(&["chunk", "features", "candidates"], false))?;
    let ran: u64 = rest.values().map(|r| r.ran).sum();
    let skipped: u64 = rest.values().map(|r| r.skipped).sum();
    assert_eq!(ran + skipped, 15);
    assert!(skipped > 0);

    let again = runner.run(&options(&["chunk", "features", "candidates"], false))?;
    assert_eq!(again.values().map(|r| r.ran).sum::<u64>(), 0);

    // Re-running everything reproduces every fact exactly.
    let verified = runner.run(&options(&["chunk", "features", "candidates"], true))?;
    assert_eq!(verified.values().map(|r| r.mismatched).sum::<u64>(), 0);
    assert_eq!(verified.values().map(|r| r.ran).sum::<u64>(), 15);

    // Two timing runs append two observations per candidates task, all deterministic.
    runner.run(&options(&["time"], false))?;
    runner.run(&options(&["time"], false))?;
    let machine = crate::machine::MachineInfo::current()?.machine_id;
    for task in run.tasks.iter().filter(|t| t.kind == TaskKind::Candidates) {
        let obs: Vec<TimeObservation> = store.observations("time", &machine, &task.key)?;
        assert_eq!(obs.len(), 2);
        assert!(obs.iter().flat_map(|o| &o.timings).all(|t| t.deterministic));
    }

    let manifest =
        crate::materialize::materialize(&run, &store, &dir.path().join("dataset"), None, "stock")?;
    assert_eq!(manifest.chunks, 5);
    assert_eq!(manifest.timed_chunks, 5);
    assert_eq!(manifest.max_runs, 2);
    let rows = fs::read_to_string(dir.path().join("dataset").join("rows-stock.csv"))?;
    // A header plus one row per chunk and candidate.
    assert_eq!(rows.lines().count(), 1 + 5 * 6);
    assert!(rows.lines().any(|l| l.contains(",default,1,")));
    Ok(())
}
