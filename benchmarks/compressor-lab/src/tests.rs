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
use crate::shard::Shard;
use crate::spec::PlanSpec;
use crate::store::Store;
use crate::table::TimingRow;

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
        cache: None,
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

    // Two timing runs append samples for every candidates task, and every rebuild matches.
    runner.run(&options(&["time"], false))?;
    runner.run(&options(&["time"], false))?;
    let machine = crate::machine::MachineInfo::current()?.machine_id;
    let samples: Vec<TimingRow> = store.read_rows(&crate::store::timings_dir(&machine))?;
    assert!(samples.iter().all(|s| s.replay_matched));
    for task in run.tasks.iter().filter(|t| t.kind == TaskKind::Candidates) {
        let runs: BTreeSet<&str> = samples
            .iter()
            .filter(|s| s.task_key == task.key.as_str())
            .map(|s| s.run_id.as_str())
            .collect();
        assert_eq!(runs.len(), 2);
    }

    // Compaction leaves the same rows and the ledger still marks every task done.
    let before =
        crate::materialize::materialize(&run, &store, &dir.path().join("before"), None, "stock")?;
    let compacted = crate::gc::compact(&store, false)?;
    assert!(compacted.segments_merged > 0);
    let after = runner.run(&options(&["chunk", "features", "candidates"], false))?;
    assert_eq!(after.values().map(|r| r.ran).sum::<u64>(), 0);
    let gc = crate::gc::collect(&store, false, true)?;
    assert_eq!(gc.orphans_deleted, 0);
    assert_eq!(
        fs::read_to_string(dir.path().join("before").join("rows-stock.csv"))?,
        {
            crate::materialize::materialize(
                &run,
                &store,
                &dir.path().join("after"),
                None,
                "stock",
            )?;
            fs::read_to_string(dir.path().join("after").join("rows-stock.csv"))?
        }
    );
    assert_eq!(before.chunks, 5);

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

fn write_parquet(path: &Path) -> anyhow::Result<()> {
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;

    let values: Vec<Option<i64>> = (0..20_000i64)
        .map(|i| (i % 7 != 0).then_some(i / 13))
        .collect();
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values))],
    )?;
    let mut writer = ArrowWriter::try_new(fs::File::create(path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn parquet_spec(mode: &str) -> String {
    format!(
        r#"
name = "parquet-{mode}"
chunk_rows = 8192
dtypes = ["int"]

[schemes]
include = ["vortex.int.for", "vortex.int.bitpacking"]

[measure]
reps = 1
compress_reps = 1

[[sources]]
kind = "parquet"
name = "local"
path = "data.parquet"
chunks = "{mode}"
"#
    )
}

#[test]
fn test_copy_and_reference_chunks() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let data = dir.path().join("data");
    fs::create_dir_all(&data)?;
    let file = data.join("data.parquet");
    write_parquet(&file)?;
    let mut opts = options(&["chunk", "features", "candidates"], false);
    opts.data_root = data.clone();

    let copy = build_plan(
        &PlanSpec::from_toml_str(&parquet_spec("copy"))?,
        &data,
        &identity("abc"),
        &registered(),
    )?;
    let reference = build_plan(
        &PlanSpec::from_toml_str(&parquet_spec("reference"))?,
        &data,
        &identity("abc"),
        &registered(),
    )?;
    let copy_store = Store::open(dir.path().join("copy"))?;
    let reference_store = Store::open(dir.path().join("reference"))?;
    Runner::new(&copy, &copy_store)?.run(&opts)?;
    Runner::new(&reference, &reference_store)?.run(&opts)?;
    assert_eq!(copy_store.list("chunks")?.len(), 3);
    assert!(reference_store.list("chunks")?.is_empty());

    // With the source gone, a copied plan still times; a referenced one fails loudly.
    fs::remove_file(&file)?;
    let mut time = options(&["time"], false);
    time.data_root = data;
    Runner::new(&copy, &copy_store)?.run(&time)?;
    assert!(
        Runner::new(&reference, &reference_store)?
            .run(&time)
            .is_err()
    );

    // GC keeps the copied chunks while the plan is live, and lets them go once it is retired.
    assert_eq!(crate::gc::collect(&copy_store, true, true)?.chunks_live, 3);
    copy_store.put(&format!("plans/{}/RETIRED", copy.plan.plan_id), Vec::new())?;
    assert_eq!(crate::gc::collect(&copy_store, true, true)?.chunks_live, 0);
    Ok(())
}
