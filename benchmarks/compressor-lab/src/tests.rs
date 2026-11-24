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
use crate::store::timings_dir;
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
        &crate::fetch::RawData::new(Some(std::env::temp_dir().join("vx-lab-test-cache")), None),
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
        data: crate::load::DataLocation {
            data_root: ".".into(),
            raw: crate::fetch::RawData::new(
                Some(std::env::temp_dir().join("vx-lab-test-cache")),
                None,
            ),
        },
        decode_reps: 2,
        compress_reps: 1,
        keep_going: false,
        skip: Default::default(),
        timing: crate::measure::TimingConfig {
            max_samples: 3,
            min_samples: 3,
            ..Default::default()
        },
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

    // Two timing runs append timings from two runs per candidates task, every plan replaying to
    // the encoding the candidates stage recorded.
    runner.run(&options(&["time"], false))?;
    runner.run(&options(&["time"], false))?;
    let machine = crate::machine::MachineInfo::current()?.machine_id;
    let timings: Vec<TimingRow> = store.read_rows(&timings_dir(&machine))?;
    assert!(timings.iter().all(|t| t.replay_matched));
    for task in run.tasks.iter().filter(|t| t.kind == TaskKind::Candidates) {
        let key = task.key.to_string();
        let runs: BTreeSet<&str> = timings
            .iter()
            .filter(|t| t.task_key == key && t.op == "decode")
            .map(|t| t.run_id.as_str())
            .collect();
        assert_eq!(runs.len(), 2);
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

    // Compaction leaves one segment per directory and changes no result.
    let report = crate::gc::compact(&store, false)?;
    assert!(report.segments_merged > report.segments_written);
    assert_eq!(crate::gc::compact(&store, true)?.segments_merged, 0);
    let again = runner.run(&options(&["chunk", "features", "candidates"], false))?;
    assert_eq!(again.values().map(|r| r.ran).sum::<u64>(), 0);
    crate::materialize::materialize(&run, &store, &dir.path().join("compacted"), None, "stock")?;
    assert_eq!(
        rows,
        fs::read_to_string(dir.path().join("compacted").join("rows-stock.csv"))?
    );
    Ok(())
}

#[test]
fn test_queue_workers_share_a_plan() -> anyhow::Result<()> {
    use crate::queue::Queue;
    use crate::queue::QueueConfig;
    use crate::queue::WorkOptions;

    let dir = tempfile::tempdir()?;
    let store = Store::open(dir.path().join("store"))?;
    let run = plan(SPEC, "abc")?;
    let config = QueueConfig {
        plan_id: run.plan.plan_id.clone(),
        shards: 4,
        lease_secs: 60,
        max_attempts: 2,
    };
    let queue = Queue::push(&store, &run, config.clone())?;
    // Pushing again is a no-op; pushing with another shard count is refused.
    Queue::push(&store, &run, config.clone())?;
    assert!(
        Queue::push(
            &store,
            &run,
            QueueConfig {
                shards: 8,
                ..config
            }
        )
        .is_err()
    );

    // A held lease blocks other workers; a released or expired one does not.
    assert_eq!(queue.try_claim(3, "a")?, Some(0));
    assert_eq!(queue.try_claim(3, "b")?, None);
    queue.release(3, 0, "a", 0)?;
    assert_eq!(queue.try_claim(3, "b")?, Some(1));
    // `a` has lost shard 3 to `b`, so it can no longer renew generation 0.
    assert!(!queue.renew(3, 0, "a", 0)?);
    queue.release(3, 1, "b", 0)?;

    let pulled = queue.plan()?;
    assert_eq!(pulled.plan.plan_id, run.plan.plan_id);
    let runner = Runner::new(&pulled, &store)?;
    let options = |worker: &str, max_shards| WorkOptions {
        worker: worker.to_string(),
        data: options(&[], false).data,
        wait: false,
        poll_secs: 1,
        max_shards,
    };
    let first = crate::queue::work(&queue, &pulled, &runner, &options("w1", Some(1)))?;
    assert_eq!(first.shards_done, 1);
    assert!(!first.plan_done);
    let status = crate::queue::status(&queue)?;
    assert_eq!(status.shards.done, 1);
    assert_eq!(status.state, "stalled");

    let rest = crate::queue::work(&queue, &pulled, &runner, &options("w2", None))?;
    assert_eq!(rest.shards_done, 3);
    assert!(rest.plan_done);

    let status = crate::queue::status(&queue)?;
    assert_eq!(status.state, "done");
    assert_eq!(status.shards.done, 4);
    for counts in status.stages.values() {
        assert_eq!(counts.done, counts.total);
        assert_eq!(counts.failed, 0);
    }
    assert!(status.live_workers.is_empty());
    let mut done = BTreeSet::new();
    for kind in [TaskKind::Chunk, TaskKind::Features, TaskKind::Candidates] {
        done.extend(store.ledger(kind)?.into_keys());
    }
    for task in &run.tasks {
        assert!(done.contains(&task.key));
    }
    Ok(())
}

mod sources {
    use std::fs;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use parquet::arrow::ArrowWriter;

    use super::identity;
    use super::registered;
    use crate::fetch::RawData;
    use crate::load::DataLocation;
    use crate::plan::PlannedRun;
    use crate::plan::TaskKind;
    use crate::plan::build_plan;
    use crate::runner::RunOptions;
    use crate::runner::Runner;
    use crate::source::ChunkInput;
    use crate::spec::PlanSpec;
    use crate::spec::Storage;
    use crate::store::Store;

    /// Writes a Parquet file of 10,000 rows with an int and a string column.
    fn write_parquet(path: &Path) -> anyhow::Result<()> {
        let ints = Int64Array::from_iter_values((0..10_000).map(|i| i / 7));
        let strings = StringArray::from_iter_values((0..10_000).map(|i| format!("s{}", i % 13)));
        let batch = RecordBatch::try_from_iter([
            ("id", Arc::new(ints) as _),
            ("label", Arc::new(strings) as _),
        ])?;
        let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn spec(sources: &str) -> String {
        format!(
            "name = \"sources\"\nchunk_rows = 4096\ndtypes = [\"int\"]\n\
             [schemes]\ninclude = [\"vortex.int.for\"]\n{sources}"
        )
    }

    fn plan_with(spec_text: &str, dir: &Path, raw: &RawData) -> anyhow::Result<PlannedRun> {
        build_plan(
            &PlanSpec::from_toml_str(spec_text)?,
            dir,
            &identity("abc"),
            &registered(),
            raw,
        )
    }

    fn chunk_keys(run: &PlannedRun) -> Vec<String> {
        run.tasks
            .iter()
            .filter(|t| t.kind == TaskKind::Chunk)
            .map(|t| t.key.to_string())
            .collect()
    }

    fn run_chunks(run: &PlannedRun, store: &Store, raw: RawData) -> anyhow::Result<u64> {
        let report = Runner::new(run, store)?.run(&RunOptions {
            stages: vec!["chunk".to_string()],
            shard: None,
            verify: false,
            data: DataLocation {
                data_root: ".".into(),
                raw,
            },
            decode_reps: 1,
            compress_reps: 1,
            keep_going: false,
            skip: Default::default(),
            timing: Default::default(),
        })?;
        Ok(report.values().map(|r| r.ran).sum())
    }

    #[test]
    fn test_remote_reference_matches_local_keys_and_refetches() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let upstream = dir.path().join("upstream");
        fs::create_dir_all(&upstream)?;
        write_parquet(&upstream.join("a.parquet"))?;
        let cache = dir.path().join("cache");
        let raw = RawData::new(Some(cache.clone()), None);

        let local = plan_with(
            &spec("[[sources]]\nkind = \"parquet\"\nname = \"t\"\npath = \"upstream/*.parquet\"\n"),
            dir.path(),
            &raw,
        )?;
        let remote_spec = spec(&format!(
            "[[sources]]\nkind = \"remote\"\nname = \"t\"\nstorage = \"reference\"\n\
             urls = [\"file://{}/*.parquet\"]\n",
            upstream.display()
        ));
        let remote = plan_with(&remote_spec, dir.path(), &raw)?;
        // Same bytes, same source name: the same chunks, wherever the file lives.
        assert_eq!(chunk_keys(&local), chunk_keys(&remote));
        assert_eq!(chunk_keys(&remote).len(), 3);
        // The string column is skipped because the plan only asks for ints.
        assert_eq!(remote.plan.skipped.len(), 1);

        // Deleting the cache only costs a re-fetch, verified against the recorded hash.
        fs::remove_dir_all(&cache)?;
        let store = Store::open(dir.path().join("store"))?;
        assert_eq!(run_chunks(&remote, &store, raw)?, 3);
        assert!(cache.exists());
        assert!(!dir.path().join("store/raw").exists());
        Ok(())
    }

    #[test]
    fn test_remote_copy_keeps_files_in_store() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let upstream = dir.path().join("a.parquet");
        write_parquet(&upstream)?;
        let store_root = dir.path().join("store");
        let raw = RawData::new(Some(dir.path().join("cache")), Some(store_root.clone()));
        let run = plan_with(
            &spec(&format!(
                "[[sources]]\nkind = \"remote\"\nname = \"t\"\nstorage = \"copy\"\n\
                 urls = [\"file://{}\"]\n",
                upstream.display()
            )),
            dir.path(),
            &raw,
        )?;
        let ChunkInput::Remote {
            file_fingerprint,
            storage,
            ..
        } = run
            .tasks
            .iter()
            .find_map(|t| match &t.input {
                crate::plan::TaskInput::Chunk(input) => Some(input.clone()),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("no chunk task"))?
        else {
            anyhow::bail!("expected a remote chunk");
        };
        assert_eq!(storage, Storage::Copy);
        let kept = store_root
            .join("raw")
            .join(&file_fingerprint[..2])
            .join(&file_fingerprint);
        assert!(kept.exists());

        // Upstream can disappear: the store has everything the run needs.
        fs::remove_file(&upstream)?;
        fs::remove_dir_all(dir.path().join("cache")).ok();
        let store = Store::open(&store_root)?;
        assert_eq!(run_chunks(&run, &store, raw)?, 3);
        Ok(())
    }

    #[test]
    fn test_remote_copy_keeps_files_in_an_object_store() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let upstream = dir.path().join("a.parquet");
        write_parquet(&upstream)?;
        let bucket = dir.path().join("bucket");
        let store = Store::remote_in(&bucket)?;
        let cache = dir.path().join("cache");
        let raw = RawData::for_store(Some(cache.clone()), &store);
        let run = plan_with(
            &spec(&format!(
                "[[sources]]\nkind = \"remote\"\nname = \"t\"\nstorage = \"copy\"\n\
                 urls = [\"file://{}\"]\n",
                upstream.display()
            )),
            dir.path(),
            &raw,
        )?;
        let fingerprint = run
            .tasks
            .iter()
            .find_map(|t| match &t.input {
                crate::plan::TaskInput::Chunk(ChunkInput::Remote {
                    file_fingerprint, ..
                }) => Some(file_fingerprint.clone()),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("no remote chunk task"))?;
        // Planning uploads the file once, under the same content-addressed key a local store uses.
        let kept = bucket
            .join("raw")
            .join(&fingerprint[..2])
            .join(&fingerprint);
        assert_eq!(fs::read(&kept)?, fs::read(&upstream)?);

        // Upstream and the cache can both disappear: a run fills the cache from the store.
        fs::remove_file(&upstream)?;
        fs::remove_dir_all(&cache)?;
        let fresh = RawData::for_store(Some(cache.clone()), &store);
        assert_eq!(run_chunks(&run, &store, fresh)?, 3);
        assert!(cache.join(&fingerprint[..2]).join(&fingerprint).exists());
        Ok(())
    }

    #[test]
    fn test_remote_pin_mismatch_is_an_error() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let upstream = dir.path().join("a.parquet");
        write_parquet(&upstream)?;
        let url = format!("file://{}", upstream.display());
        let text = spec(&format!(
            "[[sources]]\nkind = \"remote\"\nname = \"t\"\nstorage = \"reference\"\n\
             urls = [\"{url}\"]\n[sources.sha256]\n\"{url}\" = \"{}\"\n",
            "0".repeat(64)
        ));
        let raw = RawData::new(Some(dir.path().join("cache")), None);
        let error = plan_with(&text, dir.path(), &raw).unwrap_err();
        assert!(
            format!("{error:#}").contains("but the spec pins"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn test_remote_spec_validation() {
        let bad = [
            (
                "storage = \"generate\"\nurls = [\"https://x/a.parquet\"]",
                "reference",
            ),
            ("storage = \"reference\"\nurls = []", "at least one url"),
            (
                "storage = \"reference\"\nurls = [\"https://x/*.parquet\"]",
                "globs",
            ),
            (
                "storage = \"reference\"\nurls = [\"ftp://x/a.parquet\"]",
                "unsupported scheme",
            ),
        ];
        for (body, message) in bad {
            let text = spec(&format!(
                "[[sources]]\nkind = \"remote\"\nname = \"t\"\n{body}\n"
            ));
            let error = PlanSpec::from_toml_str(&text).unwrap_err();
            assert!(format!("{error:#}").contains(message), "{error:#}");
        }
    }

    #[test]
    fn test_max_chunks_is_a_stable_growing_sample() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let raw = RawData::new(Some(dir.path().join("cache")), None);
        let tpch = |max: u64| {
            spec(&format!(
                "[[sources]]\nkind = \"tpch\"\nname = \"tpch\"\nscale_factor = 0.01\n\
                 tables = [\"orders\"]\ngroup = \"tpc\"\nmax_chunks = {max}\n"
            ))
        };
        let all = chunk_keys(&plan_with(&tpch(1000), dir.path(), &raw)?);
        let five = chunk_keys(&plan_with(&tpch(5), dir.path(), &raw)?);
        let ten = chunk_keys(&plan_with(&tpch(10), dir.path(), &raw)?);
        assert!(all.len() > 10);
        assert_eq!(five.len(), 5);
        assert_eq!(ten.len(), 10);
        assert!(five.iter().all(|k| ten.contains(k)));
        assert_eq!(five, chunk_keys(&plan_with(&tpch(5), dir.path(), &raw)?));

        let parsed = PlanSpec::from_toml_str(&tpch(5))?;
        assert_eq!(parsed.sources[0].group(), "tpc");
        assert_eq!(parsed.sources[0].storage(), Storage::Generate);
        Ok(())
    }
}
