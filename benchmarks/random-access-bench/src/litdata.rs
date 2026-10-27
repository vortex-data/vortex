// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [LitData](https://github.com/Lightning-AI/litdata) random access, timed out of process.
//!
//! LitData has no Rust binding, so this backend spawns a Python interpreter running
//! [`LITDATA_SCRIPT`], which converts the dataset's Parquet file into LitData's optimized chunk
//! format and times `StreamingDataset` lookups. The clock lives inside that script rather than
//! around the subprocess, so process spawn, interpreter start and `import litdata` are excluded;
//! the script runs the same warm-up and time-limited loop as the Rust accessors
//! and reports every iteration back as JSON on stdout.
//!
//! The interpreter is a child of this process, so it inherits the CPU affinity and NUMA memory
//! binding that `scripts/bench-taskset.sh` applies to the benchmark binary. LitData therefore
//! runs on exactly the cores and memory node that the Vortex reader gets.

use std::ffi::OsStr;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use itertools::Itertools;
use serde::Deserialize;
use vortex_bench::Format;
use vortex_bench::idempotent;
use vortex_bench::random_access::BenchDataset;
use vortex_bench::random_access::data_path;
use vortex_bench::workspace_root;

/// Repo-relative path of the script that converts and times LitData datasets.
const LITDATA_SCRIPT: &str = "scripts/litdata-random-access.py";

/// Environment variable naming the Python interpreter that has `litdata` installed.
const PYTHON_ENV: &str = "LITDATA_PYTHON";

/// What the LitData script reports back from a timed run.
#[derive(Debug, Deserialize)]
struct TimingReport {
    /// Wall time of every timed iteration, in nanoseconds.
    runs_ns: Vec<u64>,
}

/// Materialize the LitData optimized dataset for `dataset`, converting it from Parquet if missing.
pub async fn dataset_path(dataset: &dyn BenchDataset) -> Result<PathBuf> {
    let parquet = dataset.path(Format::Parquet).await?;
    idempotent(&data_path(dataset.name(), Format::LitData), |output| {
        run_script(&["prepare".as_ref(), parquet.as_os_str(), output.as_os_str()]).map(drop)
    })
}

/// Time LitData lookups of `indices` in the optimized dataset at `path`.
///
/// Mirrors the Rust loop: cached mode warms the reused dataset handle for `warmup` first, reopen
/// mode constructs a new `StreamingDataset` inside every timed iteration, and iterations repeat
/// until `time_limit` has elapsed.
pub fn time_takes(
    path: &Path,
    indices: &[u64],
    time_limit: Duration,
    warmup: Duration,
    reopen: bool,
) -> Result<Vec<Duration>> {
    let indices = indices.iter().join(",");
    let time_limit = time_limit.as_secs_f64().to_string();
    let warmup = warmup.as_secs_f64().to_string();
    let mut args = vec![
        "bench".as_ref(),
        path.as_os_str(),
        "--indices".as_ref(),
        indices.as_ref(),
        "--time-limit-secs".as_ref(),
        time_limit.as_ref(),
        "--warmup-secs".as_ref(),
        warmup.as_ref(),
    ];
    if reopen {
        args.push("--reopen".as_ref());
    }

    let stdout = run_script(&args)?;
    let report: TimingReport = serde_json::from_slice(&stdout).with_context(|| {
        format!(
            "could not parse the report from {LITDATA_SCRIPT}: {}",
            String::from_utf8_lossy(&stdout).trim()
        )
    })?;
    if report.runs_ns.is_empty() {
        bail!("{LITDATA_SCRIPT} reported no timed iterations");
    }
    Ok(report
        .runs_ns
        .into_iter()
        .map(Duration::from_nanos)
        .collect())
}

/// Run [`LITDATA_SCRIPT`] with `args` and return its stdout.
fn run_script(args: &[&OsStr]) -> Result<Vec<u8>> {
    let python = std::env::var_os(PYTHON_ENV).unwrap_or_else(|| "python3".into());
    let output = Command::new(&python)
        .arg(workspace_root().join(LITDATA_SCRIPT))
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn {python:?} to run {LITDATA_SCRIPT}"))?;

    // A missing LitData install surfaces here: the interpreter starts fine and then fails on
    // `import litdata`, so the traceback is on stderr and the install hint belongs with it.
    if !output.status.success() {
        bail!(
            "{LITDATA_SCRIPT} exited with {}; is litdata installed for {python:?}? \
             Set {PYTHON_ENV} to pick another interpreter.\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}
