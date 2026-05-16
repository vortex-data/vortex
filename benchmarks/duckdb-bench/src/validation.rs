// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-H result validation against reference data.

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use duckdb_bench::DuckClient;
use similar::ChangeTag;
use similar::TextDiff;
use vortex_bench::Format;
use vortex_bench::tpch::benchmark::TpcHBenchmark;
use vortex_bench::tpch::tpch_query;

/// Scale factors that have committed reference CSVs under
/// `vortex-bench/tpch/results/duckdb/<sf>/`.
const SUPPORTED_SCALE_FACTORS: &[&str] = &["0.01", "0.1", "1.0", "10.0"];

/// Verify DuckDB TPC-H results against committed reference data.
///
/// Runs each requested TPC-H query via DuckDB on files of the given `format` and compares the
/// CSV output to the reference under `vortex-bench/tpch/results/duckdb/<scale_factor>/`.
///
/// Returns an error if any query's output differs from the reference, or if the benchmark's
/// scale factor has no reference data committed.
pub fn verify_duckdb_tpch_results(
    benchmark: &TpcHBenchmark,
    format: Format,
    queries: Vec<usize>,
) -> anyhow::Result<()> {
    let scale_factor = benchmark.scale_factor.as_str();
    if !SUPPORTED_SCALE_FACTORS.contains(&scale_factor) {
        anyhow::bail!(
            "No reference results committed for tpch scale factor {scale_factor}. \
             Supported: {SUPPORTED_SCALE_FACTORS:?}"
        );
    }

    let ref_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vortex-bench/tpch/results/duckdb")
        .join(scale_factor);

    let tmp_dir = format!(
        "{}/spiral-tpch-{scale_factor}-{format}",
        env::var("TMPDIR").or_else(|_| env::var("RUNNER_TEMP"))?,
    );
    if PathBuf::from(&tmp_dir).exists() {
        fs::remove_dir_all(&tmp_dir)?;
    }
    fs::create_dir_all(&tmp_dir)?;

    // `DuckClient::new_in_memory` opens a persistent duckdb file under TMPDIR — purge it so
    // table views from a prior validation run (e.g. a different scale factor) don't leak in.
    let stale_db = env::temp_dir()
        .join("vortex-duckdb-bench")
        .join("in-memory");
    if stale_db.exists() {
        fs::remove_dir_all(&stale_db)?;
    }

    let duckdb = DuckClient::new_in_memory()?;
    duckdb.register_tables(benchmark, format)?;

    let mut all_match = true;
    for q in queries {
        let query_name = format!("q{q:02}");
        let sql = tpch_query(q);

        // Most TPC-H queries are a single statement; q15 is multi-statement (CREATE VIEW
        // revenue0; SELECT ...; DROP VIEW revenue0). For multi-statement queries we execute
        // every statement other than the final SELECT/WITH as setup, then materialize the
        // final SELECT into a result table.
        let stmts: Vec<&str> = sql
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let final_idx = stmts
            .iter()
            .rposition(|s| {
                let lower = s.to_ascii_lowercase();
                lower.starts_with("select") || lower.starts_with("with")
            })
            .ok_or_else(|| anyhow::anyhow!("query {q} has no SELECT/WITH statement"))?;

        for (i, stmt) in stmts.iter().enumerate() {
            if i == final_idx {
                continue;
            }
            duckdb.execute_query(stmt)?;
        }

        let create_table = format!(
            "CREATE OR REPLACE TABLE {query_name}_result AS {}",
            stmts[final_idx]
        );
        duckdb.execute_query(&create_table)?;

        let csv_actual = format!("{tmp_dir}/{query_name}.csv");
        duckdb.execute_query(&format!(
            "COPY {query_name}_result TO '{csv_actual}' (HEADER, DELIMITER '|')"
        ))?;

        // Best-effort cleanup of view created by q15 so we don't leak it into the next query.
        drop(duckdb.execute_query("DROP VIEW IF EXISTS revenue0"));

        let csv_expected = ref_dir.join(format!("{query_name}.csv"));
        let expected = fs::read_to_string(&csv_expected)
            .map_err(|e| anyhow::anyhow!("reading reference {}: {e}", csv_expected.display()))?;
        let actual = fs::read_to_string(&csv_actual)?;

        if expected != actual {
            let diff = TextDiff::from_lines(&expected, &actual);
            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                print!("{sign}{change}");
            }
            eprintln!(
                "query output does not match reference for {query_name} \
                 (sf={scale_factor}, format={format})"
            );
            all_match = false;
        }
    }

    if !all_match {
        anyhow::bail!("not all queries matched the reference (sf={scale_factor}, format={format})");
    }

    Ok(())
}
