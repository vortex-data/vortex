// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Golden-vector cross-language pin for the `measurement_id_*` hash.
//!
//! The Python ingest writer (`scripts/_measurement_id.py`, used from PR-2.1)
//! must reproduce these exact `i64` ids so re-ingesting an existing
//! `(commit, dim-tuple)` upserts via `ON CONFLICT (measurement_id)` instead of
//! inserting a duplicate. This test is the SOURCE OF TRUTH: it computes ids with
//! the real `vortex_bench_server::db::measurement_id_*` functions, writes them to
//! the committed `scripts/measurement_id_golden.json` (only when
//! `REGEN_GOLDEN_VECTORS` is set in the environment), and ALWAYS asserts the
//! committed file matches the freshly-computed ids. `scripts/test_measurement_id.py`
//! reads the same file and asserts the Python port reproduces it, so
//! Rust == golden == Python transitively.
//!
//! Regenerate after any intentional hash change with:
//!
//! ```sh
//! REGEN_GOLDEN_VECTORS=1 cargo test -p vortex-bench-server --test measurement_id_golden
//! ```

use std::path::PathBuf;

use anyhow::Result;
use serde_json::Value;
use serde_json::json;
use vortex_bench_server::db::measurement_id_compression_size;
use vortex_bench_server::db::measurement_id_compression_time;
use vortex_bench_server::db::measurement_id_query;
use vortex_bench_server::db::measurement_id_random_access;
use vortex_bench_server::db::measurement_id_vector_search;
use vortex_bench_server::records::CompressionSize;
use vortex_bench_server::records::CompressionTime;
use vortex_bench_server::records::QueryMeasurement;
use vortex_bench_server::records::RandomAccessTime;
use vortex_bench_server::records::VectorSearchRun;

/// Path to the committed golden file (`<repo-root>/scripts/...`). The server
/// crate lives at `benchmarks-website/server`, so the repo root is two levels
/// up from `CARGO_MANIFEST_DIR`.
fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/measurement_id_golden.json")
}

/// Build a `query_measurements` fixture carrying only the hash-relevant dim
/// fields; the value/memory columns do not enter the hash, so they are zeroed.
/// The argument count mirrors the eight `query_measurements` dim columns
/// one-to-one; collapsing them into a struct would only obscure the test.
#[allow(clippy::too_many_arguments)]
fn query_vector(
    commit_sha: &str,
    dataset: &str,
    dataset_variant: Option<&str>,
    scale_factor: Option<&str>,
    query_idx: i32,
    storage: &str,
    engine: &str,
    format: &str,
) -> Value {
    let record = QueryMeasurement {
        commit_sha: commit_sha.to_string(),
        dataset: dataset.to_string(),
        dataset_variant: dataset_variant.map(str::to_string),
        scale_factor: scale_factor.map(str::to_string),
        query_idx,
        storage: storage.to_string(),
        engine: engine.to_string(),
        format: format.to_string(),
        value_ns: 0,
        all_runtimes_ns: vec![],
        peak_physical: None,
        peak_virtual: None,
        physical_delta: None,
        virtual_delta: None,
        env_triple: None,
    };
    json!({
        "table": "query_measurements",
        "fields": {
            "commit_sha": commit_sha,
            "dataset": dataset,
            "dataset_variant": dataset_variant,
            "scale_factor": scale_factor,
            "query_idx": query_idx,
            "storage": storage,
            "engine": engine,
            "format": format,
        },
        "measurement_id": measurement_id_query(&record),
    })
}

/// Build the full ordered vector set: a deterministic bulk sweep plus
/// hand-picked edge cases (multibyte UTF-8, null/non-null optionals, negative
/// and boundary `i32`, empty strings, and a spread of `f64` thresholds).
fn build_vectors() -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();

    // Bulk sweep over query_measurements covering both optionals null and
    // non-null, both storage values, and a query_idx range spanning negatives.
    for i in 0..40i32 {
        let variant = if i % 3 == 0 {
            None
        } else {
            Some(format!("variant-{i}"))
        };
        let scale_factor = if i % 2 == 0 {
            None
        } else {
            Some(format!("sf-{i}"))
        };
        let commit_sha = format!("{:040x}", (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        out.push(query_vector(
            &commit_sha,
            &format!("dataset-{i}"),
            variant.as_deref(),
            scale_factor.as_deref(),
            i - 20,
            if i % 2 == 0 { "nvme" } else { "s3" },
            "vortex",
            "vortex-file-compressed",
        ));
    }

    // Edge cases for query_measurements.
    let edge_sha = "0123456789abcdef0123456789abcdef01234567";
    out.push(query_vector(
        edge_sha,
        "tpch",
        None,
        Some("1"),
        7,
        "nvme",
        "vortex",
        "parquet",
    ));
    // Multibyte UTF-8 in several fields: forces BYTE-length prefixing, not char
    // count. A naive port using len(str) would diverge here.
    out.push(query_vector(
        edge_sha,
        "데이터셋-日本語",
        Some("café-Ω"),
        Some("σ"),
        0,
        "s3",
        "ヴ",
        "format-✓",
    ));
    // i32 boundaries (two's-complement little-endian encoding).
    out.push(query_vector(
        edge_sha,
        "boundary",
        None,
        None,
        i32::MIN,
        "nvme",
        "e",
        "f",
    ));
    out.push(query_vector(
        edge_sha,
        "boundary",
        None,
        None,
        i32::MAX,
        "nvme",
        "e",
        "f",
    ));
    // Empty strings (zero-length length-prefix) and empty optionals-as-Some("").
    out.push(query_vector(
        edge_sha,
        "",
        Some(""),
        Some(""),
        0,
        "",
        "",
        "",
    ));

    // compression_times: op encode/decode, variant null/non-null, unicode.
    for (dataset, variant, format, op) in [
        ("chimp", None, "vortex", "encode"),
        ("chimp", None, "vortex", "decode"),
        ("taxi", Some("v2"), "parquet", "encode"),
        ("압축-テスト", Some("変種"), "vortex-✓", "decode"),
    ] {
        let record = CompressionTime {
            commit_sha: edge_sha.to_string(),
            dataset: dataset.to_string(),
            dataset_variant: variant.map(str::to_string),
            format: format.to_string(),
            op: op.to_string(),
            value_ns: 0,
            all_runtimes_ns: vec![],
            env_triple: None,
        };
        out.push(json!({
            "table": "compression_times",
            "fields": {
                "commit_sha": edge_sha,
                "dataset": dataset,
                "dataset_variant": variant,
                "format": format,
                "op": op,
            },
            "measurement_id": measurement_id_compression_time(&record),
        }));
    }

    // compression_sizes: variant null/non-null.
    for (dataset, variant, format) in [
        ("chimp", None, "vortex"),
        ("taxi", Some("v2"), "parquet"),
        ("크기-サイズ", Some("変種-Ω"), "vortex-✓"),
    ] {
        let record = CompressionSize {
            commit_sha: edge_sha.to_string(),
            dataset: dataset.to_string(),
            dataset_variant: variant.map(str::to_string),
            format: format.to_string(),
            value_bytes: 0,
        };
        out.push(json!({
            "table": "compression_sizes",
            "fields": {
                "commit_sha": edge_sha,
                "dataset": dataset,
                "dataset_variant": variant,
                "format": format,
            },
            "measurement_id": measurement_id_compression_size(&record),
        }));
    }

    // random_access_times: no dataset_variant.
    for (dataset, format) in [("chimp", "vortex"), ("taxi", "parquet"), ("無作為", "✓")] {
        let record = RandomAccessTime {
            commit_sha: edge_sha.to_string(),
            dataset: dataset.to_string(),
            format: format.to_string(),
            value_ns: 0,
            all_runtimes_ns: vec![],
            env_triple: None,
        };
        out.push(json!({
            "table": "random_access_times",
            "fields": {
                "commit_sha": edge_sha,
                "dataset": dataset,
                "format": format,
            },
            "measurement_id": measurement_id_random_access(&record),
        }));
    }

    // vector_search_runs: a spread of f64 thresholds including 0.0, negative,
    // fractional, integral, and large/small magnitudes. `PI` stands in for an
    // arbitrary high-precision irrational (clippy forbids the bare literal).
    for threshold in [
        0.0f64,
        0.5,
        0.83,
        1.0,
        -1.5,
        1e-9,
        1e12,
        std::f64::consts::PI,
    ] {
        let record = VectorSearchRun {
            commit_sha: edge_sha.to_string(),
            dataset: "cohere-large-10m".to_string(),
            layout: "trained".to_string(),
            flavor: "vortex".to_string(),
            threshold,
            value_ns: 0,
            all_runtimes_ns: vec![],
            matches: 0,
            rows_scanned: 0,
            bytes_scanned: 0,
            iterations: 0,
            env_triple: None,
        };
        out.push(json!({
            "table": "vector_search_runs",
            "fields": {
                "commit_sha": edge_sha,
                "dataset": "cohere-large-10m",
                "layout": "trained",
                "flavor": "vortex",
                "threshold": threshold,
            },
            "measurement_id": measurement_id_vector_search(&record),
        }));
    }

    out
}

#[test]
fn measurement_id_golden_vectors() -> Result<()> {
    let vectors = build_vectors();
    let document = json!({
        "note": "Golden vectors for the measurement_id_* hash, generated from \
                 benchmarks-website/server/src/db.rs (the source of truth). \
                 Regenerate with REGEN_GOLDEN_VECTORS=1 cargo test -p \
                 vortex-bench-server --test measurement_id_golden. \
                 scripts/test_measurement_id.py asserts the Python port \
                 reproduces every measurement_id here.",
        "seed": 0,
        "vectors": vectors,
    });

    let path = golden_path();

    if std::env::var_os("REGEN_GOLDEN_VECTORS").is_some() {
        let pretty = serde_json::to_string_pretty(&document)?;
        std::fs::write(&path, format!("{pretty}\n"))?;
    }

    let on_disk = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "could not read golden file {}: {e}. Generate it with \
             REGEN_GOLDEN_VECTORS=1 cargo test -p vortex-bench-server --test \
             measurement_id_golden",
            path.display()
        )
    })?;
    let parsed: Value = serde_json::from_str(&on_disk)?;

    assert_eq!(
        parsed.get("vectors"),
        document.get("vectors"),
        "committed golden vectors are stale; regenerate with \
         REGEN_GOLDEN_VECTORS=1 and commit the result"
    );
    Ok(())
}
