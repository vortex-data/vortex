// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! v2-compatible group summary rollups.
//!
//! Each `collect_*_summary` runs a small set of focused SQL queries over a
//! single fact table and returns one [`Summary`] variant. The query group
//! summary is gated on a v2 dataset allowlist via
//! `query_group_has_v2_summary`.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use duckdb::ToSql;
use duckdb::params_from_iter;

use super::dto::ChartLink;
use super::dto::QueryRanking;
use super::dto::RandomAccessRanking;
use super::dto::Summary;
use crate::slug::GroupKey;

/// Compute the v2-compatible summary for one group, if its kind has one.
pub(crate) fn collect_group_summary(
    conn: &Connection,
    key: &GroupKey,
    charts: &[ChartLink],
) -> Result<Option<Summary>> {
    match key {
        GroupKey::QueryGroup {
            dataset,
            dataset_variant,
            scale_factor,
            storage,
        } if query_group_has_v2_summary(dataset) => {
            collect_query_summary(conn, dataset, dataset_variant, scale_factor, storage)
        }
        GroupKey::QueryGroup { .. } => Ok(None),
        GroupKey::CompressionTimeGroup => collect_compression_summary(conn),
        GroupKey::CompressionSizeGroup => collect_compression_size_summary(conn),
        GroupKey::RandomAccessGroup => collect_random_access_summary(conn, charts),
        GroupKey::VectorSearchGroup { .. } => Ok(None),
    }
}

fn query_group_has_v2_summary(dataset: &str) -> bool {
    matches!(
        dataset,
        "clickbench" | "statpopgen" | "polarsignals" | "tpch" | "tpcds"
    )
}

fn collect_random_access_summary(
    conn: &Connection,
    charts: &[ChartLink],
) -> Result<Option<Summary>> {
    for chart in charts {
        let mut stmt = conn.prepare(
            r#"
            SELECT r.format, CAST(r.value_ns AS DOUBLE)
              FROM random_access_times r
              JOIN commits c USING (commit_sha)
             WHERE r.dataset = ?
               AND r.value_ns > 0
               AND c.timestamp = (
                    SELECT MAX(c2.timestamp)
                      FROM random_access_times r2
                      JOIN commits c2 USING (commit_sha)
                     WHERE r2.dataset = ?
                       AND r2.value_ns > 0
               )
             ORDER BY r.value_ns, r.format
            "#,
        )?;
        let rows = stmt.query_map([chart.name.as_str(), chart.name.as_str()], |row| {
            Ok(RandomAccessRanking {
                name: row.get(0)?,
                time: row.get(1)?,
                ratio: 0.0,
            })
        })?;
        let mut rankings = rows.collect::<Result<Vec<_>, _>>()?;
        let Some(min_time) = rankings.iter().map(|r| r.time).reduce(f64::min) else {
            continue;
        };
        if min_time <= 0.0 || !min_time.is_finite() {
            continue;
        }
        for r in &mut rankings {
            r.ratio = r.time / min_time;
        }
        rankings.sort_by(|a, b| a.time.total_cmp(&b.time).then_with(|| a.name.cmp(&b.name)));
        return Ok(Some(Summary::RandomAccess {
            title: "Random Access Performance",
            rankings,
            explanation: "Random access time | Ratio to fastest (lower is better)",
        }));
    }
    Ok(None)
}

fn collect_compression_summary(conn: &Connection) -> Result<Option<Summary>> {
    let timestamp = match latest_compression_ratio_timestamp(conn, "encode")? {
        Some(ts) => ts,
        None => match latest_compression_ratio_timestamp(conn, "decode")? {
            Some(ts) => ts,
            None => return Ok(None),
        },
    };

    let compress = compression_speedups_at(conn, "encode", &timestamp)?;
    let decompress = compression_speedups_at(conn, "decode", &timestamp)?;
    if compress.is_empty() && decompress.is_empty() {
        return Ok(None);
    }

    Ok(Some(Summary::Compression {
        title: "Compression Throughput vs Parquet",
        compress_ratio: geo_mean(&compress),
        decompress_ratio: geo_mean(&decompress),
        dataset_count: compress.len(),
        explanation: "Inverse geomean of Vortex/Parquet ratios (higher is better)",
    }))
}

fn latest_compression_ratio_timestamp(conn: &Connection, op: &str) -> Result<Option<String>> {
    conn.query_row(
        r#"
        SELECT CAST(MAX(ts) AS VARCHAR)
          FROM (
            SELECT c.timestamp AS ts
              FROM compression_times v
              JOIN compression_times p
                ON p.commit_sha = v.commit_sha
               AND p.dataset = v.dataset
               AND p.dataset_variant IS NOT DISTINCT FROM v.dataset_variant
               AND p.op = v.op
              JOIN commits c ON c.commit_sha = v.commit_sha
             WHERE v.op = ?
               AND v.format = 'vortex-file-compressed'
               AND p.format = 'parquet'
               AND v.value_ns > 0
               AND p.value_ns > 0
               AND lower(v.dataset) NOT LIKE '%wide table%'
          )
        "#,
        [op],
        |row| row.get(0),
    )
    .context("latest compression ratio timestamp")
}

fn compression_speedups_at(conn: &Connection, op: &str, timestamp: &str) -> Result<Vec<f64>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT CAST(p.value_ns AS DOUBLE) / CAST(v.value_ns AS DOUBLE)
          FROM compression_times v
          JOIN compression_times p
            ON p.commit_sha = v.commit_sha
           AND p.dataset = v.dataset
           AND p.dataset_variant IS NOT DISTINCT FROM v.dataset_variant
           AND p.op = v.op
          JOIN commits c ON c.commit_sha = v.commit_sha
         WHERE v.op = ?
           AND v.format = 'vortex-file-compressed'
           AND p.format = 'parquet'
           AND v.value_ns > 0
           AND p.value_ns > 0
           AND lower(v.dataset) NOT LIKE '%wide table%'
           AND c.timestamp = CAST(? AS TIMESTAMPTZ)
         ORDER BY v.dataset, v.dataset_variant NULLS FIRST
        "#,
    )?;
    let rows = stmt.query_map([op, timestamp], |row| row.get::<_, f64>(0))?;
    rows.collect::<Result<_, _>>()
        .context("compression speedups")
}

fn collect_compression_size_summary(conn: &Connection) -> Result<Option<Summary>> {
    let Some(timestamp) = latest_compression_size_ratio_timestamp(conn)? else {
        return Ok(None);
    };
    let ratios = compression_size_ratios_at(conn, &timestamp)?;
    let Some(mean_ratio) = geo_mean(&ratios) else {
        return Ok(None);
    };
    let min_ratio = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ratio = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Ok(Some(Summary::CompressionSize {
        title: "Compression Size Summary",
        min_ratio,
        mean_ratio,
        max_ratio,
        dataset_count: ratios.len(),
        explanation: "Geomean of Vortex/Parquet size ratios (lower is better)",
    }))
}

fn latest_compression_size_ratio_timestamp(conn: &Connection) -> Result<Option<String>> {
    conn.query_row(
        r#"
        SELECT CAST(MAX(ts) AS VARCHAR)
          FROM (
            SELECT c.timestamp AS ts
              FROM compression_sizes v
              JOIN compression_sizes p
                ON p.commit_sha = v.commit_sha
               AND p.dataset = v.dataset
               AND p.dataset_variant IS NOT DISTINCT FROM v.dataset_variant
              JOIN commits c ON c.commit_sha = v.commit_sha
             WHERE v.format = 'vortex-file-compressed'
               AND p.format = 'parquet'
               AND v.value_bytes > 0
               AND p.value_bytes > 0
               AND lower(v.dataset) NOT LIKE '%wide table%'
          )
        "#,
        [],
        |row| row.get(0),
    )
    .context("latest compression-size ratio timestamp")
}

fn compression_size_ratios_at(conn: &Connection, timestamp: &str) -> Result<Vec<f64>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT CAST(v.value_bytes AS DOUBLE) / CAST(p.value_bytes AS DOUBLE)
          FROM compression_sizes v
          JOIN compression_sizes p
            ON p.commit_sha = v.commit_sha
           AND p.dataset = v.dataset
           AND p.dataset_variant IS NOT DISTINCT FROM v.dataset_variant
          JOIN commits c ON c.commit_sha = v.commit_sha
         WHERE v.format = 'vortex-file-compressed'
           AND p.format = 'parquet'
           AND v.value_bytes > 0
           AND p.value_bytes > 0
           AND lower(v.dataset) NOT LIKE '%wide table%'
           AND c.timestamp = CAST(? AS TIMESTAMPTZ)
         ORDER BY v.dataset, v.dataset_variant NULLS FIRST
        "#,
    )?;
    let rows = stmt.query_map([timestamp], |row| row.get::<_, f64>(0))?;
    rows.collect::<Result<_, _>>()
        .context("compression size ratios")
}

fn collect_query_summary(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    scale_factor: &Option<String>,
    storage: &str,
) -> Result<Option<Summary>> {
    let mut stmt = conn.prepare(
        r#"
        WITH latest AS (
            SELECT q.query_idx,
                   q.engine || ':' || q.format AS series,
                   CAST(q.value_ns AS DOUBLE) AS value_ns,
                   row_number() OVER (
                       PARTITION BY q.query_idx, q.engine, q.format
                       ORDER BY c.timestamp DESC
                   ) AS rn
              FROM query_measurements q
              JOIN commits c USING (commit_sha)
             WHERE q.dataset = ?
               AND q.dataset_variant IS NOT DISTINCT FROM ?
               AND q.scale_factor    IS NOT DISTINCT FROM ?
               AND q.storage = ?
               AND q.value_ns > 0
        )
        SELECT query_idx, series, value_ns
          FROM latest
         WHERE rn = 1
         ORDER BY query_idx, series
        "#,
    )?;
    let binds: Vec<Box<dyn ToSql>> = vec![
        Box::new(dataset.to_string()),
        Box::new(dataset_variant.clone()),
        Box::new(scale_factor.clone()),
        Box::new(storage.to_string()),
    ];
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, i32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;

    let mut queries = BTreeSet::new();
    let mut values_by_series: BTreeMap<String, BTreeMap<i32, f64>> = BTreeMap::new();
    for row in rows {
        let (query_idx, series, value_ns) = row?;
        queries.insert(query_idx);
        values_by_series
            .entry(series)
            .or_default()
            .insert(query_idx, value_ns);
    }
    if values_by_series.is_empty() {
        return Ok(None);
    }

    let mut best_by_query: BTreeMap<i32, f64> = BTreeMap::new();
    for query_idx in &queries {
        let best = values_by_series
            .values()
            .filter_map(|series_values| series_values.get(query_idx).copied())
            .fold(f64::INFINITY, f64::min);
        if best.is_finite() {
            best_by_query.insert(*query_idx, best);
        }
    }

    let mut rankings = Vec::new();
    for (name, query_values) in values_by_series {
        let total_runtime: f64 = query_values.values().sum();
        let max_runtime = query_values
            .values()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_runtime.is_finite() {
            continue;
        }
        let penalty = max_runtime.max(300_000.0) * 2.0;
        let ratios = queries
            .iter()
            .filter_map(|query_idx| {
                let base = best_by_query.get(query_idx).copied()?;
                let value = query_values.get(query_idx).copied().unwrap_or(penalty);
                Some((10.0 + value) / (10.0 + base))
            })
            .collect::<Vec<_>>();
        let Some(score) = geo_mean(&ratios) else {
            continue;
        };
        rankings.push(QueryRanking {
            name,
            score,
            total_runtime,
        });
    }
    rankings.sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then_with(|| a.name.cmp(&b.name))
    });

    if rankings.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Summary::QueryBenchmark {
            title: "Performance Summary",
            rankings,
            explanation: "Geomean of query time ratio to fastest (lower is better)",
        }))
    }
}

fn geo_mean(values: &[f64]) -> Option<f64> {
    let mut sum_ln = 0.0;
    let mut n = 0usize;
    for value in values {
        if *value > 0.0 && value.is_finite() {
            sum_ln += value.ln();
            n += 1;
        }
    }
    (n > 0).then(|| (sum_ln / n as f64).exp())
}
