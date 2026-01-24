// Allow std::collections::HashMap for this standalone prototype
#![allow(clippy::disallowed_types)]

//! Query functions for fetching benchmark data from DuckDB.
//!
//! # Timestamp Handling
//!
//! DuckDB's timestamp type doesn't map directly to chrono types via the Rust
//! bindings. We work around this by casting timestamps to VARCHAR in SQL and
//! parsing them back to `DateTime<Utc>` in Rust.

use std::collections::HashMap;

use chrono::DateTime;
use chrono::NaiveDateTime;
use chrono::Utc;
use duckdb::Result as DuckResult;

use super::DbPool;
use super::models::ChartData;
use super::models::CommitInfo;
use super::models::DataPoint;
use super::models::Series;

/// Fetches random_access benchmark data for the most recent N commits.
///
/// Returns a [`ChartData`] struct ready for rendering, with commits ordered
/// oldest-first (for left-to-right chart display) and series data for
/// Vortex, Parquet, and Lance.
pub fn get_random_access_data(db: &DbPool, limit: usize) -> DuckResult<ChartData> {
    db.query(|conn| {
        // First get commits (ordered newest first, then reversed for display)
        // Cast timestamp to string for portable reading
        let mut commits_stmt = conn.prepare(
            r#"
            SELECT commit_hash, CAST(timestamp AS VARCHAR), message, author
            FROM commits
            ORDER BY timestamp DESC
            LIMIT ?
        "#,
        )?;

        let commits: Vec<CommitInfo> = commits_stmt
            .query_map([limit], |row| {
                Ok(CommitInfo {
                    hash: row.get(0)?,
                    timestamp: {
                        let ts_str: String = row.get(1)?;
                        // Parse the timestamp string
                        NaiveDateTime::parse_from_str(&ts_str, "%Y-%m-%d %H:%M:%S")
                            .map(|ts| DateTime::from_naive_utc_and_offset(ts, Utc))
                            .unwrap_or_else(|_| Utc::now())
                    },
                    message: row.get(2)?,
                    author: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Reverse to get oldest-first for charting
        let commits: Vec<_> = commits.into_iter().rev().collect();

        // Create commit hash -> index map
        let hash_to_idx: HashMap<String, usize> = commits
            .iter()
            .enumerate()
            .map(|(i, c)| (c.hash.clone(), i))
            .collect();

        // Fetch benchmark data
        let mut bench_stmt = conn.prepare(
            r#"
            SELECT r.commit_hash, r.vortex_ns, r.parquet_ns, r.lance_ns
            FROM random_access r
            JOIN commits c ON r.commit_hash = c.commit_hash
            ORDER BY c.timestamp DESC
            LIMIT ?
        "#,
        )?;

        let mut vortex_points = Vec::new();
        let mut parquet_points = Vec::new();
        let mut lance_points = Vec::new();

        let rows = bench_stmt.query_map([limit], |row| {
            let hash: String = row.get(0)?;
            let vortex: Option<u64> = row.get(1)?;
            let parquet: Option<u64> = row.get(2)?;
            let lance: Option<u64> = row.get(3)?;
            Ok((hash, vortex, parquet, lance))
        })?;

        for row in rows {
            let (hash, vortex, parquet, lance) = row?;
            if let Some(&idx) = hash_to_idx.get(&hash) {
                if let Some(v) = vortex {
                    vortex_points.push(DataPoint {
                        commit_idx: idx,
                        value_ns: v,
                    });
                }
                if let Some(v) = parquet {
                    parquet_points.push(DataPoint {
                        commit_idx: idx,
                        value_ns: v,
                    });
                }
                if let Some(v) = lance {
                    lance_points.push(DataPoint {
                        commit_idx: idx,
                        value_ns: v,
                    });
                }
            }
        }

        Ok(ChartData {
            title: "Random Access".to_string(),
            commits,
            series: vec![
                Series {
                    name: "vortex".to_string(),
                    display_name: "Vortex".to_string(),
                    color: "#19a508".to_string(),
                    points: vortex_points,
                },
                Series {
                    name: "parquet".to_string(),
                    display_name: "Parquet".to_string(),
                    color: "#ef7f1d".to_string(),
                    points: parquet_points,
                },
                Series {
                    name: "lance".to_string(),
                    display_name: "Lance".to_string(),
                    color: "#2D936C".to_string(),
                    points: lance_points,
                },
            ],
            unit: "ms".to_string(),
        })
    })
}
