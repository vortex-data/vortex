//! Mock data generation for development and testing.

use chrono::Duration;
use chrono::Utc;
use duckdb::Result as DuckResult;
use rand::Rng;

use crate::db::DbPool;

/// Populates the database with realistic mock benchmark data.
///
/// Generates 100 commits over 100 days with benchmark results that simulate:
/// - Random variation (+/- 10%) in measurements
/// - Gradual Vortex performance improvements (2% every 10 commits)
/// - A regression at commit 50 that gets fixed at commit 55
/// - ~20% of commits missing benchmark data (simulating CI gaps)
///
/// Base performance values:
/// - Vortex: 50ms (improving over time)
/// - Parquet: 80ms (constant)
/// - Lance: 65ms (constant)
pub fn populate_mock_data(db: &DbPool) -> DuckResult<()> {
    let num_commits = 100;
    let mut rng = rand::rng();

    // Generate commits starting from 100 days ago
    let now = Utc::now();
    let start_time = now - Duration::days(100);

    db.execute(|conn| {
        // Insert commits
        let mut stmt = conn.prepare(
            "INSERT INTO commits (commit_hash, timestamp, message, author) VALUES (?, ?, ?, ?)",
        )?;

        for i in 0..num_commits {
            let timestamp = start_time + Duration::hours((i as i64) * 24);
            let hash = format!("{:040x}", i * 12345 + 67890);
            let message = format!("commit message #{}", i);
            let author = "developer@example.com";
            // Format timestamp as ISO 8601 string for DuckDB
            let timestamp_str = timestamp.format("%Y-%m-%d %H:%M:%S").to_string();

            stmt.execute(duckdb::params![hash, timestamp_str, message, author])?;
        }

        // Insert random_access benchmark data
        // Not all commits have benchmark data (simulate CI running on ~80% of commits)
        let mut bench_stmt = conn.prepare(
            "INSERT INTO random_access (commit_hash, vortex_ns, parquet_ns, lance_ns) VALUES (?, ?, ?, ?)",
        )?;

        // Base performance values (in nanoseconds)
        let mut vortex_base: f64 = 50_000_000.0; // 50ms
        let parquet_base: f64 = 80_000_000.0; // 80ms
        let lance_base: f64 = 65_000_000.0; // 65ms

        for i in 0..num_commits {
            // Skip some commits (20% have no benchmark data)
            if rng.random_ratio(2, 10) {
                continue;
            }

            let hash = format!("{:040x}", i * 12345 + 67890);

            // Add realistic variation (+/- 10%)
            let vortex_ns = (vortex_base * (1.0 + rng.random_range(-0.1..0.1))) as u64;
            let parquet_ns = (parquet_base * (1.0 + rng.random_range(-0.1..0.1))) as u64;
            let lance_ns = (lance_base * (1.0 + rng.random_range(-0.1..0.1))) as u64;

            bench_stmt.execute(duckdb::params![hash, vortex_ns, parquet_ns, lance_ns])?;

            // Simulate performance trends over time
            // Vortex gradually improves
            if i % 10 == 0 {
                vortex_base *= 0.98; // 2% improvement every 10 commits
            }
            // Occasional regression
            if i == 50 {
                vortex_base *= 1.15; // 15% regression at commit 50
            }
            if i == 55 {
                vortex_base *= 0.87; // Fixed at commit 55
            }
        }

        Ok(())
    })
}
