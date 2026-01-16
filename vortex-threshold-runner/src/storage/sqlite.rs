//! SQLite storage backend for benchmark results.
//!
//! Provides persistent storage for benchmark measurements with SQL query support.

// SQLite stores integers as i64, but our values (sizes, counts) are always positive
// and well within usize range. Truncation is acceptable for this internal tooling.
#![allow(clippy::cast_possible_truncation)]

use std::path::Path;

use rusqlite::Connection;
use rusqlite::Result as SqlResult;
use rusqlite::params;
use vortex_threshold_traits::BenchmarkResult;
use vortex_threshold_traits::CpuClass;
use vortex_threshold_traits::storage::BenchmarkStorage;
use vortex_threshold_traits::storage::MeasurementQuery;
use vortex_threshold_traits::storage::StoredMeasurement;

/// SQLite-based storage for benchmark results.
pub struct SqliteStorage {
    conn: Connection,
}

impl SqliteStorage {
    /// Opens or creates a SQLite database at the given path.
    pub fn open(path: impl AsRef<Path>) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        let storage = Self { conn };
        storage.init_schema()?;
        Ok(storage)
    }

    /// Creates an in-memory database (for testing).
    pub fn in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        let storage = Self { conn };
        storage.init_schema()?;
        Ok(storage)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            r"
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                commit TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                cpu_class TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                variant TEXT NOT NULL,
                parameter INTEGER NOT NULL,
                mean_ns REAL NOT NULL,
                stddev_ns REAL NOT NULL,
                min_ns INTEGER NOT NULL,
                max_ns INTEGER NOT NULL,
                iterations INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_measurements_algorithm
                ON measurements(algorithm);
            CREATE INDEX IF NOT EXISTS idx_measurements_cpu_class
                ON measurements(cpu_class);
            CREATE INDEX IF NOT EXISTS idx_measurements_timestamp
                ON measurements(timestamp);
            CREATE INDEX IF NOT EXISTS idx_measurements_commit
                ON measurements(commit);

            CREATE TABLE IF NOT EXISTS thresholds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                commit TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                cpu_class TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                from_variant TEXT NOT NULL,
                to_variant TEXT NOT NULL,
                threshold INTEGER NOT NULL,
                ci_low INTEGER NOT NULL,
                ci_high INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_thresholds_lookup
                ON thresholds(algorithm, from_variant, to_variant, cpu_class);
            ",
        )
    }

    /// Stores a detected threshold crossover.
    #[allow(clippy::too_many_arguments)]
    pub fn store_threshold(
        &self,
        run_id: &str,
        commit: &str,
        timestamp: u64,
        cpu_class: CpuClass,
        algorithm: &str,
        from_variant: &str,
        to_variant: &str,
        threshold: usize,
        ci_low: usize,
        ci_high: usize,
    ) -> SqlResult<()> {
        self.conn.execute(
            r"INSERT INTO thresholds
              (run_id, commit, timestamp, cpu_class, algorithm, from_variant, to_variant, threshold, ci_low, ci_high)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                run_id,
                commit,
                timestamp as i64,
                cpu_class_to_str(cpu_class),
                algorithm,
                from_variant,
                to_variant,
                threshold as i64,
                ci_low as i64,
                ci_high as i64,
            ],
        )?;
        Ok(())
    }

    /// Compares thresholds between two commits.
    pub fn compare_commits(&self, commit_a: &str, commit_b: &str) -> SqlResult<Vec<ThresholdDiff>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT
                a.algorithm, a.from_variant, a.to_variant, a.cpu_class,
                a.threshold as old_threshold, b.threshold as new_threshold
            FROM thresholds a
            JOIN thresholds b ON
                a.algorithm = b.algorithm AND
                a.from_variant = b.from_variant AND
                a.to_variant = b.to_variant AND
                a.cpu_class = b.cpu_class
            WHERE a.commit = ?1 AND b.commit = ?2
            AND a.threshold != b.threshold
            ",
        )?;

        let rows = stmt.query_map(params![commit_a, commit_b], |row| {
            Ok(ThresholdDiff {
                algorithm: row.get(0)?,
                from_variant: row.get(1)?,
                to_variant: row.get(2)?,
                cpu_class: str_to_cpu_class(&row.get::<_, String>(3)?),
                old_threshold: row.get::<_, i64>(4)? as usize,
                new_threshold: row.get::<_, i64>(5)? as usize,
            })
        })?;

        rows.collect()
    }

    /// Gets performance trend for a specific variant over time.
    pub fn get_performance_trend(
        &self,
        algorithm: &str,
        variant: &str,
        parameter: usize,
        cpu_class: CpuClass,
        limit: usize,
    ) -> SqlResult<Vec<(u64, f64)>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT timestamp, mean_ns
            FROM measurements
            WHERE algorithm = ?1 AND variant = ?2 AND parameter = ?3 AND cpu_class = ?4
            ORDER BY timestamp DESC
            LIMIT ?5
            ",
        )?;

        let rows = stmt.query_map(
            params![
                algorithm,
                variant,
                parameter as i64,
                cpu_class_to_str(cpu_class),
                limit as i64,
            ],
            |row| Ok((row.get::<_, i64>(0)? as u64, row.get(1)?)),
        )?;

        rows.collect()
    }
}

/// Difference in threshold between two commits.
#[derive(Debug, Clone)]
pub struct ThresholdDiff {
    pub algorithm: String,
    pub from_variant: String,
    pub to_variant: String,
    pub cpu_class: CpuClass,
    pub old_threshold: usize,
    pub new_threshold: usize,
}

impl ThresholdDiff {
    /// Returns the change as a percentage.
    pub fn change_percent(&self) -> f64 {
        if self.old_threshold == 0 {
            return 0.0;
        }
        ((self.new_threshold as f64 - self.old_threshold as f64) / self.old_threshold as f64)
            * 100.0
    }
}

impl BenchmarkStorage for SqliteStorage {
    type Error = rusqlite::Error;

    fn store_measurements(&self, measurements: &[StoredMeasurement]) -> SqlResult<()> {
        let tx = self.conn.unchecked_transaction()?;

        {
            let mut stmt = self.conn.prepare(
                r"INSERT INTO measurements
                  (run_id, commit, timestamp, cpu_class, algorithm, variant, parameter,
                   mean_ns, stddev_ns, min_ns, max_ns, iterations)
                  VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            )?;

            for m in measurements {
                stmt.execute(params![
                    m.run_id,
                    m.commit,
                    m.timestamp as i64,
                    cpu_class_to_str(m.cpu_class),
                    m.algorithm,
                    m.variant,
                    m.parameter as i64,
                    m.result.mean_ns,
                    m.result.stddev_ns,
                    m.result.min_ns as i64,
                    m.result.max_ns as i64,
                    m.result.iterations as i64,
                ])?;
            }
        }

        tx.commit()
    }

    fn query_measurements(&self, query: &MeasurementQuery) -> SqlResult<Vec<StoredMeasurement>> {
        let mut sql = String::from("SELECT * FROM measurements WHERE 1=1");
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(ref algo) = query.algorithm {
            sql.push_str(" AND algorithm = ?");
            params_vec.push(Box::new(algo.clone()));
        }
        if let Some(ref var) = query.variant {
            sql.push_str(" AND variant = ?");
            params_vec.push(Box::new(var.clone()));
        }
        if let Some(cpu) = query.cpu_class {
            sql.push_str(" AND cpu_class = ?");
            params_vec.push(Box::new(cpu_class_to_str(cpu).to_string()));
        }
        if let Some(ref commit) = query.commit {
            sql.push_str(" AND commit = ?");
            params_vec.push(Box::new(commit.clone()));
        }
        if let Some(since) = query.since {
            sql.push_str(" AND timestamp >= ?");
            params_vec.push(Box::new(since as i64));
        }
        if let Some(until) = query.until {
            sql.push_str(" AND timestamp <= ?");
            params_vec.push(Box::new(until as i64));
        }

        sql.push_str(" ORDER BY timestamp DESC");

        if let Some(limit) = query.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(StoredMeasurement {
                run_id: row.get(1)?,
                commit: row.get(2)?,
                timestamp: row.get::<_, i64>(3)? as u64,
                cpu_class: str_to_cpu_class(&row.get::<_, String>(4)?),
                algorithm: row.get(5)?,
                variant: row.get(6)?,
                parameter: row.get::<_, i64>(7)? as usize,
                result: BenchmarkResult {
                    mean_ns: row.get(8)?,
                    stddev_ns: row.get(9)?,
                    min_ns: row.get::<_, i64>(10)? as u64,
                    max_ns: row.get::<_, i64>(11)? as u64,
                    iterations: row.get::<_, i64>(12)? as usize,
                },
            })
        })?;

        rows.collect()
    }

    fn get_latest_threshold(
        &self,
        algorithm: &str,
        from_variant: &str,
        to_variant: &str,
        cpu_class: CpuClass,
    ) -> SqlResult<Option<usize>> {
        let mut stmt = self.conn.prepare(
            r"SELECT threshold FROM thresholds
              WHERE algorithm = ?1 AND from_variant = ?2 AND to_variant = ?3 AND cpu_class = ?4
              ORDER BY timestamp DESC LIMIT 1",
        )?;

        let result = stmt.query_row(
            params![
                algorithm,
                from_variant,
                to_variant,
                cpu_class_to_str(cpu_class),
            ],
            |row| Ok(row.get::<_, i64>(0)? as usize),
        );

        match result {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn get_threshold_history(
        &self,
        algorithm: &str,
        from_variant: &str,
        to_variant: &str,
        cpu_class: CpuClass,
        limit: usize,
    ) -> SqlResult<Vec<(u64, usize)>> {
        let mut stmt = self.conn.prepare(
            r"SELECT timestamp, threshold FROM thresholds
              WHERE algorithm = ?1 AND from_variant = ?2 AND to_variant = ?3 AND cpu_class = ?4
              ORDER BY timestamp DESC LIMIT ?5",
        )?;

        let rows = stmt.query_map(
            params![
                algorithm,
                from_variant,
                to_variant,
                cpu_class_to_str(cpu_class),
                limit as i64,
            ],
            |row| Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as usize)),
        )?;

        rows.collect()
    }
}

fn cpu_class_to_str(cpu: CpuClass) -> &'static str {
    match cpu {
        CpuClass::IntelSapphire => "IntelSapphire",
        CpuClass::IntelIceLake => "IntelIceLake",
        CpuClass::IntelSkylake => "IntelSkylake",
        CpuClass::AmdGenoa => "AmdGenoa",
        CpuClass::AmdMilan => "AmdMilan",
        CpuClass::AmdRome => "AmdRome",
        CpuClass::Graviton3 => "Graviton3",
        CpuClass::Graviton2 => "Graviton2",
        CpuClass::AppleSilicon => "AppleSilicon",
        CpuClass::Unknown => "Unknown",
    }
}

fn str_to_cpu_class(s: &str) -> CpuClass {
    match s {
        "IntelSapphire" => CpuClass::IntelSapphire,
        "IntelIceLake" => CpuClass::IntelIceLake,
        "IntelSkylake" => CpuClass::IntelSkylake,
        "AmdGenoa" => CpuClass::AmdGenoa,
        "AmdMilan" => CpuClass::AmdMilan,
        "AmdRome" => CpuClass::AmdRome,
        "Graviton3" => CpuClass::Graviton3,
        "Graviton2" => CpuClass::Graviton2,
        "AppleSilicon" => CpuClass::AppleSilicon,
        _ => CpuClass::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_query() {
        let storage = SqliteStorage::in_memory().unwrap();

        let measurements = vec![StoredMeasurement {
            run_id: "run-1".to_string(),
            commit: "abc123".to_string(),
            timestamp: 1000,
            cpu_class: CpuClass::IntelSapphire,
            algorithm: "popcount".to_string(),
            variant: "naive".to_string(),
            parameter: 1024,
            result: BenchmarkResult {
                mean_ns: 100.0,
                stddev_ns: 10.0,
                min_ns: 90,
                max_ns: 110,
                iterations: 100,
            },
        }];

        storage.store_measurements(&measurements).unwrap();

        let query = MeasurementQuery::new().algorithm("popcount");
        let results = storage.query_measurements(&query).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].algorithm, "popcount");
    }

    #[test]
    fn test_threshold_history() {
        let storage = SqliteStorage::in_memory().unwrap();

        // Store multiple thresholds over time
        for i in 0..5 {
            storage
                .store_threshold(
                    &format!("run-{}", i),
                    &format!("commit-{}", i),
                    1000 + i * 100,
                    CpuClass::IntelSapphire,
                    "popcount",
                    "naive",
                    "simd",
                    256 + i as usize * 32,
                    200,
                    300,
                )
                .unwrap();
        }

        let history = storage
            .get_threshold_history("popcount", "naive", "simd", CpuClass::IntelSapphire, 10)
            .unwrap();

        assert_eq!(history.len(), 5);
        // Most recent first
        assert_eq!(history[0].1, 256 + 4 * 32);
    }
}
