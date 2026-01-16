//! Storage traits for benchmark result persistence.
//!
//! This module defines traits for storing and querying benchmark results,
//! allowing different storage backends (SQLite, S3, etc.).

use crate::BenchmarkResult;
use crate::CpuClass;

/// A stored benchmark measurement.
#[derive(Debug, Clone)]
pub struct StoredMeasurement {
    /// Unique run identifier.
    pub run_id: String,
    /// Git commit hash.
    pub commit: String,
    /// Timestamp of the run.
    pub timestamp: u64,
    /// CPU class where benchmark ran.
    pub cpu_class: CpuClass,
    /// Algorithm name.
    pub algorithm: String,
    /// Variant name.
    pub variant: String,
    /// Parameter value (e.g., input size).
    pub parameter: usize,
    /// Benchmark result.
    pub result: BenchmarkResult,
}

/// Query filters for retrieving measurements.
#[derive(Debug, Clone, Default)]
pub struct MeasurementQuery {
    /// Filter by algorithm name.
    pub algorithm: Option<String>,
    /// Filter by variant name.
    pub variant: Option<String>,
    /// Filter by CPU class.
    pub cpu_class: Option<CpuClass>,
    /// Filter by commit hash.
    pub commit: Option<String>,
    /// Filter by minimum timestamp.
    pub since: Option<u64>,
    /// Filter by maximum timestamp.
    pub until: Option<u64>,
    /// Limit number of results.
    pub limit: Option<usize>,
}

/// Trait for benchmark result storage backends.
///
/// Note: This trait does not require `Send + Sync` since storage is typically
/// used in single-threaded CLI tools. For concurrent access, implementations
/// should use internal synchronization (e.g., `Mutex<Connection>`).
pub trait BenchmarkStorage {
    /// Error type for storage operations.
    type Error: std::error::Error + 'static;

    /// Stores a batch of measurements.
    fn store_measurements(&self, measurements: &[StoredMeasurement]) -> Result<(), Self::Error>;

    /// Queries measurements based on filters.
    fn query_measurements(
        &self,
        query: &MeasurementQuery,
    ) -> Result<Vec<StoredMeasurement>, Self::Error>;

    /// Gets the latest threshold for an algorithm/variant pair.
    fn get_latest_threshold(
        &self,
        algorithm: &str,
        from_variant: &str,
        to_variant: &str,
        cpu_class: CpuClass,
    ) -> Result<Option<usize>, Self::Error>;

    /// Gets threshold history for trend analysis.
    fn get_threshold_history(
        &self,
        algorithm: &str,
        from_variant: &str,
        to_variant: &str,
        cpu_class: CpuClass,
        limit: usize,
    ) -> Result<Vec<(u64, usize)>, Self::Error>; // (timestamp, threshold)
}

/// Query builder for fluent API.
impl MeasurementQuery {
    /// Creates a new empty query.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filters by algorithm name.
    #[must_use]
    pub fn algorithm(mut self, name: impl Into<String>) -> Self {
        self.algorithm = Some(name.into());
        self
    }

    /// Filters by variant name.
    #[must_use]
    pub fn variant(mut self, name: impl Into<String>) -> Self {
        self.variant = Some(name.into());
        self
    }

    /// Filters by CPU class.
    #[must_use]
    pub fn cpu_class(mut self, class: CpuClass) -> Self {
        self.cpu_class = Some(class);
        self
    }

    /// Filters by commit hash.
    #[must_use]
    pub fn commit(mut self, hash: impl Into<String>) -> Self {
        self.commit = Some(hash.into());
        self
    }

    /// Filters by minimum timestamp.
    #[must_use]
    pub fn since(mut self, timestamp: u64) -> Self {
        self.since = Some(timestamp);
        self
    }

    /// Limits number of results.
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }
}
