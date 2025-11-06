// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Documentation and handling of special cases in the benchmark system
//!
//! This module documents and isolates special behaviors that exist for
//! compatibility or performance reasons, making them explicit and easier
//! to understand and eventually refactor.

/// Special cases for DuckDB engine
pub mod duckdb {

    /// DuckDB requires reopening the database between iterations
    ///
    /// This is because DuckDB caches data aggressively, and we want
    /// to measure cold query performance for fair comparison.
    pub const REQUIRES_REOPEN: bool = true;

    /// DuckDB needs special handling for certain formats
    ///
    /// For example, OnDiskDuckDB format is DuckDB's internal format
    /// and requires different registration logic.
    pub fn needs_special_format_handling(format: crate::Format) -> bool {
        matches!(format, crate::Format::OnDiskDuckDB)
    }
}

/// Special cases for Arrow format
pub mod arrow {

    /// Arrow format is not a file format, but an in-memory representation
    ///
    /// When using Arrow format, we actually load Parquet files into memory
    /// as Arrow record batches. This is why Arrow appears to use Parquet
    /// files in various places.
    pub const IS_IN_MEMORY_FORMAT: bool = true;

    /// Get the actual file format for Arrow
    ///
    /// Arrow format loads data from Parquet files.
    pub fn source_format() -> crate::Format {
        crate::Format::Parquet
    }

    /// Arrow format should be renamed for clarity
    ///
    /// Consider renaming to "InMemoryParquet" or similar to make
    /// the behavior more obvious.
    pub const SUGGESTED_RENAME: &str = "InMemoryParquet";
}

/// Special cases for Lance format
pub mod lance {

    /// Lance is an optional feature
    ///
    /// Lance support is behind a feature flag and may not be available
    /// in all builds.
    #[cfg(feature = "lance")]
    pub const IS_AVAILABLE: bool = true;

    #[cfg(not(feature = "lance"))]
    pub const IS_AVAILABLE: bool = false;

    /// Lance has its own registration path
    ///
    /// Unlike other formats that use ListingTable, Lance uses
    /// LanceTableProvider directly.
    pub const USES_CUSTOM_REGISTRATION: bool = true;

    /// Lance manages its own partitioning
    ///
    /// This means the distinction between Single and Partitioned
    /// flavors (e.g., in ClickBench) doesn't apply to Lance.
    pub const MANAGES_OWN_PARTITIONING: bool = true;
}

/// Special cases for ClickBench
pub mod clickbench {

    /// ClickBench uses format subdirectories
    ///
    /// Unlike other benchmarks that might use a flat directory structure,
    /// ClickBench organizes data in subdirectories by format:
    /// - clickbench_single/parquet/
    /// - clickbench_single/vortex/
    /// - clickbench_single/lance/
    pub const USES_FORMAT_SUBDIRS: bool = true;

    /// ClickBench has its own conversion functions
    ///
    /// Due to the specific directory structure and requirements,
    /// ClickBench doesn't use the generic conversion infrastructure.
    pub const HAS_CUSTOM_CONVERSION: bool = true;
}

/// Special cases for StatPopGen
pub mod statpopgen {

    /// StatPopGen generates data programmatically
    ///
    /// Unlike other benchmarks that download or use pre-generated data,
    /// StatPopGen creates synthetic genomic data on demand.
    pub const GENERATES_SYNTHETIC_DATA: bool = true;

    /// StatPopGen uses VCF format as source
    ///
    /// The benchmark downloads VCF (Variant Call Format) files and
    /// converts them to Parquet/Vortex formats.
    pub const SOURCE_FORMAT: &str = "VCF";
}

/// Special cases for format conversion
pub mod conversion {

    /// Some formats require idempotent operations
    ///
    /// Conversions should check if the target already exists and skip
    /// if it does, making the operation idempotent.
    pub const SHOULD_BE_IDEMPOTENT: bool = true;

    /// Vortex formats have two variants
    ///
    /// - OnDiskVortex: Standard Vortex format
    /// - VortexCompact: Aggressively compressed variant
    ///
    /// The choice affects compression strategy during conversion.
    pub fn vortex_has_variants() -> bool {
        true
    }
}

/// Document why certain patterns exist
pub mod patterns {
    /// Why we use match statements on BenchmarkDataset
    ///
    /// While we've created traits to reduce coupling, some places still
    /// use match statements because:
    /// 1. Different benchmarks have fundamentally different data organizations
    /// 2. Some benchmarks need custom registration logic
    /// 3. Legacy code that hasn't been fully migrated yet
    pub const MATCH_STATEMENT_RATIONALE: &str =
        "Dataset-specific behavior that can't be easily abstracted";

    /// Why we have both SESSION and engine contexts
    ///
    /// SESSION is a global VortexSession for file I/O operations.
    /// Engine contexts (DataFusion, DuckDB) are for query execution.
    /// This separation exists because:
    /// 1. File I/O needs consistent configuration across all operations
    /// 2. Query engines have their own session/connection management
    /// 3. Some operations (like conversion) don't need a query engine
    pub const DUAL_SESSION_RATIONALE: &str =
        "Separation between file I/O configuration and query execution context";
}