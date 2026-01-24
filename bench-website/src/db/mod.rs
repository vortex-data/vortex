//! Database layer for benchmark data storage and retrieval.
//!
//! This module provides:
//! - [`DbPool`]: Thread-safe DuckDB connection wrapper
//! - Data models ([`ChartData`], [`Series`], [`CommitInfo`], [`DataPoint`])
//! - Query functions for fetching benchmark results

mod connection;
mod models;
mod queries;

pub use connection::DbPool;
pub use models::*;
pub use queries::*;
