// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unified query engine trait for polymorphic engine execution

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use datafusion::prelude::SessionContext;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::MetricsSet;

use crate::{Engine, Format};

/// Metrics from a single query execution
pub struct QueryMetrics {
    pub duration: Duration,
    pub row_count: usize,
    pub execution_plan: Option<Arc<dyn ExecutionPlan>>,
}

/// Unified interface for query execution engines
#[async_trait]
pub trait QueryEngine: Send + Sync {
    /// Execute a query and return metrics
    ///
    /// This method handles a single query execution and returns timing,
    /// row count, and optionally the execution plan (for DataFusion).
    async fn execute_query(&mut self, query: &str) -> Result<QueryMetrics>;

    /// Reset any caches or state between iterations
    ///
    /// This is called before each iteration in a multi-iteration benchmark.
    /// For engines like DuckDB that cache data, this should reopen the database.
    fn reset_caches(&mut self) -> Result<()> {
        // Default: no-op for engines that don't need cache clearing
        Ok(())
    }

    /// Get the engine type
    fn engine_type(&self) -> Engine;

    /// Check if this engine should emit execution plans
    fn should_emit_plan(&self) -> bool {
        false
    }

    /// Get all execution plans collected so far (DataFusion only)
    fn execution_plans(&self) -> &[(usize, Arc<dyn ExecutionPlan>)] {
        &[]
    }

    /// Get all metrics collected so far (DataFusion only)
    fn metrics(&self) -> &[(usize, Format, Vec<MetricsSet>)] {
        &[]
    }

    /// Add execution plan and metrics for a query (DataFusion only)
    fn add_execution_data(
        &mut self,
        _query_idx: usize,
        _plan: Arc<dyn ExecutionPlan>,
        _format: Format,
        _metrics: Vec<MetricsSet>,
    ) {
        // Default: no-op
    }

    /// Get the DataFusion SessionContext if this is a DataFusion engine
    ///
    /// This is used for table registration which is still engine-specific.
    /// Returns None for non-DataFusion engines.
    fn as_datafusion_session(&self) -> Option<&SessionContext> {
        None
    }
}
