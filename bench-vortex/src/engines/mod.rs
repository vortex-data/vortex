pub mod ddb;
pub mod df;

use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_physical_plan::ExecutionPlan;

use crate::bench_vortex::Format;
pub use crate::bench_vortex::Format;

struct DataFusionCtx {
    execution_plans: Vec<(usize, Arc<dyn ExecutionPlan>)>,
    metrics: Vec<(
        usize,
        Format,
        Vec<datafusion::physical_plan::metrics::MetricsSet>,
    )>,

    session: SessionContext,
    emit_plan: bool,
}

enum EngineCtx {
    DataFusion(DataFusionCtx),
    DuckDB(ddb::DuckDBCtx),
}

impl EngineCtx {
    fn new_with_datafusion(session_ctx: SessionContext, emit_plan: bool) -> Self {
        EngineCtx::DataFusion(DataFusionCtx {
            execution_plans: Vec::new(),
            metrics: Vec::new(),
            session: session_ctx,
            emit_plan,
        })
    }

    fn new_with_duckdb() -> anyhow::Result<Self> {
        Ok(EngineCtx::DuckDB(ddb::DuckDBCtx::new()?))
    }

    fn to_engine(&self) -> Engine {
        match &self {
            EngineCtx::DuckDB(_) => Engine::DuckDB,
            EngineCtx::DataFusion(_) => Engine::DataFusion,
        }
    }
}
