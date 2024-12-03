use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::projection::ProjectionExec;
use vortex_expr::datafusion::convert_expr_to_vortex;
use crate::memory::exec::VortexScanExec;

pub struct VortexScanProjectionPushdown {}

impl VortexScanProjectionPushdown {
    pub fn new() -> Self {
        Self {}
    }
}

impl Debug for VortexScanProjectionPushdown {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("VortexScanProjectionPushdown")
    }
}

impl PhysicalOptimizerRule for VortexScanProjectionPushdown {
    fn optimize(&self, plan: Arc<dyn ExecutionPlan>, _config: &ConfigOptions) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        match plan.as_any().downcast_ref::<ProjectionExec>() {
            Some(projection_exec) => {
                match projection_exec.input().as_any().downcast_ref::<VortexScanExec>() {
                    Some(vortex_scan) => {
                        let mut projection = Vec::with_capacity(projection_exec.expr().len());
                        for (expr, name) in projection_exec.expr() {
                            match convert_expr_to_vortex(expr.clone()) {
                                Ok(vortex_expr) => {
                                    projection.push((vortex_expr, name.clone()));
                                }
                                Err(e) => {
                                    println!("{:?}", e);
                                    // If any fails, don't push down.
                                    return Ok(plan);
                                }
                            }
                        }

                        // Push-down the projection.
                        println!("{:?}", plan);
                        Ok(Arc::new(vortex_scan.with_scan_projection(projection).map_err(|e| {
                            datafusion_common::DataFusionError::Execution(format!("vortex scan projection pushdown failed: {}", e))
                        })?))
                    }
                    None => {
                        return Ok(plan);
                    }
                }
            }
            None => {
                Ok(plan)
            }
        }
    }

    fn name(&self) -> &str {
        "VortexScanProjectionPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
