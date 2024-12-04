use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use crate::persistent::execution::VortexExec;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::ExecutionPlan;
use vortex_expr::datafusion::convert_expr_to_vortex;

pub struct VortexExecProjectionPushdown {}

impl VortexExecProjectionPushdown {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for VortexExecProjectionPushdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for VortexExecProjectionPushdown {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("VortexExecProjectionPushdown")
    }
}

impl PhysicalOptimizerRule for VortexExecProjectionPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        // println!("{:#?}", plan);
        match plan.as_any().downcast_ref::<ProjectionExec>() {
            Some(projection_exec) => {
                match projection_exec.children()[0].children()[0].as_any().downcast_ref::<VortexExec>() {
                    Some(vortex_exec) => {
                        let mut projections = Vec::with_capacity(projection_exec.expr().len());
                        let mut names = Vec::with_capacity(projection_exec.expr().len());
                        for (expr, name) in projection_exec.expr() {
                            match convert_expr_to_vortex(expr.clone()) {
                                Ok(vortex_expr) => {
                                    // println!("{:?}: {:?}", name, vortex_expr);
                                    projections.push(vortex_expr);
                                    names.push(name.clone());
                                }
                                Err(e) => {
                                    println!("{:?}", e);
                                    // If any fails, don't push down.
                                    return Ok(plan);
                                }
                            }
                        }
                        // FIXME(marko): Can't change name...
                        // External error: Must be able to project
                        let projection = vortex_expr::Pack::new_expr(projections, vec!["GT".to_string()]);

                        // Push-down the projection.
                        // println!("{:?}", plan);
                        Ok(projection_exec.children()[0].clone().with_new_children(
                            // Arc::new(projection_exec.children()[0].clone()),
                            vec![Arc::new(
                                vortex_exec.clone().with_projection(projection)
                            )],
                        )?,
                        )
                    }
                    None => Ok(plan),
                }
            }
        None => Ok(plan),
    }
}

fn name(&self) -> &str {
    "VortexExecProjectionPushdown"
}

fn schema_check(&self) -> bool {
    false
}
}
