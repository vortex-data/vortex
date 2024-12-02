use datafusion::datasource::DefaultTableSource;
use datafusion::execution::SessionStateBuilder;
use datafusion::functions::string::upper;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::prelude::SessionContext;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{col, Expr, LogicalPlanBuilder};
use datafusion_physical_plan::ExecutionPlan;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use vortex_array::array::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::IntoArrayData;
use vortex_datafusion::memory::{VortexMemTable, VortexMemTableOptions, VortexScanExec};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
        .into_array();
    let numbers = ChunkedArray::from_iter([
        PrimitiveArray::from(vec![1u32, 2, 3, 4]).into_array(),
        PrimitiveArray::from(vec![5u32, 6, 7, 8]).into_array(),
    ])
        .into_array();
    let st = StructArray::try_new(
        ["strings".into(), "numbers".into()].into(),
        vec![strings, numbers],
        8,
        Validity::NonNullable,
    )?;

    let table_provider = VortexMemTable::new(st.into_array(), VortexMemTableOptions::default());
    let logical_plan = LogicalPlanBuilder::scan(
        "vortex_tbl",
        Arc::new(DefaultTableSource::new(Arc::new(table_provider))),
        None,
    )?.build()?;
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new().with_physical_optimizer_rule(Arc::new(VortexTableScanPushdown::new())).build(),
    );

    let df = ctx.execute_logical_plan(logical_plan).await?;
    // FIXME(marko): Figure out what's the expression that we're running here!
    df.select(vec![
        Expr::ScalarFunction(ScalarFunction::new_udf(
            upper(),
            vec![col("strings")],
        ))
    ])?.show().await?;

    Ok(())
}

struct VortexTableScanPushdown {}

impl VortexTableScanPushdown {
    pub fn new() -> Self {
        Self {}
    }
}

impl Debug for VortexTableScanPushdown {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("VortexTableScanPushdown")
    }
}

impl PhysicalOptimizerRule for VortexTableScanPushdown {
    fn optimize(&self, plan: Arc<dyn ExecutionPlan>, _config: &ConfigOptions) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        let children = plan.children();
        if children.len() != 1 {
            return Ok(plan);
        }
        if let Some(_vortex_scan) = children[0].as_any().downcast_ref::<VortexScanExec>() {
            println!("{:?}", plan);
            // FIXME(marko): Re-write the expression instead a VortexScanExec.
            Ok(plan)
        } else {
            Ok(plan)
        }
    }

    fn name(&self) -> &str {
        "VortexTableScanPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
