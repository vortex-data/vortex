use std::sync::Arc;

use datafusion::datasource::DefaultTableSource;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_expr::{col, LogicalPlanBuilder};
use vortex_array::array::{ListArray, PrimitiveArray, StructArray};
use vortex_array::validity::Validity;
use vortex_array::IntoArrayData;
use vortex_datafusion::expr::list_mean;
use vortex_datafusion::memory::{
    VortexMemTable, VortexMemTableOptions, VortexScanProjectionPushdown,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let elements = PrimitiveArray::from(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
    let offsets = PrimitiveArray::from(vec![0, 2, 4, 5]);
    let list = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::AllValid,
    )
    .unwrap();
    let st = StructArray::try_new(
        ["numbers".into()].into(),
        vec![list.into_array()],
        3,
        Validity::NonNullable,
    )?;

    let table_provider = VortexMemTable::new(st.into_array(), VortexMemTableOptions::default());
    let logical_plan = LogicalPlanBuilder::scan(
        "vortex_tbl",
        Arc::new(DefaultTableSource::new(Arc::new(table_provider))),
        None,
    )?
    .build()?;
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_physical_optimizer_rule(Arc::new(VortexScanProjectionPushdown::new()))
            .build(),
    );
    let df = ctx.execute_logical_plan(logical_plan).await?;

    df.select(vec![list_mean(col("numbers"))])?.show().await?;

    Ok(())
}
