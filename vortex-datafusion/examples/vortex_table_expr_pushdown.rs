use datafusion::datasource::DefaultTableSource;
use datafusion::execution::SessionStateBuilder;
use datafusion::functions::string::upper;
use datafusion::prelude::SessionContext;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{col, Expr, LogicalPlanBuilder};
use std::sync::Arc;
use vortex_array::array::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::IntoArrayData;
use vortex_datafusion::memory::{VortexMemTable, VortexMemTableOptions, VortexScanProjectionPushdown};

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
        SessionStateBuilder::new().with_physical_optimizer_rule(Arc::new(VortexScanProjectionPushdown::new())).build(),
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
