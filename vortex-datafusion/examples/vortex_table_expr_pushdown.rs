use datafusion::datasource::DefaultTableSource;
use datafusion::prelude::SessionContext;
use datafusion_expr::LogicalPlanBuilder;
use std::sync::Arc;
use vortex_array::array::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::IntoArrayData;
use vortex_datafusion::memory::{VortexMemTable, VortexMemTableOptions};

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
    // .aggregate([col("strings")].into_iter(), [var_pop(col("numbers"), )].into_iter())?
    let logical_plan = LogicalPlanBuilder::scan(
        "vortex_tbl",
        Arc::new(DefaultTableSource::new(Arc::new(table_provider))),
        None,
    )?.select([0].into_iter())?.build()?;
    println!("{}", logical_plan);

    // TODO(marko): Rewrite the plan.

    let ctx = SessionContext::new();
    let df = ctx.execute_logical_plan(logical_plan).await?;
    df.show().await?;

    Ok(())
}
