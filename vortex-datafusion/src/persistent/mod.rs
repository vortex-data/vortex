// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Persistent implementation of a Vortex table provider.
mod cache;
mod format;
pub mod metrics;
mod opener;
mod sink;
mod source;

pub use format::{VortexFormat, VortexFormatFactory, VortexOptions};
pub use source::VortexSource;

#[cfg(test)]
/// Utility function to register Vortex with a [`SessionStateBuilder`]
fn register_vortex_format_factory(
    factory: VortexFormatFactory,
    session_state_builder: &mut datafusion::execution::SessionStateBuilder,
) {
    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            datafusion::common::GetExt::get_ext(&factory).to_uppercase(), // Has to be uppercase
            std::sync::Arc::new(datafusion::datasource::provider::DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(std::sync::Arc::new(factory));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion::arrow::array::{Int8Array, RecordBatch};
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::datasource::listing::{
        ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
    };
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::file_format::format_as_file_type;
    use datafusion_expr::LogicalPlanBuilder;
    use datafusion_physical_plan::display::DisplayableExecutionPlan;
    use insta::assert_snapshot;
    use rstest::rstest;
    use tempfile::{TempDir, tempdir};
    use tokio::fs::OpenOptions;
    use vortex::IntoArray;
    use vortex::arrays::{ChunkedArray, StructArray, VarBinArray};
    use vortex::buffer::buffer;
    use vortex::error::vortex_err;
    use vortex::file::VortexWriteOptions;
    use vortex::validity::Validity;

    use crate::VortexFormatFactory;
    use crate::persistent::{VortexFormat, register_vortex_format_factory};

    #[rstest]
    #[case(Some(1))]
    #[case(None)]
    #[tokio::test]
    async fn query_file(#[case] limit: Option<usize>) -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let strings = ChunkedArray::from_iter([
            VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
            VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        ])
        .into_array();

        let numbers = ChunkedArray::from_iter([
            buffer![1u32, 2, 3, 4].into_array(),
            buffer![5u32, 6, 7, 8].into_array(),
        ])
        .into_array();

        let st = StructArray::try_new(
            ["strings", "numbers"].into(),
            vec![strings, numbers],
            8,
            Validity::NonNullable,
        )?;

        let filepath = temp_dir.path().join("data.vortex");

        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&filepath)
            .await?;

        VortexWriteOptions::default()
            .write(&mut f, st.to_array_stream())
            .await?;

        let ctx = SessionContext::default();
        let format = Arc::new(VortexFormat::default());
        let table_url = ListingTableUrl::parse(
            temp_dir
                .path()
                .to_str()
                .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?,
        )?;
        assert!(table_url.is_collection());

        let config = ListingTableConfig::new(table_url)
            .with_listing_options(
                ListingOptions::new(format).with_session_config_options(ctx.state().config()),
            )
            .infer_schema(&ctx.state())
            .await?;

        let listing_table = Arc::new(ListingTable::try_new(config)?);

        ctx.register_table("vortex_tbl", listing_table as _)?;
        let total_row_count = ctx.table("vortex_tbl").await?.count().await?;
        assert_eq!(total_row_count, 8);

        let row_count = ctx
            .table("vortex_tbl")
            .await?
            .limit(0, limit)?
            .count()
            .await?;

        assert_eq!(row_count, limit.unwrap_or(total_row_count));

        Ok(())
    }

    #[tokio::test]
    async fn test_addition_pushdown() -> anyhow::Result<()> {
        let dir = TempDir::new()?;

        let factory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from_iter_values(0_i8..5))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path().to_str().unwrap().to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new())),
            Default::default(),
            vec![],
        )?
        .build()?;

        session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        // Validate the output by reading back the written files
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex 
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await?;

        let result = session
            .sql("SELECT a, a + 5 as five, a + 6 as six FROM written_data WHERE a + 5 > 7;")
            .await?
            .collect()
            .await?;

        assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---+------+-----+
        | a | five | six |
        +---+------+-----+
        | 3 | 8    | 9   |
        | 4 | 9    | 10  |
        +---+------+-----+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn create_table_ordered_by() -> anyhow::Result<()> {
        let dir = TempDir::new().unwrap();

        let factory: VortexFormatFactory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Vortex
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl_vx \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                WITH ORDER (c1 ASC)
                LOCATION '{}/vx/'",
                dir.path().to_str().unwrap()
            ))
            .await?;

        session
            .sql("INSERT INTO my_tbl_vx VALUES ('air', 5), ('balloon', 42)")
            .await?
            .collect()
            .await?;

        session
            .sql("INSERT INTO my_tbl_vx VALUES ('zebra', 5)")
            .await?
            .collect()
            .await?;

        session
            .sql("INSERT INTO my_tbl_vx VALUES ('texas', 2000), ('alabama', 2000)")
            .await?
            .collect()
            .await?;

        let df = session
            .sql("SELECT * FROM my_tbl_vx ORDER BY c1 ASC limit 3")
            .await?;
        let (state, plan) = df.clone().into_parts();
        let physical_plan = state.create_physical_plan(&plan).await?;

        insta::assert_snapshot!(DisplayableExecutionPlan::new(physical_plan.as_ref())
                .tree_render().to_string(), @r"
        ┌───────────────────────────┐
        │  SortPreservingMergeExec  │
        │    --------------------   │
        │  c1 ASC NULLS LASTlimit:  │
        │             3             │
        └─────────────┬─────────────┘
        ┌─────────────┴─────────────┐
        │       DataSourceExec      │
        │    --------------------   │
        │          files: 3         │
        │       format: vortex      │
        └───────────────────────────┘
        ");

        let r = df.collect().await?;

        insta::assert_snapshot!(pretty_format_batches(&r)?.to_string(), @r"
        +---------+------+
        | c1      | c2   |
        +---------+------+
        | air     | 5    |
        | alabama | 2000 |
        | balloon | 42   |
        +---------+------+
        ");

        Ok(())
    }
}
