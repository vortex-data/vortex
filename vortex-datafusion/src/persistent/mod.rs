// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Persistent implementation of a Vortex table provider.
mod access_plan;
mod cache;
mod format;
pub mod metrics;
mod opener;
pub mod reader;
mod sink;
mod source;
mod stream;

pub use access_plan::VortexAccessPlan;
pub use format::VortexFormat;
pub use format::VortexFormatFactory;
pub use format::VortexTableOptions;
pub use source::VortexSource;

#[cfg(test)]
mod tests {

    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion_physical_plan::display::DisplayableExecutionPlan;
    use insta::assert_snapshot;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexWrite;
    use vortex::io::object_store::ObjectStoreWrite;
    use vortex::session::VortexSession;

    use crate::common_tests::TestSessionContext;

    #[rstest]
    #[tokio::test]
    async fn test_query_file(#[values(Some(1), None)] limit: Option<usize>) -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        let session = VortexSession::default();

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

        let mut writer = ObjectStoreWrite::new(ctx.store.clone(), &"test.vortex".into()).await?;

        let summary = session
            .write_options()
            .write(&mut writer, st.into_array().to_array_stream())
            .await?;

        writer.shutdown().await?;

        assert_eq!(summary.row_count(), 8);

        let read_row_count = ctx
            .session
            .sql("SELECT * from '/test.vortex'")
            .await?
            .limit(0, limit)?
            .count()
            .await?;

        assert_eq!(read_row_count, limit.unwrap_or(8));

        Ok(())
    }

    #[tokio::test]
    async fn test_addition_pushdown() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();
        dbg!(&ctx.store);

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '/test/'",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO written_data VALUES (0), (1), (2), (3), (4)")
            .await?
            .collect()
            .await?;

        let result = ctx
            .session
            .sql("SELECT a, a + 5 as five, a + 6 as six FROM written_data WHERE a + 5 > 7")
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
        let ctx = TestSessionContext::default();

        // Vortex
        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl_vx \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                WITH ORDER (c1 ASC)
                LOCATION '/test/'",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl_vx VALUES ('air', 5), ('balloon', 42)")
            .await?
            .collect()
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl_vx VALUES ('zebra', 5)")
            .await?
            .collect()
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl_vx VALUES ('texas', 2000), ('alabama', 2000)")
            .await?
            .collect()
            .await?;

        let df = ctx
            .session
            .sql("SELECT * FROM my_tbl_vx ORDER BY c1 ASC limit 3")
            .await?;

        let physical_plan = ctx
            .session
            .state()
            .create_physical_plan(df.logical_plan())
            .await?;

        insta::assert_snapshot!(DisplayableExecutionPlan::new(physical_plan.as_ref())
                .tree_render().to_string(), @r"
        ┌───────────────────────────┐
        │  SortPreservingMergeExec  │
        │    --------------------   │
        │     c1 ASC NULLS LAST     │
        │                           │
        │          limit: 3         │
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

    /// Doc example: demonstrates creating, writing, reading, and filtering a Vortex table.
    #[tokio::test]
    async fn doc_example() -> anyhow::Result<()> {
        // [setup]
        use std::sync::Arc;

        use datafusion::datasource::provider::DefaultTableFactory;
        use datafusion::execution::SessionStateBuilder;
        use datafusion::prelude::SessionContext;
        use datafusion_common::GetExt;
        use object_store::memory::InMemory;

        use crate::VortexFormatFactory;

        let factory = Arc::new(VortexFormatFactory::new());
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_table_factory(
                factory.get_ext().to_uppercase(),
                Arc::new(DefaultTableFactory::new()),
            )
            .with_file_formats(vec![factory])
            .build();
        let ctx = SessionContext::new_with_state(state).enable_url_table();
        // [setup]

        // Register an in-memory object store for the test.
        let store = Arc::new(InMemory::new());
        ctx.register_object_store(&url::Url::try_from("file://").unwrap(), store);

        // [create]
        ctx.sql(
            "CREATE EXTERNAL TABLE my_table \
                (name VARCHAR NOT NULL, age INT NOT NULL) \
            STORED AS vortex \
            LOCATION '/demo/'",
        )
        .await?;
        // [create]

        // [write]
        ctx.sql(
            "INSERT INTO my_table VALUES \
                ('Alice', 30), ('Bob', 25), ('Charlie', 35), ('Diana', 28)",
        )
        .await?
        .collect()
        .await?;
        // [write]

        // [query]
        let result = ctx
            .sql("SELECT name, age FROM my_table WHERE age > 28 ORDER BY age")
            .await?
            .collect()
            .await?;
        // [query]

        assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---------+-----+
        | name    | age |
        +---------+-----+
        | Alice   | 30  |
        | Charlie | 35  |
        +---------+-----+
        ");

        Ok(())
    }
}
