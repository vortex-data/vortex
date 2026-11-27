// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::anyhow;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::Decimal128Array;
use arrow_array::Int32Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StructArray;
use arrow_array::TimestampMicrosecondArray;
use arrow_array::cast::AsArray;
use arrow_array::record_batch as arrow_record_batch;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use arrow_schema::extension::Json as ArrowJson;
use arrow_schema::extension::Uuid as ArrowUuid;
use datafusion::assert_batches_eq;
use datafusion::prelude::SessionContext;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::buffer::NullBuffer;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::source::DataSource;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Expr;
use datafusion_expr::Operator;
use datafusion_expr::ScalarUDF;
use datafusion_expr::col;
use datafusion_expr::lit;
use datafusion_functions::core::expr_ext::FieldAccessor;
use datafusion_functions::core::get_field;
use datafusion_functions::string::expr_fn::upper;
use datafusion_functions::string::upper::UpperFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::planner::logical2physical;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::displayable;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::expressions::lit as df_lit;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use futures::TryStreamExt;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::Nullability;
use vortex::dtype::session::DTypeSessionExt;
use vortex::editions::ComponentKind;
use vortex::editions::Edition;
use vortex::editions::EditionId;
use vortex::editions::EditionInclusion;
use vortex::editions::EditionSessionExt;
use vortex::expr::Expression;
use vortex::expr::lit as vortex_lit;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::scan::DataSourceRef;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ext_metadata::encode_arrow_ext_metadata;

use super::VortexDataSource;
use crate::common_tests::AnyStorage;
use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;
use crate::convert::ProcessedProjection;
use crate::row_index;

async fn vortex_source() -> anyhow::Result<VortexDataSource> {
    let batch =
        arrow_array::record_batch!(("a", Int32, vec![1, 2, 3]), ("b", Int32, vec![30, 10, 20]))?;
    vortex_source_from_batch(batch).await
}

async fn vortex_source_from_batch(batch: RecordBatch) -> anyhow::Result<VortexDataSource> {
    vortex_source_from_batch_with_session(batch, &VortexSession::default()).await
}

async fn vortex_source_from_batch_with_session(
    batch: RecordBatch,
    session: &VortexSession,
) -> anyhow::Result<VortexDataSource> {
    let schema = batch.schema();
    let data_source = data_source_from_batch(batch, session).await?;

    Ok(VortexDataSource::builder(data_source, session.clone())
        .with_arrow_schema(schema)
        .build()
        .await?)
}

/// Build a file-backed source so tests can supply a schema independently of the imported batch.
async fn data_source_from_batch(
    batch: RecordBatch,
    session: &VortexSession,
) -> anyhow::Result<DataSourceRef> {
    let schema = batch.schema();
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(buffer)?;
    Ok(file.data_source()?)
}

#[tokio::test]
async fn projection_filters_use_scan_schema() -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("b", 1)), "b"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                Operator::Modulo,
                Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
            )),
            "odd",
        ),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let supported: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("b", 0)),
        Operator::Gt,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(15)))),
    ));
    let unsupported: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("odd", 1)),
        Operator::Eq,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))),
    ));

    let pushed =
        projected.try_pushdown_filters(vec![supported, unsupported], &ConfigOptions::new())?;
    assert!(matches!(
        pushed.filters.as_slice(),
        [PushedDown::Yes, PushedDown::No]
    ));
    let updated = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not added to the source"))?;
    let plan = Arc::new(DataSourceExec::new(updated)) as Arc<dyn ExecutionPlan>;
    let ctx = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert_batches_eq!(
        [
            "+----+-----+",
            "| b  | odd |",
            "+----+-----+",
            "| 30 | 1   |",
            "| 20 | 1   |",
            "+----+-----+",
        ],
        &batches
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn projection_preserves_arrow_types(
    #[values(false, true)] fallback: bool,
) -> anyhow::Result<()> {
    let batch = arrow_array::record_batch!(
        ("a", Int32, vec![1, 2, 3]),
        ("s", Utf8, vec![Some("one"), None, Some("three")])
    )?;
    let source = vortex_source_from_batch(batch).await?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("s", 1)), "text"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                if fallback {
                    Operator::Modulo
                } else {
                    Operator::Plus
                },
                Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
            )),
            "n",
        ),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let plan = Arc::new(DataSourceExec::new(projected)) as Arc<dyn ExecutionPlan>;
    let expected_schema = plan.schema();
    let ctx = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    for batch in &batches {
        assert_eq!(batch.schema(), expected_schema);
    }
    let expected = if fallback {
        arrow_array::record_batch!(
            ("text", Utf8, vec![Some("one"), None, Some("three")]),
            ("n", Int32, vec![1, 0, 1])
        )?
    } else {
        arrow_array::record_batch!(
            ("text", Utf8, vec![Some("one"), None, Some("three")]),
            ("n", Int32, vec![3, 4, 5])
        )?
    };
    let actual = concat_batches(&expected_schema, &batches)?;
    assert_eq!(actual.columns(), expected.columns());
    Ok(())
}

#[tokio::test]
async fn statistics_follow_filter_and_fetch() -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let filter = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("a", 0)),
        Operator::Gt,
        df_expr::lit(1_i32),
    ));
    let filtered = source
        .try_pushdown_filters(vec![filter], &ConfigOptions::new())?
        .updated_node
        .expect("filter accepted");
    let limited = filtered.with_fetch(Some(1)).expect("fetch accepted");
    assert_eq!(
        limited.partition_statistics(None)?.num_rows,
        Precision::Inexact(1)
    );
    let rejected = limited.try_pushdown_filters(
        vec![
            df_expr::lit(false),
            Arc::new(DynamicFilterPhysicalExpr::new(vec![], df_expr::lit(false))),
        ],
        &ConfigOptions::new(),
    )?;
    assert!(matches!(
        rejected.filters.as_slice(),
        [PushedDown::No, PushedDown::No]
    ));
    assert!(rejected.updated_node.is_none());
    Ok(())
}

#[rstest]
#[case(Operator::And, vec![30, 20], vec![30], vec![20])]
#[case(Operator::Or, vec![30, 10, 20], vec![30, 20], vec![30, 20])]
#[tokio::test]
async fn dynamic_filters_refresh_projected_bounds_and_keep_residuals(
    #[case] operator: Operator,
    #[case] unrestricted: Vec<i32>,
    #[case] above_bound: Vec<i32>,
    #[case] equal_bound: Vec<i32>,
) -> anyhow::Result<()> {
    let source = vortex_source()
        .await?
        .try_swapping_with_projection(&ProjectionExprs::from(vec![ProjectionExpr::new(
            Arc::new(df_expr::Column::new("b", 1)),
            "value",
        )]))?
        .unwrap();
    let column = Arc::new(df_expr::Column::new("value", 0)) as Arc<dyn PhysicalExpr>;
    let above = |value| {
        Arc::new(df_expr::BinaryExpr::new(
            Arc::clone(&column),
            Operator::Gt,
            df_expr::lit(value),
        )) as Arc<dyn PhysicalExpr>
    };
    let bound = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::clone(&column)],
        df_expr::lit(true),
    ));
    let pushed = source.try_pushdown_filters(
        vec![Arc::new(df_expr::BinaryExpr::new(
            above(15_i32),
            operator,
            Arc::clone(&bound) as Arc<dyn PhysicalExpr>,
        ))],
        &ConfigOptions::new(),
    )?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::No]));
    let source = pushed.updated_node.unwrap();
    assert!(source.with_fetch(Some(1)).is_none());
    let mut visited = false;
    source.apply_expressions(&mut |expression| {
        expression.apply(|expr| {
            visited |= expr.is::<DynamicFilterPhysicalExpr>();
            Ok(TreeNodeRecursion::Continue)
        })
    })?;
    assert!(visited);
    // An unsupported dynamic predicate must not erase the static AND restriction.
    let unsupported = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::BinaryExpr::new(
            Arc::clone(&column),
            Operator::Modulo,
            df_expr::lit(2_i32),
        )),
        Operator::Eq,
        df_expr::lit(1_i32),
    ));
    for (predicate, expected) in [
        (df_expr::lit(true), unrestricted.clone()),
        (above(25_i32), above_bound),
        (
            Arc::new(df_expr::BinaryExpr::new(
                Arc::clone(&column),
                Operator::Eq,
                df_expr::lit(20_i32),
            )) as Arc<dyn PhysicalExpr>,
            equal_bound,
        ),
        (unsupported as Arc<dyn PhysicalExpr>, unrestricted),
    ] {
        let stream = source.open(0, SessionContext::new().task_ctx())?;
        let schema = stream.schema();
        bound.update(predicate)?;
        let batch = concat_batches(&schema, &stream.try_collect::<Vec<_>>().await?)?;
        assert_eq!(
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .as_ref(),
            expected
        );
    }
    Ok(())
}

/// Extension dtypes covered by both static and dynamic filter tests.
enum ExtensionFilterType {
    Uuid,
    Json,
}

/// The struct that holds the value column in the nested extension tests.
#[derive(Clone, Copy, Debug)]
enum Parent {
    Plain,
    /// An extension type without plugins, so that its export is its storage.
    Extension,
}

/// Input values and Arrow metadata shared by the static and dynamic extension filter tests.
struct ExtensionFilterFixture {
    session: VortexSession,
    batch: RecordBatch,
    /// Storage-typed literals in batch order, with a null in the final row.
    values: Vec<ScalarValue>,
}

impl ExtensionFilterFixture {
    fn new(extension_type: ExtensionFilterType) -> anyhow::Result<Self> {
        let session = VortexSession::default();
        vortex_json::initialize(&session);

        let (field, values) = match extension_type {
            ExtensionFilterType::Uuid => {
                let mut field = Field::new("value", DataType::FixedSizeBinary(16), true);
                field.try_with_extension_type(ArrowUuid)?;

                (
                    field,
                    vec![
                        ScalarValue::FixedSizeBinary(16, Some(b"0123456789abcde1".to_vec())), //
                        ScalarValue::FixedSizeBinary(16, Some(b"0123456789abcde2".to_vec())), //
                        ScalarValue::FixedSizeBinary(16, Some(b"0123456789abcde3".to_vec())), //
                        ScalarValue::FixedSizeBinary(16, None),                               //
                    ],
                )
            }
            ExtensionFilterType::Json => {
                let mut field = Field::new("value", DataType::Utf8, true);
                field.try_with_extension_type(ArrowJson::default())?;

                (
                    field,
                    vec![
                        ScalarValue::Utf8(Some(r#"{"x":1}"#.into())), //
                        ScalarValue::Utf8(Some(r#"{"x":2}"#.into())), //
                        ScalarValue::Utf8(Some(r#"{"x":3}"#.into())), //
                        ScalarValue::Utf8(None),                      //
                    ],
                )
            }
        };

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                field,
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
                ScalarValue::iter_to_array(values.iter().cloned())?,
            ],
        )?;

        Ok(Self {
            session,
            batch,
            values,
        })
    }

    fn column(&self) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Column::new(self.batch.schema().field(1).name(), 1))
    }

    fn literal(&self, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Literal::new(self.values[index].clone()))
    }

    fn comparison(&self, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::BinaryExpr::new(
            self.column(),
            Operator::Eq,
            self.literal(index),
        ))
    }

    fn membership(&self) -> anyhow::Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(df_expr::InListExpr::try_new(
            self.column(),
            vec![self.literal(1), self.literal(1)],
            false,
            &self.batch.schema(),
        )?))
    }

    async fn source(&self) -> anyhow::Result<VortexDataSource> {
        vortex_source_from_batch_with_session(self.batch.clone(), &self.session).await
    }

    /// Moves the value column into a `parent` struct, beside a `key` field that copies `id`.
    ///
    /// [`Parent::Extension`] registers [`AnyStorage`] and enables a test edition that permits it,
    /// so the fixture can write and read `parent` as an extension value.
    fn nested_batch(&self, parent: Parent) -> anyhow::Result<RecordBatch> {
        let schema = self.batch.schema();
        let fields = Fields::from(vec![
            Field::new("key", DataType::Int32, false),
            schema.field(1).clone(),
        ]);
        let mut parent_field = Field::new_struct("parent", fields.clone(), false);
        if matches!(parent, Parent::Extension) {
            let storage = self
                .session
                .arrow()
                .from_arrow_datatype(parent_field.data_type(), Nullability::NonNullable)?;
            let parent_dtype = AnyStorage::dtype(storage)?;
            self.session.dtypes().register(AnyStorage);

            let edition = EditionId::new("datafusion-test", 2026, 11, 0);
            let editions = self.session.editions();
            editions.declare_edition(Edition {
                id: edition,
                immutable: true,
                min_library_version: None,
            })?;
            editions.declare_inclusion(EditionInclusion::new(
                ComponentKind::DType,
                &parent_dtype.id(),
                edition,
            ))?;
            self.session.enable_edition(edition)?;

            parent_field
                .metadata_mut()
                .extend(encode_arrow_ext_metadata(&parent_dtype)?);
        }

        let parents = StructArray::try_new(
            fields,
            vec![
                Arc::clone(self.batch.column(0)),
                Arc::clone(self.batch.column(1)),
            ],
            None,
        )?;
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(vec![schema.field(0).clone(), parent_field])),
            vec![Arc::clone(self.batch.column(0)), Arc::new(parents)],
        )?)
    }
}

/// Reads `parent.<name>` from a batch that [`ExtensionFilterFixture::nested_batch`] built.
fn parent_field_expr(schema: &Schema, name: &str) -> anyhow::Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(ScalarFunctionExpr::try_new(
        get_field(),
        vec![
            Arc::new(df_expr::Column::new("parent", 1)),
            df_expr::lit(name),
        ],
        schema,
        Arc::new(ConfigOptions::new()),
    )?))
}

/// Collects row identifiers so the tests compare the scan restriction separately from its residual.
async fn scan_ids(plan: Arc<dyn ExecutionPlan>) -> anyhow::Result<Vec<i32>> {
    let batches = datafusion_physical_plan::collect(plan, SessionContext::new().task_ctx()).await?;

    Ok(batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
        })
        .collect())
}

#[rstest]
#[case::uuid(ExtensionFilterType::Uuid)]
#[case::json(ExtensionFilterType::Json)]
#[tokio::test]
async fn test_extension_comparison_literals_are_pushed_down(
    #[case] extension_type: ExtensionFilterType,
    #[values(false, true)] reverse_operands: bool,
) -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(extension_type)?;
    let (left, right) = if reverse_operands {
        (fixture.literal(1), fixture.column())
    } else {
        (fixture.column(), fixture.literal(1))
    };

    let filter = Arc::new(df_expr::BinaryExpr::new(left, Operator::Eq, right));
    let pushed = fixture
        .source()
        .await?
        .try_pushdown_filters(vec![filter], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));
    let source = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not pushed"))?;

    assert_eq!(scan_ids(Arc::new(DataSourceExec::new(source))).await?, [2]);

    Ok(())
}

#[rstest]
#[case::uuid(ExtensionFilterType::Uuid)]
#[case::json(ExtensionFilterType::Json)]
#[tokio::test]
async fn test_extension_membership_literals_are_pushed_down(
    #[case] extension_type: ExtensionFilterType,
) -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(extension_type)?;
    let pushed = fixture
        .source()
        .await?
        .try_pushdown_filters(vec![fixture.membership()?], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));
    let source = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not pushed"))?;

    assert_eq!(scan_ids(Arc::new(DataSourceExec::new(source))).await?, [2]);

    Ok(())
}

#[rstest]
#[case::uuid(ExtensionFilterType::Uuid)]
#[case::json(ExtensionFilterType::Json)]
#[tokio::test]
async fn test_dynamic_extension_literals_refresh_and_keep_residuals(
    #[case] extension_type: ExtensionFilterType,
) -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(extension_type)?;
    let bound = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![fixture.column()],
        df_expr::lit(true),
    ));
    let pushed = fixture.source().await?.try_pushdown_filters(
        vec![Arc::clone(&bound) as Arc<dyn PhysicalExpr>],
        &ConfigOptions::new(),
    )?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::No]));
    let source = pushed
        .updated_node
        .ok_or_else(|| anyhow!("dynamic filter was not pushed"))?;
    let unsupported = Arc::new(df_expr::BinaryExpr::new(
        fixture.column(),
        Operator::IsNotDistinctFrom,
        fixture.literal(1),
    ));

    for (predicate, expected_scan, expected_result) in [
        (df_expr::lit(true), vec![1, 2, 3, 4], vec![1, 2, 3, 4]), //
        (fixture.comparison(0), vec![1], vec![1]),                //
        (fixture.comparison(2), vec![3], vec![3]),                //
        (fixture.comparison(3), vec![], vec![]),                  //
        (fixture.membership()?, vec![2], vec![2]),                //
        (
            unsupported as Arc<dyn PhysicalExpr>,
            vec![1, 2, 3, 4],
            vec![2],
        ), //
    ] {
        bound.update(predicate)?;
        let scan = Arc::new(DataSourceExec::new(Arc::clone(&source))) as Arc<dyn ExecutionPlan>;
        assert_eq!(scan_ids(Arc::clone(&scan)).await?, expected_scan);

        let residual = Arc::new(FilterExec::try_new(
            Arc::clone(&bound) as Arc<dyn PhysicalExpr>,
            scan.reset_state()?,
        )?);
        assert_eq!(scan_ids(residual).await?, expected_result);
    }

    Ok(())
}

/// The V2 source has no schema adapter, so a projection over an extension column must produce
/// batches with exactly the plan schema.
#[tokio::test]
async fn test_extension_projection_preserves_arrow_type() -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(ExtensionFilterType::Json)?;
    let projection = ProjectionExprs::from(vec![ProjectionExpr::new(fixture.column(), "value")]);
    let projected = fixture
        .source()
        .await?
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let plan = Arc::new(DataSourceExec::new(projected)) as Arc<dyn ExecutionPlan>;
    let plan_schema = plan.schema();

    let batches = datafusion_physical_plan::collect(plan, SessionContext::new().task_ctx()).await?;
    for batch in &batches {
        assert_eq!(batch.schema(), plan_schema);
    }

    let actual = concat_batches(&plan_schema, &batches)?;
    assert_eq!(actual.column(0), fixture.batch.column(1));

    Ok(())
}

#[tokio::test]
async fn test_build_rejects_extension_column_over_plain() -> anyhow::Result<()> {
    let session = VortexSession::default();
    vortex_json::initialize(&session);

    let batch = arrow_array::record_batch!(("value", Utf8, vec![r#"{"x":1}"#]))?;
    let data_source = data_source_from_batch(batch, &session).await?;

    let mut json_field = Field::new("value", DataType::Utf8, true);
    json_field.try_with_extension_type(ArrowJson::default())?;

    let error = VortexDataSource::builder(data_source, session)
        .with_arrow_schema(Arc::new(Schema::new(vec![json_field])))
        .build()
        .await
        .expect_err("the plain column must not scan as Json");
    assert!(
        error.to_string().contains(
            "Field value must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type arrow.json"
        ),
        "{error}"
    );

    Ok(())
}

/// A comparison on an extension field of a plain struct pushes down like one on a column. Through
/// extension storage, the plugin that exports the field keeps the comparison in DataFusion.
#[rstest]
#[case::plain(Parent::Plain, true)]
#[case::extension(Parent::Extension, false)]
#[tokio::test]
async fn test_nested_extension_comparison_pushdown(
    #[case] parent: Parent,
    #[case] pushed_down: bool,
    #[values(ExtensionFilterType::Uuid, ExtensionFilterType::Json)]
    extension_type: ExtensionFilterType,
) -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(extension_type)?;
    let batch = fixture.nested_batch(parent)?;
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        parent_field_expr(&batch.schema(), "value")?,
        Operator::Eq,
        fixture.literal(1),
    ));
    let source = vortex_source_from_batch_with_session(batch, &fixture.session).await?;

    let pushed =
        source.try_pushdown_filters(vec![Arc::clone(&predicate)], &ConfigOptions::new())?;
    assert_eq!(
        matches!(pushed.filters.as_slice(), [PushedDown::Yes]),
        pushed_down
    );

    let scan = Arc::new(DataSourceExec::new(
        pushed.updated_node.unwrap_or_else(|| Arc::new(source)),
    ));
    let residual = Arc::new(FilterExec::try_new(predicate, scan)?);
    assert_eq!(scan_ids(residual).await?, [2]);

    Ok(())
}

/// A plain field of an extension struct reads through its storage in the scan, both in a filter
/// and in a projection.
#[tokio::test]
async fn test_get_field_through_extension_storage_is_pushed_down() -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(ExtensionFilterType::Uuid)?;
    let batch = fixture.nested_batch(Parent::Extension)?;
    let key = parent_field_expr(&batch.schema(), "key")?;
    let source = vortex_source_from_batch_with_session(batch, &fixture.session).await?;

    let filter: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::clone(&key),
        Operator::Eq,
        df_expr::lit(2i32),
    ));
    let pushed = source.try_pushdown_filters(vec![filter], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));
    let filtered = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not pushed"))?;
    assert_eq!(
        scan_ids(Arc::new(DataSourceExec::new(filtered))).await?,
        [2]
    );

    let projection = ProjectionExprs::from(vec![ProjectionExpr::new(key, "key")]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    assert_eq!(
        scan_ids(Arc::new(DataSourceExec::new(projected))).await?,
        [1, 2, 3, 4]
    );

    Ok(())
}

struct AdvisoryConverter(Arc<AtomicUsize>);

impl ExpressionConverter for AdvisoryConverter {
    fn try_convert(
        &self,
        _expr: &Arc<dyn PhysicalExpr>,
        _schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        Ok(None)
    }

    fn try_convert_dynamic(
        &self,
        _expr: &Arc<dyn PhysicalExpr>,
        _schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Some(vortex_lit(true)))
    }
}

#[tokio::test]
async fn dynamic_filter_uses_advisory_conversion_and_keeps_residual() -> anyhow::Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let session = VortexSession::default();
    let batch = arrow_record_batch!(("a", Int32, vec![1, 2, 3]))?;
    let schema = batch.schema();
    let source = VortexDataSource::builder(data_source_from_batch(batch, &session).await?, session)
        .with_arrow_schema(schema)
        .with_expression_converter(Arc::new(AdvisoryConverter(Arc::clone(&calls))))
        .build()
        .await?;
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![], df_lit(true)));
    let pushed = source.try_pushdown_filters(vec![filter], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::No]));
    assert!(pushed.updated_node.is_some());
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let exact = source.try_pushdown_filters(vec![df_lit(true)], &ConfigOptions::new())?;
    assert!(exact.updated_node.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    Ok(())
}

/// Swap `projection` into `source`, then [`display_and_collect`] the projected source.
async fn swap_projection(
    source: &VortexDataSource,
    projection: &ProjectionExprs,
) -> anyhow::Result<(String, Vec<RecordBatch>)> {
    let projected = source
        .try_swapping_with_projection(projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    display_and_collect(projected).await
}

/// Return the display of a plan that scans `source`, and the batches that the plan produces.
///
/// Every batch must have the schema that the plan declares.
async fn display_and_collect(
    source: Arc<dyn DataSource>,
) -> anyhow::Result<(String, Vec<RecordBatch>)> {
    let plan = Arc::new(DataSourceExec::new(source)) as Arc<dyn ExecutionPlan>;
    let display = displayable(plan.as_ref()).one_line().to_string();
    let schema = plan.schema();
    let batches = datafusion_physical_plan::collect(plan, SessionContext::new().task_ctx()).await?;
    for batch in &batches {
        assert_eq!(batch.schema(), schema);
    }

    Ok((display, batches))
}

/// Projects `id`, a CASE over `s.inner.label` that Vortex cannot evaluate, the leaf itself, and the
/// row index.
///
/// The CASE keeps the projection in DataFusion. It parses the label of the second row as an
/// integer, which fails for the other labels, so DataFusion must evaluate it only on the rows that
/// the CASE selects.
fn nested_label_projection(schema: &Schema) -> anyhow::Result<ProjectionExprs> {
    let s: PhysicalExprRef = Arc::new(df_expr::Column::new_with_schema("s", schema)?);
    let id: PhysicalExprRef = Arc::new(df_expr::Column::new_with_schema("id", schema)?);

    // One flattened call, as DataFusion's simplifier writes it. Nested calls would read all of
    // `s.inner`.
    let label: PhysicalExprRef = Arc::new(ScalarFunctionExpr::try_new(
        get_field(),
        vec![s, df_expr::lit("inner"), df_expr::lit("label")],
        schema,
        Arc::new(ConfigOptions::new()),
    )?);
    let upper_label = Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(ScalarUDF::from(UpperFunc::new())),
        vec![Arc::clone(&label)],
        schema,
        Arc::new(ConfigOptions::new()),
    )?);
    let parsed_label = Arc::new(df_expr::CastExpr::new(
        Arc::clone(&label),
        DataType::Int32,
        None,
    ));
    let formatted_label = Arc::new(df_expr::CastExpr::new(parsed_label, DataType::Utf8, None));
    let is_second_row = Arc::new(df_expr::BinaryExpr::new(
        Arc::clone(&id),
        Operator::Eq,
        df_expr::lit(2i32),
    ));
    let formatted = Arc::new(df_expr::CaseExpr::try_new(
        None,
        vec![(is_second_row, formatted_label)],
        Some(upper_label),
    )?);

    Ok(ProjectionExprs::from(vec![
        ProjectionExpr::new(id, "id"),
        ProjectionExpr::new(formatted, "formatted"),
        ProjectionExpr::new(label, "original"),
        ProjectionExpr::new(row_index(), "position"),
    ]))
}

/// How a source stores the struct `s`, and how its Arrow schema declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OuterNullability {
    /// Stored and declared non-nullable.
    NonNullable,
    /// Stored and declared nullable, with a null in the first row.
    Nullable,
    /// Stored nullable without nulls, and declared non-nullable.
    NullableInStorage,
    /// Stored non-nullable, and declared nullable.
    NullableInSchema,
}

impl OuterNullability {
    fn is_stored_nullable(self) -> bool {
        matches!(self, Self::Nullable | Self::NullableInStorage)
    }

    fn is_declared_nullable(self) -> bool {
        matches!(self, Self::Nullable | Self::NullableInSchema)
    }
}

/// The raw fallback reads `s.inner.label` without `s.inner.unused`, and DataFusion computes the
/// same values from it as from the whole struct, in every combination of nullable fields.
#[rstest]
#[tokio::test]
async fn test_raw_projection_reads_struct_leaf(
    #[values(
        OuterNullability::NonNullable,
        OuterNullability::Nullable,
        OuterNullability::NullableInStorage,
        OuterNullability::NullableInSchema
    )]
    outer: OuterNullability,
    #[values(false, true)] inner_nullable: bool,
    #[values(false, true)] label_nullable: bool,
) -> anyhow::Result<()> {
    let label_field = Field::new("label", DataType::Utf8, label_nullable)
        .with_metadata([("source".to_owned(), "test".to_owned())]);
    let labels = StringArray::from(vec![
        Some("x"),
        Some("20"),
        (!label_nullable).then_some("z"),
        Some("w"),
    ]);
    let inner = StructArray::try_new(
        vec![
            label_field.clone(),
            Field::new("unused", DataType::Int32, false),
        ]
        .into(),
        vec![
            Arc::new(labels),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        ],
        inner_nullable.then(|| NullBuffer::from(vec![true, true, true, false])),
    )?;

    let outer_validity = match outer {
        OuterNullability::Nullable => Some(NullBuffer::from(vec![false, true, true, true])),
        OuterNullability::NullableInStorage => Some(NullBuffer::new_valid(4)),
        OuterNullability::NonNullable | OuterNullability::NullableInSchema => None,
    };
    let s = StructArray::try_new(
        vec![Field::new(
            "inner",
            inner.data_type().clone(),
            inner_nullable,
        )]
        .into(),
        vec![Arc::new(inner)],
        outer_validity,
    )?;

    let s_field = |nullable| Field::new("s", s.data_type().clone(), nullable);
    let id_field = Field::new("id", DataType::Int32, false);
    let stored_schema = Arc::new(Schema::new(vec![
        id_field.clone(),
        s_field(outer.is_stored_nullable()),
    ]));
    let schema = Arc::new(Schema::new(vec![
        id_field,
        s_field(outer.is_declared_nullable()),
    ]));

    let batch = RecordBatch::try_new(
        stored_schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(s)],
    )?;
    let session = VortexSession::default();
    let data_source = data_source_from_batch(batch, &session).await?;
    let source = VortexDataSource::builder(data_source, session)
        .with_arrow_schema(Arc::clone(&schema))
        .build()
        .await?;

    let (display, batches) = swap_projection(&source, &nested_label_projection(&schema)?).await?;

    assert!(
        display.contains(
            "projection=pack(id: $.id, s: pack(inner: $.s.inner{label}), \
             __vortex_row_index: #row_idx)"
        ),
        "{display}"
    );

    let first_row = if outer == OuterNullability::Nullable {
        "| 1  |           |          | 0        |"
    } else {
        "| 1  | X         | x        | 0        |"
    };
    let third_row = if label_nullable {
        "| 3  |           |          | 2        |"
    } else {
        "| 3  | Z         | z        | 2        |"
    };
    let fourth_row = if inner_nullable {
        "| 4  |           |          | 3        |"
    } else {
        "| 4  | W         | w        | 3        |"
    };
    assert_batches_eq!(
        [
            "+----+-----------+----------+----------+",
            "| id | formatted | original | position |",
            "+----+-----------+----------+----------+",
            first_row,
            "| 2  | 20        | 20       | 1        |",
            third_row,
            fourth_row,
            "+----+-----------+----------+----------+",
        ],
        &batches
    );

    for batch in &batches {
        assert_eq!(batch.schema().field(2).metadata(), label_field.metadata());
    }

    Ok(())
}

fn strings(values: [&str; 2]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

/// A source of `s` and `t`, two non-nullable structs that share the child name `a`.
async fn two_struct_source() -> anyhow::Result<(VortexDataSource, SchemaRef)> {
    let utf8 = |name: &str| Field::new(name, DataType::Utf8, false);
    let s = StructArray::try_new(
        vec![utf8("a"), utf8("b")].into(),
        vec![strings(["sa1", "sa2"]), strings(["sb1", "sb2"])],
        None,
    )?;
    let t = StructArray::try_new(
        vec![utf8("a"), utf8("x")].into(),
        vec![strings(["ta1", "ta2"]), strings(["tx1", "tx2"])],
        None,
    )?;
    let batch = RecordBatch::try_from_iter([
        ("s", Arc::new(s) as ArrayRef),
        ("t", Arc::new(t) as ArrayRef),
    ])?;
    let schema = batch.schema();

    Ok((vortex_source_from_batch(batch).await?, schema))
}

/// Plan each logical expression in `projection` against `schema`, under its alias.
fn projection_of(schema: &Schema, projection: &[(Expr, &str)]) -> ProjectionExprs {
    ProjectionExprs::from(
        projection
            .iter()
            .map(|(expr, alias)| ProjectionExpr::new(logical2physical(expr, schema), *alias))
            .collect::<Vec<_>>(),
    )
}

/// Each struct keeps only its own child `a`, although both structs name a child `a`.
#[tokio::test]
async fn test_raw_projection_reads_same_named_children() -> anyhow::Result<()> {
    let (source, schema) = two_struct_source().await?;
    let projection = projection_of(
        &schema,
        &[
            (upper(col("s").field("a")), "upper_s_a"),
            (upper(col("t").field("a")), "upper_t_a"),
        ],
    );

    let (display, batches) = swap_projection(&source, &projection).await?;

    assert!(
        display.contains("projection=pack(s: $.s{a}, t: $.t{a})"),
        "{display}"
    );
    assert_batches_eq!(
        [
            "+-----------+-----------+",
            "| upper_s_a | upper_t_a |",
            "+-----------+-----------+",
            "| SA1       | TA1       |",
            "| SA2       | TA2       |",
            "+-----------+-----------+",
        ],
        &batches
    );

    Ok(())
}

/// A struct that the projection also reads whole keeps all its children.
#[tokio::test]
async fn test_raw_projection_reads_struct_used_whole() -> anyhow::Result<()> {
    let (source, schema) = two_struct_source().await?;
    let projection = projection_of(
        &schema,
        &[
            (col("s"), "s"),
            (upper(col("s").field("a")), "upper_s_a"),
            (upper(col("t").field("x")), "upper_t_x"),
        ],
    );

    let (display, batches) = swap_projection(&source, &projection).await?;

    assert!(
        display.contains("projection=pack(s: $.s, t: $.t{x})"),
        "{display}"
    );
    assert_batches_eq!(
        [
            "+------------------+-----------+-----------+",
            "| s                | upper_s_a | upper_t_x |",
            "+------------------+-----------+-----------+",
            "| {a: sa1, b: sb1} | SA1       | TX1       |",
            "| {a: sa2, b: sb2} | SA2       | TX2       |",
            "+------------------+-----------+-----------+",
        ],
        &batches
    );

    Ok(())
}

/// Nested `get_field` calls read the result of the inner call whole, since the inner call must
/// return the struct type that DataFusion planned for it.
#[tokio::test]
async fn test_raw_projection_reads_result_of_inner_call_whole() -> anyhow::Result<()> {
    let inner = StructArray::try_from(vec![
        ("label", strings(["a", "b"])),
        ("unused", strings(["c", "d"])),
    ])?;
    let s = StructArray::try_from(vec![
        ("inner", Arc::new(inner) as ArrayRef),
        ("x", strings(["e", "f"])),
    ])?;
    let batch = RecordBatch::try_from_iter([("s", Arc::new(s) as ArrayRef)])?;
    let schema = batch.schema();
    let source = vortex_source_from_batch(batch).await?;
    let projection = projection_of(
        &schema,
        &[(upper(col("s").field("inner").field("label")), "label")],
    );

    let (display, batches) = swap_projection(&source, &projection).await?;

    assert!(
        display.contains("projection=pack(s: $.s{inner})"),
        "{display}"
    );
    assert_batches_eq!(
        [
            "+-------+",
            "| label |",
            "+-------+",
            "| A     |",
            "| B     |",
            "+-------+"
        ],
        &batches
    );

    Ok(())
}

/// Fields read inside `LIKE` and inside an `IN` list are kept. The list element that is not a
/// literal keeps the projection in DataFusion.
#[tokio::test]
async fn test_raw_projection_reads_children_of_like_and_in() -> anyhow::Result<()> {
    let (source, schema) = two_struct_source().await?;
    let projection = projection_of(
        &schema,
        &[
            (col("s").field("a").like(lit("%1")), "like_s_a"),
            (
                col("s")
                    .field("a")
                    .in_list(vec![col("t").field("x"), lit("sa2")], false),
                "in_list",
            ),
        ],
    );

    let (display, batches) = swap_projection(&source, &projection).await?;

    assert!(
        display.contains("projection=pack(s: $.s{a}, t: $.t{x})"),
        "{display}"
    );
    assert_batches_eq!(
        [
            "+----------+---------+",
            "| like_s_a | in_list |",
            "+----------+---------+",
            "| true     | false   |",
            "| false    | true    |",
            "+----------+---------+",
        ],
        &batches
    );

    Ok(())
}

/// A filter on a kept field converts against the narrowed scan output and reads the field through
/// the narrowed struct.
#[tokio::test]
async fn test_filter_pushes_down_over_pruned_struct() -> anyhow::Result<()> {
    let (source, schema) = two_struct_source().await?;
    let projection = projection_of(
        &schema,
        &[
            (col("s").field("a"), "s_a"),
            (upper(col("t").field("x")), "upper_t_x"),
        ],
    );
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let filter: PhysicalExprRef = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("s_a", 0)),
        Operator::Eq,
        df_expr::lit("sa2"),
    ));

    let pushed = projected.try_pushdown_filters(vec![filter], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));

    let filtered = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not added to the source"))?;
    let (display, batches) = display_and_collect(filtered).await?;

    assert!(
        display.contains(
            "projection=pack(s: $.s{a}, t: $.t{x}), \
             filter=(pack(s: $.s{a}, t: $.t{x}).s.a = \"sa2\")"
        ),
        "{display}"
    );
    assert_batches_eq!(
        [
            "+-----+-----------+",
            "| s_a | upper_t_x |",
            "+-----+-----------+",
            "| sa2 | TX2       |",
            "+-----+-----------+",
        ],
        &batches
    );

    Ok(())
}

/// Kept leaves of other types keep their Arrow types, such as a timestamp's time zone and a
/// decimal's precision and scale.
#[tokio::test]
async fn test_raw_projection_reads_typed_leaves() -> anyhow::Result<()> {
    let timestamps =
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2]).with_timezone("UTC")) as ArrayRef;
    let decimals = Arc::new(Decimal128Array::from(vec![123, 456]).with_precision_and_scale(10, 2)?)
        as ArrayRef;
    let lists = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1)]),
        Some(vec![Some(2), None]),
    ])) as ArrayRef;
    let unused = Arc::new(Int32Array::from(vec![0, 0])) as ArrayRef;
    let s = StructArray::try_from(vec![
        ("ts", Arc::clone(&timestamps)),
        ("dec", Arc::clone(&decimals)),
        ("list", Arc::clone(&lists)),
        ("unused", unused),
    ])?;
    let batch = RecordBatch::try_from_iter([
        ("a", Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef),
        ("s", Arc::new(s) as ArrayRef),
    ])?;
    let schema = batch.schema();
    let source = vortex_source_from_batch(batch).await?;
    let projection = projection_of(
        &schema,
        &[
            (col("s").field("ts"), "ts"),
            (col("s").field("dec"), "dec"),
            (col("s").field("list"), "list"),
            (col("a") % lit(2), "odd"),
        ],
    );

    let (display, batches) = swap_projection(&source, &projection).await?;

    assert!(
        display.contains("projection=pack(a: $.a, s: $.s{ts, dec, list})"),
        "{display}"
    );
    let [batch] = batches.as_slice() else {
        return Err(anyhow!("expected one batch, got {}", batches.len()));
    };
    assert_eq!(
        batch.columns(),
        [
            timestamps,
            decimals,
            lists,
            Arc::new(Int32Array::from(vec![1, 0])) as ArrayRef,
        ]
    );

    Ok(())
}

/// The default converter, but every projection stays in DataFusion.
#[derive(Default)]
struct RawProjectionConverter(DefaultExpressionConverter);

impl ExpressionConverter for RawProjectionConverter {
    fn try_convert(&self, expr: &PhysicalExprRef, schema: &Schema) -> DFResult<Option<Expression>> {
        self.0.try_convert(expr, schema)
    }

    fn split_projection(
        &self,
        projection: ProjectionExprs,
        input_schema: &Schema,
        _output_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        self.no_pushdown_projection(projection, input_schema)
    }
}

/// A custom converter that keeps every projection in DataFusion also reads only the used
/// children.
#[tokio::test]
async fn test_custom_converter_reads_used_children() -> anyhow::Result<()> {
    let input = StructArray::try_new(
        vec![
            Field::new("value", DataType::Int32, false),
            Field::new("unused", DataType::Int32, false),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![10, 20])),
        ],
        None,
    )?;
    let batch = RecordBatch::try_from_iter([("s", Arc::new(input) as ArrayRef)])?;
    let schema = batch.schema();
    let session = VortexSession::default();
    let data_source = data_source_from_batch(batch, &session).await?;
    let source = VortexDataSource::builder(data_source, session)
        .with_arrow_schema(Arc::clone(&schema))
        .with_expression_converter(Arc::new(RawProjectionConverter::default()))
        .build()
        .await?;
    let projection = projection_of(&schema, &[(col("s").field("value"), "value")]);

    let (display, batches) = swap_projection(&source, &projection).await?;

    assert!(
        display.contains("projection=pack(s: $.s{value})"),
        "{display}"
    );
    assert_batches_eq!(
        [
            "+-------+",
            "| value |",
            "+-------+",
            "| 1     |",
            "| 2     |",
            "+-------+"
        ],
        &batches
    );

    Ok(())
}
