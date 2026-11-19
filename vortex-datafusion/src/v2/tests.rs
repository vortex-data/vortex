// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::Int32Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
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
use datafusion_expr::Operator;
use datafusion_expr::ScalarUDF;
use datafusion_functions::core::get_field;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions::string::upper::UpperFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::displayable;
use datafusion_physical_plan::expressions as df_expr;
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

#[rstest]
#[case::non_nullable(false, false, false, true)]
#[case::nullable_outer(true, true, false, true)]
#[case::nullable_outer_non_nullable_leaf(true, true, false, false)]
#[case::nullable_outer_in_storage(true, false, false, true)]
#[case::nullable_inner(false, false, true, true)]
#[case::non_nullable_leaf_in_nullable_storage(true, false, false, false)]
#[tokio::test]
async fn test_partial_projection_reads_struct_leaf(
    #[case] stored_outer_nullable: bool,
    #[case] declared_outer_nullable: bool,
    #[case] inner_nullable: bool,
    #[case] label_nullable: bool,
) -> anyhow::Result<()> {
    let label_field = Field::new("label", DataType::Utf8, label_nullable)
        .with_metadata([("source".to_owned(), "test".to_owned())]);
    let labels = if label_nullable {
        StringArray::from(vec![Some("x"), Some("20"), None])
    } else {
        StringArray::from(vec![Some("x"), Some("20"), Some("z")])
    };
    let inner = StructArray::try_new(
        vec![
            label_field.clone(),
            Field::new("unused", DataType::Int32, false),
        ]
        .into(),
        vec![
            Arc::new(labels),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
        inner_nullable.then(|| NullBuffer::from(vec![false, true, true])),
    )?;
    let outer_has_null = stored_outer_nullable && (declared_outer_nullable || label_nullable);
    let outer = StructArray::try_new(
        vec![Field::new(
            "inner",
            inner.data_type().clone(),
            inner_nullable,
        )]
        .into(),
        vec![Arc::new(inner)],
        stored_outer_nullable.then(|| {
            if outer_has_null {
                NullBuffer::from(vec![false, true, true])
            } else {
                NullBuffer::new_valid(3)
            }
        }),
    )?;
    let stored_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", outer.data_type().clone(), stored_outer_nullable),
    ]));
    let batch = RecordBatch::try_new(
        stored_schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(outer) as ArrayRef,
        ],
    )?;
    let schema = Arc::new(Schema::new(vec![
        batch.schema().field(0).clone(),
        batch
            .schema()
            .field(1)
            .clone()
            .with_nullable(declared_outer_nullable),
    ]));
    let session = VortexSession::default();
    let data_source = data_source_from_batch(batch, &session).await?;
    let source = VortexDataSource::builder(data_source, session)
        .with_arrow_schema(Arc::clone(&schema))
        .build()
        .await?;

    let function = |udf: ScalarUDF, args| -> anyhow::Result<PhysicalExprRef> {
        Ok(Arc::new(ScalarFunctionExpr::try_new(
            Arc::new(udf),
            args,
            &schema,
            Arc::new(ConfigOptions::new()),
        )?))
    };
    let id: PhysicalExprRef = Arc::new(df_expr::Column::new("id", 0));
    let label = function(
        GetFieldFunc::new().into(),
        vec![
            Arc::new(df_expr::Column::new("s", 1)),
            df_expr::lit("inner"),
            df_expr::lit("label"),
        ],
    )?;
    let upper = function(UpperFunc::new().into(), vec![Arc::clone(&label)])?;
    // The cast would fail on "x" if the scan evaluated it before the CASE selected rows.
    let parse_label = Arc::new(df_expr::CastExpr::new(
        Arc::clone(&label),
        DataType::Int32,
        None,
    ));
    let format_label: PhysicalExprRef =
        Arc::new(df_expr::CastExpr::new(parse_label, DataType::Utf8, None));
    let is_second_row: PhysicalExprRef = Arc::new(df_expr::BinaryExpr::new(
        Arc::clone(&id),
        Operator::Eq,
        df_expr::lit(2i32),
    ));
    let formatted = Arc::new(df_expr::CaseExpr::try_new(
        None,
        vec![(is_second_row, format_label)],
        Some(upper),
    )?);
    let inner = function(
        GetFieldFunc::new().into(),
        vec![
            Arc::new(df_expr::Column::new("s", 1)),
            df_expr::lit("inner"),
        ],
    )?;
    let original = function(
        GetFieldFunc::new().into(),
        vec![inner, df_expr::lit("label")],
    )?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(id, "id"),
        ProjectionExpr::new(formatted, "formatted"),
        ProjectionExpr::new(original, "original"),
        ProjectionExpr::new(row_index(), "position"),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let plan = Arc::new(DataSourceExec::new(projected)) as Arc<dyn ExecutionPlan>;
    let display = displayable(plan.as_ref()).one_line().to_string();
    let declared_leaf_nullable = declared_outer_nullable || inner_nullable || label_nullable;
    let stored_leaf_nullable = stored_outer_nullable || inner_nullable || label_nullable;
    let expected_scan = if declared_leaf_nullable != stored_leaf_nullable {
        "projection=pack(id: $.id, s: $.s, __vortex_row_index: #row_idx)"
    } else {
        "projection=pack(id: $.id, __vortex_row_index: #row_idx, label: $.s.inner.label)"
    };
    assert!(display.contains(expected_scan), "{display}");

    let batches = datafusion_physical_plan::collect(plan, SessionContext::new().task_ctx()).await?;
    let row0 = if outer_has_null || inner_nullable {
        "| 1  |           |          | 0        |"
    } else {
        "| 1  | X         | x        | 0        |"
    };
    let row2 = if label_nullable {
        "| 3  |           |          | 2        |"
    } else {
        "| 3  | Z         | z        | 2        |"
    };
    assert_batches_eq!(
        [
            "+----+-----------+----------+----------+",
            "| id | formatted | original | position |",
            "+----+-----------+----------+----------+",
            row0,
            "| 2  | 20        | 20       | 1        |",
            row2,
            "+----+-----------+----------+----------+",
        ],
        &batches
    );
    for batch in batches {
        assert_eq!(batch.schema().field(2).metadata(), label_field.metadata());
    }

    Ok(())
}

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

#[tokio::test]
async fn test_custom_converter_keeps_raw_projection() -> anyhow::Result<()> {
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
    let field: PhysicalExprRef = Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(GetFieldFunc::new().into()),
        vec![
            Arc::new(df_expr::Column::new("s", 0)),
            df_expr::lit("value"),
        ],
        &schema,
        Arc::new(ConfigOptions::new()),
    )?);
    let converter = RawProjectionConverter::default();
    assert!(converter.try_convert(&field, &schema)?.is_some());

    let session = VortexSession::default();
    let data_source = data_source_from_batch(batch, &session).await?;
    let source = VortexDataSource::builder(data_source, session)
        .with_arrow_schema(schema)
        .with_expression_converter(Arc::new(converter))
        .build()
        .await?;
    let projection = vec![ProjectionExpr::new(field, "value")].into();
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let plan = DataSourceExec::new(projected);
    let display = displayable(&plan).one_line().to_string();
    assert!(display.contains("projection=pack(s: $.s)"), "{display}");

    Ok(())
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
