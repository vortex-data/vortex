// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::common::DFSchema;
use datafusion::execution::context::ExecutionProps;
use datafusion::logical_expr::simplify::SimplifyContext;
use datafusion::optimizer::simplify_expressions::ExprSimplifier;
use datafusion::physical_expr::create_physical_expr;
use datafusion::prelude::SessionContext;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::HigherOrderUDF;
use datafusion_expr::Operator as DFOperator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions_nested::array_transform::ArrayTransform;
use datafusion_physical_expr::HigherOrderFunctionExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use rstest::rstest;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::root;

use super::*;
use crate::convert::DefaultExpressionConverter;
use crate::row_index;

/// `{id: i64, s: {a: utf8?, b: i64?, inner: {c: utf8?}}?, tags: list<i64>?}`
fn schema() -> Schema {
    let inner = Field::new_struct("inner", vec![Field::new("c", DataType::Utf8, true)], true);
    Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new_struct(
            "s",
            vec![
                Field::new("a", DataType::Utf8, true),
                Field::new("b", DataType::Int64, true),
                inner,
            ],
            true,
        ),
        Field::new_list("tags", Field::new_list_field(DataType::Int64, true), true),
    ])
}

/// Plans the SQL expression `sql` against `schema` as DataFusion does.
fn plan(sql: &str, schema: &Schema) -> DFResult<Arc<dyn PhysicalExpr>> {
    let df_schema = Arc::new(DFSchema::try_from(schema.clone())?);
    let simplifier = ExprSimplifier::new(
        SimplifyContext::builder()
            .with_schema(Arc::clone(&df_schema))
            .build(),
    );
    let expr = SessionContext::new().parse_sql_expr(sql, &df_schema)?;
    let expr = simplifier.simplify(simplifier.coerce(expr, &df_schema)?)?;
    create_physical_expr(
        &expr,
        &df_schema,
        &ExecutionProps::new(),
        &Default::default(),
    )
}

/// Plans a projection of `(sql, alias)` pairs against `schema`.
fn projection(exprs: &[(&str, &str)], schema: &Schema) -> DFResult<ProjectionExprs> {
    exprs
        .iter()
        .map(|(sql, alias)| Ok(ProjectionExpr::new(plan(sql, schema)?, *alias)))
        .collect::<DFResult<Vec<_>>>()
        .map(ProjectionExprs::from)
}

/// Splits `projection` with the default converter.
fn split(projection: ProjectionExprs, schema: &Schema) -> DFResult<ProcessedProjection> {
    let output_schema = projection.project_schema(schema)?;
    DefaultExpressionConverter::default().split_projection(projection, schema, &output_schema)
}

/// Formats each leftover expression with its alias.
fn leftover(processed: &ProcessedProjection) -> Vec<String> {
    processed
        .leftover_projection
        .iter()
        .map(|projection| format!("{} AS {}", projection.expr, projection.alias))
        .collect()
}

/// The names and Arrow types of the scan output.
fn scan_fields(processed: &ProcessedProjection) -> Vec<(String, DataType)> {
    processed
        .scan_reference_schema
        .fields()
        .iter()
        .map(|field| (field.name().clone(), field.data_type().clone()))
        .collect()
}

/// The scan projection that reads `columns`.
fn scan(columns: impl IntoIterator<Item = (&'static str, Expression)>) -> Expression {
    pack(columns, Nullability::NonNullable)
}

/// Field access along `path` into the struct column `s`.
fn s_field(path: &[&str]) -> Expression {
    path.iter()
        .fold(get_item("s", root()), |expr, name| get_item(*name, expr))
}

/// Field access under an unsupported function is read in the scan, once for each distinct
/// subexpression, while columns read whole keep their names and constants stay in DataFusion.
#[test]
fn test_unsupported_function_reads_fields_in_scan() -> DFResult<()> {
    let schema = schema();
    let processed = split(
        projection(
            &[
                ("id", "id"),
                ("upper(get_field(s, 'a'))", "up"),
                ("get_field(s, 'a')", "a"),
                (
                    "concat(get_field(s, 'a'), get_field(s, 'inner', 'c'))",
                    "ac",
                ),
                ("1", "one"),
            ],
            &schema,
        )?,
        &schema,
    )?;
    assert_eq!(
        processed.scan_projection,
        scan([
            ("id", get_item("id", root())),
            ("__vortex_expr_0", s_field(&["a"])),
            ("__vortex_expr_1", s_field(&["inner", "c"])),
        ])
    );
    assert_eq!(
        scan_fields(&processed),
        [
            ("id".to_owned(), DataType::Int64),
            ("__vortex_expr_0".to_owned(), DataType::Utf8),
            ("__vortex_expr_1".to_owned(), DataType::Utf8),
        ]
    );
    assert_eq!(
        leftover(&processed),
        [
            "id@0 AS id",
            "upper(__vortex_expr_0@1) AS up",
            "__vortex_expr_0@1 AS a",
            "concat(__vortex_expr_0@1, __vortex_expr_1@2) AS ac",
            "1 AS one",
        ]
    );
    Ok(())
}

/// DataFusion evaluates a projection expression on every row, so the scan evaluates a fallible
/// expression that converts whole. Under an unsupported function, DataFusion evaluates it.
#[rstest]
#[case::whole("id + 1", "__vortex_expr_0@0 AS n", "__vortex_expr_0")]
#[case::nested("abs(id + 1)", "abs(id@0 + 1) AS n", "id")]
fn test_fallible_expression_is_pushed_only_whole(
    #[case] sql: &str,
    #[case] expected: &str,
    #[case] column: &str,
) -> DFResult<()> {
    let schema = schema();
    let processed = split(
        projection(&[(sql, "n"), ("upper(get_field(s, 'a'))", "up")], &schema)?,
        &schema,
    )?;
    assert_eq!(leftover(&processed)[0], expected);
    assert_eq!(scan_fields(&processed)[0].0, column);
    Ok(())
}

/// The scan supplies the row index, which DataFusion cannot evaluate, also under an unsupported
/// expression.
#[test]
fn test_row_index_under_unsupported_expression() -> DFResult<()> {
    let schema = schema();
    let parity: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        row_index(),
        DFOperator::Modulo,
        Arc::new(df_expr::Literal::new(ScalarValue::UInt64(Some(2)))),
    ));
    let processed = split(
        ProjectionExprs::from(vec![
            ProjectionExpr::new(parity, "parity"),
            ProjectionExpr::new(row_index(), "row"),
        ]),
        &schema,
    )?;
    assert_eq!(
        processed.scan_projection,
        scan([("__vortex_expr_0", row_idx::row_idx())])
    );
    assert!(!processed.scan_reference_schema.field(0).is_nullable());
    assert_eq!(
        leftover(&processed),
        [
            "__vortex_expr_0@0 % 2 AS parity",
            "__vortex_expr_0@0 AS row"
        ]
    );
    Ok(())
}

/// Generated scan names skip the names of input columns, which columns read whole keep.
#[test]
fn test_generated_names_avoid_input_columns() -> DFResult<()> {
    let mut fields = schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "__vortex_expr_0",
        DataType::Utf8,
        true,
    )));
    let schema = Schema::new(fields);
    let processed = split(
        projection(
            &[
                ("upper(get_field(s, 'a'))", "up"),
                ("upper(__vortex_expr_0)", "taken"),
            ],
            &schema,
        )?,
        &schema,
    )?;
    assert_eq!(
        scan_fields(&processed),
        [
            ("__vortex_expr_1".to_owned(), DataType::Utf8),
            ("__vortex_expr_0".to_owned(), DataType::Utf8),
        ]
    );
    assert_eq!(
        leftover(&processed),
        [
            "upper(__vortex_expr_1@0) AS up",
            "upper(__vortex_expr_0@1) AS taken",
        ]
    );
    Ok(())
}

/// Fails to convert field access, and converts other expressions as the default converter does.
struct FailingFieldAccess(DefaultExpressionConverter);

impl ExpressionConverter for FailingFieldAccess {
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        let field_access = expr
            .downcast_ref::<ScalarFunctionExpr>()
            .is_some_and(|function| {
                ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(function).is_some()
            });
        if field_access {
            return Err(exec_datafusion_err!("field access fails to convert"));
        }
        self.0.try_convert(expr, schema)
    }
}

/// An expression whose conversion fails stays in DataFusion instead of failing the split, as a
/// subexpression and as a whole projection expression, whether or not another expression of the
/// projection is unsupported.
#[rstest]
#[case::subexpression(
    &[("upper(get_field(s, 'a'))", "up")],
    &["upper(get_field(s@0, a)) AS up"]
)]
#[case::after_unsupported(
    &[("upper(get_field(s, 'a'))", "up"), ("get_field(s, 'b')", "b")],
    &["upper(get_field(s@0, a)) AS up", "get_field(s@0, b) AS b"]
)]
#[case::before_unsupported(
    &[("get_field(s, 'b')", "b"), ("upper(get_field(s, 'a'))", "up")],
    &["get_field(s@0, b) AS b", "upper(get_field(s@0, a)) AS up"]
)]
#[case::alone(&[("get_field(s, 'b')", "b")], &["get_field(s@0, b) AS b"])]
fn test_failed_conversion_stays_in_datafusion(
    #[case] exprs: &[(&str, &str)],
    #[case] expected: &[&str],
) -> DFResult<()> {
    let schema = schema();
    let projection = projection(exprs, &schema)?;
    let output_schema = projection.project_schema(&schema)?;
    let processed = FailingFieldAccess(DefaultExpressionConverter::default()).split_projection(
        projection,
        &schema,
        &output_schema,
    )?;
    assert_eq!(
        processed.scan_projection,
        scan([("s", get_item("s", root()))])
    );
    assert_eq!(leftover(&processed), expected);
    Ok(())
}

/// `{name: utf8?, m: map<i64, utf8?>?, k: i64?}`
fn map_schema() -> Schema {
    let entries = Field::new_struct(
        "entries",
        vec![
            Field::new("key", DataType::Int64, false),
            Field::new("value", DataType::Utf8, true),
        ],
        false,
    );
    Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("m", DataType::Map(Arc::new(entries), false), true),
        Field::new("k", DataType::Int64, true),
    ])
}

/// Map access, whose keys need not be strings, is unsupported rather than a failed conversion, so
/// it stays in DataFusion next to an unsupported expression, in either order.
#[rstest]
#[case::literal_key_after_unsupported(
    &[("upper(name)", "u"), ("get_field(m, 1)", "v")],
    &["upper(name@0) AS u", "get_field(m@1, 1) AS v"]
)]
#[case::literal_key_before_unsupported(
    &[("get_field(m, 1)", "v"), ("upper(name)", "u")],
    &["get_field(m@0, 1) AS v", "upper(name@1) AS u"]
)]
#[case::column_key(&[("upper(get_field(m, k))", "u")], &["upper(get_field(m@0, k@1)) AS u"])]
fn test_map_access_stays_in_datafusion(
    #[case] exprs: &[(&str, &str)],
    #[case] expected: &[&str],
) -> DFResult<()> {
    let schema = map_schema();
    let projection = projection(exprs, &schema)?;
    for projection in projection.iter() {
        assert!(
            DefaultExpressionConverter::default()
                .try_convert(&projection.expr, &schema)?
                .is_none()
        );
    }
    let processed = split(projection, &schema)?;
    assert_eq!(leftover(&processed), expected);
    Ok(())
}

/// DataFusion numbers lambda parameters after the input columns, so a lambda that stays in
/// DataFusion keeps the raw projection.
#[test]
fn test_lambda_in_datafusion_reads_raw_columns() -> DFResult<()> {
    let schema = schema();
    let DataType::List(element) = schema.field(2).data_type() else {
        return Err(exec_datafusion_err!("tags must be a list"));
    };
    let variable: Arc<dyn PhysicalExpr> = Arc::new(df_expr::LambdaVariable::new(
        schema.fields().len(),
        Arc::new(element.as_ref().clone().with_name("x")),
    ));
    let body = Arc::new(df_expr::BinaryExpr::new(
        variable,
        DFOperator::Modulo,
        Arc::new(df_expr::Literal::new(ScalarValue::Int64(Some(2)))),
    ));
    let transform: Arc<dyn PhysicalExpr> = Arc::new(HigherOrderFunctionExpr::try_new_with_schema(
        Arc::new(HigherOrderUDF::new_from_impl(ArrayTransform::new())),
        vec![
            Arc::new(df_expr::Column::new("tags", 2)),
            Arc::new(df_expr::LambdaExpr::try_new(vec!["x".to_owned()], body)?),
        ],
        &schema,
        Arc::new(ConfigOptions::new()),
    )?);
    let source = ProjectionExprs::from(vec![
        ProjectionExpr::new(transform, "odd"),
        ProjectionExpr::new(plan("upper(get_field(s, 'a'))", &schema)?, "up"),
    ]);
    let processed = split(source.clone(), &schema)?;
    assert_eq!(
        processed.scan_projection,
        scan([
            ("s", get_item("s", root())),
            ("tags", get_item("tags", root()))
        ])
    );
    assert_eq!(processed.leftover_projection, source);
    Ok(())
}

/// Residual filters and the projection after them share one scan output. DataFusion evaluates
/// both on some rows only, so the scan evaluates only infallible subexpressions.
#[test]
fn test_filtered_projection_pushes_infallible_subexpressions() -> DFResult<()> {
    let schema = schema();
    let filters = vec![plan("get_field(s, 'b') % 2 = 0", &schema)?];
    let (processed, filters) = split_filtered_projection(
        &DefaultExpressionConverter::default(),
        projection(
            &[
                ("id + 1", "next"),
                ("get_field(s, 'b')", "b"),
                ("upper(get_field(s, 'a'))", "up"),
            ],
            &schema,
        )?,
        filters,
        &schema,
    )?;
    assert_eq!(
        processed.scan_projection,
        scan([
            ("__vortex_expr_0", s_field(&["b"])),
            ("id", get_item("id", root())),
            ("__vortex_expr_1", s_field(&["a"])),
        ])
    );
    assert_eq!(
        filters.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["__vortex_expr_0@0 % 2 = 0"]
    );
    assert_eq!(
        leftover(&processed),
        [
            "id@1 + 1 AS next",
            "__vortex_expr_0@0 AS b",
            "upper(__vortex_expr_1@2) AS up",
        ]
    );
    Ok(())
}

/// A residual filter can read the row index, which DataFusion cannot evaluate.
#[test]
fn test_residual_filter_reads_row_index_from_scan() -> DFResult<()> {
    let schema = schema();
    let even: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::BinaryExpr::new(
            row_index(),
            DFOperator::Modulo,
            Arc::new(df_expr::Literal::new(ScalarValue::UInt64(Some(2)))),
        )),
        DFOperator::Eq,
        Arc::new(df_expr::Literal::new(ScalarValue::UInt64(Some(0)))),
    ));
    let (processed, filters) = split_filtered_projection(
        &DefaultExpressionConverter::default(),
        projection(&[("id", "id")], &schema)?,
        vec![even],
        &schema,
    )?;
    assert_eq!(
        processed.scan_projection,
        scan([
            ("__vortex_expr_0", row_idx::row_idx()),
            ("id", get_item("id", root())),
        ])
    );
    assert_eq!(
        filters.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["__vortex_expr_0@0 % 2 = 0"]
    );
    assert_eq!(leftover(&processed), ["id@1 AS id"]);
    Ok(())
}

/// Without projection pushdown, residual filters and the projection read raw columns.
#[test]
fn test_raw_filtered_projection_reads_filter_columns() -> DFResult<()> {
    let schema = schema();
    let filters = vec![plan("get_field(s, 'b') % 2 = 0", &schema)?];
    let source = projection(&[("id", "id")], &schema)?;
    let (processed, rewritten) = raw_filtered_projection(source.clone(), filters.clone(), &schema)?;
    assert_eq!(
        processed.scan_projection,
        scan([("id", get_item("id", root())), ("s", get_item("s", root()))])
    );
    assert_eq!(processed.leftover_projection, source);
    assert_eq!(rewritten, filters);
    Ok(())
}
