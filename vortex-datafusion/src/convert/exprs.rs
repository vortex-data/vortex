// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion_expr::Operator as DFOperator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_expr_common::physical_expr::{PhysicalExprRef, is_dynamic_physical_expr};
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::compute::LikeOptions;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::expr::{
    Binary, Expression, Like, Operator, VTableExt, and, cast, get_item, is_null, list_contains,
    lit, not, root,
};
use vortex::scalar::Scalar;

use crate::convert::{FromDataFusion, TryFromDataFusion};

/// Tries to convert the expressions into a vortex conjunction. Will return Ok(None) iff the input conjunction is empty.
pub(crate) fn make_vortex_predicate(
    predicate: &[&Arc<dyn PhysicalExpr>],
) -> VortexResult<Option<Expression>> {
    let exprs = predicate
        .iter()
        .map(|e| Expression::try_from_df(e.as_ref()))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(exprs.into_iter().reduce(and))
}

// TODO(joe): Don't return an error when we have an unsupported node, bubble up "TRUE" as in keep
//  for that node, up to any `and` or `or` node.
impl TryFromDataFusion<dyn PhysicalExpr> for Expression {
    fn try_from_df(df: &dyn PhysicalExpr) -> VortexResult<Self> {
        if let Some(binary_expr) = df.as_any().downcast_ref::<df_expr::BinaryExpr>() {
            let left = Expression::try_from_df(binary_expr.left().as_ref())?;
            let right = Expression::try_from_df(binary_expr.right().as_ref())?;
            let operator = Operator::try_from_df(binary_expr.op())?;

            return Ok(Binary.new_expr(operator, [left, right]));
        }

        if let Some(col_expr) = df.as_any().downcast_ref::<df_expr::Column>() {
            return Ok(get_item(col_expr.name().to_owned(), root()));
        }

        if let Some(like) = df.as_any().downcast_ref::<df_expr::LikeExpr>() {
            let child = Expression::try_from_df(like.expr().as_ref())?;
            let pattern = Expression::try_from_df(like.pattern().as_ref())?;
            return Ok(Like.new_expr(
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
                [child, pattern],
            ));
        }

        if let Some(literal) = df.as_any().downcast_ref::<df_expr::Literal>() {
            let value = Scalar::from_df(literal.value());
            return Ok(lit(value));
        }

        if let Some(cast_expr) = df.as_any().downcast_ref::<df_expr::CastExpr>() {
            let cast_dtype = DType::from_arrow((cast_expr.cast_type(), Nullability::Nullable));
            let child = Expression::try_from_df(cast_expr.expr().as_ref())?;
            return Ok(cast(child, cast_dtype));
        }

        if let Some(is_null_expr) = df.as_any().downcast_ref::<df_expr::IsNullExpr>() {
            let arg = Expression::try_from_df(is_null_expr.arg().as_ref())?;
            return Ok(is_null(arg));
        }

        if let Some(is_not_null_expr) = df.as_any().downcast_ref::<df_expr::IsNotNullExpr>() {
            let arg = Expression::try_from_df(is_not_null_expr.arg().as_ref())?;
            return Ok(not(is_null(arg)));
        }

        if let Some(in_list) = df.as_any().downcast_ref::<df_expr::InListExpr>() {
            let value = Expression::try_from_df(in_list.expr().as_ref())?;
            let list_elements: Vec<_> = in_list
                .list()
                .iter()
                .map(|e| {
                    if let Some(lit) = e.as_any().downcast_ref::<df_expr::Literal>() {
                        Ok(Scalar::from_df(lit.value()))
                    } else {
                        Err(vortex_err!("Failed to cast sub-expression"))
                    }
                })
                .try_collect()?;

            let list = Scalar::list(
                list_elements[0].dtype().clone(),
                list_elements,
                Nullability::Nullable,
            );
            let expr = list_contains(lit(list), value);

            return Ok(if in_list.negated() { not(expr) } else { expr });
        }

        if let Some(scalar_fn) = df.as_any().downcast_ref::<ScalarFunctionExpr>() {
            return try_convert_scalar_function(scalar_fn);
        }

        vortex_bail!("Couldn't convert DataFusion physical {df} expression to a vortex expression")
    }
}

/// Attempts to convert a DataFusion ScalarFunctionExpr to a Vortex expression.
fn try_convert_scalar_function(scalar_fn: &ScalarFunctionExpr) -> VortexResult<Expression> {
    if let Some(get_field_fn) = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn) {
        let source_expr = get_field_fn
            .args()
            .first()
            .ok_or_else(|| vortex_err!("get_field missing source expression"))?
            .as_ref();
        let field_name_expr = get_field_fn
            .args()
            .get(1)
            .ok_or_else(|| vortex_err!("get_field missing field name argument"))?;
        let field_name = field_name_expr
            .as_any()
            .downcast_ref::<df_expr::Literal>()
            .ok_or_else(|| vortex_err!("get_field field name must be a literal"))?
            .value()
            .try_as_str()
            .flatten()
            .ok_or_else(|| vortex_err!("get_field field name must be a UTF-8 string"))?;
        return Ok(get_item(
            field_name.to_string(),
            Expression::try_from_df(source_expr)?,
        ));
    }

    tracing::debug!(
        function_name = scalar_fn.name(),
        "Unsupported ScalarFunctionExpr"
    );
    vortex_bail!("Unsupported ScalarFunctionExpr: {}", scalar_fn.name())
}

impl TryFromDataFusion<DFOperator> for Operator {
    fn try_from_df(value: &DFOperator) -> VortexResult<Self> {
        match value {
            DFOperator::Eq => Ok(Operator::Eq),
            DFOperator::NotEq => Ok(Operator::NotEq),
            DFOperator::Lt => Ok(Operator::Lt),
            DFOperator::LtEq => Ok(Operator::Lte),
            DFOperator::Gt => Ok(Operator::Gt),
            DFOperator::GtEq => Ok(Operator::Gte),
            DFOperator::And => Ok(Operator::And),
            DFOperator::Or => Ok(Operator::Or),
            DFOperator::Plus => Ok(Operator::Add),
            DFOperator::Minus => Ok(Operator::Sub),
            DFOperator::Multiply => Ok(Operator::Mul),
            DFOperator::Divide => Ok(Operator::Div),
            DFOperator::IsDistinctFrom
            | DFOperator::IsNotDistinctFrom
            | DFOperator::RegexMatch
            | DFOperator::RegexIMatch
            | DFOperator::RegexNotMatch
            | DFOperator::RegexNotIMatch
            | DFOperator::LikeMatch
            | DFOperator::ILikeMatch
            | DFOperator::NotLikeMatch
            | DFOperator::NotILikeMatch
            | DFOperator::BitwiseAnd
            | DFOperator::BitwiseOr
            | DFOperator::BitwiseXor
            | DFOperator::BitwiseShiftRight
            | DFOperator::BitwiseShiftLeft
            | DFOperator::StringConcat
            | DFOperator::AtArrow
            | DFOperator::ArrowAt
            | DFOperator::Modulo
            | DFOperator::Arrow
            | DFOperator::LongArrow
            | DFOperator::HashArrow
            | DFOperator::HashLongArrow
            | DFOperator::AtAt
            | DFOperator::IntegerDivide
            | DFOperator::HashMinus
            | DFOperator::AtQuestion
            | DFOperator::Question
            | DFOperator::QuestionAnd
            | DFOperator::QuestionPipe => {
                tracing::debug!(operator = %value, "Can't pushdown binary_operator operator");
                Err(vortex_err!("Unsupported datafusion operator {value}"))
            }
        }
    }
}

pub(crate) fn can_be_pushed_down(df_expr: &PhysicalExprRef, schema: &Schema) -> bool {
    // We currently do not support pushdown of dynamic expressions in DF.
    // See issue: https://github.com/vortex-data/vortex/issues/4034
    if is_dynamic_physical_expr(df_expr) {
        return false;
    }

    let expr = df_expr.as_any();
    if let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() {
        can_binary_be_pushed_down(binary, schema)
    } else if let Some(col) = expr.downcast_ref::<df_expr::Column>() {
        schema
            .field_with_name(col.name())
            .ok()
            .is_some_and(|field| supported_data_types(field.data_type()))
    } else if let Some(like) = expr.downcast_ref::<df_expr::LikeExpr>() {
        can_be_pushed_down(like.expr(), schema) && can_be_pushed_down(like.pattern(), schema)
    } else if let Some(lit) = expr.downcast_ref::<df_expr::Literal>() {
        supported_data_types(&lit.value().data_type())
    } else if let Some(cast) = expr.downcast_ref::<df_expr::CastExpr>() {
        supported_data_types(cast.cast_type()) && can_be_pushed_down(cast.expr(), schema)
    } else if let Some(is_null) = expr.downcast_ref::<df_expr::IsNullExpr>() {
        can_be_pushed_down(is_null.arg(), schema)
    } else if let Some(is_not_null) = expr.downcast_ref::<df_expr::IsNotNullExpr>() {
        can_be_pushed_down(is_not_null.arg(), schema)
    } else if let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>() {
        can_be_pushed_down(in_list.expr(), schema)
            && in_list.list().iter().all(|e| can_be_pushed_down(e, schema))
    } else if let Some(scalar_fn) = expr.downcast_ref::<ScalarFunctionExpr>() {
        can_scalar_fn_be_pushed_down(scalar_fn, schema)
    } else {
        tracing::debug!(%df_expr, "DataFusion expression can't be pushed down");
        false
    }
}

fn can_binary_be_pushed_down(binary: &df_expr::BinaryExpr, schema: &Schema) -> bool {
    let is_op_supported = Operator::try_from_df(binary.op()).is_ok();
    is_op_supported
        && can_be_pushed_down(binary.left(), schema)
        && can_be_pushed_down(binary.right(), schema)
}

fn supported_data_types(dt: &DataType) -> bool {
    use DataType::*;

    // For dictionary types, check if the value type is supported.
    if let Dictionary(_, value_type) = dt {
        return supported_data_types(value_type.as_ref());
    }

    let is_supported = dt.is_null()
        || dt.is_numeric()
        || matches!(
            dt,
            Boolean
                | Utf8
                | LargeUtf8
                | Utf8View
                | Binary
                | LargeBinary
                | BinaryView
                | Date32
                | Date64
                | Timestamp(_, _)
                | Time32(_)
                | Time64(_)
        );

    if !is_supported {
        log::debug!("DataFusion data type {dt:?} is not supported");
    }

    is_supported
}

/// Checks if a GetField scalar function can be pushed down.
fn can_scalar_fn_be_pushed_down(scalar_fn: &ScalarFunctionExpr, schema: &Schema) -> bool {
    let Some(get_field_fn) = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn)
    else {
        // Only get_field pushdown is supported.
        return false;
    };

    let args = get_field_fn.args();
    if args.len() != 2 {
        tracing::debug!(
            "Expected 2 arguments for GetField, not pushing down {} arguments",
            args.len()
        );
        return false;
    }
    let source_expr = &args[0];
    let field_name_expr = &args[1];
    let Some(field_name) = field_name_expr
        .as_any()
        .downcast_ref::<df_expr::Literal>()
        .and_then(|lit| lit.value().try_as_str().flatten())
    else {
        return false;
    };

    let Ok(source_dt) = source_expr.data_type(schema) else {
        tracing::debug!(
            field_name = field_name,
            schema = ?schema,
            source_expr = ?source_expr,
            "Failed to get source type for GetField, not pushing down"
        );
        return false;
    };
    let DataType::Struct(fields) = source_dt else {
        tracing::debug!(
            field_name = field_name,
            schema = ?schema,
            source_expr = ?source_expr,
            "Failed to get source type as struct for GetField, not pushing down"
        );
        return false;
    };
    fields.find(field_name).is_some()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit as ArrowTimeUnit};
    use datafusion::functions::core::getfield::GetFieldFunc;
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{Operator as DFOperator, ScalarUDF};
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::expressions as df_expr;
    use insta::assert_snapshot;
    use rstest::rstest;
    use vortex::expr::{Expression, Operator};

    use super::*;

    #[rstest::fixture]
    fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
            Field::new("active", DataType::Boolean, false),
            Field::new(
                "created_at",
                DataType::Timestamp(ArrowTimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "unsupported_list",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
        ])
    }

    #[test]
    fn test_make_vortex_predicate_empty() {
        let result = make_vortex_predicate(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_make_vortex_predicate_single() {
        let col_expr = Arc::new(df_expr::Column::new("test", 0)) as Arc<dyn PhysicalExpr>;
        let result = make_vortex_predicate(&[&col_expr]).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_make_vortex_predicate_multiple() {
        let col1 = Arc::new(df_expr::Column::new("col1", 0)) as Arc<dyn PhysicalExpr>;
        let col2 = Arc::new(df_expr::Column::new("col2", 1)) as Arc<dyn PhysicalExpr>;
        let result = make_vortex_predicate(&[&col1, &col2]).unwrap();
        assert!(result.is_some());
        // Result should be an AND expression combining the two columns
    }

    #[rstest]
    #[case::eq(DFOperator::Eq, Operator::Eq)]
    #[case::not_eq(DFOperator::NotEq, Operator::NotEq)]
    #[case::lt(DFOperator::Lt, Operator::Lt)]
    #[case::lte(DFOperator::LtEq, Operator::Lte)]
    #[case::gt(DFOperator::Gt, Operator::Gt)]
    #[case::gte(DFOperator::GtEq, Operator::Gte)]
    #[case::and(DFOperator::And, Operator::And)]
    #[case::or(DFOperator::Or, Operator::Or)]
    #[case::plus(DFOperator::Plus, Operator::Add)]
    #[case::plus(DFOperator::Minus, Operator::Sub)]
    #[case::plus(DFOperator::Multiply, Operator::Mul)]
    #[case::plus(DFOperator::Divide, Operator::Div)]
    fn test_operator_conversion_supported(
        #[case] df_op: DFOperator,
        #[case] expected_vortex_op: Operator,
    ) {
        let result = Operator::try_from_df(&df_op).unwrap();
        assert_eq!(result, expected_vortex_op);
    }

    #[rstest]
    #[case::modulo(DFOperator::Modulo)]
    #[case::bitwise_and(DFOperator::BitwiseAnd)]
    #[case::regex_match(DFOperator::RegexMatch)]
    #[case::like_match(DFOperator::LikeMatch)]
    fn test_operator_conversion_unsupported(#[case] df_op: DFOperator) {
        let result = Operator::try_from_df(&df_op);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported datafusion operator")
        );
    }

    #[test]
    fn test_expr_from_df_column() {
        let col_expr = df_expr::Column::new("test_column", 0);
        let result = Expression::try_from_df(&col_expr).unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r#"
        vortex.get_item "test_column"
        └── input: vortex.root
        "#);
    }

    #[test]
    fn test_expr_from_df_literal() {
        let literal_expr = df_expr::Literal::new(ScalarValue::Int32(Some(42)));
        let result = Expression::try_from_df(&literal_expr).unwrap();

        assert_snapshot!(result.display_tree().to_string(), @"vortex.literal 42i32");
    }

    #[test]
    fn test_expr_from_df_binary() {
        let left = Arc::new(df_expr::Column::new("left", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = df_expr::BinaryExpr::new(left, DFOperator::Eq, right);

        let result = Expression::try_from_df(&binary_expr).unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r#"
        vortex.binary =
        ├── lhs: vortex.get_item "left"
        │   └── input: vortex.root
        └── rhs: vortex.literal 42i32
        "#);
    }

    #[rstest]
    #[case::like_normal(false, false)]
    #[case::like_negated(true, false)]
    #[case::like_case_insensitive(false, true)]
    #[case::like_negated_case_insensitive(true, true)]
    fn test_expr_from_df_like(#[case] negated: bool, #[case] case_insensitive: bool) {
        let expr = Arc::new(df_expr::Column::new("text_col", 0)) as Arc<dyn PhysicalExpr>;
        let pattern = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            "test%".to_string(),
        )))) as Arc<dyn PhysicalExpr>;
        let like_expr = df_expr::LikeExpr::new(negated, case_insensitive, expr, pattern);

        let result = Expression::try_from_df(&like_expr).unwrap();
        let like_expr = result.as_::<Like>();
        assert_eq!(
            like_expr.data(),
            &LikeOptions {
                negated,
                case_insensitive
            }
        );
    }

    #[rstest]
    // Supported types
    #[case::null(DataType::Null, true)]
    #[case::boolean(DataType::Boolean, true)]
    #[case::int8(DataType::Int8, true)]
    #[case::int16(DataType::Int16, true)]
    #[case::int32(DataType::Int32, true)]
    #[case::int64(DataType::Int64, true)]
    #[case::uint8(DataType::UInt8, true)]
    #[case::uint16(DataType::UInt16, true)]
    #[case::uint32(DataType::UInt32, true)]
    #[case::uint64(DataType::UInt64, true)]
    #[case::float32(DataType::Float32, true)]
    #[case::float64(DataType::Float64, true)]
    #[case::utf8(DataType::Utf8, true)]
    #[case::utf8_view(DataType::Utf8View, true)]
    #[case::binary(DataType::Binary, true)]
    #[case::binary_view(DataType::BinaryView, true)]
    #[case::date32(DataType::Date32, true)]
    #[case::date64(DataType::Date64, true)]
    #[case::timestamp_ms(DataType::Timestamp(ArrowTimeUnit::Millisecond, None), true)]
    #[case::timestamp_us(
        DataType::Timestamp(ArrowTimeUnit::Microsecond, Some(Arc::from("UTC"))),
        true
    )]
    #[case::time32_s(DataType::Time32(ArrowTimeUnit::Second), true)]
    #[case::time64_ns(DataType::Time64(ArrowTimeUnit::Nanosecond), true)]
    // Unsupported types
    #[case::list(
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        false
    )]
    #[case::struct_type(DataType::Struct(vec![Field::new("field", DataType::Int32, true)].into()), false)]
    // Dictionary types - should be supported if value type is supported
    #[case::dict_utf8(
        DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
        true
    )]
    #[case::dict_int32(
        DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Int32)),
        true
    )]
    #[case::dict_unsupported(
        DataType::Dictionary(
            Box::new(DataType::UInt32),
            Box::new(DataType::List(Arc::new(Field::new("item", DataType::Int32, true))))
        ),
        false
    )]
    fn test_supported_data_types(#[case] data_type: DataType, #[case] expected: bool) {
        assert_eq!(supported_data_types(&data_type), expected);
    }

    #[rstest]
    fn test_can_be_pushed_down_column_supported(test_schema: Schema) {
        let col_expr = Arc::new(df_expr::Column::new("id", 0)) as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down(&col_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_column_unsupported_type(test_schema: Schema) {
        let col_expr =
            Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down(&col_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_column_not_found(test_schema: Schema) {
        let col_expr = Arc::new(df_expr::Column::new("nonexistent", 99)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down(&col_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_literal_supported(test_schema: Schema) {
        let lit_expr =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down(&lit_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_literal_unsupported(test_schema: Schema) {
        // Use a simpler unsupported type - Duration is not supported
        let unsupported_literal = ScalarValue::DurationSecond(Some(42));
        let lit_expr =
            Arc::new(df_expr::Literal::new(unsupported_literal)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down(&lit_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_binary_supported(test_schema: Schema) {
        let left = Arc::new(df_expr::Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = Arc::new(df_expr::BinaryExpr::new(left, DFOperator::Eq, right))
            as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down(&binary_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_binary_unsupported_operator(test_schema: Schema) {
        let left = Arc::new(df_expr::Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = Arc::new(df_expr::BinaryExpr::new(
            left,
            DFOperator::AtQuestion,
            right,
        )) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down(&binary_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_binary_unsupported_operand(test_schema: Schema) {
        let left = Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = Arc::new(df_expr::BinaryExpr::new(left, DFOperator::Eq, right))
            as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down(&binary_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_like_supported(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("name", 1)) as Arc<dyn PhysicalExpr>;
        let pattern = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            "test%".to_string(),
        )))) as Arc<dyn PhysicalExpr>;
        let like_expr =
            Arc::new(df_expr::LikeExpr::new(false, false, expr, pattern)) as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down(&like_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_like_unsupported_operand(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;
        let pattern = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            "test%".to_string(),
        )))) as Arc<dyn PhysicalExpr>;
        let like_expr =
            Arc::new(df_expr::LikeExpr::new(false, false, expr, pattern)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down(&like_expr, &test_schema));
    }

    #[test]
    fn test_expr_from_df_get_field() {
        let struct_col = Arc::new(df_expr::Column::new("my_struct", 0)) as Arc<dyn PhysicalExpr>;
        let field_name = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            "field1".to_string(),
        )))) as Arc<dyn PhysicalExpr>;
        let get_field_expr = ScalarFunctionExpr::new(
            "get_field",
            Arc::new(ScalarUDF::from(GetFieldFunc::new())),
            vec![struct_col, field_name],
            Arc::new(Field::new("field1", DataType::Utf8, true)),
            Arc::new(ConfigOptions::new()),
        );
        let result = Expression::try_from_df(&get_field_expr).unwrap();
        assert_snapshot!(result.display_tree().to_string(), @r#"
        vortex.get_item "field1"
        └── input: vortex.get_item "my_struct"
            └── input: vortex.root
        "#);
    }

    #[rstest]
    #[case::valid_field("field1", true)]
    #[case::missing_field("nonexistent_field", false)]
    fn test_can_be_pushed_down_get_field(#[case] field_name: &str, #[case] expected: bool) {
        let struct_fields = Fields::from(vec![
            Field::new("field1", DataType::Utf8, true),
            Field::new("field2", DataType::Int32, true),
        ]);
        let schema = Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(struct_fields),
            true,
        )]);

        let struct_col = Arc::new(df_expr::Column::new("my_struct", 0)) as Arc<dyn PhysicalExpr>;
        let field_name_lit = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            field_name.to_string(),
        )))) as Arc<dyn PhysicalExpr>;

        let get_field_expr = Arc::new(ScalarFunctionExpr::new(
            "get_field",
            Arc::new(ScalarUDF::from(GetFieldFunc::new())),
            vec![struct_col, field_name_lit],
            Arc::new(Field::new(field_name, DataType::Utf8, true)),
            Arc::new(ConfigOptions::new()),
        )) as Arc<dyn PhysicalExpr>;

        assert_eq!(can_be_pushed_down(&get_field_expr, &schema), expected);
    }
}
