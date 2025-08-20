// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};
use datafusion_physical_plan::expressions as df_expr;
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::expr::{BinaryExpr, ExprRef, LikeExpr, Operator, and, get_item, lit, root};
use vortex::scalar::Scalar;

use crate::convert::{FromDataFusion, TryFromDataFusion};

const SUPPORTED_BINARY_OPS: &[DFOperator] = &[
    DFOperator::Eq,
    DFOperator::NotEq,
    DFOperator::Gt,
    DFOperator::GtEq,
    DFOperator::Lt,
    DFOperator::LtEq,
];

/// Tries to convert the expressions into a vortex conjunction. Will return Ok(None) iff the input conjunction is empty.
pub(crate) fn make_vortex_predicate(
    predicate: &[&Arc<dyn PhysicalExpr>],
) -> VortexResult<Option<ExprRef>> {
    let exprs = predicate
        .iter()
        .map(|e| ExprRef::try_from_df(e.as_ref()))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(exprs.into_iter().reduce(and))
}

// TODO(joe): Don't return an error when we have an unsupported node, bubble up "TRUE" as in keep
//  for that node, up to any `and` or `or` node.
impl TryFromDataFusion<dyn PhysicalExpr> for ExprRef {
    fn try_from_df(df: &dyn PhysicalExpr) -> VortexResult<Self> {
        if let Some(binary_expr) = df.as_any().downcast_ref::<df_expr::BinaryExpr>() {
            let left = ExprRef::try_from_df(binary_expr.left().as_ref())?;
            let right = ExprRef::try_from_df(binary_expr.right().as_ref())?;
            let operator = Operator::try_from_df(binary_expr.op())?;

            return Ok(BinaryExpr::new_expr(left, operator, right));
        }

        if let Some(col_expr) = df.as_any().downcast_ref::<df_expr::Column>() {
            return Ok(get_item(col_expr.name().to_owned(), root()));
        }

        if let Some(like) = df.as_any().downcast_ref::<df_expr::LikeExpr>() {
            let child = ExprRef::try_from_df(like.expr().as_ref())?;
            let pattern = ExprRef::try_from_df(like.pattern().as_ref())?;
            return Ok(LikeExpr::new_expr(
                child,
                pattern,
                like.negated(),
                like.case_insensitive(),
            ));
        }

        if let Some(literal) = df.as_any().downcast_ref::<df_expr::Literal>() {
            let value = Scalar::from_df(literal.value());
            return Ok(lit(value));
        }

        vortex_bail!("Couldn't convert DataFusion physical {df} expression to a vortex expression")
    }
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
            | DFOperator::Plus
            | DFOperator::Minus
            | DFOperator::Multiply
            | DFOperator::Divide
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
                Err(vortex_err!("Unsupported datafusion operator {value}"))
            }
        }
    }
}

pub(crate) fn can_be_pushed_down(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    let expr = expr.as_any();
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
    } else {
        log::debug!("DataFusion expression can't be pushed down: {expr:?}");
        false
    }
}

fn can_binary_be_pushed_down(binary: &df_expr::BinaryExpr, schema: &Schema) -> bool {
    let is_op_supported =
        binary.op().is_logic_operator() || SUPPORTED_BINARY_OPS.contains(binary.op());
    is_op_supported
        && can_be_pushed_down(binary.left(), schema)
        && can_be_pushed_down(binary.right(), schema)
}

fn supported_data_types(dt: &DataType) -> bool {
    use DataType::*;
    let is_supported = dt.is_null()
        || dt.is_numeric()
        || matches!(
            dt,
            Boolean
                | Utf8
                | Utf8View
                | Binary
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::expressions as df_expr;
    use rstest::rstest;
    use vortex::expr::{BinaryVTable, ExprRef, GetItemVTable, LikeVTable, LiteralVTable, Operator};

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
    fn test_operator_conversion_supported(
        #[case] df_op: DFOperator,
        #[case] expected_vortex_op: Operator,
    ) {
        let result = Operator::try_from_df(&df_op).unwrap();
        // We can't directly compare operators, so let's check they convert successfully
        // and have the expected behavior by converting back or through other means
        match (&result, &expected_vortex_op) {
            (Operator::Eq, Operator::Eq) => (),
            (Operator::NotEq, Operator::NotEq) => (),
            (Operator::Lt, Operator::Lt) => (),
            (Operator::Lte, Operator::Lte) => (),
            (Operator::Gt, Operator::Gt) => (),
            (Operator::Gte, Operator::Gte) => (),
            (Operator::And, Operator::And) => (),
            (Operator::Or, Operator::Or) => (),
            _ => panic!(
                "Operator conversion mismatch: expected {:?}, got {:?}",
                expected_vortex_op, result
            ),
        }
    }

    #[rstest]
    #[case::plus(DFOperator::Plus)]
    #[case::minus(DFOperator::Minus)]
    #[case::multiply(DFOperator::Multiply)]
    #[case::divide(DFOperator::Divide)]
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
        let result = ExprRef::try_from_df(&col_expr).unwrap();

        // Verify it's a column reference (get_item expression)
        // We can't easily inspect the internal structure, but we can verify it converts without error
        assert!(result.is::<GetItemVTable>());
    }

    #[test]
    fn test_expr_from_df_literal() {
        let literal_expr = df_expr::Literal::new(ScalarValue::Int32(Some(42)));
        let result = ExprRef::try_from_df(&literal_expr).unwrap();

        // Verify it's a literal expression
        assert!(result.is::<LiteralVTable>());
    }

    #[test]
    fn test_expr_from_df_binary() {
        let left = Arc::new(df_expr::Column::new("left", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = df_expr::BinaryExpr::new(left, DFOperator::Eq, right);

        let result = ExprRef::try_from_df(&binary_expr).unwrap();

        // Verify it's a binary expression
        assert!(result.is::<BinaryVTable>());
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

        let result = ExprRef::try_from_df(&like_expr).unwrap();

        // Verify it's a like expression
        assert!(dbg!(result).is::<LikeVTable>());
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
        let binary_expr = Arc::new(df_expr::BinaryExpr::new(left, DFOperator::Plus, right))
            as Arc<dyn PhysicalExpr>;

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
}
