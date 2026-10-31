// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::CaseExpr;
use datafusion_physical_expr::expressions::CastExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_common::HashMap;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::binary;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::transform::replace;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::operators::Operator;

use crate::convert::ExpressionConverter;
use crate::convert::exprs::is_infallible;

/// A file's value range of each column, used to drop dynamic bounds the file already satisfies.
pub(crate) type ColumnRanges = HashMap<String, (f64, f64)>;

/// A bound whose comparison keeps at least this share of a file's value range constrains too
/// little to be worth evaluating per row, and prunes too few zones to be worth proving.
const UNINFORMATIVE_SHARE: f64 = 0.9;

/// Reads the value ranges of `columns` from a file's statistics, laid out by `file_schema`.
pub(crate) fn column_ranges<'a>(
    statistics: &Statistics,
    file_schema: &arrow_schema::Schema,
    columns: impl IntoIterator<Item = &'a str>,
) -> ColumnRanges {
    columns
        .into_iter()
        .filter_map(|name| {
            let stats = statistics
                .column_statistics
                .get(file_schema.index_of(name).ok()?)?;
            let min = scalar_to_f64(stats.min_value.get_value()?)?;
            let max = scalar_to_f64(stats.max_value.get_value()?)?;
            Some((name.to_string(), (min, max)))
        })
        .collect()
}

pub(crate) fn convert_dynamic_filter(
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
    ranges: Option<Arc<ColumnRanges>>,
) -> Expression {
    // A converter can retain a comparison whose producer guarantees a fixed shape.
    if let Ok(Some(converted)) = converter.try_convert(&expression, &schema)
        && is_infallible(&converted)
    {
        return converted;
    }

    // AND/OR preserve a conservative true fallback for unsupported children; NOT does not.
    if let Some(expr) = expression.downcast_ref::<BinaryExpr>() {
        let operator = match expr.op() {
            DFOperator::And => Some(Operator::And),
            DFOperator::Or => Some(Operator::Or),
            _ => None,
        };
        if let Some(operator) = operator {
            return binary(
                operator,
                convert_dynamic_filter(
                    Arc::clone(expr.left()),
                    Arc::clone(&schema),
                    Arc::clone(&converter),
                    ranges.clone(),
                ),
                convert_dynamic_filter(Arc::clone(expr.right()), schema, converter, ranges),
            );
        }
    }
    if !DynamicFilterTracking::classify(&expression).contains_dynamic_filter() {
        return lit(true);
    }

    // A DataFusion dynamic filter may replace its entire body, regardless of its current shape.
    let mut columns = collect_columns(&expression).into_iter().collect::<Vec<_>>();
    columns.sort_unstable_by_key(|column| column.index());
    let inputs = columns
        .iter()
        .map(|column| get_item(column.name(), root()))
        .collect::<Vec<_>>();
    let scope_projection = pack(
        columns
            .iter()
            .enumerate()
            .map(|(idx, column)| (column.name(), get_item(input_field_name(idx), root()))),
        Nullability::NonNullable,
    );
    dynamic_expression(
        Arc::new(DataFusionDynamicFilter {
            expression,
            schema,
            scope_projection,
            converter,
            ranges,
        }),
        inputs,
    )
}

struct DataFusionDynamicFilter {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    scope_projection: Expression,
    converter: Arc<dyn ExpressionConverter>,
    ranges: Option<Arc<ColumnRanges>>,
}

impl fmt::Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}

impl fmt::Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.expression)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let converted = snapshot_physical_expr(Arc::clone(&self.expression))
            .ok()
            .and_then(|snapshot| {
                relax_snapshot(
                    &snapshot,
                    &self.schema,
                    self.converter.as_ref(),
                    self.ranges.as_deref(),
                )
            });
        // The producing DataFusion operator retains exact enforcement when conversion fails.
        let expression = converted.map_or_else(
            || lit(true),
            |expr| replace(expr, &root(), self.scope_projection.clone()),
        );
        expression.bind(scope)
    }
}

/// Whether the current snapshot of `expression` keeps any constraint Vortex can evaluate.
pub(crate) fn dynamic_filter_is_convertible(
    expression: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    converter: &dyn ExpressionConverter,
    ranges: Option<&ColumnRanges>,
) -> bool {
    snapshot_physical_expr(Arc::clone(expression))
        .ok()
        .and_then(|snapshot| relax_snapshot(&snapshot, schema, converter, ranges))
        .is_some()
}

/// Converts a dynamic filter's snapshot into a Vortex expression that every row the snapshot
/// keeps also satisfies, or `None` when nothing convertible constrains the rows.
///
/// The producing operator still enforces the snapshot exactly, so the scan may keep more rows
/// than the snapshot does but never fewer. A conjunct Vortex cannot evaluate, such as a hash
/// join's `hash_lookup`, is dropped from its conjunction instead of discarding the bounds beside
/// it, and a partitioned join's `CASE hash_repartition % n WHEN i THEN bounds_i ... END` keeps
/// the union of its branches' bounds. With `ranges`, a bound comparison that keeps nearly all
/// of the file's value range is dropped too.
fn relax_snapshot(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    converter: &dyn ExpressionConverter,
    ranges: Option<&ColumnRanges>,
) -> Option<Expression> {
    let binary_expr = expr.downcast_ref::<BinaryExpr>();
    let relax = |child| relax_snapshot(child, schema, converter, ranges);
    if let Some(binary_expr) = binary_expr
        && *binary_expr.op() == DFOperator::And
    {
        return match (relax(binary_expr.left()), relax(binary_expr.right())) {
            (Some(left), Some(right)) => Some(binary(Operator::And, left, right)),
            (Some(only), None) | (None, Some(only)) => Some(only),
            (None, None) => None,
        };
    }
    if let (Some(binary_expr), Some(ranges)) = (binary_expr, ranges)
        && bound_share(binary_expr, ranges).is_some_and(|share| share >= UNINFORMATIVE_SHARE)
    {
        return None;
    }
    if let Ok(Some(converted)) = converter.try_convert(expr, schema)
        && is_infallible(&converted)
    {
        return Some(converted);
    }
    if let Some(binary_expr) = binary_expr
        && *binary_expr.op() == DFOperator::Or
    {
        return Some(binary(
            Operator::Or,
            relax(binary_expr.left())?,
            relax(binary_expr.right())?,
        ));
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        // A row the CASE keeps is kept by the branch it takes, so it satisfies the union of the
        // branches' results whichever branch its operand selects.
        let mut branches = Vec::with_capacity(case.when_then_expr().len() + 1);
        for (_, then) in case.when_then_expr() {
            if !is_false_literal(then) {
                branches.push(relax(then)?);
            }
        }
        if let Some(otherwise) = case.else_expr()
            && !is_false_literal(otherwise)
        {
            branches.push(relax(otherwise)?);
        }
        return Some(
            branches
                .into_iter()
                .reduce(|left, right| binary(Operator::Or, left, right))
                .unwrap_or_else(|| lit(false)),
        );
    }
    None
}

fn is_false_literal(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<Literal>().is_some_and(|literal| {
        matches!(
            literal.value(),
            ScalarValue::Boolean(Some(false) | None)
        )
    })
}

/// The share of a column's file value range that a `column <op> literal` bound keeps, if the
/// expression is such a bound on a column with a known numeric range.
fn bound_share(expr: &BinaryExpr, ranges: &ColumnRanges) -> Option<f64> {
    let (column, literal, op) = match (bound_column(expr.left()), bound_literal(expr.right())) {
        (Some(column), Some(literal)) => (column, literal, *expr.op()),
        _ => (
            bound_column(expr.right())?,
            bound_literal(expr.left())?,
            expr.op().swap()?,
        ),
    };
    let &(min, max) = ranges.get(column.name())?;
    let value = scalar_to_f64(literal.value())?;
    let span = max - min;
    let kept = match op {
        DFOperator::Gt | DFOperator::GtEq => max - value,
        DFOperator::Lt | DFOperator::LtEq => value - min,
        _ => return None,
    };
    Some(if span > 0.0 {
        (kept / span).clamp(0.0, 1.0)
    } else if kept >= 0.0 {
        1.0
    } else {
        0.0
    })
}

fn bound_column(expr: &Arc<dyn PhysicalExpr>) -> Option<&Column> {
    expr.downcast_ref::<Column>().or_else(|| {
        expr.downcast_ref::<CastExpr>()
            .and_then(|cast| cast.expr().downcast_ref::<Column>())
    })
}

fn bound_literal(expr: &Arc<dyn PhysicalExpr>) -> Option<&Literal> {
    expr.downcast_ref::<Literal>()
}

/// A numeric scalar's position on a line, for comparing bounds with a file's value range.
fn scalar_to_f64(value: &ScalarValue) -> Option<f64> {
    #[expect(clippy::cast_precision_loss, reason = "a share of a range tolerates rounding")]
    Some(match value {
        ScalarValue::Int8(Some(v)) => f64::from(*v),
        ScalarValue::Int16(Some(v)) => f64::from(*v),
        ScalarValue::Int32(Some(v)) | ScalarValue::Date32(Some(v)) => f64::from(*v),
        ScalarValue::Int64(Some(v))
        | ScalarValue::Date64(Some(v))
        | ScalarValue::TimestampSecond(Some(v), _)
        | ScalarValue::TimestampMillisecond(Some(v), _)
        | ScalarValue::TimestampMicrosecond(Some(v), _)
        | ScalarValue::TimestampNanosecond(Some(v), _) => *v as f64,
        ScalarValue::UInt8(Some(v)) => f64::from(*v),
        ScalarValue::UInt16(Some(v)) => f64::from(*v),
        ScalarValue::UInt32(Some(v)) => f64::from(*v),
        ScalarValue::UInt64(Some(v)) => *v as f64,
        ScalarValue::Float32(Some(v)) => f64::from(*v),
        ScalarValue::Float64(Some(v)) => *v,
        ScalarValue::Decimal128(Some(v), _, scale) => *v as f64 / 10f64.powi(i32::from(*scale)),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::lit as df_lit;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::session::VortexSession;

    use super::ColumnRanges;
    use super::relax_snapshot;
    use crate::convert::exprs::DefaultExpressionConverter;

    fn compare(op: DFOperator, value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("value", 0)),
            op,
            df_lit(value),
        ))
    }

    fn and(left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(left, DFOperator::And, right))
    }

    /// `value % 2 = 1`, which the converter does not support.
    fn unsupported() -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("value", 0)),
                DFOperator::Modulo,
                df_lit(2_i32),
            )),
            DFOperator::Eq,
            df_lit(1_i32),
        ))
    }

    #[rstest]
    // Bounds spanning the file's whole range constrain nothing.
    #[case(1, 99, false)]
    #[case(-5, 200, false)]
    // A bound keeping a fifth of the range is kept.
    #[case(10, 30, true)]
    // One informative side is enough.
    #[case(0, 50, true)]
    fn bounds_covering_the_file_range_are_dropped(
        #[case] low: i32,
        #[case] high: i32,
        #[case] kept: bool,
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let converter = DefaultExpressionConverter::new(VortexSession::default());
        let ranges = ColumnRanges::from_iter([("value".to_string(), (0.0, 100.0))]);
        let snapshot = and(
            and(compare(DFOperator::GtEq, low), compare(DFOperator::LtEq, high)),
            unsupported(),
        );
        let relaxed = relax_snapshot(&snapshot, &schema, &converter, Some(&ranges));
        assert_eq!(relaxed.is_some(), kept);
        // Without a file range every convertible bound is kept.
        assert!(relax_snapshot(&snapshot, &schema, &converter, None).is_some());
    }

    #[test]
    fn unsupported_snapshot_constrains_nothing() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let converter = DefaultExpressionConverter::new(VortexSession::default());
        assert!(relax_snapshot(&unsupported(), &schema, &converter, None).is_none());
    }
}
