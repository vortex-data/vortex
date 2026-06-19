// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_expr::Operator as DFOperator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions::string::octet_length::OctetLengthFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::is_dynamic_physical_expr;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::arrow::FromArrowType;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::byte_length;
use vortex::expr::cast;
use vortex::expr::get_item;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::lit;
use vortex::expr::nested_case_when;
use vortex::expr::not;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::operators::Operator;

use crate::convert::FromDataFusion;

/// Result of splitting a projection into Vortex expressions and leftover DataFusion projections.
pub struct ProcessedProjection {
    pub scan_projection: Expression,
    pub leftover_projection: ProjectionExprs,
}

/// Tries to convert the expressions into a vortex conjunction. Will return Ok(None) iff the input conjunction is empty.
pub(crate) fn make_vortex_predicate(
    expr_convertor: &dyn ExpressionConvertor,
    predicate: &[Arc<dyn PhysicalExpr>],
) -> DFResult<Option<Expression>> {
    let exprs = predicate
        .iter()
        .map(|e| expr_convertor.convert(e.as_ref()))
        .collect::<DFResult<Vec<_>>>()?;

    Ok(and_collect(exprs))
}

/// Trait for converting DataFusion expressions to Vortex ones.
pub trait ExpressionConvertor: Send + Sync {
    /// Can an expression be pushed down given a specific schema
    fn can_be_pushed_down(&self, expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool;

    /// Try and convert a DataFusion [`PhysicalExpr`] into a Vortex [`Expression`].
    fn convert(&self, expr: &dyn PhysicalExpr) -> DFResult<Expression>;

    /// Split a projection into Vortex expressions that can be pushed down and leftover
    /// DataFusion projections that need to be evaluated after the scan.
    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> DFResult<ProcessedProjection>;

    /// Create a projection that reads only the required columns without pushing down
    /// any expressions. All projection logic is applied after the scan.
    fn no_pushdown_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        // Get all unique column indices referenced by the projection
        let column_indices = source_projection.column_indices();

        // Create scan projection that reads the required columns
        let scan_columns: Vec<(String, Expression)> = column_indices
            .into_iter()
            .map(|idx| {
                let field = input_schema.field(idx);
                let name = field.name().clone();
                (name.clone(), get_item(name, root()))
            })
            .collect();

        Ok(ProcessedProjection {
            scan_projection: pack(scan_columns, Nullability::NonNullable),
            leftover_projection: source_projection,
        })
    }
}

/// The default [`ExpressionConvertor`] implementation.
#[derive(Default)]
pub struct DefaultExpressionConvertor {}

impl DefaultExpressionConvertor {
    /// Attempts to convert DataFusion's `octet_length` function to Vortex `byte_length`.
    fn try_convert_octet_length(&self, scalar_fn: &ScalarFunctionExpr) -> DFResult<Expression> {
        let [input] = scalar_fn.args() else {
            return Err(exec_datafusion_err!(
                "octet_length requires exactly one argument"
            ));
        };

        let input = self.convert(input.as_ref())?;
        let return_dtype =
            DType::from_arrow((scalar_fn.return_type(), scalar_fn.nullable().into()));
        Ok(cast(byte_length(input), return_dtype))
    }

    /// Attempts to convert a DataFusion ScalarFunctionExpr to a Vortex expression.
    fn try_convert_scalar_function(&self, scalar_fn: &ScalarFunctionExpr) -> DFResult<Expression> {
        if let Some(octet_length_fn) =
            ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(scalar_fn)
        {
            return self.try_convert_octet_length(octet_length_fn);
        }

        if let Some(get_field_fn) = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn)
        {
            // DataFusion's GetFieldFunc flattens nested field access into a single call
            // with multiple field name arguments. For example, `outer.inner.leaf` becomes
            // get_field(Column("outer"), "inner", "leaf"). We build a chain of get_item
            // calls for each field name in the path.
            let (source_expr, field_names) = get_field_fn
                .args()
                .split_first()
                .ok_or_else(|| exec_datafusion_err!("get_field missing source expression"))?;

            let mut result = self.convert(source_expr.as_ref())?;
            for expr in field_names {
                let field_name = expr
                    .downcast_ref::<df_expr::Literal>()
                    .ok_or_else(|| exec_datafusion_err!("get_field field name must be a literal"))?
                    .value()
                    .try_as_str()
                    .flatten()
                    .ok_or_else(|| {
                        exec_datafusion_err!("get_field field name must be a UTF-8 string")
                    })?;
                result = get_item(field_name.to_string(), result);
            }
            return Ok(result);
        }

        Err(exec_datafusion_err!(
            "Unsupported ScalarFunctionExpr: {}",
            scalar_fn.name()
        ))
    }

    /// Attempts to convert a DataFusion CaseExpr to a Vortex expression.
    fn try_convert_case_expr(&self, case_expr: &df_expr::CaseExpr) -> DFResult<Expression> {
        // DataFusion CaseExpr has:
        // - expr(): Optional base expression (for "CASE expr WHEN ..." form)
        // - when_then_expr(): Vec of (when, then) pairs
        // - else_expr(): Optional else expression

        // We don't support the "CASE expr WHEN value1 THEN result1" form yet
        if case_expr.expr().is_some() {
            return Err(exec_datafusion_err!(
                "CASE expr WHEN form is not yet supported, only searched CASE is supported"
            ));
        }

        let when_then_pairs = case_expr.when_then_expr();
        if when_then_pairs.is_empty() {
            return Err(exec_datafusion_err!(
                "CASE expression must have at least one WHEN clause"
            ));
        }

        // Convert all when/then pairs to (condition, value) tuples
        let mut pairs = Vec::with_capacity(when_then_pairs.len());
        for (when_expr, then_expr) in when_then_pairs {
            let condition = self.convert(when_expr.as_ref())?;
            let value = self.convert(then_expr.as_ref())?;
            pairs.push((condition, value));
        }

        // Convert optional else expression
        let else_value = case_expr
            .else_expr()
            .map(|e| self.convert(e.as_ref()))
            .transpose()?;

        // Build a single n-ary CASE WHEN expression from DataFusion WHEN/THEN pairs
        Ok(nested_case_when(pairs, else_value))
    }
}

impl ExpressionConvertor for DefaultExpressionConvertor {
    fn can_be_pushed_down(&self, expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
        can_be_pushed_down_impl(expr, schema)
    }

    fn convert(&self, df: &dyn PhysicalExpr) -> DFResult<Expression> {
        // TODO(joe): Don't return an error when we have an unsupported node, bubble up "TRUE" as in keep
        //  for that node, up to any `and` or `or` node.
        if let Some(binary_expr) = df.downcast_ref::<df_expr::BinaryExpr>() {
            let left = self.convert(binary_expr.left().as_ref())?;
            let right = self.convert(binary_expr.right().as_ref())?;
            let operator = try_operator_from_df(binary_expr.op())?;

            return Ok(Binary.new_expr(operator, [left, right]));
        }

        if let Some(col_expr) = df.downcast_ref::<df_expr::Column>() {
            return Ok(get_item(col_expr.name().to_owned(), root()));
        }

        if let Some(like) = df.downcast_ref::<df_expr::LikeExpr>() {
            let child = self.convert(like.expr().as_ref())?;
            let pattern = self.convert(like.pattern().as_ref())?;
            return Ok(Like.new_expr(
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
                [child, pattern],
            ));
        }

        if let Some(literal) = df.downcast_ref::<df_expr::Literal>() {
            let value = Scalar::from_df(literal.value());
            return Ok(lit(value));
        }

        if let Some(cast_expr) = df.downcast_ref::<df_expr::CastExpr>() {
            let cast_dtype = DType::from_arrow(cast_expr.target_field().as_ref());
            let child = self.convert(cast_expr.expr().as_ref())?;
            return Ok(cast(child, cast_dtype));
        }

        if let Some(is_null_expr) = df.downcast_ref::<df_expr::IsNullExpr>() {
            let arg = self.convert(is_null_expr.arg().as_ref())?;
            return Ok(is_null(arg));
        }

        if let Some(is_not_null_expr) = df.downcast_ref::<df_expr::IsNotNullExpr>() {
            let arg = self.convert(is_not_null_expr.arg().as_ref())?;
            return Ok(is_not_null(arg));
        }

        if let Some(in_list) = df.downcast_ref::<df_expr::InListExpr>() {
            let value = self.convert(in_list.expr().as_ref())?;
            let list_elements: Vec<_> = in_list
                .list()
                .iter()
                .map(|e| {
                    if let Some(lit) = e.downcast_ref::<df_expr::Literal>() {
                        Ok(Scalar::from_df(lit.value()))
                    } else {
                        Err(exec_datafusion_err!("Failed to cast sub-expression"))
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

        if let Some(scalar_fn) = df.downcast_ref::<ScalarFunctionExpr>() {
            return self.try_convert_scalar_function(scalar_fn);
        }

        if let Some(case_expr) = df.downcast_ref::<df_expr::CaseExpr>() {
            return self.try_convert_case_expr(case_expr);
        }

        Err(exec_datafusion_err!(
            "Couldn't convert DataFusion physical {df} expression to a vortex expression"
        ))
    }

    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        let mut scan_projection = vec![];
        let mut leftover_projection: Vec<ProjectionExpr> = vec![];

        for projection_expr in source_projection.iter() {
            let r = projection_expr.expr.apply(|node| {
                // We only pull column children of scalar functions that we can't push into the scan.
                if let Some(scalar_fn_expr) = node.downcast_ref::<ScalarFunctionExpr>()
                    && !can_scalar_fn_be_pushed_down(scalar_fn_expr, input_schema)
                {
                    scan_projection.extend(
                        collect_columns(node)
                            .into_iter()
                            .map(|c| (c.name().to_string(), get_item(c.name(), root()))),
                    );

                    leftover_projection.push(projection_expr.clone());
                    return Ok(TreeNodeRecursion::Stop);
                }

                // DataFusion assumes different decimal types can be coerced.
                // Vortex expects a perfect match so we don't push it down.
                if let Some(binary_expr) = node.downcast_ref::<df_expr::BinaryExpr>()
                    && binary_expr.op().is_numerical_operators()
                    && binary_expr.left().data_type(input_schema)?.is_decimal()
                    && binary_expr.right().data_type(input_schema)?.is_decimal()
                {
                    scan_projection.extend(
                        collect_columns(node)
                            .into_iter()
                            .map(|c| (c.name().to_string(), get_item(c.name(), root()))),
                    );

                    leftover_projection.push(projection_expr.clone());
                    return Ok(TreeNodeRecursion::Stop);
                }

                Ok(TreeNodeRecursion::Continue)
            })?;

            // if we didn't stop early
            if matches!(r, TreeNodeRecursion::Continue) {
                scan_projection.push((
                    projection_expr.alias.clone(),
                    self.convert(projection_expr.expr.as_ref())?,
                ));
                leftover_projection.push(ProjectionExpr {
                    expr: Arc::new(df_expr::Column::new_with_schema(
                        projection_expr.alias.as_str(),
                        output_schema,
                    )?),
                    alias: projection_expr.alias.clone(),
                });
            }
        }

        Ok(ProcessedProjection {
            scan_projection: pack(scan_projection, Nullability::NonNullable),
            leftover_projection: leftover_projection.into(),
        })
    }
}

fn try_operator_from_df(value: &DFOperator) -> DFResult<Operator> {
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
        | DFOperator::QuestionPipe
        | DFOperator::Colon => {
            tracing::debug!(operator = %value, "Can't pushdown binary_operator operator");
            Err(exec_datafusion_err!(
                "Unsupported datafusion operator {value}"
            ))
        }
    }
}

fn can_be_pushed_down_impl(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    // We currently do not support pushdown of dynamic expressions in DF.
    // See issue: https://github.com/vortex-data/vortex/issues/4034
    if is_dynamic_physical_expr(expr) {
        return false;
    }

    if let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() {
        can_binary_be_pushed_down(binary, schema)
    } else if let Some(col) = expr.downcast_ref::<df_expr::Column>() {
        schema
            .field_with_name(col.name())
            .ok()
            .is_some_and(|field| supported_data_types(field.data_type()))
    } else if let Some(like) = expr.downcast_ref::<df_expr::LikeExpr>() {
        can_be_pushed_down_impl(like.expr(), schema)
            && can_be_pushed_down_impl(like.pattern(), schema)
    } else if let Some(lit) = expr.downcast_ref::<df_expr::Literal>() {
        supported_data_types(&lit.value().data_type())
    } else if let Some(cast_expr) = expr.downcast_ref::<df_expr::CastExpr>() {
        // CastExpr child must be an expression type that convert() can handle
        is_convertible_expr(cast_expr.expr())
    } else if let Some(is_null) = expr.downcast_ref::<df_expr::IsNullExpr>() {
        can_be_pushed_down_impl(is_null.arg(), schema)
    } else if let Some(is_not_null) = expr.downcast_ref::<df_expr::IsNotNullExpr>() {
        can_be_pushed_down_impl(is_not_null.arg(), schema)
    } else if let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>() {
        can_be_pushed_down_impl(in_list.expr(), schema)
            && in_list
                .list()
                .iter()
                .all(|e| can_be_pushed_down_impl(e, schema))
    } else if let Some(scalar_fn) = expr.downcast_ref::<ScalarFunctionExpr>() {
        can_scalar_fn_be_pushed_down(scalar_fn, schema)
    } else if let Some(case_expr) = expr.downcast_ref::<df_expr::CaseExpr>() {
        can_case_be_pushed_down(case_expr, schema)
    } else {
        tracing::debug!(%expr, "DataFusion expression can't be pushed down");
        false
    }
}

/// Checks if an expression type is one that convert() can handle.
/// This is less restrictive than can_be_pushed_down since it only checks
/// expression types, not data type support.
fn is_convertible_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    // Expression types that convert() handles
    expr.downcast_ref::<df_expr::BinaryExpr>().is_some()
        || expr.downcast_ref::<df_expr::Column>().is_some()
        || expr.downcast_ref::<df_expr::LikeExpr>().is_some()
        || expr.downcast_ref::<df_expr::Literal>().is_some()
        || expr
            .downcast_ref::<df_expr::CastExpr>()
            .is_some_and(|e| is_convertible_expr(e.expr()))
        || expr.downcast_ref::<df_expr::IsNullExpr>().is_some()
        || expr.downcast_ref::<df_expr::IsNotNullExpr>().is_some()
        || expr.downcast_ref::<df_expr::InListExpr>().is_some()
        || expr.downcast_ref::<ScalarFunctionExpr>().is_some_and(|sf| {
            ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(sf).is_some()
                || ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(sf).is_some()
        })
}

fn can_binary_be_pushed_down(binary: &df_expr::BinaryExpr, schema: &Schema) -> bool {
    let is_op_supported = try_operator_from_df(binary.op()).is_ok();
    is_op_supported
        && can_be_pushed_down_impl(binary.left(), schema)
        && can_be_pushed_down_impl(binary.right(), schema)
}

fn can_case_be_pushed_down(case_expr: &df_expr::CaseExpr, schema: &Schema) -> bool {
    // We only support the "searched CASE" form (CASE WHEN cond THEN result ...)
    // not the "simple CASE" form (CASE expr WHEN value THEN result ...)
    if case_expr.expr().is_some() {
        return false;
    }

    // Check all when/then pairs
    for (when_expr, then_expr) in case_expr.when_then_expr() {
        if !can_be_pushed_down_impl(when_expr, schema)
            || !can_be_pushed_down_impl(then_expr, schema)
        {
            return false;
        }
    }

    // Check the optional else clause
    if let Some(else_expr) = case_expr.else_expr()
        && !can_be_pushed_down_impl(else_expr, schema)
    {
        return false;
    }

    true
}

fn supported_data_types(dt: &DataType) -> bool {
    use DataType::*;

    // For dictionary types, check if the value type is supported.
    if let Dictionary(_, value_type) = dt {
        return supported_data_types(value_type.as_ref());
    }

    let is_supported = dt.is_null()
        || dt.is_numeric()
        || dt.is_binary()
        || dt.is_string()
        || matches!(
            dt,
            Boolean | Date32 | Date64 | Timestamp(_, _) | Time32(_) | Time64(_)
        );

    if !is_supported {
        tracing::debug!("DataFusion data type {dt:?} is not supported");
    }

    is_supported
}

/// Checks if a scalar function can be pushed down.
/// Currently GetFieldFunc and OctetLengthFunc are supported.
fn can_scalar_fn_be_pushed_down(scalar_fn: &ScalarFunctionExpr, schema: &Schema) -> bool {
    if ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn).is_some() {
        return true;
    }

    ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(scalar_fn)
        .is_some_and(|octet_length| can_octet_length_be_pushed_down(octet_length, schema))
}

fn can_octet_length_be_pushed_down(scalar_fn: &ScalarFunctionExpr, schema: &Schema) -> bool {
    let [input] = scalar_fn.args() else {
        return false;
    };

    input.data_type(schema).as_ref().is_ok_and(|data_type| {
        let dt = if let DataType::Dictionary(_, value_type) = data_type {
            value_type.as_ref()
        } else {
            data_type
        };

        dt.is_binary() || dt.is_string()
    }) && can_be_pushed_down_impl(input, schema)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::TimeUnit as ArrowTimeUnit;
    use datafusion::arrow::array::AsArray;
    use datafusion::arrow::datatypes::Int32Type;
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_expr::ScalarUDF;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::expressions as df_expr;
    use insta::assert_snapshot;
    use rstest::rstest;

    use super::*;
    use crate::common_tests::TestSessionContext;

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

    fn octet_length_expr(input: Arc<dyn PhysicalExpr>, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(ScalarUDF::from(OctetLengthFunc::new())),
                vec![input],
                schema,
                Arc::new(ConfigOptions::new()),
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_make_vortex_predicate_empty() {
        let expr_convertor = DefaultExpressionConvertor::default();
        let result = make_vortex_predicate(&expr_convertor, &[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_make_vortex_predicate_single() {
        let expr_convertor = DefaultExpressionConvertor::default();
        let col_expr = Arc::new(df_expr::Column::new("test", 0)) as Arc<dyn PhysicalExpr>;
        let result = make_vortex_predicate(&expr_convertor, &[col_expr]).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_make_vortex_predicate_multiple() {
        let expr_convertor = DefaultExpressionConvertor::default();
        let col1 = Arc::new(df_expr::Column::new("col1", 0)) as Arc<dyn PhysicalExpr>;
        let col2 = Arc::new(df_expr::Column::new("col2", 1)) as Arc<dyn PhysicalExpr>;
        let result = make_vortex_predicate(&expr_convertor, &[col1, col2]).unwrap();
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
        let result = try_operator_from_df(&df_op).unwrap();
        assert_eq!(result, expected_vortex_op);
    }

    #[rstest]
    #[case::modulo(DFOperator::Modulo)]
    #[case::bitwise_and(DFOperator::BitwiseAnd)]
    #[case::regex_match(DFOperator::RegexMatch)]
    #[case::like_match(DFOperator::LikeMatch)]
    fn test_operator_conversion_unsupported(#[case] df_op: DFOperator) {
        let result = try_operator_from_df(&df_op);
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
        let result = DefaultExpressionConvertor::default()
            .convert(&col_expr)
            .unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r"
        vortex.get_item(test_column)
        └── input: vortex.root()
        ");
    }

    #[test]
    fn test_expr_from_df_literal() {
        let literal_expr = df_expr::Literal::new(ScalarValue::Int32(Some(42)));
        let result = DefaultExpressionConvertor::default()
            .convert(&literal_expr)
            .unwrap();

        assert_snapshot!(result.display_tree().to_string(), @"vortex.literal(42i32)");
    }

    #[test]
    fn test_expr_from_df_binary() {
        let left = Arc::new(df_expr::Column::new("left", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = df_expr::BinaryExpr::new(left, DFOperator::Eq, right);

        let result = DefaultExpressionConvertor::default()
            .convert(&binary_expr)
            .unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r"
        vortex.binary(=)
        ├── lhs: vortex.get_item(left)
        │   └── input: vortex.root()
        └── rhs: vortex.literal(42i32)
        ");
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

        let result = DefaultExpressionConvertor::default()
            .convert(&like_expr)
            .unwrap();
        let like_opts = result.as_::<Like>();
        assert_eq!(
            like_opts,
            &LikeOptions {
                negated,
                case_insensitive
            }
        );
    }

    #[rstest]
    fn test_expr_from_df_octet_length(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("name", 1)) as Arc<dyn PhysicalExpr>;
        let octet_length = octet_length_expr(expr, &test_schema);

        let result = DefaultExpressionConvertor::default()
            .convert(octet_length.as_ref())
            .unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r"
        vortex.cast(i32?)
        └── input: vortex.byte_length()
            └── input: vortex.get_item(name)
                └── input: vortex.root()
        ");
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
    #[case::struct_type(DataType::Struct(vec![Field::new("field", DataType::Int32, true)].into()
    ), false)]
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

        assert!(can_be_pushed_down_impl(&col_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_column_unsupported_type(test_schema: Schema) {
        let col_expr =
            Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&col_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_column_not_found(test_schema: Schema) {
        let col_expr = Arc::new(df_expr::Column::new("nonexistent", 99)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&col_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_literal_supported(test_schema: Schema) {
        let lit_expr =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down_impl(&lit_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_literal_unsupported(test_schema: Schema) {
        // Use a simpler unsupported type - Duration is not supported
        let unsupported_literal = ScalarValue::DurationSecond(Some(42));
        let lit_expr =
            Arc::new(df_expr::Literal::new(unsupported_literal)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&lit_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_binary_supported(test_schema: Schema) {
        let left = Arc::new(df_expr::Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = Arc::new(df_expr::BinaryExpr::new(left, DFOperator::Eq, right))
            as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down_impl(&binary_expr, &test_schema));
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

        assert!(!can_be_pushed_down_impl(&binary_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_binary_unsupported_operand(test_schema: Schema) {
        let left = Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = Arc::new(df_expr::BinaryExpr::new(left, DFOperator::Eq, right))
            as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&binary_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_like_supported(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("name", 1)) as Arc<dyn PhysicalExpr>;
        let pattern = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            "test%".to_string(),
        )))) as Arc<dyn PhysicalExpr>;
        let like_expr =
            Arc::new(df_expr::LikeExpr::new(false, false, expr, pattern)) as Arc<dyn PhysicalExpr>;

        assert!(can_be_pushed_down_impl(&like_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_like_unsupported_operand(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;
        let pattern = Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(
            "test%".to_string(),
        )))) as Arc<dyn PhysicalExpr>;
        let like_expr =
            Arc::new(df_expr::LikeExpr::new(false, false, expr, pattern)) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&like_expr, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_octet_length_supported(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("name", 1)) as Arc<dyn PhysicalExpr>;
        let octet_length = octet_length_expr(expr, &test_schema);

        assert!(can_be_pushed_down_impl(&octet_length, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_octet_length_unsupported_operand(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("unsupported_list", 5)) as Arc<dyn PhysicalExpr>;
        let octet_length = Arc::new(ScalarFunctionExpr::new(
            "octet_length",
            Arc::new(ScalarUDF::from(OctetLengthFunc::new())),
            vec![expr],
            Arc::new(Field::new("octet_length", DataType::Int32, true)),
            Arc::new(ConfigOptions::new()),
        )) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&octet_length, &test_schema));
    }

    // https://github.com/vortex-data/vortex/issues/6211
    #[tokio::test]
    async fn test_cast_int_to_string() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(r#"copy (select 1 as id) to 'example.vortex'"#)
            .await?
            .show()
            .await?;

        ctx.session
            .sql(r#"select cast(id as string) as sid from 'example.vortex' where id > 0"#)
            .await?
            .show()
            .await?;

        ctx.session
            .sql(r#"select id from 'example.vortex' where cast (id as string) == '1'"#)
            .await?
            .show()
            .await?;

        // This fails as it pushes string cast to the scan
        ctx.session
            .sql(r#"select cast(id as string) from 'example.vortex'"#)
            .await?
            .collect()
            .await?;

        Ok(())
    }

    /// Test that applying a CASE expression to an Arrow RecordBatch using DataFusion
    /// matches the result of applying the converted Vortex expression.
    #[test]
    fn test_case_when_datafusion_vortex_equivalence() {
        use datafusion::arrow::array::Int32Array;
        use datafusion::arrow::array::RecordBatch;
        use datafusion_physical_expr::expressions::CaseExpr;
        use vortex::VortexSessionDefault;
        use vortex::array::ArrayRef;
        use vortex::array::Canonical;
        use vortex::array::VortexSessionExecute as _;
        use vortex::array::arrow::FromArrowArray;
        use vortex::session::VortexSession;

        // Create test data
        let values = Arc::new(Int32Array::from(vec![1, 5, 10, 15, 20]));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![values]).unwrap();

        // Build a DataFusion CASE expression:
        // CASE WHEN value > 10 THEN 100 WHEN value > 5 THEN 50 ELSE 0 END
        let col_value = Arc::new(df_expr::Column::new("value", 0)) as Arc<dyn PhysicalExpr>;
        let lit_10 =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(10)))) as Arc<dyn PhysicalExpr>;
        let lit_5 =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let lit_100 =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(100)))) as Arc<dyn PhysicalExpr>;
        let lit_50 =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(50)))) as Arc<dyn PhysicalExpr>;
        let lit_0 =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(0)))) as Arc<dyn PhysicalExpr>;

        // WHEN value > 10 THEN 100
        let when1 = Arc::new(df_expr::BinaryExpr::new(
            Arc::clone(&col_value),
            DFOperator::Gt,
            lit_10,
        )) as Arc<dyn PhysicalExpr>;
        // WHEN value > 5 THEN 50
        let when2 = Arc::new(df_expr::BinaryExpr::new(col_value, DFOperator::Gt, lit_5))
            as Arc<dyn PhysicalExpr>;

        let case_expr =
            CaseExpr::try_new(None, vec![(when1, lit_100), (when2, lit_50)], Some(lit_0)).unwrap();

        // Apply DataFusion expression
        let df_result = case_expr.evaluate(&batch).unwrap();
        let df_array = df_result.into_array(batch.num_rows()).unwrap();

        // Convert to Vortex expression
        let expr_convertor = DefaultExpressionConvertor::default();
        let vortex_expr = expr_convertor.try_convert_case_expr(&case_expr).unwrap();

        // Convert batch to Vortex array
        let vortex_array: ArrayRef = ArrayRef::from_arrow(&batch, false).unwrap();

        // Apply Vortex expression
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();
        let vortex_result = vortex_array
            .apply(&vortex_expr)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        // Convert back to Arrow for comparison
        let vortex_as_arrow = vortex_result.into_primitive().as_slice::<i32>().to_vec();

        // Convert DataFusion result to Vec for comparison
        let df_as_arrow: Vec<i32> = df_array.as_primitive::<Int32Type>().values().to_vec();

        // Compare results
        // Expected: [0, 0, 50, 100, 100] for values [1, 5, 10, 15, 20]
        // value=1: not > 10, not > 5 -> ELSE 0
        // value=5: not > 10, not > 5 -> ELSE 0
        // value=10: not > 10, > 5 -> 50
        // value=15: > 10 -> 100
        // value=20: > 10 -> 100
        assert_eq!(df_as_arrow, vec![0, 0, 50, 100, 100]);
        assert_eq!(vortex_as_arrow, df_as_arrow);
    }
}
