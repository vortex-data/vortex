// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_expr::Operator as DFOperator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions::string::octet_length::OctetLengthFunc;
use datafusion_functions_nested::length::ArrayLength;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::expressions as df_expr;
use vortex::VortexSessionDefault;
use vortex::dtype::DType as VortexDType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::expr;
use vortex::expr::BoundExpression;
use vortex::scalar::Scalar;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::byte_length::ByteLength;
use vortex::scalar_fn::fns::case_when::CaseWhen;
use vortex::scalar_fn::fns::case_when::CaseWhenOptions;
use vortex::scalar_fn::fns::cast::Cast;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::is_not_null::IsNotNull;
use vortex::scalar_fn::fns::is_null::IsNull;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::list_contains::ListContains;
use vortex::scalar_fn::fns::list_length::ListLength;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::not::Not;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use crate::convert::scalar_from_df;

/// A split projection whose scan portion has already been type-checked.
pub struct ProcessedProjection {
    /// Projection evaluated by the Vortex scan.
    pub scan_projection: BoundExpression,
    /// Projection evaluated by DataFusion after the Vortex scan.
    pub leftover_projection: ProjectionExprs,
}

fn df_bound(result: VortexResult<BoundExpression>) -> DFResult<BoundExpression> {
    result.map_err(|e| exec_datafusion_err!("Failed to construct bound Vortex expression: {e}"))
}

/// Tries to convert the expressions into a vortex conjunction. Will return Ok(None) iff the input conjunction is empty.
pub(crate) fn make_vortex_predicate(
    expr_convertor: &dyn ExpressionConvertor,
    predicate: &[Arc<dyn PhysicalExpr>],
    scope: &VortexDType,
) -> DFResult<Option<BoundExpression>> {
    let exprs = predicate
        .iter()
        .map(|e| expr_convertor.convert(e.as_ref(), scope))
        .collect::<DFResult<Vec<_>>>()?;

    expr::and_collect(exprs)
        .map(|expression| df_bound(expression.optimize_recursive()))
        .transpose()
}

/// Trait for converting DataFusion expressions to Vortex ones.
/// Scan methods require bound trees so implementations own their coercion and overload choices.
///
/// # Implementing a custom convertor
///
/// ```
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use datafusion_common::Result as DFResult;
/// use datafusion_physical_expr::PhysicalExpr;
/// use datafusion_physical_expr::projection::ProjectionExprs;
/// use vortex::expr::BoundExpression;
/// use vortex::dtype::DType;
/// use vortex_datafusion::convert::DefaultExpressionConvertor;
/// use vortex_datafusion::convert::ExpressionConvertor;
/// use vortex_datafusion::convert::ProcessedProjection;
///
/// struct CustomExpressionConvertor(DefaultExpressionConvertor);
///
/// impl ExpressionConvertor for CustomExpressionConvertor {
///     fn can_be_pushed_down(&self, expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
///         self.0.can_be_pushed_down(expr, schema)
///     }
///
///     fn convert(&self, expr: &dyn PhysicalExpr, scope: &DType) -> DFResult<BoundExpression> {
///         self.0.convert(expr, scope)
///     }
///
///     fn split_projection(
///         &self,
///         source_projection: ProjectionExprs,
///         input_schema: &Schema,
///         output_schema: &Schema,
///         scope: &DType,
///     ) -> DFResult<ProcessedProjection> {
///         self.0.split_projection(source_projection, input_schema, output_schema, scope)
///     }
/// }
///
/// let _convertor: Arc<dyn ExpressionConvertor> = Arc::new(CustomExpressionConvertor(
///     DefaultExpressionConvertor::default(),
/// ));
/// ```
pub trait ExpressionConvertor: Send + Sync {
    /// Can an expression be pushed down given a specific schema
    fn can_be_pushed_down(&self, expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool;

    /// Lower a DataFusion physical expression to a typed execution tree.
    fn convert(&self, expr: &dyn PhysicalExpr, scope: &VortexDType) -> DFResult<BoundExpression>;

    /// Split and type-check a projection for a Vortex scan.
    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
        scope: &VortexDType,
    ) -> DFResult<ProcessedProjection>;

    /// Construct a typed scan projection that reads only columns needed by DataFusion.
    fn no_pushdown_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        scope: &VortexDType,
    ) -> DFResult<ProcessedProjection> {
        let columns = source_projection
            .column_indices()
            .into_iter()
            .map(|idx| {
                let name = input_schema.field(idx).name().clone();
                let value = df_bound(GetItem.try_new_bound_expr(
                    name.clone().into(),
                    [BoundExpression::new_root(scope.clone())],
                ))?;
                Ok((name, value))
            })
            .collect::<DFResult<Vec<_>>>()?;
        Ok(ProcessedProjection {
            scan_projection: expr::pack(columns, Nullability::NonNullable),
            leftover_projection: source_projection,
        })
    }
}

/// The default [`ExpressionConvertor`] implementation.
pub struct DefaultExpressionConvertor {
    /// Session used to resolve Arrow → Vortex dtypes through the extension
    /// plugin registry, so registered extension types (e.g. UUID ⇄
    /// `FixedSizeBinary[16]`) convert correctly instead of hitting the static,
    /// non-plugin-aware `DType::from_arrow`.
    session: VortexSession,
}

impl Default for DefaultExpressionConvertor {
    fn default() -> Self {
        Self {
            session: VortexSession::default(),
        }
    }
}

impl DefaultExpressionConvertor {
    /// Create a convertor that resolves Arrow extension types using `session`'s
    /// dtype registry.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }
}

impl ExpressionConvertor for DefaultExpressionConvertor {
    fn can_be_pushed_down(&self, expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
        can_be_pushed_down_impl(expr, schema)
            && self
                .session
                .arrow()
                .from_arrow_schema(schema)
                .is_ok_and(|scope| self.convert(expr.as_ref(), &scope).is_ok())
    }

    fn convert(&self, df: &dyn PhysicalExpr, scope: &VortexDType) -> DFResult<BoundExpression> {
        if let Some(binary_expr) = df.downcast_ref::<df_expr::BinaryExpr>() {
            let lhs = self.convert(binary_expr.left().as_ref(), scope)?;
            let rhs = self.convert(binary_expr.right().as_ref(), scope)?;
            return df_bound(
                Binary.try_new_bound_expr(try_operator_from_df(binary_expr.op())?, [lhs, rhs]),
            );
        }

        if let Some(column) = df.downcast_ref::<df_expr::Column>() {
            return df_bound(GetItem.try_new_bound_expr(
                column.name().to_owned().into(),
                [BoundExpression::new_root(scope.clone())],
            ));
        }

        if let Some(like) = df.downcast_ref::<df_expr::LikeExpr>() {
            let value = self.convert(like.expr().as_ref(), scope)?;
            let pattern = self.convert(like.pattern().as_ref(), scope)?;
            return df_bound(Like.try_new_bound_expr(
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
                [value, pattern],
            ));
        }

        if let Some(literal) = df.downcast_ref::<df_expr::Literal>() {
            return df_bound(
                Literal.try_new_bound_expr(scalar_from_df(literal.value(), &self.session), []),
            );
        }

        if let Some(cast_expr) = df.downcast_ref::<df_expr::CastExpr>() {
            let target = self
                .session
                .arrow()
                .from_arrow_field(cast_expr.target_field().as_ref())
                .map_err(|e| exec_datafusion_err!("Failed to convert cast target to dtype: {e}"))?;
            let child = self.convert(cast_expr.expr().as_ref(), scope)?;
            return df_bound(Cast.try_new_bound_expr(target, [child]));
        }

        if let Some(is_null) = df.downcast_ref::<df_expr::IsNullExpr>() {
            let child = self.convert(is_null.arg().as_ref(), scope)?;
            return df_bound(IsNull.try_new_bound_expr(EmptyOptions, [child]));
        }

        if let Some(is_not_null) = df.downcast_ref::<df_expr::IsNotNullExpr>() {
            let child = self.convert(is_not_null.arg().as_ref(), scope)?;
            return df_bound(IsNotNull.try_new_bound_expr(EmptyOptions, [child]));
        }

        if let Some(in_list) = df.downcast_ref::<df_expr::InListExpr>() {
            let value = self.convert(in_list.expr().as_ref(), scope)?;
            let list_elements = in_list
                .list()
                .iter()
                .map(|expr| {
                    expr.downcast_ref::<df_expr::Literal>()
                        .map(|literal| scalar_from_df(literal.value(), &self.session))
                        .ok_or_else(|| exec_datafusion_err!("IN-list member is not a literal"))
                })
                .collect::<DFResult<Vec<_>>>()?;
            let first = list_elements
                .first()
                .ok_or_else(|| exec_datafusion_err!("IN-list must not be empty"))?;
            let list = Scalar::list(
                Arc::new(first.dtype().clone()),
                list_elements,
                Nullability::Nullable,
            );
            let list = df_bound(Literal.try_new_bound_expr(list, []))?;
            let contains = df_bound(ListContains.try_new_bound_expr(EmptyOptions, [list, value]))?;
            return if in_list.negated() {
                df_bound(Not.try_new_bound_expr(EmptyOptions, [contains]))
            } else {
                Ok(contains)
            };
        }

        if let Some(scalar_fn) = df.downcast_ref::<ScalarFunctionExpr>() {
            if let Some(function) =
                ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(scalar_fn)
            {
                let [arg] = function.args() else {
                    return Err(exec_datafusion_err!("octet_length requires one argument"));
                };
                let input = self.convert(arg.as_ref(), scope)?;
                let result = df_bound(ByteLength.try_new_bound_expr(EmptyOptions, [input]))?;
                let target = self
                    .session
                    .arrow()
                    .from_arrow_field(&Field::new(
                        "",
                        function.return_type().clone(),
                        function.nullable(),
                    ))
                    .map_err(|e| exec_datafusion_err!("Failed to convert return type: {e}"))?;
                return df_bound(Cast.try_new_bound_expr(target, [result]));
            }
            if let Some(function) = ScalarFunctionExpr::try_downcast_func::<ArrayLength>(scalar_fn)
            {
                let input = array_length_input(function)
                    .ok_or_else(|| exec_datafusion_err!("unsupported array_length arguments"))?;
                let input = self.convert(input.as_ref(), scope)?;
                let result = df_bound(ListLength.try_new_bound_expr(EmptyOptions, [input]))?;
                let target = self
                    .session
                    .arrow()
                    .from_arrow_field(&Field::new(
                        "",
                        function.return_type().clone(),
                        function.nullable(),
                    ))
                    .map_err(|e| exec_datafusion_err!("Failed to convert return type: {e}"))?;
                return df_bound(Cast.try_new_bound_expr(target, [result]));
            }
            if let Some(function) = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn)
            {
                let (source, fields) = function
                    .args()
                    .split_first()
                    .ok_or_else(|| exec_datafusion_err!("get_field missing source"))?;
                let mut result = self.convert(source.as_ref(), scope)?;
                for field in fields {
                    let name = field
                        .downcast_ref::<df_expr::Literal>()
                        .and_then(|literal| literal.value().try_as_str().flatten())
                        .ok_or_else(|| exec_datafusion_err!("get_field name must be a string"))?;
                    result =
                        df_bound(GetItem.try_new_bound_expr(name.to_string().into(), [result]))?;
                }
                return Ok(result);
            }
            return Err(exec_datafusion_err!(
                "Unsupported scalar function: {}",
                scalar_fn.name()
            ));
        }

        if let Some(case_expr) = df.downcast_ref::<df_expr::CaseExpr>() {
            if case_expr.expr().is_some() || case_expr.when_then_expr().is_empty() {
                return Err(exec_datafusion_err!("unsupported CASE form"));
            }
            let pair_count = u32::try_from(case_expr.when_then_expr().len())
                .map_err(|e| exec_datafusion_err!("too many CASE branches: {e}"))?;
            let mut children = Vec::new();
            for (when, then) in case_expr.when_then_expr() {
                children.push(self.convert(when.as_ref(), scope)?);
                children.push(self.convert(then.as_ref(), scope)?);
            }
            let else_expr = case_expr
                .else_expr()
                .map(|expr| self.convert(expr.as_ref(), scope))
                .transpose()?;
            let has_else = else_expr.is_some();
            children.extend(else_expr);
            return df_bound(CaseWhen.try_new_bound_expr(
                CaseWhenOptions {
                    num_when_then_pairs: pair_count,
                    has_else,
                },
                children,
            ));
        }

        Err(exec_datafusion_err!(
            "Unsupported DataFusion physical expression: {df}"
        ))
    }

    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
        scope: &VortexDType,
    ) -> DFResult<ProcessedProjection> {
        let mut scan_projection = Vec::new();
        let mut leftover_projection = Vec::new();

        for projection_expr in source_projection.iter() {
            let visit = projection_expr.expr.apply(|node| {
                let unsupported_function = node
                    .downcast_ref::<ScalarFunctionExpr>()
                    .is_some_and(|function| !can_scalar_fn_be_pushed_down(function, input_schema));
                let decimal_arithmetic =
                    if let Some(binary) = node.downcast_ref::<df_expr::BinaryExpr>() {
                        binary.op().is_numerical_operators()
                            && binary.left().data_type(input_schema)?.is_decimal()
                            && binary.right().data_type(input_schema)?.is_decimal()
                    } else {
                        false
                    };
                if unsupported_function || decimal_arithmetic {
                    for column in collect_columns(node) {
                        let name = column.name().to_string();
                        let bound = df_bound(GetItem.try_new_bound_expr(
                            name.clone().into(),
                            [BoundExpression::new_root(scope.clone())],
                        ))?;
                        scan_projection.push((name, bound));
                    }
                    leftover_projection.push(projection_expr.clone());
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })?;

            if matches!(visit, TreeNodeRecursion::Continue) {
                scan_projection.push((
                    projection_expr.alias.clone(),
                    self.convert(projection_expr.expr.as_ref(), scope)?,
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
            scan_projection: expr::pack(scan_projection, Nullability::NonNullable),
            leftover_projection: leftover_projection.into(),
        })
    }

    fn no_pushdown_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        scope: &VortexDType,
    ) -> DFResult<ProcessedProjection> {
        let columns = source_projection
            .column_indices()
            .into_iter()
            .map(|idx| {
                let name = input_schema.field(idx).name().clone();
                let value = df_bound(GetItem.try_new_bound_expr(
                    name.clone().into(),
                    [BoundExpression::new_root(scope.clone())],
                ))?;
                Ok((name, value))
            })
            .collect::<DFResult<Vec<_>>>()?;
        Ok(ProcessedProjection {
            scan_projection: expr::pack(columns, Nullability::NonNullable),
            leftover_projection: source_projection,
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
    if DynamicFilterTracking::classify(expr).contains_dynamic_filter() {
        return false;
    }

    if let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() {
        can_binary_be_pushed_down(binary, schema)
    } else if let Some(col) = expr.downcast_ref::<df_expr::Column>() {
        schema
            .field_with_name(col.name())
            .is_ok_and(|field| supported_data_types(field.data_type()))
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
                || ScalarFunctionExpr::try_downcast_func::<ArrayLength>(sf).is_some()
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
/// Currently GetFieldFunc, OctetLengthFunc, and ArrayLength are supported.
fn can_scalar_fn_be_pushed_down(scalar_fn: &ScalarFunctionExpr, schema: &Schema) -> bool {
    if ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn).is_some() {
        return true;
    }

    if ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(scalar_fn)
        .is_some_and(|octet_length| can_octet_length_be_pushed_down(octet_length, schema))
    {
        return true;
    }

    ScalarFunctionExpr::try_downcast_func::<ArrayLength>(scalar_fn)
        .is_some_and(|array_length| can_array_length_be_pushed_down(array_length, schema))
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

fn can_array_length_be_pushed_down(scalar_fn: &ScalarFunctionExpr, schema: &Schema) -> bool {
    let Some(input) = array_length_input(scalar_fn) else {
        return false;
    };

    // The argument must resolve to a list type. We gate on the resolved data type rather than
    // `can_be_pushed_down_impl`, since list columns are intentionally rejected there. We still
    // require the argument to be a convertible expression (e.g. a column or struct field access).
    input.data_type(schema).as_ref().is_ok_and(|data_type| {
        matches!(
            data_type,
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
        )
    }) && is_convertible_expr(input)
}

/// Returns the list argument of an `array_length` call if the call is a form we can rewrite to
/// `list_length`: either the single-argument form `array_length(arr)`, or the two-argument form
/// with an explicit first dimension `array_length(arr, 1)`, which is equivalent. Higher
/// dimensions recurse into nested lists and are not supported.
fn array_length_input(scalar_fn: &ScalarFunctionExpr) -> Option<&Arc<dyn PhysicalExpr>> {
    match scalar_fn.args() {
        [input] => Some(input),
        [input, dimension] if is_dimension_one(dimension) => Some(input),
        _ => None,
    }
}

/// Returns true if `expr` is an `Int64` literal equal to 1. DataFusion coerces the `array_length`
/// dimension argument to `Int64`, so that is the only form we need to recognize; any other literal
/// simply isn't pushed down.
fn is_dimension_one(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<df_expr::Literal>()
        .is_some_and(|literal| matches!(literal.value(), ScalarValue::Int64(Some(1))))
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
    use vortex::dtype::StructFields;

    use super::*;
    use crate::common_tests::TestSessionContext;

    fn predicate_scope() -> VortexDType {
        VortexDType::Struct(
            StructFields::from_iter([
                ("test", VortexDType::Bool(Nullability::NonNullable)),
                ("col1", VortexDType::Bool(Nullability::NonNullable)),
                ("col2", VortexDType::Bool(Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

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
                "tags",
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

    fn array_length_expr(
        args: Vec<Arc<dyn PhysicalExpr>>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(ScalarUDF::from(ArrayLength::new())),
                args,
                schema,
                Arc::new(ConfigOptions::new()),
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_make_vortex_predicate_empty() {
        let expr_convertor = DefaultExpressionConvertor::default();
        let result = make_vortex_predicate(&expr_convertor, &[], &predicate_scope()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_make_vortex_predicate_single() {
        let expr_convertor = DefaultExpressionConvertor::default();
        let col_expr = Arc::new(df_expr::Column::new("test", 0)) as Arc<dyn PhysicalExpr>;
        let result =
            make_vortex_predicate(&expr_convertor, &[col_expr], &predicate_scope()).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_make_vortex_predicate_multiple() {
        let expr_convertor = DefaultExpressionConvertor::default();
        let col1 = Arc::new(df_expr::Column::new("col1", 0)) as Arc<dyn PhysicalExpr>;
        let col2 = Arc::new(df_expr::Column::new("col2", 1)) as Arc<dyn PhysicalExpr>;
        let result =
            make_vortex_predicate(&expr_convertor, &[col1, col2], &predicate_scope()).unwrap();
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
        let scope = VortexDType::Struct(
            StructFields::from_iter([("test_column", VortexDType::Bool(Nullability::NonNullable))]),
            Nullability::NonNullable,
        );
        let result = DefaultExpressionConvertor::default()
            .convert(&col_expr, &scope)
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
            .convert(&literal_expr, &VortexDType::Null)
            .unwrap();

        assert_snapshot!(result.display_tree().to_string(), @"vortex.literal(42i32)");
    }

    #[test]
    fn test_expr_from_df_binary() {
        let left = Arc::new(df_expr::Column::new("left", 0)) as Arc<dyn PhysicalExpr>;
        let right =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let binary_expr = df_expr::BinaryExpr::new(left, DFOperator::Eq, right);
        let scope = VortexDType::Struct(
            StructFields::from_iter([(
                "left",
                VortexDType::Primitive(vortex::dtype::PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );

        let result = DefaultExpressionConvertor::default()
            .convert(&binary_expr, &scope)
            .unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r"
        vortex.binary(=)
        ├── lhs: vortex.get_item(left)
        │   └── input: vortex.root()
        └── rhs: vortex.literal(42i32)
        ");
    }

    #[test]
    fn datafusion_conversion_does_not_apply_authored_literal_coercion() {
        let column = Arc::new(df_expr::Column::new("value", 0)) as Arc<dyn PhysicalExpr>;
        let literal =
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;
        let expr = df_expr::BinaryExpr::new(column, DFOperator::Eq, literal);
        let scope = VortexDType::Struct(
            StructFields::from_iter([(
                "value",
                VortexDType::Primitive(vortex::dtype::PType::I64, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );

        assert!(
            DefaultExpressionConvertor::default()
                .convert(&expr, &scope)
                .is_err()
        );
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
        let scope = VortexDType::Struct(
            StructFields::from_iter([("text_col", VortexDType::Utf8(Nullability::Nullable))]),
            Nullability::NonNullable,
        );

        let result = DefaultExpressionConvertor::default()
            .convert(&like_expr, &scope)
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
        let convertor = DefaultExpressionConvertor::default();
        let scope = convertor
            .session
            .arrow()
            .from_arrow_schema(&test_schema)
            .unwrap();

        let result = convertor.convert(octet_length.as_ref(), &scope).unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r"
        vortex.cast(i32?)
        └── input: vortex.byte_length()
            └── input: vortex.get_item(name)
                └── input: vortex.root()
        ");
    }

    #[rstest]
    fn test_expr_from_df_array_length(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
        let array_length = array_length_expr(vec![expr], &test_schema);
        let convertor = DefaultExpressionConvertor::default();
        let scope = convertor
            .session
            .arrow()
            .from_arrow_schema(&test_schema)
            .unwrap();

        let result = convertor.convert(array_length.as_ref(), &scope).unwrap();

        assert_snapshot!(result.display_tree().to_string(), @r"
        vortex.cast(u64?)
        └── input: vortex.list.length()
            └── input: vortex.get_item(tags)
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
        let col_expr = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;

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
        let left = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
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
        let expr = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
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
        let expr = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
        let octet_length = Arc::new(ScalarFunctionExpr::new(
            "octet_length",
            Arc::new(ScalarUDF::from(OctetLengthFunc::new())),
            vec![expr],
            Arc::new(Field::new("octet_length", DataType::Int32, true)),
            Arc::new(ConfigOptions::new()),
        )) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&octet_length, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_array_length_supported(test_schema: Schema) {
        let expr = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
        let array_length = array_length_expr(vec![expr], &test_schema);

        assert!(can_be_pushed_down_impl(&array_length, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_array_length_unsupported_operand(test_schema: Schema) {
        // `array_length` over a non-list column cannot be pushed down.
        let expr = Arc::new(df_expr::Column::new("name", 1)) as Arc<dyn PhysicalExpr>;
        let array_length = Arc::new(ScalarFunctionExpr::new(
            "array_length",
            Arc::new(ScalarUDF::from(ArrayLength::new())),
            vec![expr],
            Arc::new(Field::new("array_length", DataType::UInt64, true)),
            Arc::new(ConfigOptions::new()),
        )) as Arc<dyn PhysicalExpr>;

        assert!(!can_be_pushed_down_impl(&array_length, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_array_length_dimension_one_supported(test_schema: Schema) {
        // `array_length(arr, 1)` is the first-dimension length, equivalent to `list_length`.
        let list = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
        let dimension =
            Arc::new(df_expr::Literal::new(ScalarValue::Int64(Some(1)))) as Arc<dyn PhysicalExpr>;
        let array_length = array_length_expr(vec![list, dimension], &test_schema);

        assert!(can_be_pushed_down_impl(&array_length, &test_schema));
    }

    #[rstest]
    fn test_can_be_pushed_down_array_length_higher_dimension_not_supported(test_schema: Schema) {
        // Dimensions other than 1 recurse into nested lists, which `list_length` does not model,
        // so they must not be pushed down.
        let list = Arc::new(df_expr::Column::new("tags", 5)) as Arc<dyn PhysicalExpr>;
        let dimension =
            Arc::new(df_expr::Literal::new(ScalarValue::Int64(Some(2)))) as Arc<dyn PhysicalExpr>;
        let array_length = array_length_expr(vec![list, dimension], &test_schema);

        assert!(!can_be_pushed_down_impl(&array_length, &test_schema));
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

    /// A cast whose target is a UUID-tagged `FixedSizeBinary(16)` must resolve
    /// through the dtype extension registry (UUID is registered on the default
    /// session) instead of the static, non-plugin-aware `DType::from_arrow`,
    /// which does not support `FixedSizeBinary` and previously panicked here.
    #[test]
    fn test_cast_to_uuid_resolves_via_registry() -> anyhow::Result<()> {
        use arrow_schema::extension::Uuid;

        let mut uuid_field = Field::new("id", DataType::FixedSizeBinary(16), true);
        uuid_field.try_with_extension_type(Uuid)?;

        let child = Arc::new(df_expr::Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let cast = df_expr::CastExpr::new_with_target_field(child, Arc::new(uuid_field), None);

        // Must convert without panicking — the static path would `unimplemented!()`.
        let scope = VortexDType::Struct(
            StructFields::from_iter([(
                "id",
                VortexDType::Primitive(vortex::dtype::PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        DefaultExpressionConvertor::default().convert(&cast, &scope)?;
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

        // Convert batch to Vortex array
        let session = VortexSession::default();
        let vortex_array: ArrayRef = session
            .arrow()
            .from_arrow_record_batch(batch.clone(), &batch.schema())
            .unwrap();
        let vortex_expr = expr_convertor
            .convert(&case_expr, vortex_array.dtype())
            .unwrap();

        // Apply Vortex expression
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
