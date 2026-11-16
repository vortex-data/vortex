// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recognition and construction of each supported expression.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_common::format::DEFAULT_FORMAT_OPTIONS;
use datafusion_expr::Operator as DFOperator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions::string::contains::ContainsFunc;
use datafusion_functions::string::ends_with::EndsWithFunc;
use datafusion_functions::string::lower::LowerFunc;
use datafusion_functions::string::octet_length::OctetLengthFunc;
use datafusion_functions::string::starts_with::StartsWithFunc;
use datafusion_functions::unicode::strpos::StrposFunc;
use datafusion_functions_nested::array_sum::ArraySum;
use datafusion_functions_nested::length::ArrayLength;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::expr::Expression;
use vortex::expr::byte_length;
use vortex::expr::cast;
use vortex::expr::get_item;
use vortex::expr::in_list_of;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_length;
use vortex::expr::list_sum_opts;
use vortex::expr::lit;
use vortex::expr::lower;
use vortex::expr::nested_case_when;
use vortex::expr::not;
use vortex::expr::root;
use vortex::expr::strpos;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;

use super::Conversion;
use super::Convertible;
use super::ConvertibleCtx;
use super::ConvertibleFixedArity;
use super::ConvertibleVariableArity;
use super::DefaultExpressionConverter;
use super::Unconverted;
use super::arrow_dtype;
use super::escape_like;
use super::is_infallible;
use super::next_child;
use super::node_dtype;
use super::recur_and_construct;

pub(super) fn convert_node<'a>(
    expr: &'a Arc<dyn PhysicalExpr>,
    ctx: &ConvertibleCtx<'_>,
    recur: &mut impl FnMut(&'a Arc<dyn PhysicalExpr>) -> Conversion<Expression>,
) -> Conversion<Expression> {
    macro_rules! try_convert {
        ($($convert:ty),+ $(,)?) => {
            $(
                match <$convert>::maybe_match(expr, ctx) {
                    Err(Unconverted::Unsupported) => {},
                    Ok((children, convertible)) => {
                        return recur_and_construct(children, convertible, ctx, recur);
                    }
                    Err(error) => return Err(error),
                }
            )+
        };
    }
    try_convert!(
        ConvertBinary<'a>,
        ConvertColumn<'a>,
        ConvertLambdaVariable<'a>,
        ConvertLiteral,
        ConvertCast,
        ConvertNot,
        ConvertIsNull,
        ConvertIsNotNull,
        ConvertLike,
        ConvertInList<'a>,
        ConvertGetField<'a>,
        ConvertOctetLength,
        ConvertArrayLength,
        ConvertLower,
        ConvertStrpos,
        ConvertContains,
        ConvertStartsWith,
        ConvertEndsWith,
        ConvertArraySum,
        ConvertCase,
    );
    Err(Unconverted::Unsupported)
}

/// Convert a DataFusion `BinaryExpr` with checked variable arity.
pub type ConvertBinary<'a> = ConvertibleVariableArity<BinaryConversion<'a>>;

/// Matching and construction state used by [`ConvertBinary`].
pub struct BinaryConversion<'a> {
    operator: Operator,
    left: BinaryOperand<'a>,
    right: BinaryOperand<'a>,
}

impl<'a> Convertible<'a> for BinaryConversion<'a> {
    type DFExpr = df_expr::BinaryExpr;

    fn maybe_convert(
        expr: &'a Self::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let operator = try_operator_from_df(expr.op()).ok_or(Unconverted::Unsupported)?;
        let left_type = expr.left().data_type(ctx.schema)?;
        let right_type = expr.right().data_type(ctx.schema)?;
        if matches!(operator, Operator::And | Operator::Or) {
            if left_type != DataType::Boolean || right_type != DataType::Boolean {
                return Err(exec_datafusion_err!("Boolean operator requires Boolean operands: {expr}").into());
            }
        } else if !supported_data_types(&left_type) || !supported_data_types(&right_type) {
            return Err(Unconverted::Unsupported);
        }
        let mut children = Vec::with_capacity(2);
        let left = BinaryOperand::select(expr.left(), expr.right(), operator, ctx.schema, &mut children)?;
        let right = BinaryOperand::select(expr.right(), expr.left(), operator, ctx.schema, &mut children)?;
        Ok((children, Self { operator, left, right }))
    }

    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { operator, left, right } = self;
        let mut children = children.into_iter();
        let left = left.construct(&mut children, ctx)?;
        let right = right.construct(&mut children, ctx)?;
        // DataFusion may evaluate the RHS only on rows selected by the LHS.
        if matches!(operator, Operator::And | Operator::Or)
            && !(is_infallible(&left) && is_infallible(&right))
        {
            return Err(Unconverted::Unsupported);
        }
        if !node_dtype(&left, ctx)?
            .eq_ignore_nullability(&node_dtype(&right, ctx)?)
        {
            return Err(Unconverted::Unsupported);
        }
        Ok(Binary.new_expr(operator, [left, right]))
    }
}

/// Convert a DataFusion column reference.
pub struct ConvertColumn<'a> {
    name: &'a str,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertColumn<'a> {
    type DFExpr = df_expr::Column;
    type Children = [Expression; 0];

    fn maybe_convert_fixed(
        expr: &'a df_expr::Column,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ctx.scope.is_some() {
            return Err(Unconverted::Unsupported);
        }
        let name = expr.name();
        Ok((Vec::new(), Self { name }))
    }

    fn construct_fixed(
        self,
        []: [Expression; 0],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { name } = self;
        Ok(get_item(name, root()))
    }
}

/// Convert a lambda parameter reference within the supplied scope.
pub struct ConvertLambdaVariable<'a> {
    name: &'a str,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertLambdaVariable<'a> {
    type DFExpr = df_expr::LambdaVariable;
    type Children = [Expression; 0];

    fn maybe_convert_fixed(
        expr: &'a df_expr::LambdaVariable,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let Some(params) = ctx.scope else {
            return Err(Unconverted::Unsupported);
        };
        if params.find(expr.name()).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let name = expr.name();
        Ok((Vec::new(), Self { name }))
    }

    fn construct_fixed(
        self,
        []: [Expression; 0],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { name } = self;
        Ok(get_item(name, root()))
    }
}

/// Import a DataFusion literal through the session's Arrow plugins.
pub struct ConvertLiteral {
    scalar: Scalar,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertLiteral {
    type DFExpr = df_expr::Literal;
    type Children = [Expression; 0];

    fn maybe_convert_fixed(
        expr: &'a df_expr::Literal,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let field = expr.return_field(ctx.schema)?;
        let scalar = import_literal(expr, &field, ctx.session)?;
        Ok((Vec::new(), Self { scalar }))
    }

    fn construct_fixed(
        self,
        []: [Expression; 0],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { scalar } = self;
        Ok(lit(scalar))
    }
}

/// Convert a supported DataFusion cast.
pub struct ConvertCast {
    pushdown: CastPushdown,
    dtype: DType,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertCast {
    type DFExpr = df_expr::CastExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a df_expr::CastExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let pushdown = classify_cast(expr, ctx.schema)?;
        let dtype = arrow_dtype(expr.target_field(), ctx.session)?
            .with_nullability(Nullability::from(expr.nullable(ctx.schema)?));
        Ok((vec![expr.expr()], Self { pushdown, dtype }))
    }

    fn construct_fixed(
        self,
        [child]: [Expression; 1],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { pushdown, dtype } = self;
        let child_dtype = node_dtype(&child, ctx)?;
        match pushdown {
            CastPushdown::Drop if child_dtype.eq_ignore_nullability(&dtype) => Ok(child),
            CastPushdown::Drop => Err(Unconverted::Unsupported),
            CastPushdown::Keep => {
                // Matching storage types do not imply matching extension semantics.
                if (child_dtype.is_extension() || dtype.is_extension())
                    && !child_dtype.eq_ignore_nullability(&dtype)
                { return Err(Unconverted::Unsupported); }
                Ok(cast(child, dtype))
            }
        }
    }
}

/// Convert logical NOT.
pub struct ConvertNot;

impl<'a> ConvertibleFixedArity<'a> for ConvertNot {
    type DFExpr = df_expr::NotExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a df_expr::NotExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if expr.arg().data_type(ctx.schema)? != DataType::Boolean {
            return Err(exec_datafusion_err!("NOT requires a Boolean operand: {expr}").into());
        }
        Ok((vec![expr.arg()], Self))
    }

    fn construct_fixed(
        self,
        [arg]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        Ok(not(arg))
    }
}

/// Convert IS NULL.
pub struct ConvertIsNull;

impl<'a> ConvertibleFixedArity<'a> for ConvertIsNull {
    type DFExpr = df_expr::IsNullExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a df_expr::IsNullExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        Ok((vec![expr.arg()], Self))
    }

    fn construct_fixed(
        self,
        [arg]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        Ok(is_null(arg))
    }
}

/// Convert IS NOT NULL.
pub struct ConvertIsNotNull;

impl<'a> ConvertibleFixedArity<'a> for ConvertIsNotNull {
    type DFExpr = df_expr::IsNotNullExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a df_expr::IsNotNullExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        Ok((vec![expr.arg()], Self))
    }

    fn construct_fixed(
        self,
        [arg]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        Ok(is_not_null(arg))
    }
}

/// Convert LIKE and ILIKE expressions.
pub struct ConvertLike {
    options: LikeOptions,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertLike {
    type DFExpr = df_expr::LikeExpr;
    type Children = [Expression; 2];

    fn maybe_convert_fixed(
        expr: &'a df_expr::LikeExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let options = LikeOptions { negated: expr.negated(), case_insensitive: expr.case_insensitive() };
        Ok((vec![expr.expr(), expr.pattern()], Self { options }))
    }

    fn construct_fixed(
        self,
        [child, pattern]: [Expression; 2],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { options } = self;
        ensure_string(&child, ctx)?;
        ensure_string(&pattern, ctx)?;
        Ok(Like.new_expr(options, [child, pattern]))
    }
}

/// Convert a DataFusion `InListExpr` with checked variable arity.
pub type ConvertInList<'a> = ConvertibleVariableArity<InListConversion<'a>>;

/// Matching and construction state used by [`ConvertInList`].
pub struct InListConversion<'a> {
    negated: bool,
    elements: InListElements<'a>,
}

impl<'a> Convertible<'a> for InListConversion<'a> {
    type DFExpr = df_expr::InListExpr;

    fn maybe_convert(
        expr: &'a Self::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if expr.is_empty()
            || !expr.list().iter().all(|expr| expr.is::<df_expr::Literal>())
            || !supported_data_types(&expr.expr().data_type(ctx.schema)?)
        { return Err(Unconverted::Unsupported); }
        let field = expr.expr().return_field(ctx.schema)?;
        // Equal values with different extension metadata must remain distinct.
        let literals = expr.list().iter()
            .unique_by(|element| element.downcast_ref::<df_expr::Literal>().cloned())
            .collect_vec();
        let mut children = vec![expr.expr()];
        let elements = if field.extension_type_name().is_some() {
            InListElements::Imported { literals, field }
        } else {
            children.extend(literals);
            InListElements::Children
        };
        Ok((children, Self { negated: expr.negated(), elements }))
    }

    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { negated, elements } = self;
        let mut children = children.into_iter();
        let value = next_child(&mut children)?;
        if !is_infallible(&value) {
            return Err(Unconverted::Unsupported);
        }
        let value_dtype = node_dtype(&value, ctx)?;
        let elements = elements.construct(children, ctx)?;
        let membership = in_list_of(value, value_dtype, elements).map_err(|_| Unconverted::Unsupported)?;
        Ok(if negated {
            not(membership)
        } else {
            membership
        })
    }
}

/// Convert field access, capturing literal field names as configuration.
pub struct ConvertGetField<'a> {
    names: Vec<&'a str>,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertGetField<'a> {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [source, paths @ ..] = expr.args() else {
            return Err(exec_datafusion_err!("get_field requires a source and field path").into());
        };
        if paths.is_empty() {
            return Err(exec_datafusion_err!("get_field requires a field path").into());
        }
        let mut source_type = source.data_type(ctx.schema)?;
        let mut names = Vec::with_capacity(paths.len());
        for path in paths {
            let name = path.downcast_ref::<df_expr::Literal>()
                .and_then(|literal| literal.value().try_as_str().flatten())
                .ok_or_else(|| exec_datafusion_err!("get_field path must be a non-null string literal"))?;
            let DataType::Struct(fields) = &source_type else {
                return Err(Unconverted::Unsupported);
            };
            let (_, field) = fields.find(name).ok_or_else(|| exec_datafusion_err!("get_field references missing field {name}"))?;
            source_type = field.data_type().clone();
            names.push(name);
        }
        Ok((vec![source], Self { names }))
    }

    fn construct_fixed(
        self,
        [mut source]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { names } = self;
        for name in names {
            source = get_item(name, source);
        }
        Ok(source)
    }
}

/// Convert octet_length to the native byte length expression.
pub struct ConvertOctetLength {
    dtype: DType,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertOctetLength {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input] = expr.args() else {
            return Err(exec_datafusion_err!("octet_length requires exactly one argument").into());
        };
        let data_type = input.data_type(ctx.schema)?;
        let data_type = match &data_type { DataType::Dictionary(_, value) => value.as_ref(), data_type => data_type };
        if !data_type.is_binary() && !data_type.is_string() {
            return Err(Unconverted::Unsupported);
        }
        let dtype = arrow_dtype(&Field::new("", expr.return_type().clone(), expr.nullable()), ctx.session)?;
        Ok((vec![input], Self { dtype }))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { dtype } = self;
        Ok(cast(byte_length(input), dtype))
    }
}

/// Convert supported array_length calls to native list length.
pub struct ConvertArrayLength {
    dtype: DType,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertArrayLength {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<ArrayLength>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let input = array_length_input(expr)?.ok_or(Unconverted::Unsupported)?;
        if !matches!(input.data_type(ctx.schema)?, DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)) {
            return Err(Unconverted::Unsupported);
        }
        let dtype = arrow_dtype(&Field::new("", expr.return_type().clone(), expr.nullable()), ctx.session)?;
        Ok((vec![input], Self { dtype }))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { dtype } = self;
        Ok(cast(list_length(input), dtype))
    }
}

/// Convert lower on string expressions.
pub struct ConvertLower;

impl<'a> ConvertibleFixedArity<'a> for ConvertLower {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<LowerFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input] = expr.args() else {
            return Err(exec_datafusion_err!("lower requires exactly one argument").into());
        };
        Ok((vec![input], Self))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        ensure_string(&input, ctx)?;
        Ok(lower(input))
    }
}

/// Convert strpos, preserving the DataFusion return width.
pub struct ConvertStrpos {
    narrow: bool,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertStrpos {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 2];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<StrposFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input, search] = expr.args() else {
            return Err(exec_datafusion_err!("strpos requires exactly two arguments").into());
        };
        let narrow = match expr.return_type() { DataType::Int32 => true, DataType::Int64 => false, _ => return Err(Unconverted::Unsupported) };
        Ok((vec![input, search], Self { narrow }))
    }

    fn construct_fixed(
        self,
        [input, search]: [Expression; 2],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { narrow } = self;
        ensure_string(&input, ctx)?;
        ensure_string(&search, ctx)?;
        let position = strpos(input, search);
        if narrow {
            let nullable = node_dtype(&position, ctx)?.nullability();
            Ok(cast(position, DType::Primitive(PType::I32, nullable)))
        } else { Ok(position) }
    }
}

/// Convert contains with a literal needle to a native LIKE expression.
pub struct ConvertContains {
    pattern: String,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertContains {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<ContainsFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input, needle] = expr.args() else {
            return Err(exec_datafusion_err!("{} requires exactly two arguments", expr.name()).into());
        };
        let Some(Some(needle)) = needle.downcast_ref::<df_expr::Literal>().and_then(|literal| literal.value().try_as_str()) else {
            return Err(Unconverted::Unsupported);
        };
        let needle = escape_like(needle);
        let pattern = format!("%{needle}%");
        Ok((vec![input], Self { pattern }))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { pattern } = self;
        ensure_string(&input, ctx)?;
        Ok(Like.new_expr(LikeOptions::default(), [input, lit(pattern)]))
    }
}

/// Convert starts_with with a literal needle to a native LIKE expression.
pub struct ConvertStartsWith {
    pattern: String,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertStartsWith {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<StartsWithFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input, needle] = expr.args() else {
            return Err(exec_datafusion_err!("{} requires exactly two arguments", expr.name()).into());
        };
        let Some(Some(needle)) = needle.downcast_ref::<df_expr::Literal>().and_then(|literal| literal.value().try_as_str()) else {
            return Err(Unconverted::Unsupported);
        };
        let needle = escape_like(needle);
        let pattern = format!("{needle}%");
        Ok((vec![input], Self { pattern }))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { pattern } = self;
        ensure_string(&input, ctx)?;
        Ok(Like.new_expr(LikeOptions::default(), [input, lit(pattern)]))
    }
}

/// Convert ends_with with a literal needle to a native LIKE expression.
pub struct ConvertEndsWith {
    pattern: String,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertEndsWith {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<EndsWithFunc>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input, needle] = expr.args() else {
            return Err(exec_datafusion_err!("{} requires exactly two arguments", expr.name()).into());
        };
        let Some(Some(needle)) = needle.downcast_ref::<df_expr::Literal>().and_then(|literal| literal.value().try_as_str()) else {
            return Err(Unconverted::Unsupported);
        };
        let needle = escape_like(needle);
        let pattern = format!("%{needle}");
        Ok((vec![input], Self { pattern }))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { pattern } = self;
        ensure_string(&input, ctx)?;
        Ok(Like.new_expr(LikeOptions::default(), [input, lit(pattern)]))
    }
}

/// Convert array_sum with matching null and NaN handling.
pub struct ConvertArraySum;

impl<'a> ConvertibleFixedArity<'a> for ConvertArraySum {
    type DFExpr = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn maybe_convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if ScalarFunctionExpr::try_downcast_func::<ArraySum>(expr).is_none() {
            return Err(Unconverted::Unsupported);
        }
        let [input] = expr.args() else {
            return Err(exec_datafusion_err!("array_sum requires exactly one argument").into());
        };
        Ok((vec![input], Self))
    }

    fn construct_fixed(
        self,
        [input]: [Expression; 1],
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        Ok(list_sum_opts(input, NumericalAggregateOpts::include_nans()))
    }
}

/// Convert a DataFusion `CaseExpr` with checked variable arity.
pub type ConvertCase = ConvertibleVariableArity<CaseConversion>;

/// Matching and construction state used by [`ConvertCase`].
pub struct CaseConversion {
    num_pairs: usize,
}

impl<'a> Convertible<'a> for CaseConversion {
    type DFExpr = df_expr::CaseExpr;

    fn maybe_convert(
        expr: &'a Self::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        if expr.expr().is_some() {
            return Err(Unconverted::Unsupported);
        }
        let num_pairs = expr.when_then_expr().len();
        let mut children = Vec::with_capacity(num_pairs * 2 + 1);
        for (condition, value) in expr.when_then_expr() {
            if condition.data_type(ctx.schema)? != DataType::Boolean {
                return Err(exec_datafusion_err!("CASE WHEN must be Boolean").into());
            }
            children.extend([condition, value]);
        }
        children.extend(expr.else_expr());
        Ok((children, Self { num_pairs }))
    }

    fn construct(
        self,
        children: Vec<Expression>,
        _ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let Self { num_pairs } = self;
        let mut children = children.into_iter();
        let pairs = children.by_ref().take(num_pairs * 2).tuples().collect_vec();
        let otherwise = children.next();
        Ok(nested_case_when(pairs, otherwise))
    }
}

fn ensure_string(expr: &Expression, ctx: &ConvertibleCtx<'_>) -> Conversion<()> {
    if !node_dtype(expr, ctx)?.is_utf8() {
        return Err(Unconverted::Unsupported);
    }
    Ok(())
}

enum BinaryOperand<'a> {
    Child,
    Literal { literal: &'a df_expr::Literal, field: FieldRef },
}

impl<'a> BinaryOperand<'a> {
    fn select(expr: &'a Arc<dyn PhysicalExpr>, other: &Arc<dyn PhysicalExpr>, operator: Operator, schema: &Schema, children: &mut Vec<&'a Arc<dyn PhysicalExpr>>) -> Conversion<Self> {
        if operator.is_comparison()
            && let Some(literal) = expr.downcast_ref::<df_expr::Literal>()
            && literal.return_field(schema)?.extension_type_name().is_none()
        {
            let field = other.return_field(schema)?;
            if field.extension_type_name().is_some() {
                return Ok(Self::Literal { literal, field });
            }
        }
        children.push(expr);
        Ok(Self::Child)
    }

    fn construct(
        self,
        children: &mut impl Iterator<Item = Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        match self {
            Self::Child => next_child(children),
            Self::Literal { literal, field } => Ok(lit(import_literal_for_operand(literal, &field, ctx)?)),
        }
    }
}

enum InListElements<'a> {
    Children,
    Imported { literals: Vec<&'a Arc<dyn PhysicalExpr>>, field: FieldRef },
}

impl InListElements<'_> {
    fn construct(
        self,
        children: impl Iterator<Item = Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Vec<Scalar>> {
        match self {
            Self::Children => children.map(|expr| expr.as_opt::<Literal>().cloned().ok_or(Unconverted::Unsupported)).collect(),
            Self::Imported { literals, field } => literals.into_iter().map(|expr| {
                let literal = expr.downcast_ref::<df_expr::Literal>().ok_or(Unconverted::Unsupported)?;
                import_literal_for_operand(literal, &field, ctx)
            }).collect(),
        }
    }
}

fn import_literal_for_operand(
    literal: &df_expr::Literal,
    operand_field: &Field,
    ctx: &ConvertibleCtx<'_>,
) -> Conversion<Scalar> {
    DefaultExpressionConverter::new(ctx.session.clone())
        .import_literal_for_operand(literal, operand_field, ctx.schema)
}

fn import_literal(literal: &df_expr::Literal, field: &Field, session: &VortexSession) -> Conversion<Scalar> {
    DefaultExpressionConverter::new(session.clone()).import_literal(literal, field)
}

/// How a supported DataFusion cast is converted.
enum CastPushdown {
    /// The Arrow cast cannot fail and keeps the Vortex dtype, so the converter omits it.
    Drop,
    /// The cast becomes a Vortex `cast`.
    Keep,
}

/// Classifies a cast for pushdown; unsupported casts stay in DataFusion.
fn classify_cast(cast: &df_expr::CastExpr, schema: &Schema) -> Conversion<CastPushdown> {
    let options = cast.cast_options();
    if options.safe || options.format_options != DEFAULT_FORMAT_OPTIONS {
        return Err(Unconverted::Unsupported);
    }
    let source = cast.expr().data_type(schema)?;
    classify_cast_types(&source, cast.cast_type())
}

fn classify_cast_types(source: &DataType, target: &DataType) -> Conversion<CastPushdown> {
    use DataType::*;
    Ok(match (source, target) {
        _ if source == target => CastPushdown::Drop,
        // Views preserve narrow-offset strings without concatenating their values.
        (Utf8, Utf8View) => CastPushdown::Drop,
        // Decoding to views does not concatenate dictionary values into a narrow offset buffer.
        (Dictionary(_, values), Utf8View) if matches!(values.as_ref(), Utf8 | Utf8View) => {
            CastPushdown::Drop
        }
        // Run expansion gathers dictionary keys, then decodes them directly to views.
        (RunEndEncoded(run_ends, values), Utf8View)
            if matches!(run_ends.data_type(), Int16 | Int32)
                && matches!(values.data_type(), Dictionary(_, strings) if matches!(strings.as_ref(), Utf8 | Utf8View)) =>
        {
            CastPushdown::Drop
        }
        // Arrow expands runs by gathering with `Int32` indices, which fit Int16/Int32 run ends.
        // Gathering a dictionary only moves keys and reuses the values, so nothing can overflow.
        (RunEndEncoded(run_ends, values), Dictionary(..))
            if matches!(run_ends.data_type(), Int16 | Int32) && values.data_type() == target =>
        {
            CastPushdown::Drop
        }
        (Int8, Int16 | Int32 | Int64)
        | (Int16, Int32 | Int64)
        | (Int32, Int64)
        | (UInt8, UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64)
        | (UInt16, UInt32 | UInt64 | Int32 | Int64)
        | (UInt32, UInt64 | Int64)
        | (Float32, Float64) => CastPushdown::Keep,
        // Arrow and Vortex both convert with `as f64`: integers round to nearest and never fail,
        // booleans become 1.0 or 0.0.
        (Boolean | Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64, Float64) => {
            CastPushdown::Keep
        }
        // A list cast casts each element. Vortex has no `FixedSizeList` to `List` cast.
        (List(from) | LargeList(from), List(to) | LargeList(to))
            if to.is_nullable() || !from.is_nullable() =>
        {
            classify_cast_types(from.data_type(), to.data_type())?
        }
        _ => return Err(Unconverted::Unsupported),
    })
}

pub(super) fn try_operator_from_df(value: &DFOperator) -> Option<Operator> {
    match value {
        DFOperator::Eq => Some(Operator::Eq),
        DFOperator::NotEq => Some(Operator::NotEq),
        DFOperator::Lt => Some(Operator::Lt),
        DFOperator::LtEq => Some(Operator::Lte),
        DFOperator::Gt => Some(Operator::Gt),
        DFOperator::GtEq => Some(Operator::Gte),
        DFOperator::And => Some(Operator::And),
        DFOperator::Or => Some(Operator::Or),
        DFOperator::Plus => Some(Operator::Add),
        DFOperator::Minus => Some(Operator::Sub),
        DFOperator::Multiply => Some(Operator::Mul),
        DFOperator::Divide => Some(Operator::Div),
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
        | DFOperator::Colon => None,
    }
}

pub(super) fn supported_data_types(dt: &DataType) -> bool {
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

/// Returns the list argument of an `array_length` call that can be rewritten to `list_length`:
/// the single-argument form `array_length(arr)` or the equivalent explicit first dimension
/// `array_length(arr, 1)`. Other dimensions are unsupported (`None`); other arities are errors.
fn array_length_input(scalar_fn: &ScalarFunctionExpr) -> DFResult<Option<&Arc<dyn PhysicalExpr>>> {
    match scalar_fn.args() {
        [input] => Ok(Some(input)),
        [input, dimension]
            if dimension
                .downcast_ref::<df_expr::Literal>()
                .is_some_and(|literal| matches!(literal.value(), ScalarValue::Int64(Some(1)))) =>
        {
            Ok(Some(input))
        }
        [_, _] => Ok(None),
        _ => Err(exec_datafusion_err!(
            "array_length requires one or two arguments"
        )),
    }
}

