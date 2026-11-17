// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recognition and construction of each supported expression.

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
use datafusion_physical_expr::PhysicalExprRef;
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

use crate::convert::exprs::convertible::may;
use crate::convert::exprs::is_infallible;

use super::Constructible;
use super::Convertible;
use super::ConvertibleCtx;
use super::ConvertibleFixedArity;
use super::MayConvert;
use super::MustConvert;
use super::Unsupported;

type PER = PhysicalExprRef;
type Ctx<'a> = ConvertibleCtx<'a>;

/// Select the first supported converter, borrowing its inputs and boxing its construction state.
pub fn default_convert<'a>(
    expr: &'a PER,
    ctx: &Ctx<'_>,
) -> MayConvert<(Vec<&'a PhysicalExprRef>, Box<dyn Constructible + 'a>)> {
    ConvertBinary::recognize_boxed(expr, ctx)?;
    ConvertColumn::recognize_boxed(expr, ctx)?;
    ConvertLambdaVariable::recognize_boxed(expr, ctx)?;
    ConvertLiteral::recognize_boxed(expr, ctx)?;
    ConvertCast::recognize_boxed(expr, ctx)?;
    ConvertNot::recognize_boxed(expr, ctx)?;
    ConvertIsNull::recognize_boxed(expr, ctx)?;
    ConvertIsNotNull::recognize_boxed(expr, ctx)?;
    ConvertLike::recognize_boxed(expr, ctx)?;
    ConvertInList::recognize_boxed(expr, ctx)?;
    ConvertGetField::recognize_boxed(expr, ctx)?;
    ConvertOctetLength::recognize_boxed(expr, ctx)?;
    ConvertArrayLength::recognize_boxed(expr, ctx)?;
    ConvertLower::recognize_boxed(expr, ctx)?;
    ConvertStrpos::recognize_boxed(expr, ctx)?;
    ConvertContains::recognize_boxed(expr, ctx)?;
    ConvertStartsWith::recognize_boxed(expr, ctx)?;
    ConvertEndsWith::recognize_boxed(expr, ctx)?;
    ConvertArraySum::recognize_boxed(expr, ctx)?;
    ConvertCase::recognize_boxed(expr, ctx)?;
    Ok(Unsupported)
}

/// Convert a supported node using the caller's recursion for its selected children.
///
/// Inputs borrow from `expr`; converters remain on the stack and are consumed after recursion.
pub fn default_convert_recursive<'a>(
    expr: &'a PER,
    ctx: &Ctx<'_>,
    mut recur: impl FnMut(&'a PER) -> MustConvert<Expression>,
) -> MayConvert<Expression> {
    may(ConvertBinary::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertColumn::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertLambdaVariable::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertLiteral::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertCast::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertNot::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertIsNull::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertIsNotNull::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertLike::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertInList::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertGetField::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertOctetLength::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertArrayLength::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertLower::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertStrpos::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertContains::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertStartsWith::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertEndsWith::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertArraySum::convert_recursive(expr, ctx, &mut recur))?;
    may(ConvertCase::convert_recursive(expr, ctx, &mut recur))?;
    Ok(Unsupported)
}

/// Matching and construction state used by [`ConvertBinary`].
pub struct ConvertBinary {
    operator: Operator,
    left: Option<(df_expr::Literal, FieldRef)>,
    right: Option<(df_expr::Literal, FieldRef)>,
}

impl<'a> Convertible<'a> for ConvertBinary {
    type DF = df_expr::BinaryExpr;

    fn convert(expr: &'a Self::DF, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        let operator = try_operator_from_df(expr.op())?;
        let left_type = expr.left().data_type(ctx.schema).map_err(Err)?;
        let right_type = expr.right().data_type(ctx.schema).map_err(Err)?;

        if matches!(operator, Operator::And | Operator::Or)
            && !(left_type == DataType::Boolean && right_type == DataType::Boolean)
        {
            return Err(Err(exec_datafusion_err!(
                "Boolean operator requires Boolean operands: {expr}"
            )));
        }

        if !supported_data_types(&left_type) || !supported_data_types(&right_type) {
            return Err(Ok(Unsupported));
        }

        if !operator.is_comparison() {
            let children = vec![expr.left(), expr.right()];
            let convertible = ConvertBinary {
                operator,
                left: None,
                right: None,
            };
            return Ok((children, convertible));
        }

        let left_lit = recognize_literal(expr.left(), expr.right(), &ctx.schema).map_err(Err)?;
        let right_lit = recognize_literal(expr.right(), expr.left(), &ctx.schema).map_err(Err)?;
        let (children, left, right) = match (left_lit, right_lit) {
            (Some(left), Some(right)) => (Vec::new(), Some(left), Some(right)),
            (Some(left), None) => (vec![expr.right()], Some(left), None),
            (None, Some(right)) => (vec![expr.left()], None, Some(right)),
            (None, None) => (vec![expr.left(), expr.right()], None, None),
        };

        let convertible = ConvertBinary {
            operator,
            left,
            right,
        };
        Ok((children, convertible))
    }

    fn construct(self, children: Vec<Expression>, ctx: &Ctx) -> MustConvert<Expression> {
        let Self {
            operator,
            left,
            right,
        } = self;

        let n = children.len();
        let (left, right) = match (left, right) {
            (None, None) => {
                let Ok([left, right]) = <[_; 2]>::try_from(children) else {
                    return Err(Err(exec_datafusion_err!("Wrong number of converted binary operands: {n}")));
                };
                (left, right)
            }
            (None, Some(right)) => {
                let Ok([left]) = <[_; 1]>::try_from(children) else {
                    return Err(Err(exec_datafusion_err!("Wrong number of converted binary operands: {n}")));
                };
                let right = lit(ctx.import_literal_for_operand(&right.0, &right.1)?);
                (left, right)
            }
            (Some(left), None) => {
                let Ok([right]) = <[_; 1]>::try_from(children) else {
                    return Err(Err(exec_datafusion_err!("Wrong number of converted binary operands: {n}")));
                };
                let left = lit(ctx.import_literal_for_operand(&left.0, &left.1)?);
                (left, right)
            }
            (Some(left), Some(right)) => {
                let Ok([]) = <[_; 0]>::try_from(children) else {
                    return Err(Err(exec_datafusion_err!("Wrong number of converted binary operands: {n}")));
                };
                let left = lit(ctx.import_literal_for_operand(&left.0, &left.1)?);
                let right = lit(ctx.import_literal_for_operand(&right.0, &right.1)?);
                (left, right)
            }
        };

        // DataFusion may evaluate the RHS only on rows selected by the LHS.
        if matches!(operator, Operator::And | Operator::Or)
            && !(is_infallible(&left) && is_infallible(&right))
        {
            return Err(Ok(Unsupported));
        }
        if !ctx
            .node_dtype(&left)?
            .eq_ignore_nullability(&ctx.node_dtype(&right)?)
        {
            return Err(Ok(Unsupported));
        }

        Ok(Binary.new_expr(operator, [left, right]))
    }
}

fn recognize_literal(
    expr: &PER,
    other: &PER,
    schema: &Schema,
) -> DFResult<Option<(df_expr::Literal, FieldRef)>> {
    if let Some(literal) = expr.downcast_ref::<df_expr::Literal>()
        && literal
            .return_field(schema)?
            .extension_type_name()
            .is_none()
    {
        let field = other.return_field(schema)?;
        if field.extension_type_name().is_some() {
            return Ok(Some((literal.clone(), field)));
        }
    }

    Ok(None)
}

/// Convert a DataFusion column reference.
pub struct ConvertColumn<'a> {
    name: &'a str,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertColumn<'a> {
    type DF = df_expr::Column;
    type Children = [Expression; 0];

    fn convert_fixed(expr: &'a df_expr::Column, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if ctx.scope.is_some() {
            return Err(Ok(Unsupported));
        }
        // FIXME(DK): this should resolve by index into the schema
        let name = expr.name();
        Ok((Vec::new(), Self { name }))
    }

    fn construct_fixed(self, []: [Expression; 0], _ctx: &Ctx) -> MustConvert<Expression> {
        let Self { name } = self;
        Ok(get_item(name, root()))
    }
}

/// Convert a lambda parameter reference within the supplied scope.
pub struct ConvertLambdaVariable<'a> {
    name: &'a str,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertLambdaVariable<'a> {
    type DF = df_expr::LambdaVariable;
    type Children = [Expression; 0];

    fn convert_fixed(expr: &'a Self::DF, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        let Some(params) = ctx.scope else {
            return Err(Ok(Unsupported));
        };
        if params.find(expr.name()).is_none() {
            return Err(Ok(Unsupported));
        }
        let name = expr.name();
        Ok((Vec::new(), Self { name }))
    }

    fn construct_fixed(self, []: [Expression; 0], _ctx: &Ctx) -> MustConvert<Expression> {
        let Self { name } = self;
        Ok(get_item(name, root()))
    }
}

/// Import a DataFusion literal through the session's Arrow plugins.
pub struct ConvertLiteral {
    scalar: Scalar,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertLiteral {
    type DF = df_expr::Literal;
    type Children = [Expression; 0];

    fn convert_fixed(expr: &'a Self::DF, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        let field = expr.return_field(ctx.schema).map_err(Err)?;
        let scalar = ctx.import_literal(expr, &field)?;
        Ok((Vec::new(), Self { scalar }))
    }

    fn construct_fixed(self, []: [Expression; 0], _ctx: &Ctx) -> MustConvert<Expression> {
        let Self { scalar } = self;
        Ok(lit(scalar))
    }
}

/// Convert a supported DataFusion cast.
pub struct ConvertCast {
    pushdown: CastPushdown,
    dtype: DType,
}

/// How a supported DataFusion cast is converted.
enum CastPushdown {
    /// The Arrow cast cannot fail and keeps the Vortex dtype, so the converter omits it.
    Drop,
    /// The cast becomes a Vortex `cast`.
    Keep,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertCast {
    type DF = df_expr::CastExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a df_expr::CastExpr, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        let pushdown = classify_cast(expr, ctx.schema)?;
        let dtype = ctx
            .arrow_dtype(expr.target_field())?
            .with_nullability(Nullability::from(expr.nullable(ctx.schema).map_err(Err)?));
        Ok((vec![expr.expr()], Self { pushdown, dtype }))
    }

    fn construct_fixed(self, [child]: [Expression; 1], ctx: &Ctx) -> MustConvert<Expression> {
        let Self { pushdown, dtype } = self;
        let child_dtype = ctx.node_dtype(&child)?;
        let dtype_already_matches = child_dtype.eq_ignore_nullability(&dtype);
        match pushdown {
            // TODO(DK): why are we able to ignore nullability?
            CastPushdown::Drop if dtype_already_matches => Ok(child),
            CastPushdown::Drop => Err(Ok(Unsupported)),
            CastPushdown::Keep => {
                let either_is_extension = child_dtype.is_extension() || dtype.is_extension();
                // Matching storage types do not imply matching extension semantics.
                if either_is_extension && !dtype_already_matches {
                    return Err(Ok(Unsupported));
                }
                Ok(cast(child, dtype))
            }
        }
    }
}

/// Classifies a cast for pushdown; unsupported casts stay in DataFusion.
fn classify_cast(cast: &df_expr::CastExpr, schema: &Schema) -> MustConvert<CastPushdown> {
    let options = cast.cast_options();
    if options.safe || options.format_options != DEFAULT_FORMAT_OPTIONS {
        return Err(Ok(Unsupported));
    }
    let source = cast.expr().data_type(schema).map_err(Err)?;
    classify_cast_types(&source, cast.cast_type())
}

fn classify_cast_types(source: &DataType, target: &DataType) -> MustConvert<CastPushdown> {
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
        _ => return Err(Ok(Unsupported)),
    })
}

/// Convert a Boolean negation.
pub struct ConvertNot;

impl<'a> ConvertibleFixedArity<'a> for ConvertNot {
    type DF = df_expr::NotExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a Self::DF, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if expr.arg().data_type(ctx.schema).map_err(Err)? != DataType::Boolean {
            return Err(Err(exec_datafusion_err!(
                "NOT requires a Boolean operand: {expr}"
            )));
        }
        Ok((vec![expr.arg()], Self))
    }

    fn construct_fixed(self, [arg]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
        Ok(not(arg))
    }
}

/// Convert an IS NULL predicate.
pub struct ConvertIsNull;

impl<'a> ConvertibleFixedArity<'a> for ConvertIsNull {
    type DF = df_expr::IsNullExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a Self::DF, _ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        Ok((vec![expr.arg()], Self))
    }

    fn construct_fixed(self, [arg]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
        Ok(is_null(arg))
    }
}

/// Convert an IS NOT NULL predicate.
pub struct ConvertIsNotNull;

impl<'a> ConvertibleFixedArity<'a> for ConvertIsNotNull {
    type DF = df_expr::IsNotNullExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a Self::DF, _ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        Ok((vec![expr.arg()], Self))
    }

    fn construct_fixed(self, [arg]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
        Ok(is_not_null(arg))
    }
}

/// Convert LIKE and ILIKE expressions.
pub struct ConvertLike {
    options: LikeOptions,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertLike {
    type DF = df_expr::LikeExpr;
    type Children = [Expression; 2];

    fn convert_fixed(expr: &'a Self::DF, _ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        let options = LikeOptions {
            negated: expr.negated(),
            case_insensitive: expr.case_insensitive(),
        };
        Ok((vec![expr.expr(), expr.pattern()], Self { options }))
    }

    fn construct_fixed(
        self,
        [child, pattern]: [Expression; 2],
        ctx: &Ctx,
    ) -> MustConvert<Expression> {
        let Self { options } = self;
        ensure_string(&child, ctx)?;
        ensure_string(&pattern, ctx)?;
        Ok(Like.new_expr(options, [child, pattern]))
    }
}

/// Matching and construction state used by [`ConvertInList`].
pub struct ConvertInList {
    negated: bool,
    elements: Option<Vec<Scalar>>,
    n_children: usize,
}

impl<'a> Convertible<'a> for ConvertInList {
    type DF = df_expr::InListExpr;

    fn convert(expr: &'a Self::DF, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if expr.is_empty()
            || !expr.list().iter().all(|expr| expr.is::<df_expr::Literal>())
            || !supported_data_types(&expr.expr().data_type(ctx.schema).map_err(Err)?)
        {
            return Err(Ok(Unsupported));
        }
        let field = expr.expr().return_field(ctx.schema).map_err(Err)?;

        // Some Arrow types, like FixedSizeBinary, cannot be represented in Vortex; however, a
        // FixedSizeBinary with Arrow metadata indicating it is a "UUID" array, _can_ be represented
        // in Vortex as a UUID array.
        //
        // At `convert` time, we have the needle's FieldRef and therefore know its Arrow
        // metadata. DF frequently loses metadata, so we make a best effort to propagate that UUID
        // information from the needle to the elements.
        //
        // When our caller recursively converts the children, they lack this context, so we do it
        // for them.
        let literals = expr
            .list()
            .iter()
            .map(|element| element.downcast_ref::<df_expr::Literal>())
            .map(|maybe_literal| match maybe_literal {
                None => Err(Ok(Unsupported)),
                Some(literal) => ctx.import_literal_for_operand(literal, &field),
            })
            .collect::<MustConvert<Vec<_>>>();

        let mut children = vec![expr.expr()];
        let elements = match literals {
            Ok(literals) => Some(literals),
            Err(Ok(Unsupported)) => {
                children.extend(expr.list());
                None
            }
            Err(Err(err)) => return Err(Err(err)),
        };
        let convertible = Self {
            negated: expr.negated(),
            elements,
            n_children: children.len(),
        };
        Ok((children, convertible))
    }

    fn construct(self, children: Vec<Expression>, ctx: &Ctx) -> MustConvert<Expression> {
        let Self {
            negated,
            elements,
            n_children,
        } = self;

        let n = children.len();
        if n_children != n {
            return Err(Err(exec_datafusion_err!("Expected {n_children} converted children, found {n}")));
        }

        let mut children = children.into_iter();
        let value = children.next().expect("checked length above");
        if !is_infallible(&value) {
            return Err(Ok(Unsupported));
        }
        let value_dtype = ctx.node_dtype(&value)?;

        let elements = if let Some(literals) = elements {
            literals
        } else {
            let literals = children
                .map(|expr| expr.as_opt::<Literal>().cloned())
                .collect::<Option<Vec<_>>>();
            let Some(literals) = literals else {
                return Err(Ok(Unsupported));
            };
            literals
        };

        let membership = in_list_of(value, value_dtype, elements).map_err(|_| Ok(Unsupported))?;
        Ok(if negated { not(membership) } else { membership })
    }
}

/// Convert field access, capturing literal field names as configuration.
pub struct ConvertGetField<'a> {
    names: Vec<&'a str>,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertGetField<'a> {
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a ScalarFunctionExpr, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<GetFieldFunc>() {
            return Err(Ok(Unsupported));
        }
        let [source, paths @ ..] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "get_field requires a source and field path"
            )));
        };
        if paths.is_empty() {
            return Err(Err(exec_datafusion_err!("get_field requires a field path")));
        }
        let mut source_type = source.data_type(ctx.schema).map_err(Err)?;
        let mut names = Vec::with_capacity(paths.len());
        for path in paths {
            let Some(name) = path
                .downcast_ref::<df_expr::Literal>()
                .and_then(|literal| literal.value().try_as_str().flatten())
            else {
                return Err(Err(exec_datafusion_err!(
                    "get_field path must be a non-null string literal"
                )));
            };
            let DataType::Struct(fields) = &source_type else {
                return Err(Ok(Unsupported));
            };
            let Some((_, field)) = fields.find(name) else {
                return Err(Err(exec_datafusion_err!(
                    "get_field references missing field {name}"
                )));
            };
            source_type = field.data_type().clone();
            names.push(name);
        }
        Ok((vec![source], Self { names }))
    }

    fn construct_fixed(self, [mut source]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
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
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a ScalarFunctionExpr, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<OctetLengthFunc>() {
            return Err(Ok(Unsupported));
        }
        let [input] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "octet_length requires exactly one argument"
            )));
        };
        let data_type = input.data_type(ctx.schema).map_err(Err)?;
        let data_type = match &data_type {
            DataType::Dictionary(_, value) => value.as_ref(),
            data_type => data_type,
        };
        if !data_type.is_binary() && !data_type.is_string() {
            return Err(Ok(Unsupported));
        }
        let dtype =
            ctx.arrow_dtype(&Field::new("", expr.return_type().clone(), expr.nullable()))?;
        Ok((vec![input], Self { dtype }))
    }

    fn construct_fixed(self, [input]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
        let Self { dtype } = self;
        Ok(cast(byte_length(input), dtype))
    }
}

/// Convert supported array_length calls to native list length.
pub struct ConvertArrayLength {
    dtype: DType,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertArrayLength {
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(expr: &'a ScalarFunctionExpr, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<ArrayLength>() {
            return Err(Ok(Unsupported));
        }
        let input = array_length_input(expr)?;
        let input_dtype = input.data_type(ctx.schema).map_err(Err)?;
        if !matches!(
            input_dtype,
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
        ) {
            return Err(Ok(Unsupported));
        }
        let dtype =
            ctx.arrow_dtype(&Field::new("", expr.return_type().clone(), expr.nullable()))?;
        Ok((vec![input], Self { dtype }))
    }

    fn construct_fixed(self, [input]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
        let Self { dtype } = self;
        Ok(cast(list_length(input), dtype))
    }
}

/// Convert lower on string expressions.
pub struct ConvertLower;

impl<'a> ConvertibleFixedArity<'a> for ConvertLower {
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &Ctx,
    ) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<LowerFunc>() {
            return Err(Ok(Unsupported));
        }
        let [input] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "lower requires exactly one argument"
            )));
        };
        Ok((vec![input], Self))
    }

    fn construct_fixed(self, [input]: [Expression; 1], ctx: &Ctx) -> MustConvert<Expression> {
        ensure_string(&input, ctx)?;
        Ok(lower(input))
    }
}

/// Convert strpos, preserving the DataFusion return width.
pub struct ConvertStrpos {
    narrow: bool,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertStrpos {
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 2];

    fn convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &Ctx,
    ) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<StrposFunc>() {
            return Err(Ok(Unsupported));
        }
        let [input, search] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "strpos requires exactly two arguments"
            )));
        };
        let narrow = match expr.return_type() {
            DataType::Int32 => true,
            DataType::Int64 => false,
            _ => return Err(Ok(Unsupported)),
        };
        Ok((vec![input, search], Self { narrow }))
    }

    fn construct_fixed(
        self,
        [input, search]: [Expression; 2],
        ctx: &Ctx,
    ) -> MustConvert<Expression> {
        let Self { narrow } = self;
        ensure_string(&input, ctx)?;
        ensure_string(&search, ctx)?;
        let position = strpos(input, search);
        if narrow {
            let nullable = ctx.node_dtype(&position)?.nullability();
            Ok(cast(position, DType::Primitive(PType::I32, nullable)))
        } else {
            Ok(position)
        }
    }
}

/// Convert contains with a literal needle to a native LIKE expression.
pub struct ConvertContains {
    pattern: String,
}

impl<'a> ConvertibleFixedArity<'a> for ConvertContains {
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &Ctx,
    ) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<ContainsFunc>() {
            return Err(Ok(Unsupported));
        }
        let [input, needle] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "{} requires exactly two arguments",
                expr.name()
            )
            .into()));
        };
        let Some(Some(needle)) = needle
            .downcast_ref::<df_expr::Literal>()
            .and_then(|literal| literal.value().try_as_str())
        else {
            return Err(Ok(Unsupported));
        };
        let needle = escape_like(needle);
        let pattern = format!("%{needle}%");
        Ok((vec![input], Self { pattern }))
    }

    fn construct_fixed(self, [input]: [Expression; 1], ctx: &Ctx) -> MustConvert<Expression> {
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
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &Ctx,
    ) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<StartsWithFunc>() {
            return Err(Ok(Unsupported));
        }
        let [input, needle] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "{} requires exactly two arguments",
                expr.name()
            )
            .into()));
        };
        let Some(Some(needle)) = needle
            .downcast_ref::<df_expr::Literal>()
            .and_then(|literal| literal.value().try_as_str())
        else {
            return Err(Ok(Unsupported));
        };
        let needle = escape_like(needle);
        let pattern = format!("{needle}%");
        Ok((vec![input], Self { pattern }))
    }

    fn construct_fixed(self, [input]: [Expression; 1], ctx: &Ctx) -> MustConvert<Expression> {
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
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &Ctx,
    ) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<EndsWithFunc>() {
            return Err(Ok(Unsupported));
        }
        let [input, needle] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "{} requires exactly two arguments",
                expr.name()
            )
            .into()));
        };
        let Some(Some(needle)) = needle
            .downcast_ref::<df_expr::Literal>()
            .and_then(|literal| literal.value().try_as_str())
        else {
            return Err(Ok(Unsupported));
        };
        let needle = escape_like(needle);
        let pattern = format!("%{needle}");
        Ok((vec![input], Self { pattern }))
    }

    fn construct_fixed(self, [input]: [Expression; 1], ctx: &Ctx) -> MustConvert<Expression> {
        let Self { pattern } = self;
        ensure_string(&input, ctx)?;
        Ok(Like.new_expr(LikeOptions::default(), [input, lit(pattern)]))
    }
}

/// Convert array_sum with matching null and NaN handling.
pub struct ConvertArraySum;

impl<'a> ConvertibleFixedArity<'a> for ConvertArraySum {
    type DF = ScalarFunctionExpr;
    type Children = [Expression; 1];

    fn convert_fixed(
        expr: &'a ScalarFunctionExpr,
        _ctx: &Ctx,
    ) -> MustConvert<(Vec<&'a PER>, Self)> {
        if !expr.fun().inner().is::<ArraySum>() {
            return Err(Ok(Unsupported));
        }
        let [input] = expr.args() else {
            return Err(Err(exec_datafusion_err!(
                "array_sum requires exactly one argument"
            )
            .into()));
        };
        Ok((vec![input], Self))
    }

    fn construct_fixed(self, [input]: [Expression; 1], _ctx: &Ctx) -> MustConvert<Expression> {
        Ok(list_sum_opts(input, NumericalAggregateOpts::include_nans()))
    }
}

/// Matching and construction state used by [`ConvertCase`].
pub struct ConvertCase {
    num_pairs: usize,
    n_children: usize,
}

impl<'a> Convertible<'a> for ConvertCase {
    type DF = df_expr::CaseExpr;

    fn convert(expr: &'a Self::DF, ctx: &Ctx) -> MustConvert<(Vec<&'a PER>, Self)> {
        if expr.expr().is_some() {
            return Err(Ok(Unsupported));
        }
        let num_pairs = expr.when_then_expr().len();
        let mut children = Vec::with_capacity(num_pairs * 2 + 1);
        for (condition, value) in expr.when_then_expr() {
            if condition.data_type(ctx.schema).map_err(Err)? != DataType::Boolean {
                return Err(Err(exec_datafusion_err!("CASE WHEN must be Boolean").into()));
            }
            children.extend([condition, value]);
        }
        children.extend(expr.else_expr());
        let n_children = children.len();
        Ok((
            children,
            Self {
                num_pairs,
                n_children,
            },
        ))
    }

    fn construct(self, children: Vec<Expression>, _ctx: &Ctx) -> MustConvert<Expression> {
        let Self {
            num_pairs,
            n_children,
        } = self;
        let n = children.len();
        if n != n_children {
            return Err(Err(exec_datafusion_err!("Expected {n_children} converted children, found {n}")));
        }
        let mut children = children.into_iter();
        let pairs = children.by_ref().take(num_pairs * 2).tuples().collect_vec();
        let otherwise = children.next();
        Ok(nested_case_when(pairs, otherwise))
    }
}

/// Escapes the LIKE wildcards and escape character in `literal`.
fn escape_like(literal: &str) -> String {
    let mut escaped = String::with_capacity(literal.len());
    for c in literal.chars() {
        if matches!(c, '%' | '_' | '\\') {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

fn ensure_string(expr: &Expression, ctx: &Ctx) -> MustConvert<()> {
    if !ctx.node_dtype(expr)?.is_utf8() {
        return Err(Ok(Unsupported));
    }
    Ok(())
}

pub(crate) fn try_operator_from_df(value: &DFOperator) -> MustConvert<Operator> {
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
        | DFOperator::Colon => Err(Ok(Unsupported)),
    }
}

pub(crate) fn supported_data_types(dt: &DataType) -> bool {
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
fn array_length_input(scalar_fn: &ScalarFunctionExpr) -> MustConvert<&PER> {
    match scalar_fn.args() {
        [input] => Ok(input),
        [input, dimension]
            if dimension
                .downcast_ref::<df_expr::Literal>()
                .is_some_and(|literal| matches!(literal.value(), ScalarValue::Int64(Some(1)))) =>
        {
            Ok(input)
        }
        [_, _] => Err(Ok(Unsupported)),
        _ => Err(Err(exec_datafusion_err!(
            "array_length requires one or two arguments"
        ))),
    }
}
