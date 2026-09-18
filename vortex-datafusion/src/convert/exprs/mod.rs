// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_common::format::DEFAULT_FORMAT_OPTIONS;
use datafusion_expr::Operator as DFOperator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions::string::octet_length::OctetLengthFunc;
use datafusion_functions_nested::length::ArrayLength;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::analysis::label_infallible;
use vortex::expr::and_collect;
use vortex::expr::byte_length;
use vortex::expr::cast;
use vortex::expr::get_item;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_length;
use vortex::expr::lit;
use vortex::expr::nested_case_when;
use vortex::expr::or_collect;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

#[cfg(test)]
mod tests;

/// Result of splitting a projection into Vortex expressions and leftover DataFusion projections.
pub struct ProcessedProjection {
    /// Projection evaluated by the Vortex scan.
    pub scan_projection: Expression,
    /// Arrow reference types and metadata for the scan output.
    pub scan_reference_schema: Schema,
    /// Projection evaluated by DataFusion after the Vortex scan.
    pub leftover_projection: ProjectionExprs,
}

/// Trait for converting DataFusion expressions to Vortex ones.
///
/// Custom converters implement a single schema-aware decision. Conversion should preserve
/// DataFusion values, nulls, and evaluation errors; see [`DefaultExpressionConverter`] for
/// the arithmetic and nested field access exceptions. Unsupported expressions remain in
/// DataFusion, including when a file's schema adapter introduces them.
///
/// # Implementing a custom converter
///
/// ```
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use datafusion_common::Result as DFResult;
/// use datafusion_physical_expr::PhysicalExpr;
/// use vortex::expr::Expression;
/// use vortex_datafusion::convert::DefaultExpressionConverter;
/// use vortex_datafusion::convert::ExpressionConverter;
///
/// struct CustomExpressionConverter(DefaultExpressionConverter);
///
/// impl ExpressionConverter for CustomExpressionConverter {
///     fn try_convert(
///         &self,
///         expr: &Arc<dyn PhysicalExpr>,
///         schema: &Schema,
///     ) -> DFResult<Option<Expression>> {
///         self.0.try_convert(expr, schema)
///     }
/// }
///
/// let _converter: Arc<dyn ExpressionConverter> = Arc::new(CustomExpressionConverter(
///     DefaultExpressionConverter::default(),
/// ));
/// ```
pub trait ExpressionConverter: Send + Sync {
    /// Convert an expression for native evaluation against this schema.
    ///
    /// Returns None for valid but unsupported expressions. Malformed expressions and
    /// conversion failures return errors. Callers must retain unsupported exact predicates
    /// for DataFusion evaluation.
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>>;

    /// Split a projection into Vortex expressions that can be pushed down and leftover
    /// DataFusion projections that need to be evaluated after the scan.
    ///
    /// If any expression is unsupported, evaluate the complete projection in DataFusion
    /// over deduplicated raw inputs. This avoids mixing input names with output aliases.
    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        // Duplicate output names cannot identify native pack fields unambiguously.
        if !source_projection
            .iter()
            .map(|projection| &projection.alias)
            .all_unique()
        {
            return self.no_pushdown_projection(source_projection, input_schema);
        }
        let mut scan_projection = Vec::with_capacity(source_projection.as_ref().len());
        for projection in source_projection.as_ref() {
            let Some(expr) = self.try_convert(&projection.expr, input_schema)? else {
                return self.no_pushdown_projection(source_projection, input_schema);
            };
            scan_projection.push((projection.alias.clone(), expr));
        }
        // The output schema names its fields after the projection aliases.
        let output_indices = (0..scan_projection.len()).collect_vec();
        Ok(ProcessedProjection {
            scan_projection: pack(scan_projection, Nullability::NonNullable),
            scan_reference_schema: output_schema.clone(),
            leftover_projection: ProjectionExprs::from_indices(&output_indices, output_schema),
        })
    }

    /// Create a projection that reads only the required columns without pushing down
    /// any expressions. All projection logic is applied after the scan.
    fn no_pushdown_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        let (scan_projection, scan_reference_schema) =
            raw_projection(&source_projection.column_indices(), input_schema)?;
        Ok(ProcessedProjection {
            scan_projection,
            scan_reference_schema,
            leftover_projection: source_projection,
        })
    }
}

/// Read the raw columns at `indices` in file order, without involving custom expression
/// conversion. Returns the scan projection and the Arrow schema of its output.
pub(crate) fn raw_projection(
    indices: &[usize],
    input_schema: &Schema,
) -> DFResult<(Expression, Schema)> {
    let schema = input_schema.project(indices)?;
    let scan_columns = schema.fields().iter().map(|field| {
        (
            field.name().clone(),
            get_item(field.name().as_str(), root()),
        )
    });
    Ok((pack(scan_columns, Nullability::NonNullable), schema))
}

/// Why an expression was not converted.
enum Unconverted {
    /// A valid expression that Vortex cannot evaluate; it stays in DataFusion.
    Unsupported,
    /// A malformed expression or a failed conversion.
    Failed(DataFusionError),
}

impl From<DataFusionError> for Unconverted {
    fn from(error: DataFusionError) -> Self {
        Self::Failed(error)
    }
}

/// Conversion result where `?` propagates both unsupported expressions and errors.
type Conversion<T> = Result<T, Unconverted>;

/// The default [`ExpressionConverter`] implementation.
///
/// Supported arithmetic is pushed down using Vortex semantics, including its checked
/// integer arithmetic. Matching DataFusion's overflow behavior is deferred to a future patch.
/// Nested struct field access also uses Vortex semantics: a null parent produces a null child,
/// whereas DataFusion extracts the child without applying parent validity. These differences
/// are accepted for both filters and projections.
/// Other expressions require compatible SQL semantics or remain in DataFusion.
pub struct DefaultExpressionConverter {
    /// Session used to resolve Arrow → Vortex dtypes through the extension
    /// plugin registry, so registered extension types (e.g. UUID ⇄
    /// `FixedSizeBinary[16]`) convert correctly instead of hitting the static,
    /// non-plugin-aware `DType::from_arrow`.
    session: VortexSession,
}

impl Default for DefaultExpressionConverter {
    fn default() -> Self {
        Self {
            session: VortexSession::default(),
        }
    }
}

impl DefaultExpressionConverter {
    /// Create a converter that resolves Arrow extension types using `session`'s
    /// dtype registry.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Resolve an Arrow field through the session registry; unknown types are unsupported.
    fn arrow_dtype(&self, field: &Field) -> Conversion<DType> {
        self.session
            .arrow()
            .from_arrow_field(field)
            .map_err(|_| Unconverted::Unsupported)
    }

    /// Convert `expr` and check that it returns the dtype DataFusion expects.
    fn convert_checked(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Conversion<Expression> {
        let columns = collect_columns(expr);
        let mut column_indices = Vec::with_capacity(columns.len());
        for column in columns {
            let field = schema.fields().get(column.index()).ok_or_else(|| {
                exec_datafusion_err!(
                    "Column {}@{} is out of bounds",
                    column.name(),
                    column.index()
                )
            })?;
            if field.name() != column.name() {
                return Err(exec_datafusion_err!(
                    "Column {}@{} refers to field {}",
                    column.name(),
                    column.index(),
                    field.name()
                )
                .into());
            }
            column_indices.push(column.index());
        }
        column_indices.sort_unstable();
        column_indices.dedup();
        let input_dtype = self
            .session
            .arrow()
            .from_arrow_schema(
                &schema
                    .project(&column_indices)
                    .map_err(DataFusionError::from)?,
            )
            .map_err(|_| Unconverted::Unsupported)?;
        let converted = self.convert_expr(expr, schema, &input_dtype)?;
        let expected_dtype = self.arrow_dtype(expr.return_field(schema)?.as_ref())?;
        if !converted_dtype(&converted, &input_dtype)?.eq_ignore_nullability(&expected_dtype) {
            return Err(Unconverted::Unsupported);
        }
        Ok(converted)
    }

    fn convert_expr(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        if let Some(binary_expr) = expr.downcast_ref::<df_expr::BinaryExpr>() {
            let operator =
                try_operator_from_df(binary_expr.op()).ok_or(Unconverted::Unsupported)?;
            let boolean_operator = matches!(operator, Operator::And | Operator::Or);
            let left_type = binary_expr.left().data_type(schema)?;
            let right_type = binary_expr.right().data_type(schema)?;
            if boolean_operator {
                if left_type != DataType::Boolean || right_type != DataType::Boolean {
                    return Err(exec_datafusion_err!(
                        "Boolean operator requires Boolean operands: {expr}"
                    )
                    .into());
                }
            } else if !supported_data_types(&left_type) || !supported_data_types(&right_type) {
                return Err(Unconverted::Unsupported);
            }
            let left = self.convert_expr(binary_expr.left(), schema, input_dtype)?;
            let right = self.convert_expr(binary_expr.right(), schema, input_dtype)?;
            // DataFusion may evaluate the RHS only on rows selected by the LHS.
            if boolean_operator && !(is_infallible(&left) && is_infallible(&right)) {
                return Err(Unconverted::Unsupported);
            }
            if !converted_dtype(&left, input_dtype)?
                .eq_ignore_nullability(&converted_dtype(&right, input_dtype)?)
            {
                return Err(Unconverted::Unsupported);
            }
            return Ok(Binary.new_expr(operator, [left, right]));
        }

        if let Some(col_expr) = expr.downcast_ref::<df_expr::Column>() {
            return Ok(get_item(col_expr.name(), root()));
        }

        if let Some(literal) = expr.downcast_ref::<df_expr::Literal>() {
            let field = literal.return_field(schema)?;
            let array = literal.value().to_array()?;
            if array.len() != 1 {
                return Err(exec_datafusion_err!(
                    "Literal must contain exactly one value, found {}",
                    array.len()
                )
                .into());
            }
            // Literals of unknown Arrow types stay in DataFusion; conversion failures are errors.
            self.arrow_dtype(&field)?;
            let scalar = self
                .session
                .arrow()
                .from_arrow_array(array, &field)
                .and_then(|array| array.execute_scalar(0, &mut self.session.create_execution_ctx()))
                .map_err(|e| exec_datafusion_err!("Failed to convert literal {expr}: {e}"))?;
            return Ok(lit(scalar));
        }

        if let Some(cast_expr) = expr.downcast_ref::<df_expr::CastExpr>() {
            if !supported_cast(cast_expr, schema)? {
                return Err(Unconverted::Unsupported);
            }
            let child = self.convert_expr(cast_expr.expr(), schema, input_dtype)?;
            // DataFusion casts preserve input nulls even when the target field is declared
            // non-nullable. Match the expression's runtime nullability rather than narrowing
            // to the target field's metadata.
            let cast_dtype = self
                .arrow_dtype(cast_expr.target_field())?
                .with_nullability(Nullability::from(cast_expr.nullable(schema)?));
            // Matching Arrow storage types do not imply matching extension semantics.
            let child_dtype = converted_dtype(&child, input_dtype)?;
            if (child_dtype.is_extension() || cast_dtype.is_extension())
                && !child_dtype.eq_ignore_nullability(&cast_dtype)
            {
                return Err(Unconverted::Unsupported);
            }
            return Ok(cast(child, cast_dtype));
        }

        if let Some(is_null_expr) = expr.downcast_ref::<df_expr::IsNullExpr>() {
            let arg = self.convert_expr(is_null_expr.arg(), schema, input_dtype)?;
            return Ok(is_null(arg));
        }

        if let Some(is_not_null_expr) = expr.downcast_ref::<df_expr::IsNotNullExpr>() {
            let arg = self.convert_expr(is_not_null_expr.arg(), schema, input_dtype)?;
            return Ok(is_not_null(arg));
        }

        if let Some(like) = expr.downcast_ref::<df_expr::LikeExpr>() {
            if !like.expr().data_type(schema)?.is_string()
                || !like.pattern().data_type(schema)?.is_string()
            {
                return Err(Unconverted::Unsupported);
            }
            let child = self.convert_expr(like.expr(), schema, input_dtype)?;
            let pattern = self.convert_expr(like.pattern(), schema, input_dtype)?;
            return Ok(Like.new_expr(
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
                [child, pattern],
            ));
        }

        if let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>() {
            return self.convert_in_list(in_list, schema, input_dtype);
        }

        if let Some(scalar_fn) = expr.downcast_ref::<ScalarFunctionExpr>() {
            return self.convert_scalar_function(scalar_fn, schema, input_dtype);
        }

        if let Some(case_expr) = expr.downcast_ref::<df_expr::CaseExpr>() {
            return self.convert_case_expr(case_expr, schema, input_dtype);
        }

        Err(Unconverted::Unsupported)
    }

    fn convert_in_list(
        &self,
        in_list: &df_expr::InListExpr,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        if in_list.is_empty()
            || !in_list
                .list()
                .iter()
                .all(|expr| expr.is::<df_expr::Literal>())
            || !supported_data_types(&in_list.expr().data_type(schema)?)
        {
            return Err(Unconverted::Unsupported);
        }
        let value = self.convert_expr(in_list.expr(), schema, input_dtype)?;
        // Boolean rewrites may skip evaluating the input, particularly for all-null lists.
        if !is_infallible(&value) {
            return Err(Unconverted::Unsupported);
        }
        let value_dtype = converted_dtype(&value, input_dtype)?;
        let operator = if in_list.negated() {
            Operator::NotEq
        } else {
            Operator::Eq
        };
        let mut comparisons = Vec::with_capacity(in_list.len());
        for element in in_list.list() {
            let element = self.convert_expr(element, schema, input_dtype)?;
            let element_dtype = converted_dtype(&element, input_dtype)?;
            if element_dtype == DType::Null {
                comparisons.push(lit(None::<bool>));
            } else if value_dtype.eq_ignore_nullability(&element_dtype) {
                comparisons.push(Binary.new_expr(operator, [value.clone(), element]));
            } else {
                return Err(Unconverted::Unsupported);
            }
        }
        // Kleene AND/OR preserve SQL IN/NOT IN nulls; list_contains does not.
        let membership = if in_list.negated() {
            and_collect(comparisons)
        } else {
            or_collect(comparisons)
        };
        membership.ok_or(Unconverted::Unsupported)
    }

    fn convert_scalar_function(
        &self,
        scalar_fn: &ScalarFunctionExpr,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        if ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn).is_some() {
            // DataFusion's GetFieldFunc flattens nested field access into a single call
            // with multiple field name arguments, e.g. `outer.inner.leaf` becomes
            // get_field(Column("outer"), "inner", "leaf").
            let [source, paths @ ..] = scalar_fn.args() else {
                return Err(
                    exec_datafusion_err!("get_field requires a source and field path").into(),
                );
            };
            if paths.is_empty() {
                return Err(exec_datafusion_err!("get_field requires a field path").into());
            }
            let mut source_type = source.data_type(schema)?;
            let mut names = Vec::with_capacity(paths.len());
            for path in paths {
                let name = path
                    .downcast_ref::<df_expr::Literal>()
                    .and_then(|literal| literal.value().try_as_str().flatten())
                    .ok_or_else(|| {
                        exec_datafusion_err!("get_field path must be a non-null string literal")
                    })?;
                let DataType::Struct(fields) = &source_type else {
                    return Err(Unconverted::Unsupported);
                };
                let (_, field) = fields.find(name).ok_or_else(|| {
                    exec_datafusion_err!("get_field references missing field {name}")
                })?;
                source_type = field.data_type().clone();
                names.push(name);
            }
            let mut result = self.convert_expr(source, schema, input_dtype)?;
            for name in names {
                result = get_item(name, result);
            }
            return Ok(result);
        }

        let (input, length): (_, fn(Expression) -> Expression) =
            if ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(scalar_fn).is_some() {
                let [input] = scalar_fn.args() else {
                    return Err(
                        exec_datafusion_err!("octet_length requires exactly one argument").into(),
                    );
                };
                let data_type = input.data_type(schema)?;
                let data_type = match &data_type {
                    DataType::Dictionary(_, value) => value.as_ref(),
                    data_type => data_type,
                };
                if !data_type.is_binary() && !data_type.is_string() {
                    return Err(Unconverted::Unsupported);
                }
                (input, byte_length)
            } else if ScalarFunctionExpr::try_downcast_func::<ArrayLength>(scalar_fn).is_some() {
                let input = array_length_input(scalar_fn)?.ok_or(Unconverted::Unsupported)?;
                if !matches!(
                    input.data_type(schema)?,
                    DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
                ) {
                    return Err(Unconverted::Unsupported);
                }
                (input, list_length)
            } else {
                return Err(Unconverted::Unsupported);
            };
        let input = self.convert_expr(input, schema, input_dtype)?;
        let return_dtype = self.arrow_dtype(&Field::new(
            "",
            scalar_fn.return_type().clone(),
            scalar_fn.nullable(),
        ))?;
        Ok(cast(length(input), return_dtype))
    }

    fn convert_case_expr(
        &self,
        case_expr: &df_expr::CaseExpr,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        // Only the searched form (CASE WHEN cond THEN value ...) is supported.
        if case_expr.expr().is_some() {
            return Err(Unconverted::Unsupported);
        }
        let mut pairs = Vec::with_capacity(case_expr.when_then_expr().len());
        for (when_expr, then_expr) in case_expr.when_then_expr() {
            if when_expr.data_type(schema)? != DataType::Boolean {
                return Err(exec_datafusion_err!("CASE WHEN must be Boolean").into());
            }
            let condition = self.convert_expr(when_expr, schema, input_dtype)?;
            let value = self.convert_expr(then_expr, schema, input_dtype)?;
            pairs.push((condition, value));
        }
        let else_value = case_expr
            .else_expr()
            .map(|e| self.convert_expr(e, schema, input_dtype))
            .transpose()?;
        let case = nested_case_when(pairs, else_value);
        // Vortex may evaluate branch values on rows excluded by the condition.
        if !is_infallible(&case) {
            return Err(Unconverted::Unsupported);
        }
        Ok(case)
    }
}

impl ExpressionConverter for DefaultExpressionConverter {
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        // We currently do not support pushdown of dynamic expressions in DF.
        // See issue: https://github.com/vortex-data/vortex/issues/4034
        if DynamicFilterTracking::classify(expr).contains_dynamic_filter() {
            return Ok(None);
        }
        match self.convert_checked(expr, schema) {
            Ok(converted) => Ok(Some(converted)),
            Err(Unconverted::Unsupported) => Ok(None),
            Err(Unconverted::Failed(error)) => Err(error),
        }
    }
}

/// Whether evaluating `expr` cannot fail, so Vortex may evaluate it on rows DataFusion skips.
fn is_infallible(expr: &Expression) -> bool {
    label_infallible(expr).get(expr) == Some(&true)
}

/// The Vortex return dtype of a converted expression; unresolvable dtypes are unsupported.
fn converted_dtype(expr: &Expression, input_dtype: &DType) -> Conversion<DType> {
    expr.return_dtype(input_dtype)
        .map_err(|_| Unconverted::Unsupported)
}

fn supported_cast(cast: &df_expr::CastExpr, schema: &Schema) -> DFResult<bool> {
    use DataType::*;
    let options = cast.cast_options();
    if options.safe || options.format_options != DEFAULT_FORMAT_OPTIONS {
        return Ok(false);
    }
    let source = cast.expr().data_type(schema)?;
    let target = cast.cast_type();
    Ok(source == *target
        || matches!(
            (&source, target),
            (Int8, Int16 | Int32 | Int64)
                | (Int16, Int32 | Int64)
                | (Int32, Int64)
                | (UInt8, UInt16 | UInt32 | UInt64)
                | (UInt16, UInt32 | UInt64)
                | (UInt32, UInt64)
                | (Float32, Float64)
        ))
}

fn try_operator_from_df(value: &DFOperator) -> Option<Operator> {
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
