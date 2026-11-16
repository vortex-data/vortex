// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts DataFusion physical expressions for native Vortex evaluation.
//!
//! Unsupported expressions stay in DataFusion. Comparison and membership literals retain their own
//! extension metadata, or inherit missing metadata from the operand they are compared with. This
//! supplies the logical dtype that Vortex requires without adding implicit casts.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_functions_nested::array_transform::ArrayTransform;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::HigherOrderFunctionExpr;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::expr::Expression;
use vortex::expr::Lambda;
use vortex::expr::get_item;
use vortex::expr::list_transform;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::traversal::NodeExt;
use vortex::expr::traversal::Transformed;
use vortex::expr::var;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

pub(crate) mod convertible;

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
#[derive(Debug)]
pub enum Unconverted {
    /// A valid expression that Vortex cannot evaluate; it stays in DataFusion.
    Unsupported,
    /// A malformed expression or a failed conversion.
    Failed(DataFusionError),
}

impl fmt::Display for Unconverted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("Unsupported expression"),
            Self::Failed(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl Error for Unconverted {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Unsupported => None,
            Self::Failed(error) => Some(error),
        }
    }
}

impl From<DataFusionError> for Unconverted {
    fn from(error: DataFusionError) -> Self {
        Self::Failed(error)
    }
}

/// Conversion result where `?` propagates both unsupported expressions and errors.
pub type Conversion<T> = Result<T, Unconverted>;

/// Shared inputs for matching and constructing a native expression.
#[derive(Clone, Copy)]
pub struct ConvertibleCtx<'a> {
    /// The DataFusion input schema.
    pub schema: &'a Schema,
    /// Session used to resolve Arrow types and import literals.
    pub session: &'a VortexSession,
    /// Lambda parameters read by the native root, or `None` for scan input columns.
    pub scope: Option<&'a StructFields>,
}

impl<'a> ConvertibleCtx<'a> {
    /// Create a context for expressions over scan input columns.
    pub fn new(schema: &'a Schema, session: &'a VortexSession) -> Self {
        Self {
            schema,
            session,
            scope: None,
        }
    }
}

/// Recognition and construction of one built-in expression form.
///
/// Inputs are borrowed and converter state is consumed after recursion. Matching and construction
/// both receive the same [`ConvertibleCtx`]. Concrete converters can be combined directly:
///
/// ```
/// use std::sync::Arc;
/// use datafusion_common::Result as DFResult;
/// use datafusion_physical_expr::PhysicalExpr;
/// use vortex::expr::Expression;
/// use vortex_datafusion::convert::{Convertible, ConvertibleCtx, ConvertColumn, ConvertNot};
///
/// fn convert(
///     expr: &Arc<dyn PhysicalExpr>,
///     ctx: &ConvertibleCtx<'_>,
/// ) -> DFResult<Option<Expression>> {
///     let mut recur = |child| convert(child, ctx);
///     if let Some(converted) = ConvertNot::try_convert(expr, ctx, &mut recur)? {
///         return Ok(Some(converted));
///     }
///     ConvertColumn::try_convert(expr, ctx, recur)
/// }
/// ```
pub trait Convertible<'a>: Sized {
    /// The DataFusion expression matched by this converter.
    type DFExpr: PhysicalExpr;

    /// Downcast and select the inputs to convert before constructing this node.
    ///
    /// `Unconverted::Unsupported` means no match; other conversion errors must be propagated.
    fn maybe_match(
        expr: &'a Arc<dyn PhysicalExpr>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let expr = expr
            .downcast_ref::<Self::DFExpr>()
            .ok_or(Unconverted::Unsupported)?;
        Self::maybe_convert(expr, ctx)
    }

    /// Select inputs and capture the configuration of a typed DataFusion node.
    fn maybe_convert(
        expr: &'a Self::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)>;

    /// Construct this node from its converted inputs, using the supplied context.
    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression>;

    /// Try this converter, using the caller's recursion to convert its selected inputs.
    ///
    /// Returns `None` for an unsupported node, child, or constructed expression.
    fn try_convert(
        expr: &'a Arc<dyn PhysicalExpr>,
        ctx: &ConvertibleCtx<'_>,
        mut recur: impl FnMut(&'a Arc<dyn PhysicalExpr>) -> DFResult<Option<Expression>>,
    ) -> DFResult<Option<Expression>> {
        conversion_result(Self::maybe_match(expr, ctx).and_then(
            |(children, convertible)| {
                recur_and_construct(
                    children,
                    convertible,
                    ctx,
                    &mut |child| recur(child)?.ok_or(Unconverted::Unsupported),
                )
            },
        ))
    }
}

/// Typed construction for fixed-arity converters, adapted to [`Convertible`] below.
pub trait ConvertibleFixedArity<'a>: Sized {
    /// The DataFusion expression matched by this converter.
    type DFExpr: PhysicalExpr;
    /// The converted children, represented by an `[Expression; N]` array.
    type Children: TryFrom<Vec<Expression>>;

    /// Select inputs and capture the configuration of a typed DataFusion node.
    fn maybe_convert_fixed(
        expr: &'a Self::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)>;

    /// Construct from the fixed-size array, using the supplied context.
    fn construct_fixed(
        self,
        children: Self::Children,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression>;
}

impl<'a, T: ConvertibleFixedArity<'a>> Convertible<'a> for T {
    type DFExpr = <T as ConvertibleFixedArity<'a>>::DFExpr;

    fn maybe_convert(
        expr: &'a <Self as Convertible<'a>>::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        <T as ConvertibleFixedArity<'a>>::maybe_convert_fixed(expr, ctx)
    }

    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        let count = children.len();
        let children = T::Children::try_from(children).map_err(|_| {
            exec_datafusion_err!("Cannot construct expression from {count} converted children")
        })?;
        <T as ConvertibleFixedArity<'a>>::construct_fixed(self, children, ctx)
    }
}

/// Add a selected-child count check to a variable-arity converter.
///
/// Matching records the selected input count; construction rejects a different number of results.
/// The wrapped converter and its configuration stay on the stack.
pub struct ConvertibleVariableArity<T> {
    convertible: T,
    arity: usize,
}

impl<'a, T: Convertible<'a>> Convertible<'a> for ConvertibleVariableArity<T> {
    type DFExpr = T::DFExpr;

    fn maybe_convert(
        expr: &'a Self::DFExpr,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<(Vec<&'a Arc<dyn PhysicalExpr>>, Self)> {
        let (children, convertible) = T::maybe_convert(expr, ctx)?;
        let arity = children.len();
        Ok((children, Self { convertible, arity }))
    }

    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> Conversion<Expression> {
        if children.len() != self.arity {
            return Err(exec_datafusion_err!(
                "Expected {} converted children, found {}",
                self.arity,
                children.len()
            )
            .into());
        }
        self.convertible.construct(children, ctx)
    }
}

fn recur_and_construct<'a, T: Convertible<'a>>(
    children: Vec<&'a Arc<dyn PhysicalExpr>>,
    convertible: T,
    ctx: &ConvertibleCtx<'_>,
    recur: &mut impl FnMut(&'a Arc<dyn PhysicalExpr>) -> Conversion<Expression>,
) -> Conversion<Expression> {
    let children = children
        .into_iter()
        .map(recur)
        .collect::<Conversion<Vec<_>>>()?;
    convertible.construct(children, ctx)
}

fn next_child(children: &mut impl Iterator<Item = Expression>) -> Conversion<Expression> {
    children
        .next()
        .ok_or_else(|| exec_datafusion_err!("Missing converted child").into())
}

/// Keep unsupported expressions distinct from conversion errors at public API boundaries.
fn conversion_result<T>(result: Conversion<T>) -> DFResult<Option<T>> {
    match result {
        Ok(converted) => Ok(Some(converted)),
        Err(Unconverted::Unsupported) => Ok(None),
        Err(Unconverted::Failed(error)) => Err(error),
    }
}

/// Convert one native node, using the caller's recursion for its selected inputs.
///
/// The callback receives borrowed DataFusion children in construction order. Field paths stay
/// configuration and repeated IN literals are deduplicated before traversal. Unsupported nodes
/// return `None` without calling `recur`; unsupported children or construction also return `None`.
/// Whole-tree validation and lambda-scoped traversal remain with the recursive converter.
pub fn try_convert_node<'a>(
    expr: &'a Arc<dyn PhysicalExpr>,
    ctx: &ConvertibleCtx<'_>,
    mut recur: impl FnMut(&'a Arc<dyn PhysicalExpr>) -> DFResult<Option<Expression>>,
) -> DFResult<Option<Expression>> {
    conversion_result(convertible::convert_node(
        expr,
        ctx,
        &mut |child| recur(child)?.ok_or(Unconverted::Unsupported),
    ))
}

/// Resolve an Arrow field through the session registry; unknown types are unsupported.
fn arrow_dtype(field: &Field, session: &VortexSession) -> Conversion<DType> {
    session
        .arrow()
        .from_arrow_field(field)
        .map_err(|_| Unconverted::Unsupported)
}

/// The default [`ExpressionConverter`] implementation.
///
/// Supported arithmetic is pushed down using Vortex semantics, including its checked
/// integer arithmetic. Matching DataFusion's overflow behavior is deferred to a future patch.
/// Nested struct field access also uses Vortex semantics: a null parent produces a null child,
/// whereas DataFusion extracts the child without applying parent validity. These differences
/// are accepted for both filters and projections.
/// Searched `CASE WHEN` expressions preserve short-circuit evaluation: conditions and
/// branch values only execute on the rows that reach them, including fallible branches.
/// Other expressions require compatible SQL semantics or remain in DataFusion.
pub struct DefaultExpressionConverter {
    /// Session used to resolve Arrow → Vortex dtypes through the extension
    /// plugin registry, so registered extension types (e.g. UUID ⇄
    /// `FixedSizeBinary[16]`) convert correctly instead of hitting the static,
    /// non-plugin-aware `DType::from_arrow`.
    session: VortexSession,
    /// What `$` is in the expressions this converts.
    scope: Scope,
}

/// What `$` is where an expression is converted.
enum Scope {
    /// Outside any lambda, the scan's columns: a column `c` reads `$.c`.
    Global,
    /// A lambda's parameters: a parameter `x` reads `$.x`, as Vortex runs a lambda's body. Vortex
    /// lambdas can't read columns yet.
    Lambda(StructFields),
}

impl Default for DefaultExpressionConverter {
    fn default() -> Self {
        Self::new(VortexSession::default())
    }
}

impl DefaultExpressionConverter {
    /// Create a converter that resolves Arrow extension types using `session`'s
    /// dtype registry.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            scope: Scope::Global,
        }
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
        let converted = self.convert_tree(expr, schema, &input_dtype)?;
        // `return_field` names the field by formatting the whole expression. For the IN list of a
        // hash join filter, with thousands of keys, that costs more than the conversion. A
        // predicate has no field metadata, so its type and nullability are enough.
        let expected_dtype = if expr.data_type(schema)? == DataType::Boolean {
            DType::Bool(expr.nullable(schema)?.into())
        } else {
            self.arrow_dtype(expr.return_field(schema)?.as_ref())?
        };
        if !converted_dtype(&converted, &input_dtype)?.eq_ignore_nullability(&expected_dtype) {
            return Err(Unconverted::Unsupported);
        }
        Ok(converted)
    }

    fn convert_tree(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        if let Some(function) = expr.downcast_ref::<HigherOrderFunctionExpr>() {
            let function_impl: &dyn std::any::Any = function.fun().inner().as_ref();
            if function_impl.downcast_ref::<ArrayTransform>().is_some() {
                return self.convert_array_transform(function, schema, input_dtype);
            }
            return Err(Unconverted::Unsupported);
        }
        let scope = match &self.scope {
            Scope::Global => None,
            Scope::Lambda(params) => Some(params),
        };
        let ctx = ConvertibleCtx {
            schema,
            session: &self.session,
            scope,
        };
        convertible::convert_node(
            expr,
            &ctx,
            &mut |child| self.convert_tree(child, schema, input_dtype),
        )
    }

    /// Imports a literal, using `operand_field` only when the literal has no extension metadata.
    ///
    /// The literal keeps its Arrow dtype, so different storage types or timestamp units still fail
    /// the comparison's dtype check. Explicit metadata, including an unknown extension, is preserved.
    /// The inherited extension name and payload both come from `operand_field`, including an absent
    /// payload, so the import cannot combine metadata from two different types.
    fn import_literal_for_operand(
        &self,
        literal: &df_expr::Literal,
        operand_field: &Field,
        schema: &Schema,
    ) -> Conversion<Scalar> {
        let field = literal.return_field(schema)?;
        if field.extension_type_name().is_some() || operand_field.extension_type_name().is_none() {
            return self.import_literal(literal, &field);
        }

        let mut field = field.as_ref().clone();
        for key in [EXTENSION_TYPE_NAME_KEY, EXTENSION_TYPE_METADATA_KEY] {
            field.metadata_mut().remove(key);
            if let Some(value) = operand_field.metadata().get(key) {
                field.metadata_mut().insert(key.into(), value.clone());
            }
        }

        self.import_literal(literal, &field)
    }

    /// Imports one scalar through the session's Arrow plugins, preserving `field`'s metadata.
    fn import_literal(&self, literal: &df_expr::Literal, field: &Field) -> Conversion<Scalar> {
        let array = literal.value().to_array()?;
        if array.len() != 1 {
            return Err(exec_datafusion_err!(
                "Literal must contain exactly one value, got {}",
                array.len()
            )
            .into());
        }

        self.arrow_dtype(field)?;

        self.session
            .arrow()
            .from_arrow_array(array, field)
            .and_then(|array| array.execute_scalar(0, &mut self.session.create_execution_ctx()))
            .map_err(|error| {
                Unconverted::Failed(DataFusionError::External(Box::new(
                    error.with_context(format!("Failed to convert literal {literal}")),
                )))
            })
    }

    /// Converts `array_transform` to `list_transform`.
    ///
    /// The body is converted in the lambda's scope, where it may read only its one parameter, the
    /// element.
    fn convert_array_transform(
        &self,
        function: &HigherOrderFunctionExpr,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        let [list, lambda] = function.args() else {
            return Err(
                exec_datafusion_err!("array_transform requires a list and a lambda").into(),
            );
        };
        let Some(lambda) = lambda.downcast_ref::<df_expr::LambdaExpr>() else {
            return Err(Unconverted::Unsupported);
        };
        let [param] = lambda.params() else {
            return Err(Unconverted::Unsupported);
        };
        let list = self.convert_tree(list, schema, input_dtype)?;
        let DType::List(element, _) = converted_dtype(&list, input_dtype)? else {
            return Err(Unconverted::Unsupported);
        };
        let params = StructFields::from_iter([(param.as_str(), element.as_ref().clone())]);
        let inputs = DType::Struct(params.clone(), Nullability::NonNullable);
        let body_converter = Self {
            session: self.session.clone(),
            scope: Scope::Lambda(params),
        };
        // DataFusion also types the body against the outer schema: a lambda variable carries its
        // own field.
        let body = body_converter.convert_tree(lambda.body(), schema, &inputs)?;
        let lambda = Lambda::new([param.as_str()], read_params(body)?)
            .map_err(|_| Unconverted::Unsupported)?;
        Ok(list_transform(list, lambda))
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
pub(crate) fn is_infallible(expr: &Expression) -> bool {
    expr.as_scalar()
        .is_none_or(|scalar_fn| scalar_fn.signature().is_infallible())
        && expr.children().iter().all(is_infallible)
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

/// The Vortex return dtype of a converted expression; unresolvable dtypes are unsupported.
fn converted_dtype(expr: &Expression, input_dtype: &DType) -> Conversion<DType> {
    expr.return_dtype(input_dtype)
        .map_err(|_| Unconverted::Unsupported)
}

/// Resolve only the input fields read by the converted expression. Resolving a whole Arrow
/// schema here would reject otherwise supported expressions because of unrelated fields.
fn node_dtype(expr: &Expression, ctx: &ConvertibleCtx<'_>) -> Conversion<DType> {
    let ConvertibleCtx {
        schema,
        session,
        scope,
    } = *ctx;
    if let Some(name) = expr.as_opt::<GetItem>()
        && expr.child(0).is_root()
    {
        if let Some(params) = scope {
            return params.field(name).ok_or(Unconverted::Unsupported);
        }
        let field = schema
            .field_with_name(name.as_ref())
            .map_err(|_| Unconverted::Unsupported)?;
        return DefaultExpressionConverter::new(session.clone()).arrow_dtype(field);
    }
    match expr {
        Expression::Root => match scope {
            Some(params) => Ok(DType::Struct(params.clone(), Nullability::NonNullable)),
            None => session
                .arrow()
                .from_arrow_schema(schema)
                .map_err(|_| Unconverted::Unsupported),
        },
        Expression::Variable(_) => Err(Unconverted::Unsupported),
        Expression::Scalar {
            scalar_fn,
            children,
        } => {
            let dtypes = children
                .iter()
                .map(|child| node_dtype(child, ctx))
                .collect::<Conversion<Vec<_>>>()?;
            scalar_fn
                .return_dtype(&dtypes)
                .map_err(|_| Unconverted::Unsupported)
        }
    }
}

/// Turns a body converted in a lambda's scope, where `$.x` is the parameter `x`, into a lambda's
/// body, which reads it as `@x`.
fn read_params(body: Expression) -> Conversion<Expression> {
    body.transform_down(|node| {
        if let Some(name) = node.as_opt::<GetItem>()
            && node.child(0).is_root()
        {
            return Ok(Transformed::yes(var(name.clone())));
        }
        Ok(Transformed::no(node))
    })
    .map(Transformed::into_inner)
    .map_err(|_| Unconverted::Unsupported)
}

#[cfg(test)]
mod tests;
