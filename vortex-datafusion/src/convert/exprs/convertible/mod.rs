// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protocols and context for converting one DataFusion expression at a time.

use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use datafusion_common::DataFusionError;
use datafusion_common::exec_datafusion_err;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_plan::expressions as df_expr;
use vortex::array::VortexSessionExecute as _;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::expr::Expression;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

mod impls;

pub use impls::ConvertArrayLength;
pub use impls::ConvertArraySum;
pub use impls::ConvertArrayTransform;
pub use impls::ConvertBinary;
pub use impls::ConvertCase;
pub use impls::ConvertCast;
pub use impls::ConvertColumn;
pub use impls::ConvertContains;
pub use impls::ConvertEndsWith;
pub use impls::ConvertGetField;
pub use impls::ConvertInList;
pub use impls::ConvertIsNotNull;
pub use impls::ConvertIsNull;
pub use impls::ConvertLambdaVariable;
pub use impls::ConvertLike;
pub use impls::ConvertLiteral;
pub use impls::ConvertLower;
pub use impls::ConvertNot;
pub use impls::ConvertOctetLength;
pub use impls::ConvertStartsWith;
pub use impls::ConvertStrpos;
pub use impls::default_convert;
pub use impls::default_convert_recursive;
#[cfg(test)]
pub(super) use impls::supported_data_types;
#[cfg(test)]
pub(super) use impls::try_operator_from_df;

/// A valid expression that this converter does not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported;

/// A successful conversion, an unsupported expression (`Err(Ok(Unsupported))`), or an error.
pub type MustConvert<T> = Result<T, Result<Unsupported, DataFusionError>>;

/// A converter that did not match (`Ok(Unsupported)`), or a conversion or error to return.
///
/// Using `?` tries another converter only when this one did not match.
pub type MayConvert<T> = Result<Unsupported, Result<T, DataFusionError>>;

pub(super) fn must<T>(x: MayConvert<T>) -> MustConvert<T> {
    match x {
        Ok(Unsupported) => Err(Ok(Unsupported)),
        Err(Ok(value)) => Ok(value),
        Err(Err(err)) => Err(Err(err)),
    }
}

pub(super) fn may<T>(x: MustConvert<T>) -> MayConvert<T> {
    match x {
        Ok(value) => Err(Ok(value)),
        Err(Ok(Unsupported)) => Ok(Unsupported),
        Err(Err(err)) => Err(Err(err)),
    }
}

/// A selected expression child and the scope in which it should be converted.
#[derive(Debug)]
pub enum Scoped<'a> {
    /// Convert in the parent's scope.
    Same(&'a PhysicalExprRef),
    /// Convert with these lambda parameters as the native root, replacing the enclosing scope.
    With(&'a PhysicalExprRef, StructFields),
}

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

    /// Resolve an Arrow field through the session registry; unknown types are unsupported.
    pub fn arrow_dtype(&self, field: &Field) -> MustConvert<DType> {
        self.session
            .arrow()
            .from_arrow_field(field)
            .map_err(|_| Ok(Unsupported))
    }

    /// Resolve only the input fields read by the converted expression. Resolving a whole Arrow
    /// schema here would reject otherwise supported expressions because of unrelated fields.
    pub(super) fn node_dtype(&self, expr: &Expression) -> MustConvert<DType> {
        let ConvertibleCtx {
            schema,
            session,
            scope,
        } = *self;
        if let Some(name) = expr.as_opt::<GetItem>()
            && expr.child(0).is_root()
        {
            if let Some(params) = scope {
                return params.field(name).ok_or(Ok(Unsupported));
            }
            let field = schema
                .field_with_name(name.as_ref())
                .map_err(|_| Ok(Unsupported))?;
            return self.arrow_dtype(field);
        }
        match expr {
            Expression::Root => match scope {
                Some(params) => Ok(DType::Struct(params.clone(), Nullability::NonNullable)),
                None => session
                    .arrow()
                    .from_arrow_schema(schema)
                    .map_err(|_| Ok(Unsupported)),
            },
            Expression::Variable(_) => Err(Ok(Unsupported)),
            Expression::Scalar {
                scalar_fn,
                children,
            } => {
                let dtypes = children
                    .iter()
                    .map(|child| self.node_dtype(child))
                    .collect::<MustConvert<Vec<_>>>()?;
                scalar_fn.return_dtype(&dtypes).map_err(|_| Ok(Unsupported))
            }
        }
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
    ) -> MustConvert<Scalar> {
        let field = literal.return_field(self.schema).map_err(Err)?;
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
    fn import_literal(&self, literal: &df_expr::Literal, field: &Field) -> MustConvert<Scalar> {
        let array = literal.value().to_array().map_err(Err)?;
        if array.len() != 1 {
            return Err(Err(exec_datafusion_err!(
                "Literal must contain exactly one value, got {}",
                array.len()
            )));
        }

        self.arrow_dtype(field)?;

        self.session
            .arrow()
            .from_arrow_array(array, field)
            .and_then(|array| array.execute_scalar(0, &mut self.session.create_execution_ctx()))
            .map_err(|error| {
                Err(DataFusionError::External(Box::new(error.with_context(
                    format!("Failed to convert literal {literal}"),
                ))))
            })
    }
}

/// Convert a specific DataFusion expression type to a Vortex expression.
///
/// This API is "single step". The implementor merely defines how to recognize one-level of a
/// DataFusion expression. The caller is responsible for orchestrating a recursive tree walk.
///
/// Implementors must define:
///
/// 1. The type of DataFusion expression which they convert.
///
/// 2. The DataFusion children which they need converted and any state they need for later
///    conversion.
///
/// 3. How to construct the converted Vortex expression given children which the caller has
///    converted to Vortex.
pub trait Convertible<'a>: Sized + 'a {
    /// The DataFusion expression matched by this converter.
    type DF: PhysicalExpr;

    /// Recognize (or not) an expression which we can convert to a Vortex expression.
    fn recognize(
        expr: &'a PhysicalExprRef,
        ctx: &ConvertibleCtx<'_>,
    ) -> MayConvert<(Vec<Scoped<'a>>, Self)> {
        let Some(expr) = expr.downcast_ref::<Self::DF>() else {
            return Ok(Unsupported);
        };
        may(Self::convert(expr, ctx))
    }

    /// Recognize (or not) an expression which we can convert to a Vortex expression.
    fn recognize_boxed(
        expr: &'a PhysicalExprRef,
        ctx: &ConvertibleCtx<'_>,
    ) -> MayConvert<(Vec<Scoped<'a>>, Box<dyn Constructible + 'a>)> {
        may(must(Self::recognize(expr, ctx)).map(|(children, convertible)| {
            (children, Box::new(convertible) as Box<dyn Constructible + 'a>)
        }))
    }

    /// Try this converter, using the caller's recursion to convert its selected inputs.
    ///
    /// The callback receives each child and is responsible for handling its scope.
    ///
    /// Returns `Err(Ok(Unsupported))` for an unsupported node, child, or constructed expression.
    fn convert_recursive(
        expr: &'a PhysicalExprRef,
        ctx: &ConvertibleCtx<'_>,
        recur: impl FnMut(Scoped<'a>) -> MustConvert<Expression>,
    ) -> MustConvert<Expression> {
        let (children, convertible) = must(Self::recognize(expr, ctx))?;
        let children = children
            .into_iter()
            .map(recur)
            .collect::<MustConvert<Vec<_>>>()?;
        convertible.construct(children, ctx)
    }

    /// Convert a specific DataFusion expression to a Vortex expression.
    fn convert(
        expr: &'a Self::DF,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<(Vec<Scoped<'a>>, Self)>;

    /// Given Vortex expressions for our children, construct a Vortex expression for ourselves.
    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<Expression>;
}

/// A converter whose concrete type has been erased for deferred construction.
pub trait Constructible {
    /// Given Vortex expressions for our children, construct a Vortex expression for ourselves.
    fn construct(
        self: Box<Self>,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<Expression>;
}

impl<'a, T: Convertible<'a>> Constructible for T {
    fn construct(
        self: Box<Self>,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<Expression> {
        <T as Convertible<'a>>::construct(*self, children, ctx)
    }
}

/// Typed construction for fixed-arity converters, adapted to [`Convertible`] below.
pub trait ConvertibleFixedArity<'a>: Sized {
    /// The DataFusion expression matched by this converter.
    type DF: PhysicalExpr;
    /// The converted children, represented by an `[Expression; N]` array.
    type Children: TryFrom<Vec<Expression>>;

    /// Select inputs and capture the configuration of a typed DataFusion node.
    fn convert_fixed(
        expr: &'a Self::DF,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<(Vec<Scoped<'a>>, Self)>;

    /// Construct from the fixed-size array, using the supplied context.
    fn construct_fixed(
        self,
        children: Self::Children,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<Expression>;
}

impl<'a, T: ConvertibleFixedArity<'a> + 'a> Convertible<'a> for T {
    type DF = <T as ConvertibleFixedArity<'a>>::DF;

    fn convert(
        expr: &'a <Self as Convertible<'a>>::DF,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<(Vec<Scoped<'a>>, Self)> {
        <T as ConvertibleFixedArity<'a>>::convert_fixed(expr, ctx)
    }

    fn construct(
        self,
        children: Vec<Expression>,
        ctx: &ConvertibleCtx<'_>,
    ) -> MustConvert<Expression> {
        let count = children.len();
        let children = T::Children::try_from(children).map_err(|_| {
            Err(exec_datafusion_err!(
                "Cannot construct expression from {count} converted children"
            ))
        })?;
        <T as ConvertibleFixedArity<'a>>::construct_fixed(self, children, ctx)
    }
}

