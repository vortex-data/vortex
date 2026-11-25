// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts DataFusion physical expressions for native Vortex evaluation.
//!
//! Unsupported expressions stay in DataFusion. Comparison and membership literals retain their own
//! extension metadata, or inherit missing metadata from the operand they are compared with. This
//! supplies the logical dtype that Vortex requires without adding implicit casts.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_common::tree_node::Transformed as DFTransformed;
use datafusion_common::tree_node::TreeNode;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::VortexSessionDefault;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::layout::layouts::row_idx;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use self::convertible::ConvertibleCtx;
use self::convertible::MustConvert;
use self::convertible::Scoped;
use self::convertible::Unsupported;
use self::convertible::default_convert_recursive;
use self::convertible::must;
use crate::row_index::RowIndex;

pub mod convertible;

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
        raw_projection(
            &source_projection.column_indices(),
            input_schema,
            source_projection,
        )
    }
}

/// Read raw columns and any physical row index needed by the leftover projection,
/// without involving custom expression conversion. Replace row-index leaves with scan columns.
pub(crate) fn raw_projection(
    indices: &[usize],
    input_schema: &Schema,
    source_projection: ProjectionExprs,
) -> DFResult<ProcessedProjection> {
    let schema = input_schema.project(indices)?;
    let mut scan_columns = schema
        .fields()
        .iter()
        .map(|field| {
            (
                field.name().clone(),
                get_item(field.name().as_str(), root()),
            )
        })
        .collect_vec();
    // Use a private, collision-free scan field, independently of output aliases.
    let mut row_index_name = "__vortex_row_index".to_owned();
    while input_schema.field_with_name(&row_index_name).is_ok() {
        row_index_name.push('_');
    }
    let mut needs_row_index = false;
    let leftover_projection = source_projection.try_map_exprs(|expr| {
        expr.transform_up(|expr| {
            if expr.is::<RowIndex>() {
                needs_row_index = true;
                Ok(DFTransformed::yes(
                    Arc::new(df_expr::Column::new(&row_index_name, indices.len()))
                        as Arc<dyn PhysicalExpr>,
                ))
            } else {
                Ok(DFTransformed::no(expr))
            }
        })
        .map(|transformed| transformed.data)
    })?;
    let mut fields = schema.fields().to_vec();
    if needs_row_index {
        scan_columns.push((row_index_name.clone(), row_idx::row_idx()));
        fields.push(Arc::new(Field::new(
            row_index_name,
            DataType::UInt64,
            false,
        )));
    }
    Ok(ProcessedProjection {
        scan_projection: pack(scan_columns, Nullability::NonNullable),
        scan_reference_schema: Schema::new_with_metadata(fields, schema.metadata().clone()),
        leftover_projection,
    })
}

/// The default [`ExpressionConverter`] implementation.
///
/// Supported arithmetic is pushed down using Vortex semantics, including its checked
/// integer arithmetic. Matching DataFusion's overflow behavior is deferred to a future patch.
/// Nested struct field access also uses Vortex semantics: a null parent produces a null child,
/// whereas DataFusion extracts the child without applying parent validity. These differences
/// are accepted for both filters and projections.
/// Field access through an extension can run natively when Arrow exposes its Vortex storage and
/// the selected field exports without plugins.
/// Searched `CASE WHEN` expressions preserve short-circuit evaluation: conditions and
/// branch values only execute on the rows that reach them, including fallible branches.
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
        Self::new(VortexSession::default())
    }
}

impl DefaultExpressionConverter {
    /// Create a converter that resolves Arrow extension types using `session`'s
    /// dtype registry.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Convert `expr` and check that it returns the dtype DataFusion expects.
    fn convert_checked(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> MustConvert<Expression> {
        let ctx = ConvertibleCtx::new(schema, &self.session);
        let columns = collect_columns(expr);
        let mut column_indices = Vec::with_capacity(columns.len());
        for column in columns {
            let field = schema
                .fields()
                .get(column.index())
                .ok_or_else(|| {
                    exec_datafusion_err!(
                        "Column {}@{} is out of bounds",
                        column.name(),
                        column.index()
                    )
                })
                .map_err(Err)?;
            if field.name() != column.name() {
                return Err(Err(exec_datafusion_err!(
                    "Column {}@{} refers to field {}",
                    column.name(),
                    column.index(),
                    field.name()
                )));
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
                    .map_err(DataFusionError::from)
                    .map_err(Err)?,
            )
            .map_err(|_| Ok(Unsupported))?;
        let converted = convert_tree(expr, &ctx)?;
        // `return_field` names the field by formatting the whole expression. For the IN list of a
        // hash join filter, with thousands of keys, that costs more than the conversion. A
        // predicate has no field metadata, so its type and nullability are enough.
        let expected_dtype = if expr.data_type(schema).map_err(Err)? == DataType::Boolean {
            DType::Bool(expr.nullable(schema).map_err(Err)?.into())
        } else {
            ctx.arrow_dtype(expr.return_field(schema).map_err(Err)?.as_ref())?
        };
        if !converted
            .return_dtype(&input_dtype)
            .map_err(|_| Ok(Unsupported))?
            .eq_ignore_nullability(&expected_dtype)
        {
            return Err(Ok(Unsupported));
        }
        Ok(converted)
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
            Err(Ok(Unsupported)) => Ok(None),
            Err(Err(error)) => Err(error),
        }
    }
}

/// Whether evaluating `expr` cannot fail, so Vortex may evaluate it on rows DataFusion skips.
pub(crate) fn is_infallible(expr: &Expression) -> bool {
    expr.as_scalar()
        .is_none_or(|scalar_fn| scalar_fn.signature().is_infallible())
        && expr.children().iter().all(is_infallible)
}

fn convert_tree(expr: &Arc<dyn PhysicalExpr>, ctx: &ConvertibleCtx<'_>) -> MustConvert<Expression> {
    must(default_convert_recursive(expr, ctx, |child| match child {
        Scoped::Same(expr) => convert_tree(expr, ctx),
        Scoped::With(expr, scope) => convert_tree(
            expr,
            &ConvertibleCtx {
                scope: Some(&scope),
                ..*ctx
            },
        ),
    }))
}

#[cfg(test)]
mod tests;
