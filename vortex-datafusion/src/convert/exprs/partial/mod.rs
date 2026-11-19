// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Partial projection pushdown.
//!
//! A projection expression that does not convert as a whole often contains subexpressions that do,
//! such as field access on a wide struct or extension column under an unsupported function. The
//! scan evaluates the largest such subexpressions that read its input, and DataFusion evaluates the
//! rest of the expression over the scan output. The scan then reads only the fields that the
//! projection uses, rather than whole columns.
//!
//! A pushed-down subexpression has the semantics of its conversion, including the exceptions that
//! [`DefaultExpressionConverter`] documents. Nested field access under a null parent returns null,
//! where DataFusion extracts the child without applying parent validity. So a field read under an
//! unsupported function returns the same values as the same field read on its own, which the scan
//! evaluates whole.
//!
//! The scan evaluates every row it returns, so it evaluates a fallible subexpression only where
//! DataFusion evaluates it on every row too: a whole projection expression without a residual
//! filter. Such an expression uses Vortex's checked integer arithmetic, as it does when the whole
//! projection converts. Elsewhere, such as under an unsupported function, after a residual filter
//! or in a `CASE` branch, the scan evaluates only infallible subexpressions, and DataFusion
//! evaluates the rest, including any arithmetic.
//!
//! [`DefaultExpressionConverter`]: super::DefaultExpressionConverter

use std::sync::Arc;

use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_common::tree_node::TreeNode;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::expressions as df_expr;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::layout::layouts::row_idx;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use super::ExpressionConverter;
use super::ProcessedProjection;
use super::is_infallible;
use crate::row_index::RowIndex;

/// The prefix of the names of scan columns that hold pushed-down subexpressions.
const PUSHED_PREFIX: &str = "__vortex_expr_";

/// Splits a projection whose expressions do not all convert.
///
/// `converted` holds the conversion of each projection expression. DataFusion evaluates a
/// projection on every row that the scan returns, so the scan evaluates each converted expression
/// whole, and the largest convertible subexpressions of the others.
///
/// Returns `None` when DataFusion would still evaluate a lambda over the scan output, one that no
/// subexpression evaluated in the scan contains. DataFusion numbers lambda parameters after the
/// columns of the input schema, so the caller reads the raw input columns instead, which never
/// outnumber them.
pub(super) fn split_projection_partially<C: ExpressionConverter + ?Sized>(
    converter: &C,
    source_projection: &ProjectionExprs,
    converted: Vec<Option<Expression>>,
    input_schema: &Schema,
) -> DFResult<Option<ProcessedProjection>> {
    let mut scan = ScanColumns::new(converter, input_schema);
    let leftover = source_projection
        .iter()
        .zip(converted)
        .map(|(projection, converted)| {
            let expr = scan.rewrite_converted(&projection.expr, converted, Evaluation::EveryRow)?;
            Ok(ProjectionExpr::new(expr, projection.alias.clone()))
        })
        .collect::<DFResult<Vec<_>>>()?;
    if contains_lambda(leftover.iter().map(|projection| &projection.expr))? {
        return Ok(None);
    }
    Ok(Some(scan.finish(leftover.into())))
}

/// Splits a projection and the residual filters that DataFusion evaluates before it.
///
/// Both read one scan output. The scan evaluates no filter whole, since the caller found that the
/// filters do not convert, and DataFusion evaluates the projection only on the rows that they keep.
/// So the scan evaluates only infallible subexpressions of either. Returns the residual filters
/// rewritten to read the scan output.
///
/// When DataFusion would still evaluate a lambda over the scan output, falls back to
/// [`ExpressionConverter::no_pushdown_filtered_projection`], for the reason given on
/// [`split_projection_partially`].
pub(super) fn split_filtered_projection<C: ExpressionConverter + ?Sized>(
    converter: &C,
    projection: ProjectionExprs,
    residual_filters: Vec<Arc<dyn PhysicalExpr>>,
    input_schema: &Schema,
) -> DFResult<(ProcessedProjection, Vec<Arc<dyn PhysicalExpr>>)> {
    let mut scan = ScanColumns::new(converter, input_schema);
    let rewritten_filters = residual_filters
        .iter()
        .map(|filter| scan.rewrite_converted(filter, None, Evaluation::SomeRows))
        .collect::<DFResult<Vec<_>>>()?;
    let leftover = projection
        .iter()
        .map(|projection| {
            let expr = scan.rewrite(&projection.expr)?;
            Ok(ProjectionExpr::new(expr, projection.alias.clone()))
        })
        .collect::<DFResult<Vec<_>>>()?;
    let exprs = rewritten_filters
        .iter()
        .chain(leftover.iter().map(|projection| &projection.expr));
    if contains_lambda(exprs)? {
        return converter.no_pushdown_filtered_projection(
            projection,
            residual_filters,
            input_schema,
        );
    }
    Ok((scan.finish(leftover.into()), rewritten_filters))
}

/// Whether DataFusion evaluates an expression on every row that the scan returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Evaluation {
    /// On every row, so the scan may evaluate a fallible expression, as it does a projection that
    /// it evaluates whole.
    EveryRow,
    /// Perhaps on some rows only: after a residual filter, or in an operand that DataFusion
    /// evaluates lazily, such as a `CASE` branch. The scan evaluates only infallible expressions.
    SomeRows,
}

/// The columns of a scan that DataFusion expressions read after it.
///
/// An input column that an expression reads whole keeps its name, so DataFusion resolves it by
/// name as before. A pushed-down subexpression gets a name that no input column has. Equal
/// subexpressions with equal Arrow fields share one scan column.
struct ScanColumns<'a, C: ?Sized> {
    converter: &'a C,
    input_schema: &'a Schema,
    /// The name and Vortex expression of each scan column.
    columns: Vec<(String, Expression)>,
    /// The Arrow field that DataFusion expects of each scan column, named like it.
    fields: Vec<FieldRef>,
    /// The scan column of each input column that is read whole, by input index.
    whole: HashMap<usize, usize>,
    /// The scan column of each pushed-down subexpression, by its conversion and its unnamed Arrow
    /// field.
    pushed: HashMap<(Expression, Field), usize>,
    /// The number of generated names tried, which numbers the next one.
    generated: usize,
}

impl<'a, C: ExpressionConverter + ?Sized> ScanColumns<'a, C> {
    fn new(converter: &'a C, input_schema: &'a Schema) -> Self {
        Self {
            converter,
            input_schema,
            columns: Vec::new(),
            fields: Vec::new(),
            whole: HashMap::default(),
            pushed: HashMap::default(),
            generated: 0,
        }
    }

    /// Rewrites `expr`, which DataFusion may evaluate on some rows only, to read the scan output.
    fn rewrite(&mut self, expr: &Arc<dyn PhysicalExpr>) -> DFResult<Arc<dyn PhysicalExpr>> {
        let converted =
            if expr.is::<df_expr::Column>() || expr.is::<RowIndex>() || !reads_scan(expr)? {
                None
            } else {
                try_convert_or_keep(self.converter, expr, self.input_schema)
            };
        self.rewrite_converted(expr, converted, Evaluation::SomeRows)
    }

    /// Rewrites `expr`, whose conversion is `converted`, to read the scan output, adding the scan
    /// columns that it reads.
    fn rewrite_converted(
        &mut self,
        expr: &Arc<dyn PhysicalExpr>,
        converted: Option<Expression>,
        evaluation: Evaluation,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        if let Some(column) = expr.downcast_ref::<df_expr::Column>() {
            return self.whole_column(column);
        }
        // DataFusion cannot evaluate the row index, so the scan always supplies it.
        if expr.is::<RowIndex>() {
            return self.push(expr, row_idx::row_idx());
        }
        if !reads_scan(expr)? {
            return Ok(Arc::clone(expr));
        }
        if let Some(converted) = converted
            && (evaluation == Evaluation::EveryRow || is_infallible(&converted))
        {
            return self.push(expr, converted);
        }
        // A function that applies a lambda stays in DataFusion whole, so the caller reads the raw
        // input columns. Its lambda reads parameters that the scan cannot supply.
        if expr
            .children()
            .iter()
            .any(|child| child.is::<df_expr::LambdaExpr>())
        {
            return Ok(Arc::clone(expr));
        }
        let children = expr
            .children()
            .into_iter()
            .map(|child| self.rewrite(child))
            .collect::<DFResult<Vec<_>>>()?;
        Arc::clone(expr).with_new_children(children)
    }

    /// Reads the input column `column` whole.
    fn whole_column(&mut self, column: &df_expr::Column) -> DFResult<Arc<dyn PhysicalExpr>> {
        let field = self
            .input_schema
            .fields()
            .get(column.index())
            .filter(|field| field.name() == column.name())
            .ok_or_else(|| exec_datafusion_err!("Column {column} is not in the input schema"))?;
        let index = match self.whole.entry(column.index()) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = self.columns.len();
                self.columns.push((
                    field.name().clone(),
                    get_item(field.name().as_str(), root()),
                ));
                self.fields.push(Arc::clone(field));
                *entry.insert(index)
            }
        };
        Ok(self.column(index))
    }

    /// Evaluates `expr`, which converts to `converted`, in the scan.
    fn push(
        &mut self,
        expr: &Arc<dyn PhysicalExpr>,
        converted: Expression,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        let field = expr
            .return_field(self.input_schema)?
            .as_ref()
            .clone()
            .with_name("");
        let index = match self.pushed.entry((converted, field)) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let name = generate_name(self.input_schema, &mut self.generated);
                let index = self.columns.len();
                let (converted, field) = entry.key();
                self.columns.push((name.clone(), converted.clone()));
                self.fields.push(Arc::new(field.clone().with_name(name)));
                *entry.insert(index)
            }
        };
        Ok(self.column(index))
    }

    /// The DataFusion column that reads scan column `index`.
    fn column(&self, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Column::new(&self.columns[index].0, index))
    }

    /// Returns the scan projection and its reference schema, with `leftover_projection`, which
    /// reads the scan output.
    fn finish(self, leftover_projection: ProjectionExprs) -> ProcessedProjection {
        ProcessedProjection {
            scan_projection: pack(self.columns, Nullability::NonNullable),
            scan_reference_schema: Schema::new_with_metadata(
                self.fields,
                self.input_schema.metadata().clone(),
            ),
            leftover_projection,
        }
    }
}

/// Converts `expr`, or returns `None` if it stays in DataFusion.
///
/// The conversion of a projection is an optimization, so an expression whose conversion fails
/// stays in DataFusion, like an unsupported one, which reports any error when it evaluates it.
pub(super) fn try_convert_or_keep<C: ExpressionConverter + ?Sized>(
    converter: &C,
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Option<Expression> {
    converter
        .try_convert(expr, input_schema)
        .unwrap_or_else(|error| {
            tracing::debug!("{expr} stays in DataFusion: {error}");
            None
        })
}

/// Returns the next name for a pushed-down subexpression that no column of `input_schema` has.
fn generate_name(input_schema: &Schema, generated: &mut usize) -> String {
    loop {
        let name = format!("{PUSHED_PREFIX}{generated}");
        *generated += 1;
        if input_schema.index_of(&name).is_err() {
            return name;
        }
    }
}

/// Whether `expr` reads a column or the row index of the scan.
fn reads_scan(expr: &Arc<dyn PhysicalExpr>) -> DFResult<bool> {
    expr.exists(|node| Ok(node.is::<df_expr::Column>() || node.is::<RowIndex>()))
}

/// Whether any of `exprs` contains a lambda.
fn contains_lambda<'e>(
    exprs: impl IntoIterator<Item = &'e Arc<dyn PhysicalExpr>>,
) -> DFResult<bool> {
    for expr in exprs {
        if expr.exists(|node| Ok(node.is::<df_expr::LambdaExpr>()))? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests;
