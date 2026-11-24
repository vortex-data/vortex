// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prunes struct fields from the raw inputs of a DataFusion projection.
//!
//! DataFusion still evaluates the projection. The scan substitutes columns only for field reads
//! whose Arrow field and values are the same as in the raw projection. A field read stays in
//! DataFusion when the scan still reads its raw column whole, so the scan never reads a field twice.

use std::iter;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions as df_expr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::reassign_expr_columns;
use itertools::Itertools;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::layout::layouts::row_idx;
use vortex::scalar_fn::builtin::get_item::GetItem;
use vortex::scalar_fn::builtin::pack::Pack;
use vortex_arrow::ArrowSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::ExpressionConverter;
use super::ProcessedProjection;

/// Refines a raw fallback when it can omit at least one whole struct column.
///
/// The Arrow reference schema can promise stricter nullability than the data, so
/// eligibility also uses the actual dtype returned by `scan_dtype`. The callback runs
/// only after finding a candidate projection.
pub(crate) fn push_down_projection_fields(
    projection: ProcessedProjection,
    converter: &dyn ExpressionConverter,
    scan_dtype: impl FnOnce(&Expression) -> VortexResult<DType>,
    arrow: &ArrowSession,
) -> ProcessedProjection {
    if !converter.allows_projection_field_pushdown() {
        return projection;
    }

    try_push_down_fields(&projection, converter, scan_dtype, arrow).unwrap_or(projection)
}

fn try_push_down_fields(
    projection: &ProcessedProjection,
    converter: &dyn ExpressionConverter,
    scan_dtype: impl FnOnce(&Expression) -> VortexResult<DType>,
    arrow: &ArrowSession,
) -> Option<ProcessedProjection> {
    let raw_columns = raw_projection_columns(projection)?;
    let leftover = &projection.leftover_projection;
    if !can_rewrite_projection(leftover) {
        return None;
    }

    let DType::Struct(scan_fields, _) = scan_dtype(&projection.scan_projection).ok()? else {
        return None;
    };

    let raw_schema = &projection.scan_reference_schema;
    let substitute_field_reads = |retained_columns: &[usize]| {
        let mut scan_columns = ScanColumns::from_raw(raw_schema.fields(), raw_columns);
        let mut substitute = |expr: PhysicalExprRef| {
            if get_field_function(&expr).is_none() {
                return Ok(Transformed::no(expr));
            }

            let field_read = convert_field_read(&expr, raw_schema, &scan_fields, converter, arrow)
                .filter(|read| retained_columns.binary_search(&read.raw_column).is_err());
            match field_read.and_then(|read| scan_columns.substitute(read)) {
                Some(column) => Ok(Transformed::yes(column)),
                None => Ok(Transformed::new(expr, false, TreeNodeRecursion::Jump)),
            }
        };

        // Assign the raw column indices first so rebuilding a CASE can distinguish its original
        // columns from the substituted field reads.
        let rewritten = leftover
            .clone()
            .try_map_exprs(|expr| {
                reassign_expr_columns(expr, raw_schema)?
                    .transform_down(&mut substitute)
                    .map(|transformed| transformed.data)
            })
            .ok()?;

        Some((rewritten, scan_columns))
    };

    // A substituted field is redundant when the scan still reads its raw column whole, because the
    // projection reads that column directly, reads a field of it that cannot be substituted, or
    // reads a field whose name a different read already claimed. The scan would then read and
    // export the field twice, so the second pass keeps every field read of a retained column in
    // DataFusion. Each remaining field read claimed its name without a conflict in the first
    // pass, and the second pass claims only some of those names, so each one succeeds again and
    // the second pass retains no further raw columns.
    let (rewritten, scan_columns) = substitute_field_reads(&[])?;
    let retained_columns = scan_columns.retained_raw_columns(&rewritten);
    let (rewritten, scan_columns) = if scan_columns.substitutes_field_of(&retained_columns) {
        substitute_field_reads(&retained_columns)?
    } else {
        (rewritten, scan_columns)
    };

    scan_columns.finish(rewritten, raw_schema)
}

/// A raw fallback packs unchanged input columns, with a virtual row index if requested.
fn raw_projection_columns(projection: &ProcessedProjection) -> Option<&[Expression]> {
    let fields = projection.scan_reference_schema.fields();
    let columns = projection.scan_projection.children();
    let pack = projection.scan_projection.as_opt::<Pack>()?;
    if pack.nullability.is_nullable()
        || pack.names.len() != fields.len()
        || columns.len() != fields.len()
        || !fields
            .iter()
            .any(|field| matches!(field.data_type(), DataType::Struct(_)))
        || !fields.iter().map(|field| field.name()).all_unique()
    {
        return None;
    }

    for ((name, field), column) in pack.names.iter().zip(fields).zip(columns) {
        let direct_column = column.as_opt::<GetItem>() == Some(name) && column.child(0) == &root();
        if name.as_ref() != field.name() || (!direct_column && *column != row_idx::row_idx()) {
            return None;
        }
    }

    Some(columns)
}

fn can_rewrite_projection(projection: &ProjectionExprs) -> bool {
    if !projection.iter().map(|output| &output.alias).all_unique() {
        return false;
    }

    let mut has_field_read = false;
    for output in projection.iter() {
        let has_unsupported_node = output
            .expr
            .exists(|expr| {
                has_field_read |= get_field_function(expr).is_some();
                Ok(!can_rewrite(expr))
            })
            .unwrap_or(true);
        if has_unsupported_node {
            return false;
        }
    }

    has_field_read
}

/// A column that the scan can read, and the raw column that it comes from.
struct ScanColumn {
    /// The raw column itself, or the raw struct column that contains this field.
    raw_column: usize,
    /// The Arrow field that DataFusion expects for this column.
    field: FieldRef,
    /// The Vortex expression that reads this column.
    read: Expression,
}

/// Tracks the scan column at each index in the rewritten projection.
struct ScanColumns {
    columns: Vec<ScanColumn>,
    indices: HashMap<String, usize>,
    /// Substituted fields are appended after this prefix of original columns.
    raw_field_count: usize,
}

impl ScanColumns {
    fn from_raw(fields: &[FieldRef], columns: &[Expression]) -> Self {
        let indices = fields
            .iter()
            .enumerate()
            .map(|(index, field)| (field.name().clone(), index))
            .collect();
        let columns = fields
            .iter()
            .zip(columns)
            .enumerate()
            .map(|(raw_column, (field, read))| ScanColumn {
                raw_column,
                field: Arc::clone(field),
                read: read.clone(),
            })
            .collect();

        Self {
            columns,
            indices,
            raw_field_count: fields.len(),
        }
    }

    fn substitute(&mut self, column: ScanColumn) -> Option<PhysicalExprRef> {
        let name = column.field.name().clone();
        let index = if let Some(&index) = self.indices.get(&name) {
            // UDFs receive the argument's full Arrow field. Renaming it to resolve a collision
            // could change the UDF's result, so only reuse a previously substituted read.
            if index < self.raw_field_count || self.columns[index].read != column.read {
                return None;
            }
            index
        } else {
            let index = self.columns.len();
            self.indices.insert(name.clone(), index);
            self.columns.push(column);
            index
        };

        Some(Arc::new(df_expr::Column::new(&name, index)))
    }

    /// Returns the sorted raw columns that the rewritten projection still references.
    fn retained_raw_columns(&self, projection: &ProjectionExprs) -> Vec<usize> {
        let mut used = projection.column_indices();
        used.truncate(used.partition_point(|&index| index < self.raw_field_count));
        used
    }

    /// Returns whether a substituted field comes from one of the sorted `raw_columns`.
    fn substitutes_field_of(&self, raw_columns: &[usize]) -> bool {
        self.columns[self.raw_field_count..]
            .iter()
            .any(|column| raw_columns.binary_search(&column.raw_column).is_ok())
    }

    fn finish(
        self,
        leftover_projection: ProjectionExprs,
        raw_schema: &Schema,
    ) -> Option<ProcessedProjection> {
        let used = leftover_projection.column_indices();
        let retained_raw = used.partition_point(|&index| index < self.raw_field_count);
        if retained_raw == self.raw_field_count || retained_raw == used.len() {
            return None;
        }

        let mut fields = Vec::with_capacity(used.len());
        let mut reads = Vec::with_capacity(used.len());
        for index in used {
            let column = self.columns.get(index)?;
            fields.push(Arc::clone(&column.field));
            reads.push((column.field.name().clone(), column.read.clone()));
        }

        Some(ProcessedProjection {
            scan_projection: pack(reads, Nullability::NonNullable),
            scan_reference_schema: Schema::new_with_metadata(fields, raw_schema.metadata().clone()),
            leftover_projection,
        })
    }
}

/// Allow only expressions whose children describe their input bindings. A lambda or a physical
/// expression with its own bindings needs the original fallback layout.
fn can_rewrite(expr: &PhysicalExprRef) -> bool {
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        // DataFusion passes return fields to UDFs, and the default return field uses the
        // expression's display name. These argument types preserve their fields when remapped.
        return function.args().iter().all(|arg| {
            arg.is::<df_expr::Column>()
                || arg.is::<df_expr::Literal>()
                || arg.is::<ScalarFunctionExpr>()
        });
    }

    expr.is::<df_expr::Column>()
        || expr.is::<df_expr::Literal>()
        || expr.is::<df_expr::CaseExpr>()
        || expr.is::<df_expr::BinaryExpr>()
        || expr.is::<df_expr::CastExpr>()
        || expr.is::<df_expr::TryCastExpr>()
        || expr.is::<df_expr::IsNullExpr>()
        || expr.is::<df_expr::IsNotNullExpr>()
}

fn get_field_function(expr: &PhysicalExprRef) -> Option<&ScalarFunctionExpr> {
    ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr.as_ref())
}

fn convert_field_read(
    expr: &PhysicalExprRef,
    schema: &Schema,
    scan_fields: &StructFields,
    converter: &dyn ExpressionConverter,
    arrow: &ArrowSession,
) -> Option<ScanColumn> {
    let (column, dtype) = struct_field_path(expr, schema, scan_fields)?;
    // Importing an Arrow field can erase dictionary and run-end encoding. Check the Arrow
    // type directly, since a UDF can observe that encoding even when the values are equivalent.
    let arrow_type = column.field.data_type();
    let plain_type = arrow_type.is_integer()
        || arrow_type.is_floating()
        || arrow_type.is_string()
        || matches!(
            arrow_type,
            DataType::Boolean | DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        );
    if !plain_type
        || column.field.extension_type_name().is_some()
        || arrow.from_arrow_field(&column.field).ok()? != dtype
    {
        return None;
    }

    // A custom converter may reject the read or give it different semantics.
    let converted = converter.try_convert(expr, schema).ok().flatten()?;
    (converted == column.read).then_some(column)
}

/// Resolve both flattened and nested get_field calls, propagating parent struct nullability
/// through both schemas to match DataFusion's get_field and Vortex's get_item.
fn struct_field_path(
    expr: &PhysicalExprRef,
    schema: &Schema,
    scan_fields: &StructFields,
) -> Option<(ScanColumn, DType)> {
    if let Some(raw_column) = expr.downcast_ref::<df_expr::Column>() {
        let field = schema.fields().get(raw_column.index())?;
        let dtype = scan_fields.field(field.name())?;
        let column = ScanColumn {
            raw_column: raw_column.index(),
            field: Arc::clone(field),
            read: get_item(field.name().as_str(), root()),
        };
        return Some((column, dtype));
    }

    let function = get_field_function(expr)?;
    let [source, first, rest @ ..] = function.args() else {
        return None;
    };
    let (mut column, mut dtype) = struct_field_path(source, schema, scan_fields)?;

    for path in iter::once(first).chain(rest) {
        if column.field.extension_type_name().is_some() {
            return None;
        }
        let (DataType::Struct(fields), DType::Struct(children, nullability)) =
            (column.field.data_type(), &dtype)
        else {
            return None;
        };
        let name = path
            .downcast_ref::<df_expr::Literal>()?
            .value()
            .try_as_str()
            .flatten()
            .filter(|name| !name.is_empty())?;
        let mut matching = fields.iter().filter(|field| field.name() == name);
        let child = matching.next()?;
        if matching.next().is_some() {
            return None;
        }

        let parent_field_nullable = column.field.is_nullable();
        let parent_dtype_nullable = nullability.is_nullable();
        column.field = if parent_field_nullable && !child.is_nullable() {
            Arc::new(child.as_ref().clone().with_nullable(true))
        } else {
            Arc::clone(child)
        };
        let child_dtype = children.field(name)?;
        dtype = if parent_dtype_nullable {
            child_dtype.as_nullable()
        } else {
            child_dtype
        };
        column.read = get_item(name, column.read);
    }

    (expr.return_field(schema).ok()? == column.field).then_some((column, dtype))
}
