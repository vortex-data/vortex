// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prunes struct fields from the raw inputs of a DataFusion projection.
//!
//! DataFusion still evaluates the projection. The scan substitutes columns only for field reads
//! whose Arrow field and values are the same as in the raw projection.

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
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::pack::Pack;
use vortex_arrow::ArrowSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::ExpressionConverter;
use super::ProcessedProjection;

/// Refine a raw fallback if it can omit at least one whole struct column. Resolve the actual scan
/// dtype lazily because the Arrow reference schema can describe stricter nullability than the data.
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
    let raw_schema = &projection.scan_reference_schema;
    let raw_fields = raw_schema.fields();
    let raw_columns = projection.scan_projection.children();
    let options = projection.scan_projection.as_opt::<Pack>()?;
    if options.nullability.is_nullable()
        || options.names.len() != raw_fields.len()
        || raw_columns.len() != raw_fields.len()
        || !raw_fields
            .iter()
            .any(|field| matches!(field.data_type(), DataType::Struct(_)))
        || !raw_fields.iter().map(|field| field.name()).all_unique()
        || !options
            .names
            .iter()
            .zip(raw_fields)
            .zip(raw_columns)
            .all(|((name, field), expr)| {
                name.as_ref() == field.name()
                    && (expr.as_opt::<GetItem>() == Some(name) && expr.child(0) == &root()
                        || *expr == row_idx::row_idx())
            })
    {
        return None;
    }

    let leftover = &projection.leftover_projection;
    if !leftover.iter().map(|output| &output.alias).all_unique() {
        return None;
    }

    let mut has_field_read = false;
    for output in leftover.iter() {
        if output
            .expr
            .exists(|expr| {
                has_field_read |= get_field_function(expr).is_some();
                Ok(!can_rewrite(expr))
            })
            .ok()?
        {
            return None;
        }
    }
    if !has_field_read {
        return None;
    }

    let DType::Struct(scan_fields, _) = scan_dtype(&projection.scan_projection).ok()? else {
        return None;
    };

    let mut fields = raw_fields.to_vec();
    let mut columns = raw_columns.to_vec();
    let mut indices: HashMap<String, usize> = fields
        .iter()
        .enumerate()
        .map(|(index, field)| (field.name().clone(), index))
        .collect();
    let leftover = leftover
        .clone()
        .try_map_exprs(|expr| {
            // The fallback keeps the original column indices until the caller remaps by name.
            // Assign indices before adding columns so rebuilding a CASE sees distinct inputs.
            reassign_expr_columns(expr, raw_schema)?
                .transform_down(|expr| {
                    if get_field_function(&expr).is_none() {
                        return Ok(Transformed::no(expr));
                    }

                    let keep =
                        || Transformed::new(Arc::clone(&expr), false, TreeNodeRecursion::Jump);
                    let Some((field, read)) =
                        convert_field_read(&expr, raw_schema, &scan_fields, converter, arrow)
                    else {
                        return Ok(keep());
                    };

                    // UDFs receive the argument's full Arrow field. Its name must survive the
                    // rewrite, so a collision with a different read keeps the original expression.
                    let index = if let Some(&index) = indices.get(field.name()) {
                        if index < raw_fields.len() || columns[index] != read {
                            return Ok(keep());
                        }
                        index
                    } else {
                        let index = columns.len();
                        indices.insert(field.name().clone(), index);
                        fields.push(Arc::clone(&field));
                        columns.push(read);
                        index
                    };

                    Ok(Transformed::yes(
                        Arc::new(df_expr::Column::new(field.name(), index)) as _,
                    ))
                })
                .map(|transformed| transformed.data)
        })
        .ok()?;

    let used = leftover.column_indices();
    let retained_raw = used.partition_point(|&index| index < raw_fields.len());
    if retained_raw == raw_fields.len() || retained_raw == used.len() {
        return None;
    }

    let schema = Schema::new_with_metadata(fields, raw_schema.metadata().clone());
    let scan_reference_schema = schema.project(&used).ok()?;
    let scan_projection = pack(
        used.iter()
            .map(|&index| (schema.field(index).name().clone(), columns[index].clone())),
        Nullability::NonNullable,
    );
    Some(ProcessedProjection {
        scan_projection,
        scan_reference_schema,
        leftover_projection: leftover,
    })
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
) -> Option<(FieldRef, Expression)> {
    let (field, dtype, read) = struct_field_path(expr, schema, scan_fields)?;
    // Importing an Arrow field can erase dictionary and run-end encoding. Check the Arrow
    // type directly, since a UDF can observe that encoding even when the values are equivalent.
    let arrow_type = field.data_type();
    let plain_type = arrow_type.is_integer()
        || arrow_type.is_floating()
        || arrow_type.is_string()
        || matches!(
            arrow_type,
            DataType::Boolean | DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        );
    if !plain_type
        || field.extension_type_name().is_some()
        || arrow.from_arrow_field(&field).ok()? != dtype
    {
        return None;
    }

    // A custom converter may reject the read or give it different semantics.
    let converted = converter.try_convert(expr, schema).ok().flatten()?;
    (converted == read).then_some((field, read))
}

/// Resolve both flattened and nested get_field calls. Vortex applies parent validity to a child,
/// so every traversed parent must be a non-nullable struct in both schemas.
fn struct_field_path(
    expr: &PhysicalExprRef,
    schema: &Schema,
    scan_fields: &StructFields,
) -> Option<(FieldRef, DType, Expression)> {
    if let Some(column) = expr.downcast_ref::<df_expr::Column>() {
        let field = schema.fields().get(column.index())?;
        let dtype = scan_fields.field(field.name())?;
        let read = get_item(field.name().as_str(), root());
        return Some((Arc::clone(field), dtype, read));
    }

    let function = get_field_function(expr)?;
    let [source, first, rest @ ..] = function.args() else {
        return None;
    };
    let (mut field, mut dtype, mut read) = struct_field_path(source, schema, scan_fields)?;

    for path in iter::once(first).chain(rest) {
        if field.is_nullable() || field.extension_type_name().is_some() {
            return None;
        }
        let (DataType::Struct(fields), DType::Struct(children, Nullability::NonNullable)) =
            (field.data_type(), &dtype)
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

        field = Arc::clone(child);
        dtype = children.field(name)?;
        read = get_item(name, read);
    }

    (expr.return_field(schema).ok()? == field).then_some((field, dtype, read))
}
