// SPDX-FileCopyrightText: 2016-2025 Copyright The Apache Software Foundation
// SPDX-FileCopyrightText: 2025 Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileComment: Derived from upstream file datafusion/physical-expr-adapter/src/schema_rewriter.rs at commit e571b49 at https://github.com/apache/datafusion
// SPDX-FileNotice: https://github.com/apache/datafusion/blob/e571b49e0983892597a8f92e5d1502b17a15b180/NOTICE.txt

#![allow(missing_docs)]

//! Physical expression schema rewriting utilities
//!
//! NOTE(aduffy): this is vendored until DF 52 is released, at which point this should
//!     all be deleted.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::can_cast_types;
use datafusion_common::arrow::datatypes::DataType;
use datafusion_common::arrow::datatypes::FieldRef;
use datafusion_common::arrow::datatypes::Schema;
use datafusion_common::arrow::datatypes::SchemaRef;
use datafusion_common::exec_err;
use datafusion_common::nested_struct::validate_struct_compatibility;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TransformedResult;
use datafusion_common::tree_node::TreeNode;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::CastColumnExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::{self};
use datafusion_physical_expr_adapter::PhysicalExprAdapter;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

#[derive(Debug, Clone)]
pub struct DF52PhysicalExprAdapterFactory;

impl PhysicalExprAdapterFactory for DF52PhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Arc<dyn PhysicalExprAdapter> {
        Arc::new(DF52PhysicalExprAdapter {
            logical_file_schema,
            physical_file_schema,
            partition_values: Vec::new(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct DF52PhysicalExprAdapter {
    logical_file_schema: SchemaRef,
    physical_file_schema: SchemaRef,
    partition_values: Vec<(FieldRef, ScalarValue)>,
}

impl DF52PhysicalExprAdapter {
    /// Create a new instance of the default physical expression adapter.
    ///
    /// This adapter rewrites expressions to match the physical schema of the file being scanned,
    /// handling type mismatches and missing columns by filling them with default values.
    pub fn new(logical_file_schema: SchemaRef, physical_file_schema: SchemaRef) -> Self {
        Self {
            logical_file_schema,
            physical_file_schema,
            partition_values: Vec::new(),
        }
    }
}

impl PhysicalExprAdapter for DF52PhysicalExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        let rewriter = DefaultPhysicalExprAdapterRewriter {
            logical_file_schema: &self.logical_file_schema,
            physical_file_schema: &self.physical_file_schema,
            partition_fields: &self.partition_values,
        };
        expr.transform(|expr| rewriter.rewrite_expr(Arc::clone(&expr)))
            .data()
    }

    fn with_partition_values(
        &self,
        partition_values: Vec<(FieldRef, ScalarValue)>,
    ) -> Arc<dyn PhysicalExprAdapter> {
        Arc::new(DF52PhysicalExprAdapter {
            partition_values,
            ..self.clone()
        })
    }
}

struct DefaultPhysicalExprAdapterRewriter<'a> {
    logical_file_schema: &'a Schema,
    physical_file_schema: &'a Schema,
    partition_fields: &'a [(FieldRef, ScalarValue)],
}

impl<'a> DefaultPhysicalExprAdapterRewriter<'a> {
    fn rewrite_expr(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        if let Some(transformed) = self.try_rewrite_struct_field_access(&expr)? {
            return Ok(Transformed::yes(transformed));
        }

        if let Some(column) = expr.as_any().downcast_ref::<Column>() {
            return self.rewrite_column(Arc::clone(&expr), column);
        }

        Ok(Transformed::no(expr))
    }

    /// Attempt to rewrite struct field access expressions to return null if the field does not exist in the physical schema.
    /// Note that this does *not* handle nested struct fields, only top-level struct field access.
    /// See <https://github.com/apache/datafusion/issues/17114> for more details.
    fn try_rewrite_struct_field_access(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Result<Option<Arc<dyn PhysicalExpr>>> {
        let get_field_expr =
            match ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr.as_ref()) {
                Some(expr) => expr,
                None => return Ok(None),
            };

        let source_expr = match get_field_expr.args().first() {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let field_name_expr = match get_field_expr.args().get(1) {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let lit = match field_name_expr
            .as_any()
            .downcast_ref::<expressions::Literal>()
        {
            Some(lit) => lit,
            None => return Ok(None),
        };

        let field_name = match lit.value().try_as_str().flatten() {
            Some(name) => name,
            None => return Ok(None),
        };

        let column = match source_expr.as_any().downcast_ref::<Column>() {
            Some(column) => column,
            None => return Ok(None),
        };

        let physical_field = match self.physical_file_schema.field_with_name(column.name()) {
            Ok(field) => field,
            Err(_) => return Ok(None),
        };

        let physical_struct_fields = match physical_field.data_type() {
            DataType::Struct(fields) => fields,
            _ => return Ok(None),
        };

        if physical_struct_fields
            .iter()
            .any(|f| f.name() == field_name)
        {
            return Ok(None);
        }

        let logical_field = match self.logical_file_schema.field_with_name(column.name()) {
            Ok(field) => field,
            Err(_) => return Ok(None),
        };

        let logical_struct_fields = match logical_field.data_type() {
            DataType::Struct(fields) => fields,
            _ => return Ok(None),
        };

        let logical_struct_field = match logical_struct_fields
            .iter()
            .find(|f| f.name() == field_name)
        {
            Some(field) => field,
            None => return Ok(None),
        };

        let null_value = ScalarValue::Null.cast_to(logical_struct_field.data_type())?;
        Ok(Some(expressions::lit(null_value)))
    }

    fn rewrite_column(
        &self,
        expr: Arc<dyn PhysicalExpr>,
        column: &Column,
    ) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        // Get the logical field for this column if it exists in the logical schema
        let logical_field = match self.logical_file_schema.field_with_name(column.name()) {
            Ok(field) => field,
            Err(e) => {
                // If the column is a partition field, we can use the partition value
                if let Some(partition_value) = self.get_partition_value(column.name()) {
                    return Ok(Transformed::yes(expressions::lit(partition_value)));
                }
                // This can be hit if a custom rewrite injected a reference to a column that doesn't exist in the logical schema.
                // For example, a pre-computed column that is kept only in the physical schema.
                // If the column exists in the physical schema, we can still use it.
                if let Ok(physical_field) = self.physical_file_schema.field_with_name(column.name())
                {
                    // If the column exists in the physical schema, we can use it in place of the logical column.
                    // This is nice to users because if they do a rewrite that results in something like `physical_int32_col = 123u64`
                    // we'll at least handle the casts for them.
                    physical_field
                } else {
                    // A completely unknown column that doesn't exist in either schema!
                    // This should probably never be hit unless something upstream broke, but nonetheless it's better
                    // for us to return a handleable error than to panic / do something unexpected.
                    return Err(e.into());
                }
            }
        };

        // Check if the column exists in the physical schema
        let physical_column_index = match self.physical_file_schema.index_of(column.name()) {
            Ok(index) => index,
            Err(_) => {
                if !logical_field.is_nullable() {
                    return exec_err!(
                        "Non-nullable column '{}' is missing from the physical schema",
                        column.name()
                    );
                }
                // If the column is missing from the physical schema fill it in with nulls as `SchemaAdapter` would do.
                // TODO: do we need to sync this with what the `SchemaAdapter` actually does?
                // While the default implementation fills in nulls in theory a custom `SchemaAdapter` could do something else!
                // See https://github.com/apache/datafusion/issues/16527
                let null_value = ScalarValue::Null.cast_to(logical_field.data_type())?;
                return Ok(Transformed::yes(expressions::lit(null_value)));
            }
        };
        let physical_field = self.physical_file_schema.field(physical_column_index);

        let column = match (
            column.index() == physical_column_index,
            logical_field.data_type() == physical_field.data_type(),
        ) {
            // If the column index matches and the data types match, we can use the column as is
            (true, true) => return Ok(Transformed::no(expr)),
            // If the indexes or data types do not match, we need to create a new column expression
            (true, _) => column.clone(),
            (false, _) => Column::new_with_schema(logical_field.name(), self.physical_file_schema)?,
        };

        if logical_field.data_type() == physical_field.data_type() {
            // If the data types match, we can use the column as is
            return Ok(Transformed::yes(Arc::new(column)));
        }

        // We need to cast the column to the logical data type
        // TODO: add optimization to move the cast from the column to literal expressions in the case of `col = 123`
        // since that's much cheaper to evalaute.
        // See https://github.com/apache/datafusion/issues/15780#issuecomment-2824716928
        match (physical_field.data_type(), logical_field.data_type()) {
            (DataType::Struct(physical_fields), DataType::Struct(logical_fields)) => {
                validate_struct_compatibility(physical_fields, logical_fields)?;
            }
            _ => {
                let is_compatible =
                    can_cast_types(physical_field.data_type(), logical_field.data_type());
                if !is_compatible {
                    return exec_err!(
                        "Cannot cast column '{}' from '{}' (physical data type) to '{}' (logical data type)",
                        column.name(),
                        physical_field.data_type(),
                        logical_field.data_type()
                    );
                }
            }
        }

        let cast_expr = Arc::new(CastColumnExpr::new(
            Arc::new(column),
            Arc::new(physical_field.clone()),
            Arc::new(logical_field.clone()),
            None,
        ));

        Ok(Transformed::yes(cast_expr))
    }

    fn get_partition_value(&self, column_name: &str) -> Option<ScalarValue> {
        self.partition_fields
            .iter()
            .find(|(field, _)| field.name() == column_name)
            .map(|(_, value)| value.clone())
    }
}
