// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Shared utilities and custom implementations for schema and expression adapters.
//!
//! Most of this code is taken from one of two files:
//!
//! `datafusion/physical-expr-adapter/src/schema_rewriter.rs` -> everything related to the
//! DefaultPhysicalExprAdapter
//!
//! `datafusion/datasource/src/schema_adapter.rs` -> for can_cast_field (which is crate-private)
//!
//! See https://github.com/apache/datafusion/issues/18957

use std::fmt::Debug;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::can_cast_types;
use datafusion_common::exec_err;
use datafusion_common::nested_struct::validate_struct_compatibility;
use datafusion_common::plan_err;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TransformedResult;
use datafusion_common::tree_node::TreeNode;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions;
use datafusion_physical_expr::expressions::CastExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr_adapter::PhysicalExprAdapter;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

#[derive(Debug, Clone)]
pub struct DefaultPhysicalExprAdapterFactory;

impl PhysicalExprAdapterFactory for DefaultPhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Arc<dyn PhysicalExprAdapter> {
        Arc::new(DefaultPhysicalExprAdapter::new(
            logical_file_schema,
            physical_file_schema,
        ))
    }
}
#[derive(Debug, Clone)]
pub struct DefaultPhysicalExprAdapter {
    logical_file_schema: SchemaRef,
    physical_file_schema: SchemaRef,
    partition_values: Vec<(FieldRef, ScalarValue)>,
}

impl DefaultPhysicalExprAdapter {
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

impl PhysicalExprAdapter for DefaultPhysicalExprAdapter {
    fn rewrite(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> datafusion_common::Result<Arc<dyn PhysicalExpr>> {
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
        Arc::new(Self {
            partition_values,
            ..self.clone()
        })
    }
}

/// This is derived from the standard library, with the exception that it fixes handling of
/// rewriting expressions with struct fields.
pub(crate) struct DefaultPhysicalExprAdapterRewriter<'a> {
    logical_file_schema: &'a Schema,
    physical_file_schema: &'a Schema,
    partition_fields: &'a [(FieldRef, ScalarValue)],
}

impl<'a> DefaultPhysicalExprAdapterRewriter<'a> {
    fn rewrite_expr(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> datafusion_common::Result<Transformed<Arc<dyn PhysicalExpr>>> {
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
    ) -> datafusion_common::Result<Option<Arc<dyn PhysicalExpr>>> {
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
    ) -> datafusion_common::Result<Transformed<Arc<dyn PhysicalExpr>>> {
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

        // This has been changed to replace the DF upstream version which just does can_cast_types,
        // which ignores struct fields with compatible but differing columns.
        let is_compatible = can_cast_field(physical_field, logical_field)?;
        if !is_compatible {
            return exec_err!(
                "Cannot cast column '{}' from '{}' (physical data type) to '{}' (logical data type)",
                column.name(),
                physical_field.data_type(),
                logical_field.data_type()
            );
        }

        let cast_expr = Arc::new(CastExpr::new(
            Arc::new(column),
            logical_field.data_type().clone(),
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

pub(crate) fn can_cast_field(
    file_field: &Field,
    table_field: &Field,
) -> datafusion_common::Result<bool> {
    match (file_field.data_type(), table_field.data_type()) {
        (DataType::Struct(source_fields), DataType::Struct(target_fields)) => {
            validate_struct_compatibility(source_fields, target_fields)
        }
        _ => {
            if can_cast_types(file_field.data_type(), table_field.data_type()) {
                Ok(true)
            } else {
                plan_err!(
                    "Cannot cast file schema field {} of type {:?} to table schema field of type {:?}",
                    file_field.name(),
                    file_field.data_type(),
                    table_field.data_type()
                )
            }
        }
    }
}
