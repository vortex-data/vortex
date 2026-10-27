// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::transform::replace;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;

use crate::convert::ExpressionConverter;

pub(super) fn convert_dynamic_filter(
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
) -> Expression {
    let mut columns = collect_columns(&expression).into_iter().collect::<Vec<_>>();
    columns.sort_unstable_by_key(|column| column.index());
    let inputs = columns
        .iter()
        .map(|column| get_item(column.name(), root()))
        .collect::<Vec<_>>();
    let scope_projection = pack(
        columns
            .iter()
            .enumerate()
            .map(|(idx, column)| (column.name(), get_item(input_field_name(idx), root()))),
        Nullability::NonNullable,
    );
    dynamic_expression(
        Arc::new(DataFusionDynamicFilter {
            expression,
            schema,
            scope_projection,
            converter,
        }),
        inputs,
    )
}

struct DataFusionDynamicFilter {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    scope_projection: Expression,
    converter: Arc<dyn ExpressionConverter>,
}

impl fmt::Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}

impl fmt::Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.expression)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let converted = snapshot_physical_expr(Arc::clone(&self.expression))
            .and_then(|snapshot| self.converter.try_convert(&snapshot, &self.schema))
            .ok()
            .flatten();
        // The producing DataFusion operator retains exact enforcement when conversion fails.
        let expression = converted.map_or_else(
            || lit(true),
            |expr| replace(expr, &root(), self.scope_projection.clone()),
        );
        expression.bind(scope)
    }
}
