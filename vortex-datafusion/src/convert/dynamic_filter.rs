// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use itertools::Itertools;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::bound::and_collect;
use vortex::expr::bound::lit;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::root;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::get_item::GetItem;

use crate::convert::ExpressionConverter;

/// A DataFusion predicate with dynamic filters, as the source of a Vortex dynamic expression.
///
/// The columns of the predicate become the inputs of the dynamic expression, so the scan reads
/// them before the dynamic filters have their first value. Each new generation of the dynamic
/// filters is converted again. A dynamic filter only discards rows that cannot contribute to the
/// result, so conjuncts that Vortex cannot evaluate are dropped.
pub(crate) struct DataFusionDynamicFilter {
    predicate: Arc<dyn PhysicalExpr>,
    columns: Vec<FieldName>,
    converter: Arc<dyn ExpressionConverter>,
    schema: SchemaRef,
}

impl DataFusionDynamicFilter {
    /// Create a Vortex dynamic expression for a predicate adapted to `schema`.
    pub(crate) fn expression(
        predicate: Arc<dyn PhysicalExpr>,
        converter: Arc<dyn ExpressionConverter>,
        schema: SchemaRef,
    ) -> Expression {
        let columns = collect_columns(&predicate)
            .into_iter()
            .sorted_by_key(|column| column.index())
            .map(|column| FieldName::from(column.name()))
            .collect::<Vec<_>>();
        let inputs = columns
            .iter()
            .map(|column| get_item(column.clone(), root()))
            .collect::<Vec<_>>();

        let source = Self {
            predicate,
            columns,
            converter,
            schema,
        };
        dynamic_expression(Arc::new(source), inputs)
    }

    /// Replace the column references of `expr` by the inputs of the dynamic expression. Returns
    /// `None` if `expr` reads the scope in another way.
    fn read_inputs(&self, expr: &Expression) -> Option<Expression> {
        if let Some(field) = expr.as_opt::<GetItem>()
            && expr.child(0).is_root()
        {
            let input_idx = self.columns.iter().position(|column| column == field)?;
            return Some(get_item(input_field_name(input_idx), root()));
        }

        if expr.is_root() {
            return None;
        }

        let children = expr
            .children()
            .iter()
            .map(|child| self.read_inputs(child))
            .collect::<Option<Vec<_>>>()?;
        expr.clone().with_children(children).ok()
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.predicate)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let snapshot = snapshot_physical_expr(Arc::clone(&self.predicate))
            .and_then(|snapshot| PhysicalExprSimplifier::new(&self.schema).simplify(snapshot))
            .map_err(|err| vortex_err!("Failed to snapshot dynamic filter {self}: {err}"))?;

        let conjuncts = split_conjunction(&snapshot)
            .into_iter()
            .filter_map(
                |conjunct| match self.converter.try_convert(conjunct, &self.schema) {
                    Ok(converted) => converted,
                    Err(err) => {
                        tracing::debug!(%conjunct, %err, "dropping dynamic filter conjunct");
                        None
                    }
                },
            )
            .filter_map(|conjunct| self.read_inputs(&conjunct)?.bind(scope).ok())
            .collect::<Vec<_>>();

        Ok(and_collect(conjuncts).unwrap_or_else(|| lit(true)))
    }
}

impl Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.predicate, f)
    }
}

impl Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFusionDynamicFilter")
            .field("predicate", &self.predicate)
            .field("columns", &self.columns)
            .finish()
    }
}
