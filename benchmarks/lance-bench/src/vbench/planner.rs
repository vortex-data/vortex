// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::arrow::array::Array;
use datafusion::arrow::array::AsArray;
use datafusion::arrow::array::types::Float32Type;
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::record_batch::RecordBatchOptions;
use datafusion::catalog::default_table_source::provider_as_source;
use datafusion::catalog::default_table_source::source_as_provider;
use datafusion::common::DFSchema;
use datafusion::common::Result;
use datafusion::common::ScalarValue;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::Transformed;
use datafusion::common::tree_node::TreeNode;
use datafusion::functions_nested::cosine_distance::CosineDistance;
use datafusion::functions_nested::make_array::MakeArray;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::Operator;
use datafusion::logical_expr::logical_plan::FetchType;
use datafusion::logical_expr::logical_plan::SkipType;
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_expr::create_physical_expr;
use datafusion::physical_expr::execution_props::ExecutionProps;

use super::table::RecipeTable;
use super::table::Search;

/// Only explicitly registered RecipeTables opt in to approximate candidate selection.
/// Preserve the SQL operators above the scan so DataFusion scores and orders the fetched rows.
#[derive(Debug)]
pub(super) struct VBenchAnn;

impl AnalyzerRule for VBenchAnn {
    fn name(&self) -> &str {
        "vbench_cosine_topk"
    }

    fn analyze(&self, plan: LogicalPlan, _: &ConfigOptions) -> Result<LogicalPlan> {
        Ok(plan
            .transform_up(|plan| {
                let Some(table) = candidate_scan(&plan)? else {
                    return Ok(Transformed::no(plan));
                };
                plan.transform_down(|node| match node {
                    LogicalPlan::TableScan(mut scan) => {
                        scan.source = provider_as_source(Arc::new(table.clone()));
                        Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
                    }
                    other => Ok(Transformed::no(other)),
                })
            })?
            .data)
    }
}

fn candidate_scan(plan: &LogicalPlan) -> Result<Option<RecipeTable>> {
    let LogicalPlan::Limit(limit) = plan else {
        return Ok(None);
    };
    let (SkipType::Literal(0), FetchType::Literal(Some(k)), LogicalPlan::Sort(sort)) = (
        limit.get_skip_type()?,
        limit.get_fetch_type()?,
        limit.input.as_ref(),
    ) else {
        return Ok(None);
    };
    let ([order], LogicalPlan::Projection(projection)) =
        (sort.expr.as_slice(), sort.input.as_ref())
    else {
        return Ok(None);
    };
    if !order.asc || k == 0 {
        return Ok(None);
    }
    let Expr::Column(distance) = &order.expr else {
        return Ok(None);
    };
    let Some(distance) = projection.expr.iter().find_map(|expr| match expr {
        Expr::Alias(alias) if alias.name == distance.name => Some(alias.expr.as_ref()),
        _ => None,
    }) else {
        return Ok(None);
    };
    let Expr::ScalarFunction(distance) = distance else {
        return Ok(None);
    };
    let [column, query] = distance.args.as_slice() else {
        return Ok(None);
    };
    if !distance.func.inner().is::<CosineDistance>() || !embedding_column(column) {
        return Ok(None);
    }

    let (input, min_price) = match projection.input.as_ref() {
        LogicalPlan::Filter(filter) => {
            let Expr::BinaryExpr(predicate) = &filter.predicate else {
                return Ok(None);
            };
            let (
                Expr::Column(column),
                Operator::GtEq,
                Expr::Literal(ScalarValue::Int64(Some(floor)), _),
            ) = (
                predicate.left.as_ref(),
                predicate.op,
                predicate.right.as_ref(),
            )
            else {
                return Ok(None);
            };
            if column.name != "price" {
                return Ok(None);
            }
            (filter.input.as_ref(), Some(*floor))
        }
        input => (input, None),
    };
    let LogicalPlan::TableScan(scan) = input else {
        return Ok(None);
    };
    let Ok(provider) = source_as_provider(&scan.source) else {
        return Ok(None);
    };
    let Some(table) = provider.downcast_ref::<RecipeTable>() else {
        return Ok(None);
    };
    if table.search.is_some()
        || !scan.filters.is_empty()
        || scan.fetch.is_some()
        || scan.skip.is_some()
    {
        return Ok(None);
    }
    let Some(vector) = constant_vector(query)? else {
        return Ok(None);
    };
    let mut table = table.clone();
    table.search = Some(Search {
        vector,
        k,
        min_price,
    });
    Ok(Some(table))
}

fn embedding_column(expr: &Expr) -> bool {
    match expr {
        Expr::Column(column) => column.name == "emb",
        Expr::Cast(cast)
            if matches!(cast.field.data_type(), DataType::List(item) | DataType::LargeList(item)
            if matches!(item.data_type(), DataType::Float32 | DataType::Float64)) =>
        {
            embedding_column(&cast.expr)
        }
        _ => false,
    }
}

fn vector_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(..) => true,
        Expr::Cast(cast) => vector_literal(&cast.expr),
        Expr::Negative(inner) => vector_literal(inner),
        Expr::ScalarFunction(fun) if fun.func.inner().is::<MakeArray>() => {
            fun.args.iter().all(vector_literal)
        }
        _ => false,
    }
}

fn constant_vector(expr: &Expr) -> Result<Option<Vec<f32>>> {
    if !vector_literal(expr) {
        return Ok(None);
    }
    let physical = create_physical_expr(
        expr,
        &DFSchema::empty(),
        &ExecutionProps::new(),
        &PhysicalPlanningContext::default(),
    )?;
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )?;
    let value = physical.evaluate(&batch)?.into_array(1)?;
    if value.is_null(0) {
        return Ok(None);
    }
    let values = match value.data_type() {
        DataType::List(_) => value.as_list::<i32>().value(0),
        DataType::LargeList(_) => value.as_list::<i64>().value(0),
        DataType::FixedSizeList(..) => value.as_fixed_size_list().value(0),
        _ => return Ok(None),
    };
    let floats = cast(&values, &DataType::Float32)?;
    let floats = floats.as_primitive::<Float32Type>();
    if floats.null_count() != 0
        || !floats.values().iter().all(|x| x.is_finite())
        || floats.values().iter().all(|x| *x == 0.0)
    {
        return Ok(None);
    }
    Ok(Some(floats.values().to_vec()))
}
