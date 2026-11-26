// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::catalog::TableFunctionArgs;
use datafusion::catalog::TableFunctionImpl;
use datafusion::catalog::TableProvider;
use datafusion::catalog::default_table_source::provider_as_source;
use datafusion::catalog::view::ViewTable;
use datafusion::common::Result;
use datafusion::common::ScalarValue;
use datafusion::common::plan_err;
use datafusion::functions_nested::expr_fn::cosine_distance;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::prelude::SessionContext;
use datafusion::prelude::col;
use datafusion::prelude::lit;

use super::table::RecipeTable;
use super::table::Search;

/// Register `function(vector_json, k [, min_price])` for a recipe table. The function returns
/// candidate IDs and exact cosine distances over the retrieved embeddings. Order by distance
/// in the caller. A price bound restricts candidates before the backend selects its top k.
pub async fn register_topk(ctx: &SessionContext, table: &str, function: &str) -> Result<()> {
    let provider = ctx.table_provider(table).await?;
    let Some(recipe) = provider.downcast_ref::<RecipeTable>() else {
        return plan_err!("expected a registered recipe table");
    };
    ctx.register_udtf(function, Arc::new(TopK(recipe.clone())));
    Ok(())
}

#[derive(Debug)]
struct TopK(RecipeTable);

impl TableFunctionImpl for TopK {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let (vector, k, floor) = match args.exprs() {
            [vector, k] => (vector, k, None),
            [vector, k, Expr::Literal(ScalarValue::Int64(Some(floor)), _)] => {
                (vector, k, Some(*floor))
            }
            _ => return plan_err!("expected (vector_json, k [, min_price])"),
        };
        let Expr::Literal(
            ScalarValue::Utf8(Some(json))
            | ScalarValue::Utf8View(Some(json))
            | ScalarValue::LargeUtf8(Some(json)),
            _,
        ) = vector
        else {
            return plan_err!("expected a JSON vector string");
        };
        let Expr::Literal(ScalarValue::Int64(Some(k)), _) = k else {
            return plan_err!("expected an integer k");
        };
        let Ok(k) = usize::try_from(*k) else {
            return plan_err!("k must be positive");
        };
        let Ok(vector) = serde_json::from_str::<Vec<f32>>(json) else {
            return plan_err!("expected a JSON array of floats");
        };
        let schema = self.0.schema();
        let DataType::FixedSizeList(_, dim) = schema.field_with_name("emb")?.data_type() else {
            return plan_err!("expected fixed-size embeddings");
        };
        if k == 0
            || vector.len() != usize::try_from(*dim).unwrap_or_default()
            || !vector.iter().all(|x| x.is_finite())
            || vector.iter().all(|x| *x == 0.0)
        {
            return plan_err!("expected positive k and a finite, nonzero {dim}-dimensional vector");
        }
        let scalar = ScalarValue::List(ScalarValue::new_list(
            &vector
                .iter()
                .copied()
                .map(ScalarValue::from)
                .collect::<Vec<_>>(),
            &DataType::Float32,
            false,
        ));
        let mut table = self.0.clone();
        table.search = Some(Search {
            vector,
            k,
            min_price: floor,
        });
        let plan =
            LogicalPlanBuilder::scan("candidates", provider_as_source(Arc::new(table)), None)?
                .project([
                    col("id"),
                    cosine_distance(col("emb"), lit(scalar)).alias("distance"),
                ])?
                .build()?;
        Ok(Arc::new(ViewTable::new(plan, None)))
    }
}
