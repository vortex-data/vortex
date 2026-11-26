// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
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
use lance::Dataset;
use vortex::file::VortexFile;
use vortex_index::IndexFile;
use vortex_index_ivf::IvfIndex;

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
        search_view(&self.0, string_literal(vector)?, k, floor, false)
    }
}

/// The named indexed sources available to the session's vector_search table function.
#[derive(Default, Debug)]
pub struct VectorSearch {
    tables: BTreeMap<String, RecipeTable>,
}

impl VectorSearch {
    pub fn add_vortex(
        &mut self,
        name: &str,
        file: VortexFile,
        index: &IndexFile<IvfIndex>,
        probes: usize,
    ) -> anyhow::Result<()> {
        self.tables
            .insert(name.to_owned(), super::vortex_table(file, index, probes)?);
        Ok(())
    }

    pub fn add_lance(
        &mut self,
        name: &str,
        dataset: Arc<Dataset>,
        probes: usize,
    ) -> anyhow::Result<()> {
        self.tables
            .insert(name.to_owned(), super::lance_table(dataset, probes)?);
        Ok(())
    }

    pub fn register(self, ctx: &SessionContext) {
        ctx.register_udtf("vector_search", Arc::new(self));
    }
}

impl TableFunctionImpl for VectorSearch {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let [name, column, query, k] = args.exprs() else {
            return plan_err!("expected vector_search(table, column, vector_json, k)");
        };
        let name = string_literal(name)?;
        let Some(source) = self.tables.get(name) else {
            return plan_err!("unknown vector search table {name}");
        };
        let column = string_literal(column)?;
        if column != "emb" {
            return plan_err!("no registered vector index for {name}.{column}");
        }
        search_view(source, string_literal(query)?, k, None, true)
    }
}

fn string_literal(expr: &Expr) -> Result<&str> {
    match expr {
        Expr::Literal(
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)),
            _,
        ) => Ok(value),
        _ => plan_err!("expected a string literal"),
    }
}

fn search_view(
    source: &RecipeTable,
    json: &str,
    k: &Expr,
    floor: Option<i64>,
    expose_price: bool,
) -> Result<Arc<dyn TableProvider>> {
    let Expr::Literal(ScalarValue::Int64(Some(k)), _) = k else {
        return plan_err!("expected an integer k");
    };
    let Ok(k) = usize::try_from(*k) else {
        return plan_err!("k must be positive");
    };
    let Ok(vector) = serde_json::from_str::<Vec<f32>>(json) else {
        return plan_err!("expected a JSON array of floats");
    };
    let schema = source.schema();
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
    let mut table = source.clone();
    table.push_price_filter = expose_price;
    table.search = Some(Search {
        vector,
        k,
        min_price: floor,
    });
    let mut columns = vec![
        col("id"),
        cosine_distance(col("emb"), lit(scalar)).alias("distance"),
    ];
    if expose_price {
        columns.extend([col("price"), col("emb")]);
    }
    let plan = LogicalPlanBuilder::scan("candidates", provider_as_source(Arc::new(table)), None)?
        .project(columns)?
        .build()?;
    Ok(Arc::new(ViewTable::new(plan, None)))
}
