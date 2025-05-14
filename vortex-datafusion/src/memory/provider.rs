use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::prelude::*;
use datafusion_common::{Result as DFResult, ToDFSchema};
use datafusion_expr::utils::conjunction;
use datafusion_expr::{TableProviderFilterPushDown, TableType};
use datafusion_physical_expr::{EquivalenceProperties, create_physical_expr};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::{ExecutionPlan, Partitioning, PlanProperties};
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::arrays::{ChunkedArray, ChunkedVTable};
use vortex_error::{VortexError, VortexExpect as _};
use vortex_expr::ExprRef;
use vortex_expr::datafusion::convert_expr_to_vortex;

use crate::can_be_pushed_down;
use crate::memory::exec::VortexScanExec;
use crate::memory::plans::{RowSelectorExec, TakeRowsExec};

/// A [`TableProvider`] that exposes an existing Vortex Array to the DataFusion SQL engine.
///
/// Only arrays that have a top-level [struct type](vortex_dtype::StructDType) can be exposed as
/// a table to DataFusion.
#[derive(Debug, Clone)]
pub struct VortexMemTable {
    array: ChunkedArray,
    schema_ref: SchemaRef,
}

impl VortexMemTable {
    /// Build a new table provider from an existing [struct type](vortex_dtype::StructDType) array.
    ///
    /// # Panics
    ///
    /// Creation will panic if the provided array is not of `DType::Struct` type.
    pub fn new(array: ArrayRef) -> Self {
        let arrow_schema = array
            .dtype()
            .to_arrow_schema()
            .vortex_expect("schema is inferable");
        let schema_ref = SchemaRef::new(arrow_schema);

        let array = match array.as_opt::<ChunkedVTable>() {
            Some(a) => a.clone(),
            _ => {
                let dtype = array.dtype().clone();
                ChunkedArray::try_new(vec![array], dtype)
                    .vortex_expect("Failed to wrap array as a ChunkedArray with 1 chunk")
            }
        };

        Self { array, schema_ref }
    }
}

#[async_trait]
impl TableProvider for VortexMemTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema_ref)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Plan an array scan.
    ///
    /// Currently, projection pushdown is supported, but not filter pushdown.
    /// The array is flattened directly into the nearest Arrow-compatible encoding.
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let output_projection: Vec<usize> = match projection {
            None => (0..self.schema_ref.fields().len()).collect(),
            Some(proj) => proj.clone(),
        };

        match conjunction(filters.to_vec()) {
            // If there is a filter expression, we execute in two phases, first performing a filter
            // on the input to get back row indices, and then taking the remaining struct columns
            // using the calculated indices from the filter.
            Some(expr) => {
                let df_schema = self.schema_ref.clone().to_dfschema()?;

                let filter_expr = create_physical_expr(&expr, &df_schema, state.execution_props())?;
                let filter_expr = convert_expr_to_vortex(filter_expr)?;

                make_filter_then_take_plan(
                    self.schema_ref.clone(),
                    filter_expr,
                    self.array.clone(),
                    output_projection,
                    state,
                )
            }

            // If no filters were pushed down, we materialize the entire StructArray into a
            // RecordBatch and let DataFusion process the entire query.
            _ => {
                let output_schema = Arc::new(
                    self.schema_ref
                        .project(output_projection.as_slice())
                        .map_err(VortexError::from)?,
                );
                let plan_properties = PlanProperties::new(
                    EquivalenceProperties::new(output_schema),
                    // non-pushdown scans execute in single partition, where the partition
                    // yields one RecordBatch per chunk in the input ChunkedArray
                    Partitioning::UnknownPartitioning(1),
                    EmissionType::Incremental,
                    Boundedness::Bounded,
                );

                Ok(Arc::new(VortexScanExec::try_new(
                    self.array.clone(),
                    output_projection,
                    plan_properties,
                )?))
            }
        }
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|expr| {
                if can_be_pushed_down(expr, self.schema().as_ref()) {
                    Ok(TableProviderFilterPushDown::Exact)
                } else {
                    Ok(TableProviderFilterPushDown::Unsupported)
                }
            })
            .try_collect()
    }
}

/// Construct an operator plan that executes in two stages.
///
/// The first plan stage only materializes the columns related to the provided set of filter
/// expressions. It evaluates the filters into a row selection.
///
/// The second stage receives the row selection above and dispatches a `take` on the remaining
/// columns.
fn make_filter_then_take_plan(
    schema: SchemaRef,
    filter_expr: ExprRef,
    chunked_array: ChunkedArray,
    output_projection: Vec<usize>,
    _session_state: &dyn Session,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let row_selector_op = Arc::new(RowSelectorExec::try_new(
        filter_expr,
        chunked_array.clone(),
    )?);

    Ok(Arc::new(TakeRowsExec::new(
        schema,
        &output_projection,
        row_selector_op,
        chunked_array,
    )))
}

#[cfg(test)]
mod test {
    use arrow_array::cast::AsArray as _;
    use arrow_array::types::Int64Type;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::functions_aggregate::count::count_distinct;
    use datafusion::prelude::SessionContext;
    use datafusion_common::{Column, Spans, TableReference};
    use datafusion_expr::{BinaryExpr, Expr, Operator, and, col, lit};
    use vortex_array::arrays::{PrimitiveArray, StructArray, VarBinViewArray};
    use vortex_array::{ArrayRef, IntoArray};

    use crate::{SessionContextExt as _, can_be_pushed_down};

    fn presidents_array() -> ArrayRef {
        let names = VarBinViewArray::from_iter_str([
            "Washington",
            "Adams",
            "Jefferson",
            "Madison",
            "Monroe",
            "Adams",
        ]);
        let term_start = PrimitiveArray::from_iter([1789u16, 1797, 1801, 1809, 1817, 1825]);

        StructArray::from_fields(&[
            ("president", names.into_array()),
            ("term_start", term_start.into_array()),
        ])
        .unwrap()
        .into_array()
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_datafusion_pushdown() {
        let ctx = SessionContext::new();

        let df = ctx.read_mem_vortex(presidents_array()).unwrap();

        let distinct_names = df
            .filter(col("term_start").gt_eq(lit(1795)))
            .unwrap()
            .aggregate(vec![], vec![count_distinct(col("president"))])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(distinct_names.len(), 1);

        assert_eq!(
            *distinct_names[0]
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .first()
                .unwrap(),
            4i64
        );
    }

    #[test]
    fn test_can_be_pushed_down0() {
        let e = BinaryExpr {
            left: Box::new(
                Column {
                    relation: Some(TableReference::Bare {
                        table: "orders".into(),
                    }),
                    name: "o_orderstatus".to_string(),
                    spans: Spans::new(),
                }
                .into(),
            ),
            op: Operator::Eq,
            right: Box::new(lit("F")),
        };
        let e = Expr::BinaryExpr(e);

        assert!(can_be_pushed_down(
            &e,
            &Schema::new(vec![Field::new("o_orderstatus", DataType::Utf8, true)])
        ));
    }

    #[test]
    fn test_can_be_pushed_down1() {
        let e = lit("hello");

        assert!(can_be_pushed_down(&e, &Schema::empty()));
    }

    #[test]
    fn test_can_be_pushed_down2() {
        let e = lit(3);

        assert!(can_be_pushed_down(&e, &Schema::empty()));
    }

    #[test]
    fn test_can_be_pushed_down3() {
        let e = BinaryExpr {
            left: Box::new(col("nums")),
            op: Operator::Modulo,
            right: Box::new(lit(5)),
        };
        let e = Expr::BinaryExpr(e);

        assert!(!can_be_pushed_down(
            &e,
            &Schema::new(vec![Field::new("nums", DataType::Int32, true)])
        ));
    }

    #[test]
    fn test_can_be_pushed_down4() {
        let e = and((col("a")).eq(lit(2u64)), col("b").eq(lit(true)));
        assert!(can_be_pushed_down(
            &e,
            &Schema::new(vec![
                Field::new("a", DataType::UInt64, true),
                Field::new("b", DataType::Boolean, true)
            ])
        ));
    }
}
