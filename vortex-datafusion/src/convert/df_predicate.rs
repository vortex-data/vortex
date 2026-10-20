// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::DataType;
use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions as df_expr;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::utils::reassign_expr_columns;
use itertools::Itertools;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::display::ExprDisplay;
use vortex::expr::get_item;
use vortex::expr::root;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::session::registry::CachedId;
use vortex_arrow::ArrowSessionExt;

/// A DataFusion predicate that Vortex cannot express, such as the hash table lookup in the
/// dynamic filter of a hash join, evaluated by DataFusion on Arrow batches of the columns it reads.
///
/// A scan then removes the rows the predicate rejects before it decodes the other columns, as it
/// does for the predicates it converts.
#[derive(Clone)]
pub(crate) struct DataFusionPredicate;

/// The predicate, over a batch whose column `i` is child `i` of the expression.
#[derive(Clone)]
pub(crate) struct DataFusionPredicateOptions {
    predicate: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

impl DataFusionPredicate {
    /// An expression that evaluates `predicate`, a boolean expression over `schema`, on the
    /// columns it reads. Returns `None` if the predicate reads no column or is not boolean.
    pub(crate) fn expression(
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
    ) -> Option<Expression> {
        // The dynamic filter of a partitioned join routes each row by its hash to the filter of
        // its partition, which repeats the hashing the join does for every row anyway.
        if predicate.downcast_ref::<df_expr::CaseExpr>().is_some() {
            return None;
        }
        let indices = collect_columns(predicate)
            .into_iter()
            .map(|column| column.index())
            .sorted_unstable()
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return None;
        }
        let batch_schema = Arc::new(schema.project(&indices).ok()?);
        let predicate = reassign_expr_columns(Arc::clone(predicate), &batch_schema).ok()?;
        if predicate.data_type(&batch_schema).ok()? != DataType::Boolean {
            return None;
        }
        let inputs = batch_schema
            .fields()
            .iter()
            .map(|field| get_item(field.name().as_str(), root()))
            .collect::<Vec<_>>();
        Some(DataFusionPredicate.new_expr(
            DataFusionPredicateOptions {
                predicate,
                schema: batch_schema,
            },
            inputs,
        ))
    }
}

impl ScalarFnVTable for DataFusionPredicate {
    type Options = DataFusionPredicateOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex_datafusion.predicate");
        *ID
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(options.schema.fields().len())
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(options.schema.field(child_idx).name().as_str()))
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        _expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "datafusion({})", options.predicate)
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::Nullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let session = ctx.session().clone();
        let arrow = session.arrow();
        let columns = options
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| arrow.execute_arrow(args.get(idx)?, Some(field), ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&options.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(args.row_count())),
        )
        .map_err(|err| vortex_err!("Failed to build a batch for {}: {err}", options.predicate))?;
        let result = options
            .predicate
            .evaluate(&batch)
            .and_then(|value| value.into_array(batch.num_rows()))
            .map_err(|err| vortex_err!("Failed to evaluate {}: {err}", options.predicate))?;
        arrow.from_arrow_array(result, true)
    }
}

impl Debug for DataFusionPredicateOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFusionPredicateOptions")
            .field("predicate", &self.predicate)
            .finish()
    }
}

impl Display for DataFusionPredicateOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.predicate, f)
    }
}

/// Equality and hashing use the identity of the predicate.
impl PartialEq for DataFusionPredicateOptions {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.predicate, &other.predicate)
    }
}

impl Eq for DataFusionPredicateOptions {}

impl Hash for DataFusionPredicateOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.predicate).cast::<()>().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion_expr::Operator;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::assert_arrays_eq;
    use vortex::session::VortexSession;

    use super::*;

    #[test]
    fn test_evaluates_predicates_vortex_cannot_express() -> VortexResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ]));
        // `IS NOT DISTINCT FROM` has no Vortex equivalent.
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
            Arc::new(df_expr::Column::new("b", 1)),
            Operator::IsNotDistinctFrom,
            df_expr::lit(3i32),
        ));
        let expr = DataFusionPredicate::expression(&predicate, &schema)
            .ok_or_else(|| vortex_err!("predicate reads a column"))?;

        let input = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32, 2, 3]).into_array()),
            (
                "b",
                PrimitiveArray::from_option_iter([Some(3i32), None, Some(4)]).into_array(),
            ),
        ])?
        .into_array();
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(
            input.apply(&expr)?,
            BoolArray::from_iter([Some(true), Some(false), Some(false)]),
            &mut ctx
        );
        Ok(())
    }
}
