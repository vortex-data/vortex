// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BooleanArray;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_array::UInt32Array;
use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_schema::DataType;
use arrow_schema::SchemaRef;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::take_record_batch;
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
use vortex_utils::aliases::hash_map::HashMap;

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

/// Whether evaluating `expr` in a scan that outputs `projected` columns can pay off. A predicate
/// that DataFusion evaluates costs about as much as the join it comes from, so it only pays when
/// the rows it removes save decoding columns it does not read.
pub(crate) fn saves_decoding(expr: &Expression, projected: &[&str]) -> bool {
    let Some(options) = expr.as_opt::<DataFusionPredicate>() else {
        return true;
    };
    projected.iter().any(|column| {
        options
            .schema
            .fields()
            .iter()
            .all(|field| field.name() != column)
    })
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
        let result = match options.predicate.downcast_ref::<df_expr::CaseExpr>() {
            Some(case) => evaluate_routed(case, &batch),
            None => options
                .predicate
                .evaluate(&batch)
                .and_then(|value| value.into_array(batch.num_rows())),
        }
        .map_err(|err| vortex_err!("Failed to evaluate {}: {err}", options.predicate))?;
        arrow.from_arrow_array(result, true)
    }
}

/// Evaluates `case`, which routes each row to one boolean branch by the value of its base
/// expression, in one pass over the rows.
///
/// The dynamic filter of a partitioned hash join is
/// `CASE hash(keys) % n WHEN 0 THEN filter_0 ... ELSE false END`. DataFusion evaluates such a
/// CASE by comparing every row with each WHEN value and filtering the batch for each branch, which
/// copies the batch once per partition. Here the rows of each branch are gathered once and each
/// branch evaluates only its rows. A null result is false, as the predicate filters rows.
fn evaluate_routed(case: &df_expr::CaseExpr, batch: &RecordBatch) -> DFResult<ArrowArrayRef> {
    let generic = || {
        case.evaluate(batch)
            .and_then(|value| value.into_array(batch.num_rows()))
    };
    let Some(base) = case.expr() else {
        return generic();
    };
    let mut branches = HashMap::<u64, usize>::default();
    for (idx, (when, _)) in case.when_then_expr().iter().enumerate() {
        match when
            .downcast_ref::<df_expr::Literal>()
            .map(|literal| literal.value())
        {
            Some(ScalarValue::UInt64(Some(value))) => {
                branches.entry(*value).or_insert(idx);
            }
            _ => return generic(),
        }
    }
    let otherwise = match case.else_expr() {
        None => false,
        Some(expr) => match expr
            .downcast_ref::<df_expr::Literal>()
            .map(|literal| literal.value())
        {
            Some(ScalarValue::Boolean(value)) => value.unwrap_or(false),
            _ => return generic(),
        },
    };

    let routes = base.evaluate(batch)?.into_array(batch.num_rows())?;
    let Some(routes) = routes.as_primitive_opt::<UInt64Type>() else {
        return generic();
    };
    let mut rows = vec![Vec::<u32>::new(); case.when_then_expr().len()];
    let mut result = BooleanBufferBuilder::new(batch.num_rows());
    result.append_n(batch.num_rows(), otherwise);
    for (row, route) in routes.iter().enumerate() {
        if let Some(branch) = route.and_then(|route| branches.get(&route)) {
            rows[*branch]
                .push(u32::try_from(row).map_err(|err| DataFusionError::External(Box::new(err)))?);
        }
    }
    for ((_, then), rows) in case.when_then_expr().iter().zip(rows) {
        if rows.is_empty() {
            continue;
        }
        let indices = UInt32Array::from(rows);
        let branch_batch = take_record_batch(batch, &indices)?;
        let kept = then
            .evaluate(&branch_batch)?
            .into_array(branch_batch.num_rows())?;
        let kept = kept.as_boolean_opt().ok_or_else(|| {
            DataFusionError::Internal(format!("CASE branch {then} is not boolean"))
        })?;
        for (row, kept) in indices.values().iter().zip(kept.iter()) {
            result.set_bit(*row as usize, kept.unwrap_or(false));
        }
    }
    Ok(Arc::new(BooleanArray::new(result.finish(), None)))
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
    use rstest::rstest;
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

    /// A CASE that routes rows by a partition number gives the same result as DataFusion, with
    /// rows routed to no branch taking the default and null branch results false.
    #[rstest]
    #[case::default_false(false)]
    #[case::default_true(true)]
    fn test_routed_case_matches_datafusion(#[case] default: bool) -> DFResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("route", DataType::UInt64, true),
            Field::new("a", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::UInt64Array::from(vec![
                    Some(0),
                    Some(1),
                    Some(2),
                    None,
                    Some(1),
                    Some(0),
                    Some(3),
                ])),
                Arc::new(arrow_array::Int32Array::from(vec![
                    Some(1),
                    Some(5),
                    Some(9),
                    Some(1),
                    None,
                    Some(7),
                    Some(2),
                ])),
            ],
        )?;
        let a: Arc<dyn PhysicalExpr> = Arc::new(df_expr::Column::new("a", 1));
        let compare = |op, value: i32| -> Arc<dyn PhysicalExpr> {
            Arc::new(df_expr::BinaryExpr::new(
                Arc::clone(&a),
                op,
                df_expr::lit(value),
            ))
        };
        let case = df_expr::CaseExpr::try_new(
            Some(Arc::new(df_expr::Column::new("route", 0))),
            vec![
                (df_expr::lit(0u64), compare(Operator::Lt, 5)),
                (df_expr::lit(1u64), compare(Operator::Gt, 3)),
                (df_expr::lit(2u64), df_expr::lit(false)),
            ],
            Some(df_expr::lit(default)),
        )?;

        let routed = evaluate_routed(&case, &batch)?;
        let expected = case.evaluate(&batch)?.into_array(batch.num_rows())?;
        let expected = expected
            .as_boolean()
            .iter()
            .map(|value| value.unwrap_or(false))
            .collect::<BooleanArray>();
        assert_eq!(routed.as_boolean(), &expected);
        Ok(())
    }

    #[test]
    fn test_pays_off_only_when_other_columns_are_decoded() -> VortexResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ]));
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
            Arc::new(df_expr::Column::new("b", 1)),
            Operator::IsNotDistinctFrom,
            df_expr::lit(3i32),
        ));
        let expr = DataFusionPredicate::expression(&predicate, &schema)
            .ok_or_else(|| vortex_err!("predicate reads a column"))?;

        assert!(saves_decoding(&expr, &["a", "b"]));
        assert!(!saves_decoding(&expr, &["b"]));
        Ok(())
    }
}
