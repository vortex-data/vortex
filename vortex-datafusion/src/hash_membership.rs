// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Process-local evaluation of DataFusion's runtime join membership expression.

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::plan_err;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_common::physical_expr::is_volatile;
use datafusion_physical_plan::joins::HashTableLookupExpr;
use itertools::Itertools;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::root;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

/// Wrap a DataFusion [`HashTableLookupExpr`] as a Vortex expression.
///
/// Retains the expression's shared runtime map and hashing state. Only referenced
/// columns are exported to Arrow; DataFusion evaluates the keys and probes its map.
/// The result is a non-nullable Boolean, matching DataFusion even for nullable keys.
/// Hash-backed probes may admit collisions: the join still checks actual key equality.
/// This expression owns process-local state and cannot be serialized.
///
/// # Errors
///
/// Rejects other expression types, volatile/async/live keys, invalid or ambiguous column
/// references, and input fields that the session cannot represent in Vortex.
/// Evaluation preserves DataFusion errors.
pub fn hash_membership(
    expression: Arc<dyn PhysicalExpr>,
    schema: &Schema,
    session: &VortexSession,
) -> DFResult<Expression> {
    if !expression.is::<HashTableLookupExpr>()
        || is_volatile(&expression)
        || contains_async(&expression)
        || DynamicFilterTracking::classify(&expression).contains_dynamic_filter()
    {
        return plan_err!(
            "Expected a static synchronous non-volatile DataFusion hash membership expression"
        );
    }
    let name_counts = schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .counts();
    let mut columns = collect_columns(&expression).into_iter().collect_vec();
    columns.sort_unstable_by_key(|column| column.index());
    let mut indices = Vec::with_capacity(columns.len());
    for column in &columns {
        if schema
            .fields()
            .get(column.index())
            .is_none_or(|field| field.name() != column.name())
        {
            return plan_err!("Invalid hash membership column: {column}");
        }
        if name_counts.get(column.name()).copied() != Some(1) {
            return plan_err!("Ambiguous hash membership input name: {}", column.name());
        }
        indices.push(column.index());
    }
    let schema = Arc::new(schema.project(&indices)?);
    let input_dtypes = schema
        .fields()
        .iter()
        .map(|field| {
            session
                .arrow()
                .from_arrow_field(field)
                .map_err(|error| datafusion_common::DataFusionError::External(Box::new(error)))
        })
        .collect::<DFResult<Vec<_>>>()?;
    let expression = reassign_expr_columns(expression, &schema)?;
    let inputs = schema
        .fields()
        .iter()
        .map(|field| get_item(field.name().as_str(), root()))
        .collect_vec();
    Ok(HashMembership.new_expr(
        Options {
            expression,
            schema,
            input_dtypes,
        },
        inputs,
    ))
}

fn contains_async(expression: &Arc<dyn PhysicalExpr>) -> bool {
    expression.is::<datafusion_physical_expr::async_scalar_function::AsyncFuncExpr>()
        || expression
            .children()
            .iter()
            .any(|child| contains_async(child))
}

#[derive(Clone, Debug)]
struct Options {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    input_dtypes: Vec<DType>,
}
impl PartialEq for Options {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.expression, &other.expression)
            && self.schema == other.schema
            && self.input_dtypes == other.input_dtypes
    }
}
impl Eq for Options {}
impl Hash for Options {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.expression) as *const ()).hash(state);
        self.schema.hash(state);
        self.input_dtypes.hash(state);
    }
}
impl fmt::Display for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expression.fmt(f)
    }
}

#[derive(Clone)]
struct HashMembership;
impl ScalarFnVTable for HashMembership {
    type Options = Options;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_static("datafusion.hash_membership")
    }
    fn arity(&self, options: &Options) -> Arity {
        Arity::Exact(options.schema.fields().len())
    }
    fn child_name(&self, options: &Options, index: usize) -> ChildName {
        ChildName::new_arc(options.schema.field(index).name().clone().into())
    }
    fn return_dtype(&self, options: &Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure_eq!(args.len(), options.input_dtypes.len());
        for (actual, expected) in args.iter().zip(&options.input_dtypes) {
            vortex_ensure!(
                actual.with_nullability(expected.nullability()) == *expected
                    && (!actual.is_nullable() || expected.is_nullable()),
                "Hash membership input dtype mismatch"
            );
        }
        Ok(DType::Bool(Nullability::NonNullable))
    }
    fn execute(
        &self,
        options: &Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let columns = options
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| args.get(index)?.execute_arrow(Some(field), ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&options.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(args.row_count())),
        )?;
        let result = options
            .expression
            .evaluate(&batch)
            .and_then(|value| value.into_array_of_size(args.row_count()))
            .map_err(VortexError::external)?;
        vortex_ensure_eq!(result.len(), args.row_count());
        vortex_ensure_eq!(result.data_type(), &DataType::Boolean);
        vortex_ensure_eq!(result.null_count(), 0);
        ctx.session()
            .arrow()
            .from_arrow_array(result, &Field::new("membership", DataType::Boolean, false))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::BooleanArray;
    use arrow_array::Int64Array;
    use arrow_array::StringArray;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;
    use datafusion::prelude::SessionContext;
    use datafusion_common::JoinType;
    use datafusion_common::NullEquality;
    use datafusion_common::hash_utils::create_hashes;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion_physical_expr::expressions::lit;
    use datafusion_physical_expr::split_conjunction;
    use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::joins::HashJoinExec;
    use datafusion_physical_plan::joins::Map;
    use datafusion_physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::joins::SeededRandomState;
    use datafusion_physical_plan::joins::join_hash_map::JoinHashMapU32;
    use datafusion_physical_plan::joins::utils::JoinHashMapType;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::VortexSessionExecute;

    use super::*;

    #[rstest]
    #[case::single(false)]
    #[case::composite(true)]
    fn matches_datafusion_with_nullable_keys(#[case] composite: bool) -> anyhow::Result<()> {
        let ids: ArrowArrayRef = Arc::new(Int64Array::from(vec![Some(2), None, Some(5)]));
        let labels: ArrowArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("b"), None]));
        let state = SeededRandomState::with_seed(17);
        let build = if composite {
            vec![ids, labels]
        } else {
            vec![ids]
        };
        let mut hashes = vec![0; 3];
        create_hashes(&build, state.random_state(), &mut hashes)?;
        let mut map = JoinHashMapU32::with_capacity(hashes.len());
        map.update_from_iter(Box::new(hashes.iter().enumerate()), 0);
        let mut keys: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("id", 1))];
        if composite {
            keys.push(Arc::new(Column::new("label", 2)));
        }
        let lookup: Arc<dyn PhysicalExpr> = Arc::new(HashTableLookupExpr::new(
            keys,
            state,
            Arc::new(Map::HashMap(Box::new(map))),
            "test_membership".into(),
        ));
        let batch = RecordBatch::try_from_iter(vec![
            (
                "unused",
                Arc::new(StringArray::from(vec!["payload"; 5])) as ArrowArrayRef,
            ),
            (
                "id",
                Arc::new(Int64Array::from(vec![
                    Some(2),
                    Some(3),
                    None,
                    Some(5),
                    Some(2),
                ])) as ArrowArrayRef,
            ),
            (
                "label",
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("a"),
                    Some("b"),
                    None,
                    Some("wrong"),
                ])) as ArrowArrayRef,
            ),
        ])?;
        let session = VortexSession::default();
        let expression = hash_membership(Arc::clone(&lookup), &batch.schema(), &session)?;
        assert_eq!(expression.children().len(), if composite { 2 } else { 1 });
        assert!(
            expression
                .as_scalar()
                .unwrap()
                .options()
                .serialize()?
                .is_none()
        );
        let input = session
            .arrow()
            .from_arrow_record_batch(batch.clone(), &batch.schema())?;
        let result = input.apply(&expression)?;
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));
        let actual = result.execute_arrow(
            Some(&Field::new("result", DataType::Boolean, false)),
            &mut session.create_execution_ctx(),
        )?;
        let expected = lookup
            .evaluate(&batch)?
            .into_array_of_size(batch.num_rows())?;
        assert_eq!(&actual, &expected);
        Ok(())
    }
    #[test]
    fn preserves_key_evaluation_errors() -> anyhow::Result<()> {
        let key = Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
            lit(1_i64),
            datafusion_expr::Operator::Divide,
            Arc::new(Column::new("id", 0)),
        ));
        let lookup: Arc<dyn PhysicalExpr> = Arc::new(HashTableLookupExpr::new(
            vec![key],
            SeededRandomState::with_seed(0),
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(0)))),
            "fallible_membership".into(),
        ));
        let batch = RecordBatch::try_from_iter(vec![(
            "id",
            Arc::new(Int64Array::from(vec![0])) as ArrowArrayRef,
        )])?;
        let expected = lookup.evaluate(&batch).unwrap_err();
        let session = VortexSession::default();
        let expression = hash_membership(lookup, &batch.schema(), &session)?;
        assert!(!expression.as_scalar().unwrap().signature().is_infallible());
        let input = session
            .arrow()
            .from_arrow_record_batch(batch.clone(), &batch.schema())?;
        let error = input
            .apply(&expression)?
            .execute_arrow(
                Some(&Field::new("result", DataType::Boolean, false)),
                &mut session.create_execution_ctx(),
            )
            .unwrap_err();
        assert!(error.to_string().contains(&expected.to_string()));
        Ok(())
    }
    #[test]
    fn rejects_ambiguous_original_column_names() -> anyhow::Result<()> {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("id", DataType::Int64, false),
        ]);
        let lookup: Arc<dyn PhysicalExpr> = Arc::new(HashTableLookupExpr::new(
            vec![Arc::new(Column::new("id", 1))],
            SeededRandomState::with_seed(0),
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(0)))),
            "second_id".into(),
        ));
        assert!(hash_membership(lookup, &schema, &VortexSession::default()).is_err());
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let crossed: Arc<dyn PhysicalExpr> = Arc::new(HashTableLookupExpr::new(
            vec![Arc::new(Column::new("b", 0)), Arc::new(Column::new("a", 1))],
            SeededRandomState::with_seed(0),
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(0)))),
            "crossed_columns".into(),
        ));
        assert!(hash_membership(crossed, &schema, &VortexSession::default()).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn matches_datafusion_array_map() -> anyhow::Result<()> {
        let build = RecordBatch::try_from_iter(vec![(
            "id",
            Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3), None])) as ArrowArrayRef,
        )])?;
        let probe = RecordBatch::try_from_iter(vec![(
            "id",
            Arc::new(Int64Array::from(vec![
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                None,
            ])) as ArrowArrayRef,
        )])?;
        let key = Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&key)],
            lit(true),
        ));
        let left = MemorySourceConfig::try_new_exec(&[vec![build.clone()]], build.schema(), None)?;
        let right = MemorySourceConfig::try_new_exec(&[vec![probe.clone()]], probe.schema(), None)?;
        let right = Arc::new(FilterExec::try_new(
            Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>,
            right,
        )?);
        let join = Arc::new(
            HashJoinExec::try_new(
                left,
                right,
                vec![(Arc::clone(&key), key)],
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?
            .with_dynamic_filter_expr(Arc::clone(&dynamic))?,
        );
        let mut config = SessionConfig::new().with_target_partitions(1);
        config
            .options_mut()
            .optimizer
            .hash_join_inlist_pushdown_max_distinct_values = 0;
        let context = SessionContext::new_with_state(
            SessionStateBuilder::new()
                .with_default_features()
                .with_config(config)
                .build(),
        );
        datafusion_physical_plan::collect(
            Arc::clone(&join) as Arc<dyn ExecutionPlan>,
            context.task_ctx(),
        )
        .await?;
        assert_eq!(
            join.metrics()
                .unwrap()
                .sum_by_name("array_map_created_count")
                .unwrap()
                .as_usize(),
            1
        );
        let snapshot = snapshot_physical_expr(dynamic as Arc<dyn PhysicalExpr>)?;
        let lookup = Arc::clone(
            split_conjunction(&snapshot)
                .into_iter()
                .find(|expression| expression.is::<HashTableLookupExpr>())
                .unwrap(),
        );
        let expected = lookup
            .evaluate(&probe)?
            .into_array_of_size(probe.num_rows())?;
        let session = VortexSession::default();
        let expression = hash_membership(lookup, &probe.schema(), &session)?;
        let input = session
            .arrow()
            .from_arrow_record_batch(probe.clone(), &probe.schema())?;
        let actual = input.apply(&expression)?.execute_arrow(
            Some(&Field::new("result", DataType::Boolean, false)),
            &mut session.create_execution_ctx(),
        )?;
        assert_eq!(&actual, &expected);
        assert_eq!(
            &actual,
            &(Arc::new(BooleanArray::from(vec![
                false, true, true, true, false, false
            ])) as ArrowArrayRef)
        );
        Ok(())
    }
}
