// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed, process-local DataFusion expressions inside Vortex expression trees.

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::BooleanArray;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::DataType;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::HashMap;
use datafusion_common::Result as DFResult;
use datafusion_common::plan_err;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::HigherOrderFunctionExpr;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::is_volatile;
use itertools::Itertools;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::dtype::DType;
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
use vortex::session::registry::CachedId;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

/// Wrap a synchronous, non-volatile DataFusion expression for Vortex execution.
///
/// Only referenced columns are exported to Arrow, except scoped/lambda expressions
/// retain the full original input layout to preserve parameter indices.
/// The result preserves DataFusion's
/// type, effective nullability, field metadata and evaluation errors. Runtime state
/// is retained in-process; the expression cannot be serialized.
/// Returns `None` for live/volatile/async expressions or unsupported result types.
/// Decimal outputs remain in DataFusion because Arrow permits coefficients outside
/// their declared precision; decimal intermediates in other result types are allowed.
///
/// # Errors
///
/// Rejects invalid or ambiguous source-column references.
pub fn df_expression(
    expression: Arc<dyn PhysicalExpr>,
    schema: &Schema,
    session: &VortexSession,
) -> DFResult<Option<Expression>> {
    raw_expression(expression, schema, session, false)
}

pub(crate) fn raw_expression(
    expression: Arc<dyn PhysicalExpr>,
    schema: &Schema,
    session: &VortexSession,
    advisory: bool,
) -> DFResult<Option<Expression>> {
    if !eligible(&expression) {
        return Ok(None);
    }
    let mut name_counts: HashMap<&str, usize> = HashMap::default();
    for field in schema.fields() {
        *name_counts.entry(field.name().as_str()).or_insert(0_usize) += 1;
    }
    let mut columns = collect_columns(&expression).into_iter().collect_vec();
    columns.sort_unstable_by_key(|column| column.index());
    let mut indices = Vec::with_capacity(columns.len());
    for column in &columns {
        if schema
            .fields()
            .get(column.index())
            .is_none_or(|field| field.name() != column.name())
        {
            return plan_err!("Invalid DF expression column: {column}");
        }
        if name_counts.get(column.name()).copied() != Some(1) {
            return plan_err!("Ambiguous DF expression input name: {}", column.name());
        }
        indices.push(column.index());
    }
    let scoped = contains_scoped(&expression);
    if scoped {
        // Lambda variables are numbered after the ORIGINAL input width. Keep
        // that positional boundary unchanged instead of compacting captures.
        if name_counts.values().any(|count| *count != 1) {
            return Ok(None);
        }
        indices = (0..schema.fields().len()).collect();
    }
    let projected = Arc::new(schema.project(&indices)?);
    let expression = if scoped {
        expression
    } else {
        rebind_columns(&expression, schema, &projected)?
    };
    let schema = projected;
    let inputs = schema
        .fields()
        .iter()
        .map(|field| get_item(field.name().as_str(), root()))
        .collect_vec();
    from_inputs(expression, schema, inputs, session, advisory)
}

pub(crate) fn contains_scoped(expression: &Arc<dyn PhysicalExpr>) -> bool {
    expression.is::<HigherOrderFunctionExpr>()
        || expression
            .children()
            .iter()
            .any(|child| contains_scoped(child))
}

// Compacting inputs and extracting native children must not change the Fields
// that DF UDFs observe. Default return_fields derive their names from Display.
pub(crate) fn preserve_field(
    original: &Arc<dyn PhysicalExpr>,
    rewritten: Arc<dyn PhysicalExpr>,
    original_schema: &Schema,
    rewritten_schema: &Schema,
) -> DFResult<Arc<dyn PhysicalExpr>> {
    let field = original.return_field(original_schema)?;
    let nullable = original.nullable(original_schema)?;
    if rewritten.data_type(rewritten_schema)? != *field.data_type() {
        return plan_err!("Rewritten DF expression changed its datatype");
    }
    Ok(Arc::new(BoundField {
        original: Arc::clone(original),
        inner: rewritten,
        field,
        nullable,
    }))
}

fn rebind_columns(
    original: &Arc<dyn PhysicalExpr>,
    original_schema: &Schema,
    schema: &Schema,
) -> DFResult<Arc<dyn PhysicalExpr>> {
    if original.children().is_empty() && !original.is::<Column>() {
        return Ok(Arc::clone(original));
    }
    let rewritten: Arc<dyn PhysicalExpr> = if let Some(column) = original.downcast_ref::<Column>() {
        Arc::new(Column::new(column.name(), schema.index_of(column.name())?))
    } else {
        let children = original
            .children()
            .into_iter()
            .map(|child| rebind_columns(child, original_schema, schema))
            .collect::<DFResult<Vec<_>>>()?;
        Arc::clone(original).with_new_children(children)?
    };
    preserve_field(original, rewritten, original_schema, schema)
}

#[derive(Debug)]
struct BoundField {
    inner: Arc<dyn PhysicalExpr>,
    field: FieldRef,
    nullable: bool,
    original: Arc<dyn PhysicalExpr>,
}
impl PartialEq for BoundField {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
            && self.field == other.field
            && self.nullable == other.nullable
            && Arc::ptr_eq(&self.original, &other.original)
    }
}
impl Eq for BoundField {}
impl Hash for BoundField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.inner) as *const ()).hash(state);
        self.field.hash(state);
        self.nullable.hash(state);
        (Arc::as_ptr(&self.original) as *const ()).hash(state);
    }
}
impl fmt::Display for BoundField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.original, f)
    }
}
impl PhysicalExpr for BoundField {
    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.original.fmt_sql(f)
    }
    fn return_field(&self, _schema: &Schema) -> DFResult<FieldRef> {
        Ok(Arc::clone(&self.field))
    }
    fn nullable(&self, _schema: &Schema) -> DFResult<bool> {
        Ok(self.nullable)
    }
    fn evaluate(&self, batch: &RecordBatch) -> DFResult<datafusion_expr::ColumnarValue> {
        self.inner.evaluate(batch)
    }
    fn evaluate_selection(
        &self,
        batch: &RecordBatch,
        selection: &BooleanArray,
    ) -> DFResult<datafusion_expr::ColumnarValue> {
        self.inner.evaluate_selection(batch, selection)
    }
    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.inner]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        let [inner]: [Arc<dyn PhysicalExpr>; 1] = children.try_into().map_err(|_| {
            datafusion_common::plan_datafusion_err!("Bound DF field requires one child")
        })?;
        // DF can rebind children into CASE-local schemas. Its rewrites preserve
        // types; the original binding schema is no longer valid here.
        Ok(Arc::new(Self {
            inner,
            field: Arc::clone(&self.field),
            nullable: self.nullable,
            original: Arc::clone(&self.original),
        }))
    }
}

pub(crate) fn eligible(expression: &Arc<dyn PhysicalExpr>) -> bool {
    !is_volatile(expression)
        && !contains_async(expression)
        && !DynamicFilterTracking::classify(expression).contains_dynamic_filter()
}

pub(crate) fn from_inputs(
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    inputs: Vec<Expression>,
    session: &VortexSession,
    advisory: bool,
) -> DFResult<Option<Expression>> {
    if !eligible(&expression) {
        return Ok(None);
    }
    let field = expression.return_field(&schema)?;
    let output = field
        .as_ref()
        .clone()
        .with_nullable(field.is_nullable() || expression.nullable(&schema)?);
    if contains_decimal(output.data_type()) || advisory && output.data_type() != &DataType::Boolean
    {
        return Ok(None);
    }
    let output = Arc::new(output);
    let Ok(dtype) = session.arrow().from_arrow_field(&output) else {
        return Ok(None);
    };
    if advisory && !matches!(dtype, DType::Bool(_)) {
        return Ok(None);
    }
    let Ok(input_dtypes) = schema
        .fields()
        .iter()
        .map(|field| session.arrow().from_arrow_field(field))
        .collect::<VortexResult<Vec<_>>>()
    else {
        return Ok(None);
    };
    Expression::try_new(
        DataFusionExpression.bind(Options {
            expression,
            schema,
            input_dtypes,
            output,
            dtype,
            advisory,
        }),
        inputs,
    )
    .map(Some)
    .map_err(|error| datafusion_common::DataFusionError::External(Box::new(error)))
}

fn contains_decimal(dtype: &DataType) -> bool {
    match dtype {
        DataType::Decimal32(..)
        | DataType::Decimal64(..)
        | DataType::Decimal128(..)
        | DataType::Decimal256(..) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => contains_decimal(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_decimal(field.data_type())),
        DataType::Dictionary(_, value) => contains_decimal(value),
        DataType::RunEndEncoded(_, value) => contains_decimal(value.data_type()),
        _ => false,
    }
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
    output: FieldRef,
    dtype: DType,
    advisory: bool,
}
impl PartialEq for Options {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.expression, &other.expression)
            && self.schema == other.schema
            && self.input_dtypes == other.input_dtypes
            && self.output == other.output
            && self.dtype == other.dtype
            && self.advisory == other.advisory
    }
}
impl Eq for Options {}
impl Hash for Options {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.expression) as *const ()).hash(state);
        self.schema.hash(state);
        self.input_dtypes.hash(state);
        self.output.hash(state);
        self.dtype.hash(state);
        self.advisory.hash(state);
    }
}
impl fmt::Display for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expression.fmt(f)
    }
}

#[derive(Clone)]
struct DataFusionExpression;
impl ScalarFnVTable for DataFusionExpression {
    type Options = Options;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("datafusion.df_expression");
        *ID
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
                "DF expression input dtype mismatch"
            );
        }
        Ok(options.dtype.clone())
    }
    fn execute(
        &self,
        options: &Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let inputs = (0..options.schema.fields().len())
            .map(|index| args.get(index))
            .collect::<VortexResult<Vec<_>>>()?;
        let result = (|| {
            let columns = inputs
                .into_iter()
                .zip(options.schema.fields())
                .map(|(array, field)| array.execute_arrow(Some(field), ctx))
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
            vortex_ensure_eq!(result.data_type(), options.output.data_type());
            Ok::<_, VortexError>(result)
        })();
        let result = match result {
            Ok(result) => result,
            Err(_) if options.advisory => {
                Arc::new(BooleanArray::from(vec![true; args.row_count()]))
            }
            Err(error) => return Err(error),
        };
        ctx.session()
            .arrow()
            .from_arrow_array(result, &options.output)
    }
    fn is_infallible(&self, options: &Options) -> bool {
        options.advisory
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::BooleanArray;
    use arrow_array::Int64Array;
    use arrow_array::StringArray;
    use arrow_schema::Field;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;
    use datafusion::prelude::SessionContext;
    use datafusion_common::JoinType;
    use datafusion_common::NullEquality;
    use datafusion_common::ScalarValue;
    use datafusion_common::hash_utils::create_hashes;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion_physical_expr::expressions::lit;
    use datafusion_physical_expr::split_conjunction;
    use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::joins::HashJoinExec;
    use datafusion_physical_plan::joins::HashTableLookupExpr;
    use datafusion_physical_plan::joins::Map;
    use datafusion_physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::joins::SeededRandomState;
    use datafusion_physical_plan::joins::join_hash_map::JoinHashMapU32;
    use datafusion_physical_plan::joins::utils::JoinHashMapType;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::VortexSessionExecute;
    use vortex::dtype::Nullability;
    use vortex::dtype::extension::ExtDType;
    use vortex::dtype::extension::ExtId;
    use vortex::dtype::extension::ExtVTable;
    use vortex::dtype::session::DTypeSessionExt;
    use vortex::scalar::ScalarValue as VortexScalarValue;

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
        let expression = df_expression(Arc::clone(&lookup), &batch.schema(), &session)?.unwrap();
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
        let key = Arc::new(BinaryExpr::new(
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
        let expression = df_expression(lookup, &batch.schema(), &session)?.unwrap();
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
        assert!(df_expression(lookup, &schema, &VortexSession::default()).is_err());
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
        assert!(df_expression(crossed, &schema, &VortexSession::default()).is_err());
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
        let expression = df_expression(lookup, &probe.schema(), &session)?.unwrap();
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
    #[test]
    fn typed_results_preserve_nulls_wrapping_and_scalar_row_count() -> anyhow::Result<()> {
        let batch = RecordBatch::try_from_iter(vec![(
            "id",
            Arc::new(Int64Array::from(vec![Some(i64::MAX), None, Some(0)])) as ArrowArrayRef,
        )])?;
        let expressions: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("id", 0)),
                datafusion_expr::Operator::Plus,
                lit(1_i64),
            )),
            lit(7_i64),
            Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
                Arc::new(Column::new("id", 0)),
            )),
        ];
        let session = VortexSession::default();
        for expression in expressions {
            let output = expression.return_field(&batch.schema())?;
            let expected = expression
                .evaluate(&batch)?
                .into_array_of_size(batch.num_rows())?;
            let wrapped = df_expression(expression, &batch.schema(), &session)?.unwrap();
            let input = session
                .arrow()
                .from_arrow_record_batch(batch.clone(), &batch.schema())?;
            let actual = input
                .apply(&wrapped)?
                .execute_arrow(Some(output.as_ref()), &mut session.create_execution_ctx())?;
            assert_eq!(&actual, &expected);
        }
        Ok(())
    }

    #[test]
    fn unsupported_results_and_live_or_volatile_expressions_decline() -> anyhow::Result<()> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let session = VortexSession::default();
        let decimal: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion_physical_expr::expressions::Literal::new(
                ScalarValue::Decimal128(Some(123), 10, 2),
            ));
        assert!(df_expression(Arc::clone(&decimal), &schema, &session)?.is_none());
        let comparison: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&decimal),
            datafusion_expr::Operator::Eq,
            decimal,
        ));
        assert!(df_expression(comparison, &schema, &session)?.is_some());
        let live: Arc<dyn PhysicalExpr> = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("id", 0))],
            lit(true),
        ));
        assert!(df_expression(live, &schema, &session)?.is_none());
        let random: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion_physical_expr::ScalarFunctionExpr::try_new(
                Arc::new(datafusion_expr::create_udf(
                    "volatile",
                    vec![],
                    DataType::Float64,
                    datafusion_expr::Volatility::Volatile,
                    Arc::new(|_| {
                        Ok(datafusion_expr::ColumnarValue::Scalar(
                            ScalarValue::Float64(Some(0.5)),
                        ))
                    }),
                )),
                vec![],
                &schema,
                Arc::new(datafusion_common::config::ConfigOptions::new()),
            )?);
        assert!(df_expression(random, &schema, &session)?.is_none());
        Ok(())
    }
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TaggedBool;
    impl ExtVTable for TaggedBool {
        type Metadata = u8;
        type NativeValue<'a> = &'a VortexScalarValue;

        fn id(&self) -> ExtId {
            static ID: CachedId = CachedId::new("test.df_expression_bool");
            *ID
        }
        fn serialize_metadata(&self, metadata: &u8) -> VortexResult<Vec<u8>> {
            Ok(vec![*metadata])
        }
        fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<u8> {
            vortex_ensure_eq!(bytes.len(), 1);
            Ok(bytes[0])
        }
        fn validate_dtype(dtype: &ExtDType<Self>) -> VortexResult<()> {
            vortex_ensure!(matches!(dtype.storage_dtype(), DType::Bool(_)));
            Ok(())
        }
        fn unpack_native<'a>(
            _dtype: &'a ExtDType<Self>,
            value: &'a VortexScalarValue,
        ) -> VortexResult<&'a VortexScalarValue> {
            Ok(value)
        }
    }

    #[test]
    fn boolean_result_preserves_original_field_metadata() -> anyhow::Result<()> {
        let session = VortexSession::default();
        session.dtypes().register(TaggedBool);
        let dtype = DType::Extension(
            ExtDType::try_with_vtable(TaggedBool, 7, DType::Bool(Nullability::Nullable))?.erased(),
        );
        let field = session.arrow().to_arrow_field("flag", &dtype)?;
        assert_eq!(field.data_type(), &DataType::Boolean);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![field.clone()])),
            vec![Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(false),
            ]))],
        )?;
        let expression: Arc<dyn PhysicalExpr> = Arc::new(Column::new("flag", 0));
        let wrapped = df_expression(Arc::clone(&expression), &batch.schema(), &session)?.unwrap();
        let input = session
            .arrow()
            .from_arrow_record_batch(batch.clone(), &batch.schema())?;
        let result = input.apply(&wrapped)?;
        assert_eq!(result.dtype(), &dtype);
        let actual = result.execute_arrow(Some(&field), &mut session.create_execution_ctx())?;
        let expected = expression
            .evaluate(&batch)?
            .into_array_of_size(batch.num_rows())?;
        assert_eq!(&actual, &expected);
        Ok(())
    }
}
