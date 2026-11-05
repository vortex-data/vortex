// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushes the partial stage of `COUNT(*) ... GROUP BY ...` into a Vortex scan.
//!
//! DataFusion aggregates in two stages: a partial aggregate in each input partition, then a final
//! aggregate that merges the partial states. Grouping by strings hashes every row's strings in
//! the partial stage, and again to repartition and to merge.
//!
//! A dictionary-encoded Vortex column already names each row's distinct value with an integer
//! code, and a small integer column is its own code. [`VortexCountByPushdown`] lets the scan count
//! the rows of each chunk that agree on every scanned column, by those codes alone, and emit one
//! representative row of each with its count. DataFusion evaluates the grouping expressions on
//! the representative rows, which group exactly as the rows they stand for, since the expressions
//! are deterministic functions of the scanned columns. The final aggregate merges the counts like
//! any other partial state, so a chunk whose columns have no codes emits each of its rows with a
//! count of one instead.

use std::fmt::Formatter;
use std::sync::Arc;

use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TransformedResult;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_expr::Volatility;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::CaseExpr;
use datafusion_physical_expr::expressions::CastExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::IsNotNullExpr;
use datafusion_physical_expr::expressions::IsNullExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::NotExpr;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::ExecutionPlanProperties;
use datafusion_physical_plan::InputOrderMode;
use datafusion_physical_plan::PlanProperties;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::aggregates::AggregateMode;
use datafusion_physical_plan::execution_plan::ChildrenPropertiesMode;
use datafusion_physical_plan::execution_plan::ReplaceChildrenOptions;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::dtype::DType;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::buffer::BufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_arrow::ArrowSessionExt;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::VortexSource;

/// A physical optimizer rule that pushes the partial stage of a `COUNT(*)` into the Vortex scan
/// beneath it.
///
/// It rewrites an [`AggregateExec`] in [`AggregateMode::Partial`] directly over a Vortex
/// [`DataSourceExec`] without a limit, when the aggregate computes only `COUNT` of a non-null
/// literal, which is how DataFusion plans `COUNT(*)`, and groups by deterministic expressions of
/// the scanned columns. Register it after DataFusion's own rules, so filters have already moved
/// into the scan:
///
/// ```
/// # use std::sync::Arc;
/// # use datafusion::execution::SessionStateBuilder;
/// # use vortex_datafusion::VortexCountByPushdown;
/// let state = SessionStateBuilder::new()
///     .with_default_features()
///     .with_physical_optimizer_rule(Arc::new(VortexCountByPushdown))
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct VortexCountByPushdown;

impl PhysicalOptimizerRule for VortexCountByPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            Ok(match push_down_count_by(&node) {
                Some(pushed) => Transformed::yes(pushed),
                None => Transformed::no(node),
            })
        })
        .data()
    }

    fn name(&self) -> &str {
        "VortexCountByPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// A partial `COUNT(*)` a Vortex scan computes in place of emitting its rows.
#[derive(Debug)]
pub(crate) struct CountBy {
    /// The partial aggregate's output: the grouping expressions' values, then the count.
    pub(crate) schema: SchemaRef,
    /// The grouping expressions, over the scan's output columns.
    pub(crate) group_by: Vec<Arc<dyn PhysicalExpr>>,
}

impl CountBy {
    /// Evaluates the grouping expressions over `rows`, the scan's output, each standing for
    /// `counts` of the scanned rows.
    pub(crate) fn partial_state(
        &self,
        rows: &RecordBatch,
        counts: Vec<i64>,
    ) -> DFResult<RecordBatch> {
        let mut columns = self
            .group_by
            .iter()
            .map(|expr| expr.evaluate(rows)?.into_array(rows.num_rows()))
            .collect::<DFResult<Vec<_>>>()?;
        columns.push(Arc::new(Int64Array::from(counts)));
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

fn push_down_count_by(node: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    let aggregate = node.downcast_ref::<AggregateExec>()?;
    if *aggregate.mode() != AggregateMode::Partial
        || aggregate.limit_options().is_some()
        // A sorted-mode final aggregate relies on the partial aggregate's output ordering, which
        // the representative rows would not declare.
        || *aggregate.input_order_mode() != InputOrderMode::Linear
        || !aggregate.dynamic_expressions_produced().is_empty()
        || aggregate.filter_expr().iter().any(Option::is_some)
    {
        return None;
    }

    let group_by = aggregate.group_expr();
    if !group_by.is_single() || !group_by.null_expr().is_empty() || group_by.expr().is_empty() {
        return None;
    }
    if !group_by
        .expr()
        .iter()
        .all(|(expr, _)| is_deterministic(expr))
    {
        return None;
    }

    let [count] = aggregate.aggr_expr() else {
        return None;
    };
    let counts_rows = count.fun().name() == "count"
        && !count.is_distinct()
        && count.order_bys().is_empty()
        && matches!(
            count.expressions().as_slice(),
            [arg] if arg.downcast_ref::<Literal>().is_some_and(|lit| !lit.value().is_null())
        );
    if !counts_rows {
        return None;
    }

    // Only dictionary-encoded and integer columns have codes to count rows by; grouping by any
    // other column would hand the final aggregate every row unaggregated.
    if !aggregate
        .input()
        .schema()
        .fields()
        .iter()
        .all(|field| may_have_codes(field.data_type()))
    {
        return None;
    }

    let scan = aggregate.input().downcast_ref::<DataSourceExec>()?;
    let config = scan.data_source().downcast_ref::<FileScanConfig>()?;
    let source = config.file_source.downcast_ref::<VortexSource>()?;
    if config.limit.is_some() {
        return None;
    }

    let count_by = CountBy {
        schema: aggregate.schema(),
        group_by: group_by
            .expr()
            .iter()
            .map(|(expr, _)| Arc::clone(expr))
            .collect(),
    };
    let schema = aggregate.schema();
    let mut config = config.clone();
    config.file_source = Arc::new(source.clone().with_count_by(Arc::new(count_by)));
    let scan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
    Some(Arc::new(VortexCountByExec::new(scan, schema)))
}

/// Whether a scanned column of this type can have codes: a string or binary column that may be
/// dictionary-encoded, or an integer column.
fn may_have_codes(data_type: &DataType) -> bool {
    data_type.is_integer()
        || matches!(
            data_type,
            DataType::Utf8
                | DataType::Utf8View
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::BinaryView
                | DataType::LargeBinary
        )
}

/// Whether `expr` always gives equal rows equal values, so it groups the representative rows the
/// scan emits exactly as it would group the rows they stand for.
fn is_deterministic(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let known = expr.downcast_ref::<Column>().is_some()
        || expr.downcast_ref::<Literal>().is_some()
        || expr.downcast_ref::<BinaryExpr>().is_some()
        || expr.downcast_ref::<CaseExpr>().is_some()
        || expr.downcast_ref::<CastExpr>().is_some()
        || expr.downcast_ref::<NotExpr>().is_some()
        || expr.downcast_ref::<IsNullExpr>().is_some()
        || expr.downcast_ref::<IsNotNullExpr>().is_some()
        || expr
            .downcast_ref::<ScalarFunctionExpr>()
            .is_some_and(|func| func.fun().signature().volatility == Volatility::Immutable);
    known && expr.children().into_iter().all(is_deterministic)
}

/// The partial aggregate a Vortex scan computes for [`VortexCountByPushdown`].
///
/// It runs the scan it wraps, whose source emits the partial aggregate's rows, and presents them
/// with the partial aggregate's schema.
#[derive(Debug)]
pub struct VortexCountByExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl VortexCountByExec {
    fn new(input: Arc<dyn ExecutionPlan>, schema: SchemaRef) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Self {
            input,
            schema,
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for VortexCountByExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        let fields = self.schema.fields();
        let keys = fields[..fields.len() - 1]
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>();
        write!(f, "VortexCountByExec: gby=[{}]", keys.join(", "))
    }
}

impl ExecutionPlan for VortexCountByExec {
    fn name(&self) -> &'static str {
        "VortexCountByExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children.remove(0),
            Arc::clone(&self.schema),
        )))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context)?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}

/// The most bits a chunk's packed row key may take.
const MAX_KEY_BITS: u32 = 128;

/// Counts the rows of a scanned chunk that agree on every column, by their codes, and returns
/// one representative row of each with its count. `scan_field` is the Arrow struct field of the
/// scan's rows and `scan_schema` their schema.
///
/// Returns every row with a count of one when a column has no codes, or their codes do not fit
/// a packed key.
pub(crate) fn count_chunk(
    chunk: ArrayRef,
    count_by: &CountBy,
    scan_field: &Field,
    scan_schema: &SchemaRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RecordBatch> {
    let rows = chunk.len();
    let (chunk, counts) = match row_keys(&chunk, ctx)? {
        Some(keys) => {
            let (representatives, counts) = group_rows(&keys)?;
            let representatives =
                PrimitiveArray::new(representatives.freeze(), Validity::NonNullable).into_array();
            (keys.take(&representatives)?, counts)
        }
        None => (chunk, vec![1; rows]),
    };

    let session = ctx.session().clone();
    let scanned = session
        .arrow()
        .execute_arrow(chunk, Some(scan_field), ctx)?;
    let (_, columns, _) = scanned.as_struct().clone().into_parts();
    // A scan of no columns, as for a constant grouping key, still has its rows.
    let scanned = RecordBatch::try_new_with_options(
        Arc::clone(scan_schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(counts.len())),
    )?;
    count_by
        .partial_state(&scanned, counts)
        .map_err(|e| vortex_err!("Failed to evaluate the grouping expressions: {e}"))
}

/// Row keys of a chunk: each row's codes, one per column, packed into `bits` bits, with the
/// decoded columns they were read from.
struct RowKeys {
    keys: Vec<u128>,
    bits: u32,
    chunk: StructArray,
    columns: Vec<ColumnCodes>,
}

impl RowKeys {
    /// The chunk's rows at `indices`.
    ///
    /// Takes them from the codes decoded for the keys, as taking from the chunk itself would
    /// decode its encoded codes, and any filter over them, a second time.
    fn take(&self, indices: &ArrayRef) -> VortexResult<ArrayRef> {
        let fields = self
            .columns
            .iter()
            .map(|column| column.take(indices))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new_with_dtype(
            fields,
            self.chunk.dtype().as_struct_fields().clone(),
            indices.len(),
            self.chunk.struct_validity().take(indices)?,
        )?
        .into_array())
    }
}

/// Packs each row's codes, one per column, into one key, or returns `None` if a column has no
/// codes or they do not fit.
fn row_keys(chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<RowKeys>> {
    let chunk = chunk.clone().execute::<StructArray>(ctx)?;
    let mut keys = vec![0u128; chunk.len()];
    let mut bits = 0u32;
    let mut columns = Vec::with_capacity(chunk.iter_unmasked_fields().len());
    for column in chunk.iter_unmasked_fields() {
        let Some(codes) = ColumnCodes::try_new(column, ctx)? else {
            return Ok(None);
        };
        bits += codes.width;
        if bits > MAX_KEY_BITS {
            return Ok(None);
        }
        codes.pack_into(&mut keys)?;
        columns.push(codes);
    }
    Ok(Some(RowKeys {
        keys,
        bits,
        chunk,
        columns,
    }))
}

/// A column's code for each row: a dictionary's codes, or an integer column's values less their
/// minimum.
struct ColumnCodes {
    integers: PrimitiveArray,
    /// A dictionary's values, which its codes index.
    values: Option<ArrayRef>,
    /// Subtracted from each integer to give its code.
    min: i128,
    /// How many bits the codes take.
    width: u32,
}

impl ColumnCodes {
    /// Returns `None` for a column that is neither a dictionary nor integers, and for one with
    /// nulls.
    fn try_new(column: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        let (integers, values) = match column.as_opt::<Dict>() {
            Some(dict) if !dict.codes().dtype().is_nullable() => {
                let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
                (codes, Some(dict.values().clone()))
            }
            Some(_) => return Ok(None),
            None if matches!(column.dtype(), DType::Primitive(ptype, _) if ptype.is_int()) => {
                let integers = column.clone().execute::<PrimitiveArray>(ctx)?;
                if !integers.clone().into_array().all_valid(ctx)? {
                    return Ok(None);
                }
                (integers, None)
            }
            None => return Ok(None),
        };

        let (min, span) = match &values {
            // A dictionary's codes index its values.
            Some(values) => (0, i128::try_from(values.len().max(1) - 1)?),
            None => match_each_integer_ptype!(integers.ptype(), |P| {
                let values = integers.as_slice::<P>();
                let min = values.iter().copied().min().map_or(0, i128::from);
                let max = values.iter().copied().max().map_or(0, i128::from);
                (min, max - min)
            }),
        };
        let width = (u128::BITS - u128::try_from(span)?.leading_zeros()).max(1);
        Ok(Some(Self {
            integers,
            values,
            min,
            width,
        }))
    }

    /// The column's rows at `indices`.
    fn take(&self, indices: &ArrayRef) -> VortexResult<ArrayRef> {
        let codes = self.integers.clone().into_array().take(indices.clone())?;
        Ok(match &self.values {
            Some(values) => DictArray::try_new(codes, values.clone())?.into_array(),
            None => codes,
        })
    }

    /// Shifts each key left by this column's width and adds the row's code.
    fn pack_into(&self, keys: &mut [u128]) -> VortexResult<()> {
        let width = self.width;
        let min = self.min;
        match_each_integer_ptype!(self.integers.ptype(), |P| {
            for (key, &value) in keys.iter_mut().zip(self.integers.as_slice::<P>()) {
                let code = u128::try_from(i128::from(value) - min)?;
                *key = (*key << width) | code;
            }
        });
        Ok(())
    }
}

/// The most bits a key may take to group rows in a table indexed by key.
const DENSE_KEY_BITS: u32 = 20;

/// Groups rows with equal keys, returning the first row of each group and its row count.
fn group_rows(row_keys: &RowKeys) -> VortexResult<(BufferMut<u32>, Vec<i64>)> {
    let keys = &row_keys.keys;
    let mut representatives = BufferMut::<u32>::empty();
    let mut counts = Vec::<i64>::new();

    // Narrow keys, such as a single dictionary's codes, index a table of groups directly.
    if row_keys.bits <= DENSE_KEY_BITS && (1usize << row_keys.bits) <= keys.len().max(1024) * 8 {
        let mut groups = vec![u32::MAX; 1 << row_keys.bits];
        for (row, &key) in keys.iter().enumerate() {
            let group = &mut groups[usize::try_from(key)?];
            if *group == u32::MAX {
                *group = u32::try_from(counts.len())?;
                representatives.push(u32::try_from(row)?);
                counts.push(1);
            } else {
                counts[*group as usize] += 1;
            }
        }
        return Ok((representatives, counts));
    }

    let mut groups = HashMap::<u128, usize>::with_capacity(keys.len().min(1 << 16));
    for (row, &key) in keys.iter().enumerate() {
        match groups.entry(key) {
            Entry::Occupied(group) => counts[*group.get()] += 1,
            Entry::Vacant(group) => {
                group.insert(counts.len());
                representatives.push(u32::try_from(row)?);
                counts.push(1);
            }
        }
    }
    Ok((representatives, counts))
}

/// Gives each row of a batch of the scan's output a count of one, for a file whose scan cannot
/// count its rows itself.
pub(crate) fn count_rows(batch: &RecordBatch, count_by: &CountBy) -> DFResult<RecordBatch> {
    count_by.partial_state(batch, vec![1; batch.num_rows()])
}

#[cfg(test)]
mod tests;
