// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `k` largest counts of a `COUNT(*) ... GROUP BY ... ORDER BY COUNT(*) DESC LIMIT k`, from
//! the partial counts of a [`VortexCountBySource`], without merging every group.
//!
//! The final aggregate DataFusion would run merges every group's partial counts by its grouping
//! values, so it builds a table of every distinct group, while the query keeps only `k` of them.
//! [`VortexTopCountExec`] prunes the groups that cannot be among the `k` largest first, with a
//! threshold in the spirit of TPUT (Cao and Wang, "Efficient top-K query calculation in
//! distributed networks", PODC 2004):
//!
//! 1. It hashes each partial row's grouping values and sums the counts of each hash. Every row
//!    of a group has the group's hash, so a hash's sum bounds the total of each group with that
//!    hash from above.
//! 2. It merges exactly the rows of the hashes with the largest sums, which holds every row of
//!    their groups, so their totals are exact. The `k`-th largest of those totals bounds the
//!    `k`-th largest total of all groups from below.
//! 3. A group whose hash sums to less than that bound cannot be among the `k` largest. It merges
//!    exactly the rows of every other hash, and returns the `k` largest totals.
//!
//! TPUT bounds a group's total by how many nodes report it. A partial row here does not stand
//! for a whole group per chunk, as grouping expressions can map rows with different codes to one
//! group, so the bound comes from the hashes instead.
//!
//! [`VortexCountBySource`]: super::VortexCountBySource

use std::fmt::Formatter;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::UInt32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::SchemaRef;
use arrow_schema::SortOptions;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::arrow::compute::take;
use datafusion_common::arrow::compute::take_record_batch;
use datafusion_common::arrow::row::RowConverter;
use datafusion_common::arrow::row::SortField;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::internal_err;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::ExecutionPlanProperties;
use datafusion_physical_plan::PlanProperties;
use datafusion_physical_plan::execution_plan::ChildrenPropertiesMode;
use datafusion_physical_plan::execution_plan::ReplaceChildrenOptions;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use futures::stream;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

/// The `fetch` largest counts of the groups of a [`VortexCountBySource`]'s partial counts, in
/// descending order of count, as one partition. See the [module documentation](self).
///
/// [`VortexCountBySource`]: super::VortexCountBySource
#[derive(Debug)]
pub struct VortexTopCountExec {
    /// The partial counts: the grouping values, then the count.
    input: Arc<dyn ExecutionPlan>,
    /// The final aggregate's output: the grouping values, then the count.
    schema: SchemaRef,
    fetch: usize,
    properties: Arc<PlanProperties>,
}

impl VortexTopCountExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, schema: SchemaRef, fetch: usize) -> Self {
        let count = schema.fields().len() - 1;
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new(schema.field(count).name(), count)),
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        )]);
        let properties = PlanProperties::new(
            EquivalenceProperties::new_with_orderings(Arc::clone(&schema), ordering),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Self {
            input,
            schema,
            fetch,
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for VortexTopCountExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "VortexTopCountExec: fetch={}", self.fetch)
    }
}

impl ExecutionPlan for VortexTopCountExec {
    fn name(&self) -> &'static str {
        "VortexTopCountExec"
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
            self.fetch,
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
        if partition != 0 {
            return internal_err!("VortexTopCountExec has one partition, not {partition}");
        }
        let partials = (0..self.input.output_partitioning().partition_count())
            .map(|partition| {
                let stream = self.input.execute(partition, Arc::clone(&context))?;
                Ok(SpawnedTask::spawn(hash_partials(stream)))
            })
            .collect::<DFResult<Vec<_>>>()?;
        let schema = Arc::clone(&self.schema);
        let fetch = self.fetch;
        let top = async move {
            let mut hashed = Vec::with_capacity(partials.len());
            for partial in partials {
                let partial = partial
                    .join_unwind()
                    .await
                    .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))??;
                hashed.push(partial);
            }
            top_counts(hashed, &schema, fetch)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream::once(top),
        )))
    }
}

/// A partition's partial counts, the hash of each row's grouping values, and the sum of the
/// counts of each hash.
struct HashedPartials {
    batches: Vec<RecordBatch>,
    hashes: Vec<Vec<u64>>,
    sums: HashMap<u64, i64>,
}

/// The hash state of the grouping values. Any fixed state works: the hashes only need to agree
/// within one execution.
const HASH_STATE: RandomState = RandomState::with_seed(0);

async fn hash_partials(mut stream: SendableRecordBatchStream) -> DFResult<HashedPartials> {
    let mut hashed = HashedPartials {
        batches: Vec::new(),
        hashes: Vec::new(),
        sums: HashMap::default(),
    };
    while let Some(batch) = stream.next().await.transpose()? {
        if batch.num_rows() == 0 {
            continue;
        }
        let (keys, counts) = split_counts(&batch)?;
        let mut hashes = vec![0; batch.num_rows()];
        create_hashes(keys, &HASH_STATE, &mut hashes)?;
        for (&hash, &count) in hashes.iter().zip(counts.values()) {
            *hashed.sums.entry(hash).or_default() += count;
        }
        hashed.batches.push(batch);
        hashed.hashes.push(hashes);
    }
    Ok(hashed)
}

/// The grouping columns and the count column of a batch of partial counts.
fn split_counts(batch: &RecordBatch) -> DFResult<(&[ArrowArrayRef], &Int64Array)> {
    let (counts, keys) = batch.columns().split_last().ok_or_else(|| {
        datafusion_common::internal_datafusion_err!("partial counts without a count")
    })?;
    Ok((keys, counts.as_primitive::<Int64Type>()))
}

/// The `fetch` largest totals of the groups of `partials`, in descending order.
fn top_counts(
    partials: Vec<HashedPartials>,
    schema: &SchemaRef,
    fetch: usize,
) -> DFResult<RecordBatch> {
    let mut sums = HashMap::<u64, i64>::default();
    for partial in &partials {
        for (&hash, &sum) in &partial.sums {
            *sums.entry(hash).or_default() += sum;
        }
    }
    let batches = partials
        .iter()
        .flat_map(|partial| partial.batches.iter().zip(&partial.hashes))
        .collect::<Vec<_>>();

    // The exact totals of the groups of the hashes with the largest sums bound the `fetch`-th
    // largest total from below. Widen the hashes until they hold `fetch` groups.
    let mut by_sum = sums
        .iter()
        .map(|(&hash, &sum)| (sum, hash))
        .collect::<Vec<_>>();
    let mut widest = fetch.max(1).saturating_mul(2);
    let threshold = loop {
        let largest = largest_sums(&mut by_sum, widest);
        let groups = merge_exactly(&batches, |hash| largest.contains(hash))?;
        if groups.totals.len() >= fetch {
            let mut totals = groups.totals.clone();
            let kth = totals.len() - fetch;
            totals.select_nth_unstable(kth);
            break totals[kth];
        }
        if widest >= by_sum.len() {
            break i64::MIN;
        }
        widest = widest.saturating_mul(4);
    };

    // A group whose hash sums to less than the threshold totals less than it.
    let groups = merge_exactly(&batches, |hash| sums[hash] >= threshold)?;
    let mut order = (0..groups.totals.len()).collect::<Vec<_>>();
    order.sort_by_key(|&group| std::cmp::Reverse(groups.totals[group]));
    order.truncate(fetch);

    let representatives = UInt32Array::from(
        order
            .iter()
            .map(|&group| groups.representatives[group])
            .collect::<Vec<_>>(),
    );
    let keys = &groups.rows.columns()[..groups.rows.num_columns() - 1];
    let mut columns = keys
        .iter()
        .map(|column| take(column, &representatives, None))
        .collect::<Result<Vec<_>, _>>()?;
    columns.push(Arc::new(Int64Array::from(
        order
            .iter()
            .map(|&group| groups.totals[group])
            .collect::<Vec<_>>(),
    )));
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// The hashes of the `count` largest sums, reordering `by_sum`.
fn largest_sums(by_sum: &mut [(i64, u64)], count: usize) -> HashSet<u64> {
    if count < by_sum.len() {
        by_sum.select_nth_unstable_by(count, |a, b| b.cmp(a));
    }
    by_sum.iter().take(count).map(|&(_, hash)| hash).collect()
}

/// The partial counts of some hashes, merged by their grouping values.
struct ExactGroups {
    /// The selected partial counts.
    rows: RecordBatch,
    /// A row of `rows` in each group.
    representatives: Vec<u32>,
    /// Each group's total count.
    totals: Vec<i64>,
}

/// Merges, by their grouping values, the partial counts whose hash `keep` selects. Every row of
/// a group has the same hash, so each group it returns holds all of that group's counts.
fn merge_exactly(
    batches: &[(&RecordBatch, &Vec<u64>)],
    keep: impl Fn(&u64) -> bool,
) -> DFResult<ExactGroups> {
    let Some((first, _)) = batches.first() else {
        return internal_err!("no partial counts to merge");
    };
    let schema = first.schema();
    let selected = batches
        .iter()
        .map(|(batch, hashes)| {
            let rows = UInt32Array::from(
                hashes
                    .iter()
                    .enumerate()
                    .filter(|(_, hash)| keep(hash))
                    .map(|(row, _)| u32::try_from(row))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| datafusion_common::internal_datafusion_err!("{e}"))?,
            );
            Ok(take_record_batch(batch, &rows)?)
        })
        .collect::<DFResult<Vec<_>>>()?;
    let rows = concat_batches(&schema, &selected)?;

    let (keys, counts) = split_counts(&rows)?;
    let converter = RowConverter::new(
        keys.iter()
            .map(|column| SortField::new(column.data_type().clone()))
            .collect(),
    )?;
    let converted = converter.convert_columns(keys)?;
    let mut groups = HashMap::<_, usize>::default();
    let mut representatives = Vec::new();
    let mut totals = Vec::<i64>::new();
    for (row, &count) in counts.values().iter().enumerate() {
        let group = *groups.entry(converted.row(row)).or_insert_with(|| {
            representatives.push(u32::try_from(row).unwrap_or(u32::MAX));
            totals.push(0);
            totals.len() - 1
        });
        totals[group] += count;
    }
    Ok(ExactGroups {
        rows,
        representatives,
        totals,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::SchemaRef;
    use datafusion_common::Result as DFResult;
    use datafusion_common::hash_utils::create_hashes;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::HASH_STATE;
    use super::HashedPartials;
    use super::top_counts;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("url", DataType::Utf8, true),
            Field::new("count", DataType::Int64, false),
        ]))
    }

    fn partials(rows: &[(Option<&str>, i64)]) -> DFResult<HashedPartials> {
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|(url, _)| *url).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    rows.iter().map(|(_, count)| *count).collect::<Vec<_>>(),
                )),
            ],
        )?;
        let mut hashes = vec![0; batch.num_rows()];
        create_hashes(&batch.columns()[..1], &HASH_STATE, &mut hashes)?;
        let mut sums = HashMap::default();
        for (&hash, (_, count)) in hashes.iter().zip(rows) {
            *sums.entry(hash).or_default() += count;
        }
        Ok(HashedPartials {
            batches: vec![batch],
            hashes: vec![hashes],
            sums,
        })
    }

    /// Totals spread over many small partial counts beat a single large one, and nulls group.
    #[test]
    fn top_counts_merges_partials_across_partitions() -> DFResult<()> {
        let first = partials(&[
            (Some("a"), 5),
            (Some("b"), 1),
            (Some("c"), 1),
            (None, 2),
            (Some("d"), 1),
        ])?;
        let second = partials(&[
            (Some("b"), 1),
            (Some("c"), 1),
            (None, 2),
            (Some("e"), 4),
            (Some("b"), 4),
        ])?;
        let third = partials(&[(Some("b"), 1), (Some("c"), 1), (Some("c"), 1)])?;

        let top = top_counts(vec![first, second, third], &schema(), 3)?;
        let urls = top
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let counts = top.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let top = urls
            .iter()
            .zip(counts.values().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(top, vec![(Some("b"), 7), (Some("a"), 5), (Some("c"), 4)]);
        Ok(())
    }

    /// Asking for more groups than there are returns all of them.
    #[test]
    fn top_counts_returns_every_group_when_fetch_exceeds_them() -> DFResult<()> {
        let only = partials(&[(Some("a"), 1), (Some("b"), 3), (Some("a"), 1)])?;
        let top = top_counts(vec![only], &schema(), 10)?;
        let counts = top.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(counts.values().to_vec(), vec![3, 2]);
        Ok(())
    }
}
