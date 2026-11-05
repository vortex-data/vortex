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
use datafusion_common::arrow::compute::interleave_record_batch;
use datafusion_common::arrow::compute::take;
use datafusion_common::arrow::row::RowConverter;
use datafusion_common::arrow::row::SortField;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::internal_datafusion_err;
use datafusion_common::internal_err;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::memory_pool::MemoryReservation;
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
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::parallelism::get_available_parallelism;

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
    /// The options of the descending sort by count it replaces.
    options: SortOptions,
    /// Up to this many partial rows, it merges them all rather than pruning them first.
    merge_all_rows: usize,
    properties: Arc<PlanProperties>,
}

impl VortexTopCountExec {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        fetch: usize,
        options: SortOptions,
        merge_all_rows: usize,
    ) -> Self {
        let count = schema.fields().len() - 1;
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new(schema.field(count).name(), count)),
            options,
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
            options,
            merge_all_rows,
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
            self.options,
            self.merge_all_rows,
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
                let reservation = MemoryConsumer::new(format!("VortexTopCountExec[{partition}]"))
                    .register(context.memory_pool());
                Ok(SpawnedTask::spawn(hash_partials(stream, reservation)))
            })
            .collect::<DFResult<Vec<_>>>()?;
        let merge_reservation =
            MemoryConsumer::new("VortexTopCountExec merge").register(context.memory_pool());
        let schema = Arc::clone(&self.schema);
        let fetch = self.fetch;
        let merge_all_rows = self.merge_all_rows;
        let top = async move {
            let mut hashed = Vec::with_capacity(partials.len());
            let mut reservations = Vec::with_capacity(partials.len());
            for partial in partials {
                let (partial, reservation) = partial
                    .join_unwind()
                    .await
                    .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))??;
                hashed.push(partial);
                reservations.push(reservation);
            }
            let rows = hashed
                .iter()
                .flat_map(|partial| &partial.batches)
                .map(RecordBatch::num_rows)
                .sum::<usize>();
            merge_reservation.try_grow(rows.saturating_mul(MERGE_BYTES_PER_ROW))?;
            // The merge runs threads of its own; keep it off the runtime's workers. The
            // reservations last until it returns.
            SpawnedTask::spawn_blocking(move || {
                let top = top_counts(hashed, &schema, fetch, merge_all_rows);
                drop((reservations, merge_reservation));
                top
            })
            .join_unwind()
            .await
            .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))?
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream::once(top),
        )))
    }
}

/// How many high bits of a grouping hash pick the shard that sums it.
const SHARD_BITS: u32 = 4;
const SHARDS: usize = 1 << SHARD_BITS;

/// A partial row: its batch and row, its count, and the hash of its grouping values.
#[derive(Clone, Copy)]
struct HashedRow {
    hash: u64,
    count: i64,
    batch: u32,
    row: u32,
}

/// A partition's partial counts, with each row's hash in the shard its hash picks.
struct HashedPartials {
    batches: Vec<RecordBatch>,
    /// The rows of each shard. A row's `batch` indexes `batches`.
    shards: Vec<Vec<HashedRow>>,
}

impl HashedPartials {
    fn new() -> Self {
        Self {
            batches: Vec::new(),
            shards: vec![Vec::new(); SHARDS],
        }
    }

    /// Hashes the grouping values of a batch of partial counts and files its rows by shard.
    fn push(&mut self, batch: RecordBatch) -> DFResult<()> {
        let (keys, counts) = split_counts(&batch)?;
        let mut hashes = vec![0; batch.num_rows()];
        create_hashes(keys, &HASH_STATE, &mut hashes)?;
        let index = to_u32(self.batches.len())?;
        for (row, (&hash, &count)) in hashes.iter().zip(counts.values()).enumerate() {
            self.shards[shard_of(hash)].push(HashedRow {
                hash,
                count,
                batch: index,
                row: to_u32(row)?,
            });
        }
        self.batches.push(batch);
        Ok(())
    }
}

fn shard_of(hash: u64) -> usize {
    // The shift leaves `SHARD_BITS` bits, which fit.
    usize::try_from(hash >> (u64::BITS - SHARD_BITS)).unwrap_or(0)
}

fn to_u32(value: usize) -> DFResult<u32> {
    u32::try_from(value).map_err(|e| internal_datafusion_err!("{e}"))
}

/// The hash state of the grouping values. Any fixed state works: the hashes only need to agree
/// within one execution.
const HASH_STATE: RandomState = RandomState::with_seed(0);

/// Hashes a partition's partial counts as they arrive, so the scan overlaps the hashing and
/// filing by shard.
async fn hash_partials(
    mut stream: SendableRecordBatchStream,
    reservation: MemoryReservation,
) -> DFResult<(HashedPartials, MemoryReservation)> {
    let mut hashed = HashedPartials::new();
    while let Some(batch) = stream.next().await.transpose()? {
        if batch.num_rows() > 0 {
            reservation.try_grow(
                batch.get_array_memory_size() + batch.num_rows() * size_of::<HashedRow>(),
            )?;
            hashed.push(batch)?;
        }
    }
    Ok((hashed, reservation))
}

/// The memory the merge takes for each partial row: the row's copy in its shard, the sum of
/// its hash, and its hash's entry in the shard's table.
const MERGE_BYTES_PER_ROW: usize = 2 * size_of::<HashedRow>() + 3 * size_of::<i64>();

/// The grouping columns and the count column of a batch of partial counts.
fn split_counts(batch: &RecordBatch) -> DFResult<(&[ArrowArrayRef], &Int64Array)> {
    let (counts, keys) = batch
        .columns()
        .split_last()
        .ok_or_else(|| internal_datafusion_err!("partial counts without a count"))?;
    Ok((keys, counts.as_primitive::<Int64Type>()))
}

/// Up to this many partial rows, merging them all costs less than pruning them first.
pub(super) const MERGE_ALL_ROWS: usize = 1 << 15;

/// The `fetch` largest totals of the groups of `partials`, in descending order.
fn top_counts(
    partials: Vec<HashedPartials>,
    schema: &SchemaRef,
    fetch: usize,
    merge_all_rows: usize,
) -> DFResult<RecordBatch> {
    // Number the batches of every partition together.
    let mut offsets = Vec::with_capacity(partials.len());
    let mut batches = Vec::new();
    for partial in &partials {
        offsets.push(to_u32(batches.len())?);
        batches.extend(partial.batches.iter());
    }
    let Some(first) = batches.first().filter(|_| fetch > 0) else {
        return Ok(RecordBatch::new_empty(Arc::clone(schema)));
    };
    let rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
    if rows <= merge_all_rows {
        let rows = concat_batches(&first.schema(), batches.iter().copied())?;
        return top_groups(group_exactly(rows)?, schema, fetch);
    }

    // The hashes with the `widest + 1` largest sums hold at least `widest + 1 > fetch` groups,
    // every row of each, so the `fetch`-th largest of their exact totals bounds the `fetch`-th
    // largest total from below. With no more than `widest` hashes, every group is merged.
    let widest = fetch.saturating_mul(2);
    let shards = sum_shards(&partials, &offsets, widest)?;
    let mut tops = shards
        .iter()
        .flat_map(|shard| shard.top.iter().copied())
        .collect::<Vec<_>>();
    let least = if widest < tops.len() {
        *tops.select_nth_unstable_by(widest, |a, b| b.cmp(a)).1
    } else {
        i64::MIN
    };
    let mut totals = merge_exactly(&batches, &shards, least)?.totals;
    let threshold = match totals.len().checked_sub(fetch) {
        Some(kth) => *totals.select_nth_unstable(kth).1,
        None => i64::MIN,
    };

    // A group whose hash sums to less than the threshold totals less than it.
    top_groups(merge_exactly(&batches, &shards, threshold)?, schema, fetch)
}

/// The `fetch` groups with the largest totals, in descending order.
fn top_groups(groups: ExactGroups, schema: &SchemaRef, fetch: usize) -> DFResult<RecordBatch> {
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

/// The rows of one shard of the hashes, each with the sum of the counts of its hash.
struct ShardSums {
    rows: Vec<HashedRow>,
    /// The sum of the counts of each row's hash.
    row_sums: Vec<i64>,
    /// The `widest + 1` largest sums of the shard's hashes, or all of them.
    top: Vec<i64>,
}

/// Sums the counts of each hash, with a thread for each group of shards. Every row of a hash is
/// in one shard, so each thread sums its hashes alone, in a table a fraction of the size of
/// one for every hash.
fn sum_shards(
    partials: &[HashedPartials],
    offsets: &[u32],
    widest: usize,
) -> DFResult<Vec<ShardSums>> {
    let threads = get_available_parallelism().unwrap_or(1).clamp(1, SHARDS);
    let shards = (0..SHARDS).collect::<Vec<_>>();
    std::thread::scope(|scope| {
        let workers = shards
            .chunks(SHARDS.div_ceil(threads))
            .map(|shards| {
                scope.spawn(move || {
                    shards
                        .iter()
                        .map(|&shard| sum_shard(partials, offsets, shard, widest))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| internal_datafusion_err!("hash sum panicked"))
            })
            .collect::<DFResult<Vec<_>>>()
    })
    .map(|shards| shards.into_iter().flatten().collect())
}

fn sum_shard(
    partials: &[HashedPartials],
    offsets: &[u32],
    shard: usize,
    widest: usize,
) -> ShardSums {
    let mut rows = Vec::with_capacity(
        partials
            .iter()
            .map(|partial| partial.shards[shard].len())
            .sum(),
    );
    for (partial, &offset) in partials.iter().zip(offsets) {
        rows.extend(partial.shards[shard].iter().map(|row| HashedRow {
            batch: row.batch + offset,
            ..*row
        }));
    }
    let mut sums = HashMap::<u64, i64>::with_capacity(rows.len());
    for row in &rows {
        *sums.entry(row.hash).or_default() += row.count;
    }
    let row_sums = rows.iter().map(|row| sums[&row.hash]).collect();
    let mut top = sums.into_values().collect::<Vec<_>>();
    if widest < top.len() {
        top.select_nth_unstable_by(widest, |a, b| b.cmp(a));
        top.truncate(widest + 1);
    }
    ShardSums {
        rows,
        row_sums,
        top,
    }
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

/// Merges, by their grouping values, the partial counts whose hash sums to at least `least`.
/// Every row of a group has the same hash, so each group it returns holds all of that group's
/// counts.
fn merge_exactly(
    batches: &[&RecordBatch],
    shards: &[ShardSums],
    least: i64,
) -> DFResult<ExactGroups> {
    let selected = shards
        .iter()
        .flat_map(|shard| {
            shard
                .rows
                .iter()
                .zip(&shard.row_sums)
                .filter(move |(_, sum)| **sum >= least)
                .map(|(row, _)| (row.batch as usize, row.row as usize))
        })
        .collect::<Vec<_>>();
    group_exactly(interleave_record_batch(batches, &selected)?)
}

/// Merges partial counts by their grouping values.
fn group_exactly(rows: RecordBatch) -> DFResult<ExactGroups> {
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
        let group = match groups.entry(converted.row(row)) {
            Entry::Occupied(group) => *group.get(),
            Entry::Vacant(group) => {
                representatives.push(to_u32(row)?);
                totals.push(0);
                *group.insert(totals.len() - 1)
            }
        };
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
    use datafusion_common::DataFusionError;
    use datafusion_common::Result as DFResult;
    use datafusion_common::internal_datafusion_err;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_execution::memory_pool::MemoryConsumer;
    use datafusion_execution::memory_pool::MemoryPool;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use rstest::rstest;

    use super::HashedPartials;
    use super::hash_partials;
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
        let mut partials = HashedPartials::new();
        partials.push(batch)?;
        Ok(partials)
    }

    /// Totals spread over many small partial counts beat a single large one, and nulls group.
    #[rstest]
    fn top_counts_merges_partials_across_partitions(
        #[values(0, usize::MAX)] merge_all_rows: usize,
    ) -> DFResult<()> {
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
            (None, 4),
            (Some("e"), 4),
            (Some("b"), 4),
        ])?;
        let third = partials(&[(Some("b"), 1), (Some("c"), 1), (Some("c"), 1)])?;

        let top = top_counts(vec![first, second, third], &schema(), 3, merge_all_rows)?;
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
        assert_eq!(top, vec![(Some("b"), 7), (None, 6), (Some("a"), 5)]);
        Ok(())
    }

    /// Asking for more groups than there are returns all of them.
    #[rstest]
    fn top_counts_returns_every_group_when_fetch_exceeds_them(
        #[values(0, usize::MAX)] merge_all_rows: usize,
    ) -> DFResult<()> {
        let only = partials(&[(Some("a"), 1), (Some("b"), 3), (Some("a"), 1)])?;
        let top = top_counts(vec![only], &schema(), 10, merge_all_rows)?;
        let counts = top.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(counts.values().to_vec(), vec![3, 2]);
        Ok(())
    }

    /// Partial counts beyond the memory pool fail to buffer rather than grow unaccounted.
    #[tokio::test]
    async fn hash_partials_reserves_its_batches() -> DFResult<()> {
        let batch = partials(&[(Some("a"), 1), (Some("b"), 3)])?
            .batches
            .remove(0);
        let input = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::iter([Ok(batch.clone())]),
        ));
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let err = hash_partials(input, reservation)
            .await
            .err()
            .ok_or_else(|| internal_datafusion_err!("64 bytes held a batch"))?;
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );

        let input = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::iter([Ok(batch)]),
        ));
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let (hashed, reservation) = hash_partials(input, reservation).await?;
        assert_eq!(hashed.batches.len(), 1);
        assert!(reservation.size() > 0);
        Ok(())
    }

    /// A `LIMIT 0` returns no groups rather than indexing past the merged totals.
    #[rstest]
    fn top_counts_of_zero_is_empty(#[values(0, usize::MAX)] merge_all_rows: usize) -> DFResult<()> {
        let only = partials(&[(Some("a"), 1), (Some("b"), 3)])?;
        let top = top_counts(vec![only], &schema(), 0, merge_all_rows)?;
        assert_eq!(top.num_rows(), 0);
        Ok(())
    }

    /// A scan without rows has no top counts, rather than failing to merge nothing.
    #[rstest]
    fn top_counts_of_no_partials_is_empty(
        #[values(0, usize::MAX)] merge_all_rows: usize,
    ) -> DFResult<()> {
        let top = top_counts(vec![HashedPartials::new()], &schema(), 3, merge_all_rows)?;
        assert_eq!(top.num_rows(), 0);
        Ok(())
    }
}
