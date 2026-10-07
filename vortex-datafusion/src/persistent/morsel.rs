// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Morsel-driven scans of Vortex files.
//!
//! DataFusion's [`FileStream`] drives a [`Morselizer`] one step at a time: planners do CPU work,
//! hand back one I/O future at a time, and produce [`Morsel`]s that only decode data. With
//! [`VortexTableOptions::morsel_scan`] enabled, each file becomes:
//!
//! 1. a [`FilePlanner`], whose I/O opens the footer and prepares the scan, then
//! 2. a [`SplitPlanner`], which keeps the scan's usual number of split tasks running on the
//!    Vortex runtime and whose I/O waits for the next one to finish, and
//! 3. one [`SplitMorsel`] per split with rows, which shapes the split into the plan's schema.
//!
//! A Vortex split task reads and decodes its rows together, so the decoding happens on the
//! Vortex runtime rather than in the morsel. What DataFusion gains is split granularity: it
//! applies limits between splits, and a dynamic filter that rules out the rest of a file stops
//! the remaining splits from starting.
//!
//! Without the option, each file is a single morsel wrapping the stream-based scan, as with
//! DataFusion's adapter for [`FileOpener`]s.
//!
//! [`FileStream`]: datafusion_datasource::file_stream::FileStream
//! [`FileOpener`]: datafusion_datasource::file_stream::FileOpener
//! [`VortexTableOptions::morsel_scan`]: crate::VortexTableOptions::morsel_scan

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::mem;
use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::morsel::Morsel;
use datafusion_datasource::morsel::MorselPlan;
use datafusion_datasource::morsel::MorselPlanner;
use datafusion_datasource::morsel::Morselizer;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use futures::stream::FuturesOrdered;
use futures::stream::FuturesUnordered;
use vortex::error::VortexResult;
use vortex::io::runtime::Task;

use crate::persistent::opener::BatchOutput;
use crate::persistent::opener::SplitTasks;
use crate::persistent::opener::VortexOpener;

/// Plans the files of one partition of a Vortex scan.
pub(crate) struct VortexMorselizer {
    opener: Arc<VortexOpener>,
    /// Whether to read files split by split, see [`crate::VortexTableOptions::morsel_scan`].
    split_morsels: bool,
}

impl VortexMorselizer {
    pub(crate) fn new(opener: Arc<VortexOpener>, split_morsels: bool) -> Self {
        Self {
            opener,
            split_morsels,
        }
    }
}

impl Debug for VortexMorselizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexMorselizer")
            .field("split_morsels", &self.split_morsels)
            .finish_non_exhaustive()
    }
}

impl Morselizer for VortexMorselizer {
    fn plan_file(&self, file: PartitionedFile) -> DFResult<Box<dyn MorselPlanner>> {
        Ok(Box::new(FilePlanner {
            opener: Arc::clone(&self.opener),
            file,
            split_morsels: self.split_morsels,
        }))
    }
}

/// Opens one file and prepares its scan.
struct FilePlanner {
    opener: Arc<VortexOpener>,
    file: PartitionedFile,
    split_morsels: bool,
}

impl Debug for FilePlanner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilePlanner")
            .field("file", &self.file.object_meta.location)
            .finish_non_exhaustive()
    }
}

impl MorselPlanner for FilePlanner {
    fn plan(self: Box<Self>) -> DFResult<Option<MorselPlan>> {
        let prepare = self.opener.prepare(self.file)?;
        let split_morsels = self.split_morsels;
        Ok(Some(MorselPlan::new().with_pending_planner(async move {
            let next: Box<dyn MorselPlanner> = match prepare.await? {
                None => Box::new(DonePlanner),
                Some(scan) if split_morsels => {
                    Box::new(SplitPlanner::new(scan.into_split_tasks().await?))
                }
                Some(scan) => Box::new(StreamPlanner(scan.into_stream()?)),
            };
            Ok(next)
        })))
    }
}

/// A file with nothing left to read.
#[derive(Debug)]
struct DonePlanner;

impl MorselPlanner for DonePlanner {
    fn plan(self: Box<Self>) -> DFResult<Option<MorselPlan>> {
        Ok(None)
    }
}

/// Hands over a whole file's stream as one morsel.
struct StreamPlanner(BoxStream<'static, DFResult<RecordBatch>>);

impl Debug for StreamPlanner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamPlanner").finish_non_exhaustive()
    }
}

impl MorselPlanner for StreamPlanner {
    fn plan(self: Box<Self>) -> DFResult<Option<MorselPlan>> {
        Ok(Some(
            MorselPlan::new().with_morsels(vec![Box::new(StreamMorsel(self.0))]),
        ))
    }
}

struct StreamMorsel(BoxStream<'static, DFResult<RecordBatch>>);

impl Debug for StreamMorsel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamMorsel").finish_non_exhaustive()
    }
}

impl Morsel for StreamMorsel {
    fn into_stream(self: Box<Self>) -> BoxStream<'static, DFResult<RecordBatch>> {
        self.0
    }
}

type SplitResult = VortexResult<Vec<RecordBatch>>;

/// Split tasks running on the Vortex runtime.
enum InFlight {
    /// Finishes splits in the order they started, for scans that must keep the file's order.
    Ordered(FuturesOrdered<Task<SplitResult>>),
    /// Finishes splits as soon as they are read.
    Unordered(FuturesUnordered<Task<SplitResult>>),
}

impl InFlight {
    fn len(&self) -> usize {
        match self {
            Self::Ordered(tasks) => tasks.len(),
            Self::Unordered(tasks) => tasks.len(),
        }
    }

    fn push(&mut self, task: Task<SplitResult>) {
        match self {
            Self::Ordered(tasks) => tasks.push_back(task),
            Self::Unordered(tasks) => tasks.push(task),
        }
    }

    async fn next(&mut self) -> Option<SplitResult> {
        match self {
            Self::Ordered(tasks) => tasks.next().await,
            Self::Unordered(tasks) => tasks.next().await,
        }
    }
}

/// Reads one file's splits, a bounded number at a time.
struct SplitPlanner {
    tasks: SplitTasks,
    in_flight: InFlight,
    /// The batches of the last split read, waiting to become a morsel.
    ready: Vec<RecordBatch>,
}

impl SplitPlanner {
    fn new(tasks: SplitTasks) -> Self {
        let in_flight = if tasks.ordered {
            InFlight::Ordered(FuturesOrdered::new())
        } else {
            InFlight::Unordered(FuturesUnordered::new())
        };
        Self {
            tasks,
            in_flight,
            ready: Vec::new(),
        }
    }
}

impl Debug for SplitPlanner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitPlanner")
            .field("pending", &self.tasks.tasks.len())
            .field("in_flight", &self.in_flight.len())
            .finish_non_exhaustive()
    }
}

impl MorselPlanner for SplitPlanner {
    fn plan(mut self: Box<Self>) -> DFResult<Option<MorselPlan>> {
        let batches = mem::take(&mut self.ready);
        let morsel = (!batches.is_empty()).then(|| {
            Box::new(SplitMorsel {
                batches,
                output: Arc::clone(&self.tasks.output),
            }) as Box<dyn Morsel>
        });
        // The file ends with its last morsel, or with `None` when it has none left to hand over.
        // `FileStream` only counts a file as finished, and stops timing its opening, on one of
        // these, so an empty plan here would leave its opening timer running.
        let last = |morsel: Option<Box<dyn Morsel>>| {
            morsel.map(|morsel| MorselPlan::new().with_morsels(vec![morsel]))
        };

        // A dynamic filter may have ruled out the rest of the file since the last split. Dropping
        // the planner cancels the splits still running.
        if let Some(file_pruner) = self.tasks.file_pruner.as_mut()
            && file_pruner.should_prune()?
        {
            return Ok(last(morsel));
        }

        while self.in_flight.len() < self.tasks.max_in_flight
            && let Some(task) = self.tasks.tasks.pop_front()
        {
            let task = self.tasks.handle.spawn(task);
            self.in_flight.push(task);
        }
        if self.in_flight.len() == 0 {
            return Ok(last(morsel));
        }

        let plan = MorselPlan::new().with_morsels(morsel.into_iter().collect());

        Ok(Some(plan.with_pending_planner(async move {
            if let Some(result) = self.in_flight.next().await {
                self.ready = result.map_err(|e| self.tasks.output.read_error(e))?;
            }
            Ok(self as Box<dyn MorselPlanner>)
        })))
    }
}

/// One split's rows, read and decoded by the Vortex scan.
struct SplitMorsel {
    batches: Vec<RecordBatch>,
    output: Arc<BatchOutput>,
}

impl Debug for SplitMorsel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitMorsel")
            .field(
                "rows",
                &self
                    .batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
            )
            .finish_non_exhaustive()
    }
}

impl Morsel for SplitMorsel {
    fn into_stream(self: Box<Self>) -> BoxStream<'static, DFResult<RecordBatch>> {
        let output = self.output;
        stream::iter(self.batches)
            .map(move |batch| output.finish(batch))
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use datafusion::arrow::array::Int32Array;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::Field;
    use datafusion::arrow::datatypes::Schema;
    use datafusion_datasource::TableSchema;
    use futures::TryStreamExt;
    use object_store::ObjectStore;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::error::VortexExpect;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexWrite;
    use vortex::io::object_store::ObjectStoreWrite;
    use vortex::layout::LayoutStrategy;
    use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
    use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex::layout::layouts::table::TableStrategy;
    use vortex::session::VortexSession;

    use super::*;
    use crate::persistent::opener::tests::make_opener;

    /// Drives `planner` to completion the way DataFusion's `FileStream` does, returning how many
    /// morsels it produced and the values of their batches.
    async fn drive(planner: Box<dyn MorselPlanner>) -> DFResult<(usize, Vec<i32>)> {
        let mut planners = VecDeque::from([planner]);
        let (mut morsels, mut values) = (0, Vec::new());
        while let Some(planner) = planners.pop_front() {
            let Some(mut plan) = planner.plan()? else {
                continue;
            };
            for morsel in plan.take_morsels() {
                morsels += 1;
                for batch in morsel.into_stream().try_collect::<Vec<_>>().await? {
                    let column = batch.column(0).as_any().downcast_ref::<Int32Array>();
                    values.extend(column.into_iter().flatten().flatten());
                }
            }
            planners.extend(plan.take_ready_planners());
            if let Some(pending) = plan.take_pending_planner() {
                planners.push_back(pending.await?);
            }
        }
        Ok((morsels, values))
    }

    /// Writes `chunks` chunks of 20,000 ascending `a` values as a file with one split per chunk.
    async fn write_chunked_file(
        object_store: &Arc<dyn ObjectStore>,
        chunks: i32,
    ) -> anyhow::Result<u64> {
        let rows = 20_000;
        let table = ChunkedArray::from_iter((0..chunks).map(|chunk| {
            StructArray::try_new(
                ["a"].into(),
                vec![Buffer::from_iter(chunk * rows..(chunk + 1) * rows).into_array()],
                20_000,
                Validity::NonNullable,
            )
            .map(IntoArray::into_array)
            .vortex_expect("valid chunk")
        }))
        .into_array();
        let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
        ));
        let mut writer =
            ObjectStoreWrite::new(Arc::clone(object_store), &"file.vortex".into()).await?;
        let summary = VortexSession::default()
            .write_options()
            .with_strategy(strategy)
            .write(&mut writer, table.to_array_stream())
            .await?;
        writer.shutdown().await?;
        Ok(summary.size())
    }

    /// With split morsels, each split of the file is its own morsel, read in the file's order
    /// when the scan must keep it. Otherwise the file is one morsel.
    #[rstest]
    #[case::split_morsels(true, 5)]
    #[case::file_morsel(false, 1)]
    #[tokio::test]
    async fn morsels_follow_splits(
        #[case] split_morsels: bool,
        #[case] expected_morsels: usize,
    ) -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let size = write_chunked_file(&object_store, 5).await?;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let mut opener = make_opener(object_store, TableSchema::builder(schema).build(), None);
        opener.has_output_ordering = true;
        let morselizer = VortexMorselizer::new(Arc::new(opener), split_morsels);
        let planner = morselizer.plan_file(PartitionedFile::new("file.vortex", size))?;
        let (morsels, values) = drive(planner).await?;

        assert_eq!(values, (0..100_000).collect::<Vec<_>>());
        assert_eq!(morsels, expected_morsels);
        Ok(())
    }
}
