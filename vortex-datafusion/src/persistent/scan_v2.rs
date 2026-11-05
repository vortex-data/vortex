// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DataFusion.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. The
//! opener composes the planners over a [`PreparedScan`] and hands the driver the first; the
//! driver runs it and everything it emits on planning runs it owns: the root run stepped on
//! `spawn_cpu` tasks as the stream is polled, each spawned chunk on a task of its own, or on the
//! root run when the engine's partitions already fill the cores. Every read is a `spawn_io` task
//! of the file's segment source. The DataFusion thread polling the stream does no scan work
//! itself.
//!
//! A batch longer than the engine's batch size is sliced before it is mapped to Arrow, and each
//! slice is mapped when the stream asks for it, so a chunk decodes one engine batch at a time.
//! Slicing a scan's batch is lazy and cheap: a filtered array slices its mask and child rather
//! than evaluating the filter again.

use std::collections::VecDeque;
use std::future;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use vortex::array::ArrayRef;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::request::IoService;
use vortex::io::request::IoSource;
use vortex::io::runtime::Handle;
use vortex::layout::scan::v2::PreparedScan;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

/// Orders a mapped batch among the scan's: the row its scan batch starts at, then the slice of
/// that batch it is.
type BatchKey = (u64, usize);

/// Maps a scan's batches to Arrow on the task that produced them.
type MapBatch = Arc<dyn Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync>;

/// Runs one partition's V2 scan on DataFusion's runtime.
pub(crate) struct ScanDriver {
    handle: Handle,
    /// Chunks run beside each other per worker thread; at 1 they run on the scan's own run.
    concurrency: usize,
    /// Whether batches are yielded in row order, which holds them until the scan is done.
    ordered: bool,
    /// The most rows of a batch the stream yields, or `None` to yield the scan's batches whole.
    batch_size: Option<usize>,
}

impl ScanDriver {
    pub(crate) fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        Self {
            handle,
            concurrency,
            ordered,
            batch_size: None,
        }
    }

    /// Yields batches of at most `batch_size` rows, slicing longer ones before mapping them.
    pub(crate) fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size).filter(|&batch_size| batch_size > 0);
        self
    }

    /// Runs `root`, the first planner composed over `scan`, and everything it emits, and
    /// streams their batches mapped by `map`.
    pub(crate) fn run(
        self,
        scan: Arc<PreparedScan>,
        root: Box<dyn Planner>,
        map: impl Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync + 'static,
    ) -> BoxStream<'static, VortexResult<RecordBatch>> {
        let map: MapBatch = Arc::new(map);
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let inline = in_flight <= 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let batch_size = self.batch_size;
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map));
        let batches = run_stream(
            self.handle,
            root,
            scope,
            Arc::clone(&io),
            map,
            inline,
            batch_size,
        )
        .map_ok(move |yielded| match yielded {
            Yielded::Batch(key, batch) => stream::once(future::ready(Ok((key, batch)))).boxed(),
            Yielded::Spawn(scope, chunk) => chunk_stream(
                &chunk_handle,
                chunk,
                scope,
                chunk_io.session(),
                Arc::clone(&chunk_map),
                batch_size,
            ),
        })
        .try_flatten_unordered(in_flight);
        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        let batches = batches.chain(stream::poll_fn(move |_| {
            let _ = &scan;
            Poll::Ready(None)
        }));
        if !self.ordered {
            return batches.map_ok(|(_, batch)| batch).boxed();
        }
        stream::once(async move {
            let mut held: Vec<(BatchKey, RecordBatch)> = batches.try_collect().await?;
            // Morsels finish in whatever order their reads arrive.
            held.sort_by_key(|(key, _)| *key);
            VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
        })
        .try_flatten()
        .boxed()
    }
}

/// Cuts `array` into lazy slices of at most `batch_size` rows, in row order, keyed by the row
/// the scan batch starts at. A batch that fits is yielded whole.
fn slice_batch(
    start: u64,
    array: ArrayRef,
    batch_size: Option<usize>,
) -> VortexResult<VecDeque<(BatchKey, ArrayRef)>> {
    let len = array.len();
    let Some(batch_size) = batch_size.filter(|&batch_size| len > batch_size) else {
        return Ok(VecDeque::from([((start, 0), array)]));
    };
    (0..len)
        .step_by(batch_size)
        .enumerate()
        .map(|(index, offset)| {
            let slice = array.slice(offset..(offset + batch_size).min(len))?;
            Ok(((start, index), slice))
        })
        .collect()
}

/// What the root run's stream yields: a mapped batch with its key, or a chunk a planner of the
/// run spawned.
enum Yielded {
    Batch(BatchKey, RecordBatch),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step {
    /// A mapped batch, and the slices of its scan batch still to map.
    Batch(BatchKey, RecordBatch, VecDeque<(BatchKey, ArrayRef)>),
    Spawn(WorkScope, Box<dyn Planner>),
    RootDone,
    Idle,
    Waiting,
}

/// The state of the root run's stream.
struct RunState {
    handle: Handle,
    /// Taken while a CPU step runs on its task and put back with its result.
    run: Option<Run>,
    /// The scan's reads: each root of the run reads through a session of its own, which the run
    /// clears when the root retires.
    io: Arc<dyn IoService>,
    map: MapBatch,
    /// Whether a spawned chunk runs on this run, as another root, instead of being yielded.
    inline: bool,
    batch_size: Option<usize>,
    /// Slices of the run's last batch not mapped yet: each is mapped on a step of its own, before
    /// the run advances again.
    pending: VecDeque<(BatchKey, ArrayRef)>,
}

/// Runs `root` over `scope` on a planning run of its own, reading through `io`, each CPU step
/// on a `spawn_cpu` task, and yields its mapped batches and, unless `inline`, the chunks it
/// spawns. Inline, a spawned chunk runs on this run beside the others: the run's steps already
/// interleave every chunk's work, and a run per chunk would only add a hand-off per chunk for
/// parallelism the engine provides.
fn run_stream(
    handle: Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoService>,
    map: MapBatch,
    inline: bool,
    batch_size: Option<usize>,
) -> BoxStream<'static, VortexResult<Yielded>> {
    let mut run = Run::new();
    run.admit(root, scope, io.session());
    let state = RunState {
        handle,
        run: Some(run),
        io,
        map,
        inline,
        batch_size,
        pending: VecDeque::new(),
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            if let Some((key, slice)) = state.pending.pop_front() {
                let map = Arc::clone(&state.map);
                let batch = state.handle.spawn_cpu(move || map(slice)).await?;
                return Ok(Some((Yielded::Batch(key, batch), state)));
            }
            let run = state
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let map = Arc::clone(&state.map);
            let batch_size = state.batch_size;
            let (run, step) = state
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let step = run.advance().and_then(|progress| {
                        Ok(match progress {
                            Progress::Batch(batch) => {
                                let mut slices =
                                    slice_batch(batch.scope.rows.start, batch.array, batch_size)?;
                                let (key, first) = slices
                                    .pop_front()
                                    .ok_or_else(|| vortex_err!("a batch has no slices"))?;
                                Step::Batch(key, map(first)?, slices)
                            }
                            Progress::Spawn(scope, child) => Step::Spawn(scope, child),
                            Progress::RootDone(_) => Step::RootDone,
                            Progress::Idle => Step::Idle,
                            Progress::Waiting => Step::Waiting,
                        })
                    });
                    (run, step)
                })
                .await;
            let run = state.run.insert(run);
            match step? {
                Step::Batch(key, batch, rest) => {
                    state.pending = rest;
                    return Ok(Some((Yielded::Batch(key, batch), state)));
                }
                Step::Spawn(scope, child) if state.inline => {
                    run.admit(child, scope, state.io.session());
                }
                Step::Spawn(scope, child) => {
                    return Ok(Some((Yielded::Spawn(scope, child), state)));
                }
                Step::RootDone if run.live_roots() > 0 => {}
                Step::RootDone | Step::Idle => return Ok(None),
                Step::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    })
    .boxed()
}

/// Runs a spawned chunk's `root` over `scope` on a planning run of its own, on one task that
/// drives the run to its end and awaits only its reads, and yields its mapped batches with their
/// keys. A planner the chunk spawns in turn runs as another root of the same run. Each slice of a
/// batch is mapped only once the stream has taken the one before it.
///
/// One task per chunk, not one per step: a chunk's run waits for reads many times, and a task
/// hop at each wait is latency on the chunk's critical path that no other chunk's work fills.
fn chunk_stream(
    handle: &Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map: MapBatch,
    batch_size: Option<usize>,
) -> BoxStream<'static, VortexResult<(BatchKey, RecordBatch)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, Arc::clone(&io));
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    for (key, slice) in
                        slice_batch(batch.scope.rows.start, batch.array, batch_size)?
                    {
                        let batch = map(slice);
                        let failed = batch.is_err();
                        if send.send(batch.map(|batch| (key, batch))).await.is_err() || failed {
                            return Ok(());
                        }
                    }
                }
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, Arc::clone(&io));
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return Ok(()),
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    });
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    stream::select(recv, errors).boxed()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::error::VortexResult;
    use vortex::mask::Mask;

    use super::slice_batch;

    #[rstest]
    #[case::no_batch_size(10, None, vec![10])]
    #[case::fits(10, Some(10), vec![10])]
    #[case::even(10, Some(5), vec![5, 5])]
    #[case::remainder(10, Some(4), vec![4, 4, 2])]
    #[case::one_row(3, Some(1), vec![1, 1, 1])]
    fn slice_batch_cuts_to_batch_size(
        #[case] len: i32,
        #[case] batch_size: Option<usize>,
        #[case] expected: Vec<usize>,
    ) -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0..len).into_array();
        let slices = slice_batch(7, array, batch_size)?;
        let lens: Vec<_> = slices.iter().map(|(_, slice)| slice.len()).collect();
        assert_eq!(lens, expected);
        let keys: Vec<_> = slices.iter().map(|(key, _)| *key).collect();
        assert_eq!(
            keys,
            (0..expected.len())
                .map(|index| (7, index))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    /// Slicing a filtered batch covers exactly the rows its filter keeps.
    #[test]
    fn slice_batch_covers_filtered_rows() -> VortexResult<()> {
        let mask = Mask::from_iter((0..100).map(|index| index % 3 != 0));
        let array = PrimitiveArray::from_iter(0..100i32)
            .into_array()
            .filter(mask)?;
        let kept = array.len();
        let slices = slice_batch(0, array, Some(16))?;
        assert_eq!(
            slices.iter().map(|(_, slice)| slice.len()).sum::<usize>(),
            kept
        );
        assert!(slices.iter().all(|(_, slice)| slice.len() <= 16));
        Ok(())
    }
}
