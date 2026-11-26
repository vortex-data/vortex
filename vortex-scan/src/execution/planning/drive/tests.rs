// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A plan that opens files itself: a root opens every file, then spawns one reader per file, and
//! every read goes through the one service the driver was given.

use std::sync::Arc;

use futures::TryStreamExt;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_io::VortexReadAt;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSlot;
use vortex_io::request::IoTarget;
use vortex_io::request::OpenTarget;
use vortex_io::request::ReadAtIoService;
use vortex_io::request::ReadAtOpenable;
use vortex_io::request::SourceId;
use vortex_io::request::SourcesIoService;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;

use super::*;
use crate::execution::planning::morsel::Morsel;
use crate::execution::planning::morsel::MorselOutput;
use crate::execution::planning::planner::PlannerOutput;
use crate::execution::planning::planner::State;
use crate::execution::planning::tests::scripted::ctx;

/// File `i` holds four bytes, each `i`.
fn file(i: u8) -> ReadAtOpenable {
    let bytes: Arc<dyn VortexReadAt> = Arc::new(ByteBuffer::from(vec![i; 4]));
    ReadAtOpenable::new(format!("mem://file-{i}"), bytes)
}

/// Opens every file, then spawns a [`ReadFile`] for each, with the file's index as its ordinal.
struct OpenFiles {
    sources: SourceId,
    files: Vec<ReadAtOpenable>,
    /// The id each file opened to, by file index.
    opened: Vec<Option<SourceId>>,
    requested: bool,
    next: usize,
}

impl OpenFiles {
    fn new(sources: &SourcesIoService, files: Vec<ReadAtOpenable>) -> Self {
        Self {
            sources: sources.id(),
            opened: vec![None; files.len()],
            files,
            requested: false,
            next: 0,
        }
    }
}

impl IoConsumer for OpenFiles {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        let IoResult::Opened(id) = result else {
            vortex_panic!("an open was answered with {}", result.kind());
        };
        self.opened[request.0 as usize] = Some(id);
    }
}

impl Planner for OpenFiles {
    fn state(&self) -> State {
        if !self.requested {
            State::NeedsCompute
        } else if self.opened.iter().any(Option::is_none) {
            State::Waiting
        } else if self.next < self.files.len() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.requested {
            self.requested = true;
            let opens: IoBatch = self
                .files
                .iter()
                .enumerate()
                .map(|(i, file)| IoRequest {
                    source: self.sources,
                    intent: IoIntent::Fetch,
                    request: IoRequestId(u32::try_from(i).unwrap_or(u32::MAX)),
                    target: IoTarget::Open(OpenTarget::new(file.clone())),
                })
                .collect();
            return Ok(PlannerOutput::NeedsIO(opens));
        }
        let i = self.next;
        self.next += 1;
        let source = self.opened[i].unwrap_or_else(|| vortex_panic!("file {i} is not open"));
        let scope = WorkScope {
            file_ordinal: i as u64,
            rows: 0..4,
        };
        Ok(PlannerOutput::Spawn(scope, Box::new(ReadFile::new(source))))
    }
}

/// Reads its file's four bytes, then hands them to a morsel.
struct ReadFile {
    slot: IoSlot,
    requested: bool,
    done: bool,
}

impl ReadFile {
    fn new(source: SourceId) -> Self {
        Self {
            slot: IoSlot::new(source),
            requested: false,
            done: false,
        }
    }
}

impl IoConsumer for ReadFile {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl Planner for ReadFile {
    fn state(&self) -> State {
        if self.done {
            State::Done
        } else if self.slot.is_waiting() {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.requested {
            self.requested = true;
            return Ok(PlannerOutput::NeedsIO(
                self.slot.issue(IoTarget::range(0, 4)),
            ));
        }
        let Some(IoResult::Bytes(bytes)) = self.slot.take() else {
            vortex_panic!("the read was not delivered");
        };
        self.done = true;
        let array = PrimitiveArray::from_iter(bytes.try_to_host_sync()?.iter().copied());
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..4,
        };
        Ok(PlannerOutput::Morsel(
            scope,
            Box::new(Emit(Some(array.into_array()))),
        ))
    }
}

/// Emits one array.
struct Emit(Option<ArrayRef>);

impl IoConsumer for Emit {
    fn set_io_result(&mut self, _: IoRequestId, _: IoResult) {
        vortex_panic!("Emit reads nothing");
    }
}

impl Morsel for Emit {
    fn state(&self) -> State {
        if self.0.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        Ok(self
            .0
            .take()
            .map_or(MorselOutput::Done, MorselOutput::Batch))
    }
}

fn root(sources: &SourcesIoService, files: u8) -> PlanRoot {
    PlanRoot::new(
        WorkScope {
            file_ordinal: 0,
            rows: 0..u64::from(files) * 4,
        },
        Box::new(OpenFiles::new(sources, (0..files).map(file).collect())),
    )
}

/// The first byte of each batch, which names the file it was read from.
fn files_read(batches: &[ScopedBatch]) -> VortexResult<Vec<u8>> {
    let mut files = batches
        .iter()
        .map(|batch| {
            let mut ctx = ctx();
            let array = batch.batch.clone().execute::<PrimitiveArray>(&mut ctx)?;
            Ok(array.as_slice::<u8>()[0])
        })
        .collect::<VortexResult<Vec<_>>>()?;
    files.sort_unstable();
    Ok(files)
}

#[derive(Clone, Copy, Debug)]
enum Driver {
    Sequential,
    Parallel,
}

fn run(driver: Driver, files: u8) -> VortexResult<(Vec<ScopedBatch>, SourcesIoService)> {
    let sources = SourcesIoService::new();
    let io: Arc<dyn IoService> = Arc::new(sources.clone());
    let runtime = CurrentThreadRuntime::new();
    let stream = match driver {
        Driver::Sequential => SequentialDriver::new(io).execute(root(&sources, files)),
        Driver::Parallel => ParallelDriver::new(runtime.handle(), io)
            .with_workers(3)
            .execute(root(&sources, files)),
    };
    let batches = runtime.block_on(stream.try_collect())?;
    Ok((batches, sources))
}

#[rstest]
#[case(Driver::Sequential)]
#[case(Driver::Parallel)]
fn a_planner_opens_and_reads_many_files(#[case] driver: Driver) -> VortexResult<()> {
    let (batches, sources) = run(driver, 5)?;
    assert_eq!(files_read(&batches)?, vec![0, 1, 2, 3, 4]);
    assert_eq!(sources.len(), 5, "every file opened to a source of its own");
    Ok(())
}

#[test]
fn ordered_batches_follow_their_files() -> VortexResult<()> {
    let sources = SourcesIoService::new();
    let io: Arc<dyn IoService> = Arc::new(sources.clone());
    let runtime = CurrentThreadRuntime::new();
    let batches: Vec<ScopedBatch> = runtime.block_on(
        ParallelDriver::new(runtime.handle(), io)
            .with_workers(4)
            .with_ordered(true)
            .execute(root(&sources, 8))
            .try_collect(),
    )?;
    let ordinals: Vec<u64> = batches.iter().map(|b| b.scope.file_ordinal).collect();
    let mut sorted = ordinals.clone();
    sorted.sort_unstable();
    assert_eq!(ordinals, sorted);
    Ok(())
}

#[test]
fn a_request_for_a_source_never_opened_fails_the_plan() {
    // An id another service minted, which the plan's service never registered.
    let stray = SourcesIoService::new().register(Arc::new(ReadAtIoService::new(Arc::new(
        ByteBuffer::from(vec![0u8; 4]),
    ))));
    let io: Arc<dyn IoService> = Arc::new(SourcesIoService::new());
    let runtime = CurrentThreadRuntime::new();
    let root = PlanRoot::new(
        WorkScope {
            file_ordinal: 0,
            rows: 0..4,
        },
        Box::new(ReadFile::new(stray)),
    );
    let result: VortexResult<Vec<ScopedBatch>> =
        runtime.block_on(SequentialDriver::new(io).execute(root).try_collect());
    assert!(result.is_err());
}
