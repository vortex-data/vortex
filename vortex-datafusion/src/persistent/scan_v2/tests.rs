// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;

use rstest::rstest;
use vortex::io::request::Completion;
use vortex::io::request::IoBatch;
use vortex::io::request::IoConsumer;
use vortex::io::request::IoOwnerId;
use vortex::io::request::IoRequest;
use vortex::io::request::IoRequestId;
use vortex::io::request::IoResult;
use vortex::io::request::IoSlot;
use vortex::io::request::IoSource;
use vortex::io::request::IoTarget;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::scan::planning::planner::PlannerOutput;
use vortex::scan::planning::planner::State;

use super::*;

struct QueuedIo;

impl IoService for QueuedIo {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(QueuedReads::default())
    }
}

/// Deliver one read until its owner retires, keeping a sibling's read outstanding.
#[derive(Default)]
struct QueuedReads(Mutex<VecDeque<(IoOwnerId, IoRequest)>>, AtomicBool);

impl IoSource for QueuedReads {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        self.0
            .lock()
            .unwrap()
            .extend(batch.into_iter().map(|request| (owner, request)));
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }

    fn poll_completion(&self, _cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        if self.1.swap(true, Ordering::SeqCst) {
            Poll::Pending
        } else {
            Poll::Ready(self.wait())
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        let (owner, request) = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| vortex_err!("the sibling's pending read was cancelled"))?;
        Ok(Completion {
            owner,
            request: request.request,
            result: Ok(IoResult::Size(42)),
        })
    }

    fn release(&self, owner: IoOwnerId) {
        self.0
            .lock()
            .unwrap()
            .retain(|(queued, _)| *queued != owner);
        self.1.store(false, Ordering::SeqCst);
    }

    fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
}

struct SpawnReads {
    remaining: usize,
    completed: Arc<AtomicUsize>,
}

impl IoConsumer for SpawnReads {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {
        panic!("only the spawned roots issue reads");
    }
}

impl Planner for SpawnReads {
    fn state(&self) -> State {
        if self.remaining == 0 {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        self.remaining -= 1;
        Ok(PlannerOutput::Spawn(
            WorkScope {
                file_ordinal: 0,
                rows: self.remaining as u64..self.remaining as u64 + 1,
            },
            Box::new(ReadOnce {
                slot: IoSlot::default(),
                done: false,
                completed: Arc::clone(&self.completed),
            }),
        ))
    }
}

struct ReadOnce {
    slot: IoSlot,
    done: bool,
    completed: Arc<AtomicUsize>,
}

impl IoConsumer for ReadOnce {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl Planner for ReadOnce {
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
        if let Some(result) = self.slot.take() {
            assert!(matches!(result, IoResult::Size(42)));
            self.done = true;
            self.completed.fetch_add(1, Ordering::SeqCst);
            return Ok(PlannerOutput::Done);
        }
        Ok(PlannerOutput::NeedsIO(self.slot.issue(IoTarget::Size)))
    }
}

#[rstest]
#[tokio::test]
async fn spawned_roots_do_not_cancel_sibling_reads(
    #[values(false, true)] chunk_task: bool,
) -> VortexResult<()> {
    let handle = TokioRuntime::current();
    let completed = Arc::new(AtomicUsize::new(0));
    let root = Box::new(SpawnReads {
        remaining: 2,
        completed: Arc::clone(&completed),
    });
    let scope = WorkScope {
        file_ordinal: 0,
        rows: 0..2,
    };
    let io = Arc::new(QueuedIo) as Arc<dyn IoService>;
    let map: MapBatch = Arc::new(|_| Err(vortex_err!("this test produces no batches")));
    if chunk_task {
        chunk_stream(&handle, root, scope, io, map)
            .try_for_each(|_| future::ready(Ok(())))
            .await?;
    } else {
        run_stream(handle, root, scope, io, map, true)
            .try_for_each(|_| future::ready(Ok(())))
            .await?;
    }
    assert_eq!(completed.load(Ordering::SeqCst), 2);
    Ok(())
}
