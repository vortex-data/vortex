// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;

use futures::StreamExt;
use futures::channel::oneshot;
use rstest::rstest;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::request::Completion;
use vortex::io::request::IoBatch;
use vortex::io::request::IoConsumer;
use vortex::io::request::IoIntent;
use vortex::io::request::IoOwnerId;
use vortex::io::request::IoRequest;
use vortex::io::request::IoRequestId;
use vortex::io::request::IoResult;
use vortex::io::request::IoSource;
use vortex::io::request::IoTarget;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::PlannerOutput;
use vortex::scan::planning::planner::State;
use vortex::scan::planning::planner::WorkScope;

use super::Yielded;
use super::run_stream;

struct NoIo;
impl IoSource for NoIo {
    fn submit(&self, _: IoOwnerId, _: IoBatch) -> VortexResult<()> {
        panic!("unexpected IO");
    }
    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }
    fn poll_completion(&self, _: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        panic!("unexpected IO wait");
    }
    fn wait(&self) -> VortexResult<Completion> {
        panic!("unexpected IO wait");
    }
    fn release(&self, _: IoOwnerId) {}
    fn clear(&self) {}
}

struct Outputs {
    outputs: VecDeque<VortexResult<PlannerOutput>>,
    computed: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}
impl IoConsumer for Outputs {
    fn set_io_result(&mut self, _: IoRequestId, _: IoResult) {
        panic!("unexpected IO result");
    }
}
impl Planner for Outputs {
    fn state(&self) -> State {
        State::NeedsCompute
    }
    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        self.computed.fetch_add(1, Ordering::SeqCst);
        self.outputs.pop_front().unwrap_or(Ok(PlannerOutput::Done))
    }
}
impl Drop for Outputs {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}
fn outputs(values: Vec<VortexResult<PlannerOutput>>) -> Outputs {
    Outputs {
        outputs: values.into(),
        computed: Arc::default(),
        dropped: Arc::default(),
    }
}
fn scope() -> WorkScope {
    WorkScope {
        file_ordinal: 0,
        rows: 0..1,
    }
}
fn root_stream(
    root: Outputs,
    inline: bool,
) -> futures::stream::BoxStream<'static, VortexResult<Yielded>> {
    run_stream(
        TokioRuntime::current(),
        Box::new(root),
        scope(),
        Arc::new(NoIo),
        Arc::new(|_| panic!("unexpected batch")),
        inline,
    )
}

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn root_task_handles_spawn_and_completion(#[case] inline: bool) -> VortexResult<()> {
    let child = outputs(vec![]);
    let computed = Arc::clone(&child.computed);
    let root = outputs(vec![Ok(PlannerOutput::Spawn(scope(), Box::new(child)))]);
    let mut stream = root_stream(root, inline);
    if !inline {
        assert!(matches!(
            stream.next().await.transpose()?,
            Some(Yielded::Spawn(_, _))
        ));
    }
    assert!(stream.next().await.is_none());
    assert_eq!(computed.load(Ordering::SeqCst), usize::from(inline));
    Ok(())
}

#[tokio::test]
async fn root_task_propagates_failure() -> VortexResult<()> {
    let mut stream = root_stream(outputs(vec![Err(vortex_err!("root failure"))]), false);
    let error = stream
        .next()
        .await
        .expect("expected failure")
        .err()
        .expect("expected error");
    assert!(error.to_string().contains("root failure"));
    assert!(stream.next().await.is_none());
    Ok(())
}

#[tokio::test]
async fn root_task_bounds_handoff_and_cancels_when_dropped() -> VortexResult<()> {
    let root = outputs(
        (0..100)
            .map(|_| Ok(PlannerOutput::Spawn(scope(), Box::new(outputs(vec![])))))
            .collect(),
    );
    let computed = Arc::clone(&root.computed);
    let dropped = Arc::clone(&root.dropped);
    let mut stream = root_stream(root, false);
    assert!(matches!(
        stream.next().await.transpose()?,
        Some(Yielded::Spawn(_, _))
    ));
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    // One consumed item, the bounded queue plus sender slot, and one pending send.
    assert!(computed.load(Ordering::SeqCst) <= 4);
    assert!(!dropped.load(Ordering::SeqCst));
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !dropped.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("root task did not release its planner");
    Ok(())
}

struct DelayedIo {
    ready: Mutex<oneshot::Receiver<()>>,
    request: Mutex<Option<(IoOwnerId, IoRequestId)>>,
    waiting: AtomicBool,
    cleared: AtomicBool,
}

impl IoSource for DelayedIo {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        assert_eq!(batch.len(), 1);
        *self.request.lock().expect("request lock") = Some((owner, batch[0].request));
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        self.waiting.store(true, Ordering::SeqCst);
        let mut ready = self.ready.lock().expect("receiver lock");
        match Pin::new(&mut *ready).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                result.expect("completion sender dropped");
                let (owner, request) = self
                    .request
                    .lock()
                    .expect("request lock")
                    .take()
                    .expect("missing request");
                Poll::Ready(Ok(Completion {
                    owner,
                    request,
                    result: Ok(IoResult::Size(1)),
                }))
            }
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        panic!("unexpected blocking IO");
    }
    fn release(&self, _: IoOwnerId) {}
    fn clear(&self) {
        self.request.lock().expect("request lock").take();
        self.cleared.store(true, Ordering::SeqCst);
    }
}

struct WaitForSize {
    issued: bool,
    delivered: Arc<AtomicBool>,
}

impl IoConsumer for WaitForSize {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        assert_eq!(request, IoRequestId(0));
        assert!(matches!(result, IoResult::Size(1)));
        self.delivered.store(true, Ordering::SeqCst);
    }
}

impl Planner for WaitForSize {
    fn state(&self) -> State {
        if self.issued && !self.delivered.load(Ordering::SeqCst) {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }
    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.issued {
            return Ok(PlannerOutput::Done);
        }
        self.issued = true;
        Ok(PlannerOutput::NeedsIO(vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::Size,
        }]))
    }
}

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn root_task_waits_for_io_and_clears_it_on_cancellation(
    #[case] cancel: bool,
) -> VortexResult<()> {
    let (send, recv) = oneshot::channel();
    let source = Arc::new(DelayedIo {
        ready: Mutex::new(recv),
        request: Mutex::new(None),
        waiting: AtomicBool::new(false),
        cleared: AtomicBool::new(false),
    });
    let delivered = Arc::new(AtomicBool::new(false));
    let mut stream = run_stream(
        TokioRuntime::current(),
        Box::new(WaitForSize {
            issued: false,
            delivered: Arc::clone(&delivered),
        }),
        scope(),
        Arc::clone(&source) as Arc<dyn IoSource>,
        Arc::new(|_| panic!("unexpected batch")),
        false,
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while !source.waiting.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("root task never waited for IO");
    assert!(!delivered.load(Ordering::SeqCst));
    if cancel {
        drop(stream);
    } else {
        send.send(()).expect("completion receiver dropped");
        assert!(
            tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .expect("root task did not wake")
                .is_none()
        );
        assert!(delivered.load(Ordering::SeqCst));
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        while !source.cleared.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("root task did not clear its IO scope");
    Ok(())
}
