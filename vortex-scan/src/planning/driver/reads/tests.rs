// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::task::AtomicWaker;
use rstest::rstest;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_io::request::IoConsumer;

use super::*;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

#[derive(Default)]
struct Control {
    reads: AtomicUsize,
    announcements: AtomicUsize,
    clears: AtomicUsize,
    ready: AtomicBool,
    fail: AtomicBool,
    sources: Mutex<Vec<Weak<TestSource>>>,
}

impl Control {
    fn complete(&self) {
        self.ready.store(true, Ordering::SeqCst);
        for source in self.sources.lock().iter().filter_map(Weak::upgrade) {
            source.waker.wake();
        }
    }
}

struct TestService(Arc<Control>);

impl IoService for TestService {
    fn session(&self) -> Arc<dyn IoSource> {
        let source = Arc::new(TestSource {
            control: Arc::clone(&self.0),
            ready: Mutex::default(),
            waker: AtomicWaker::new(),
        });
        self.0.sources.lock().push(Arc::downgrade(&source));
        source
    }
}

struct TestSource {
    control: Arc<Control>,
    ready: Mutex<VecDeque<Completion>>,
    waker: AtomicWaker,
}

impl IoSource for TestSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        for request in batch {
            if request.intent == IoIntent::Announce {
                self.control.announcements.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            self.control.reads.fetch_add(1, Ordering::SeqCst);
            self.ready.lock().push_back(Completion {
                owner,
                request: request.request,
                result: if self.control.fail.load(Ordering::SeqCst) {
                    Err(vortex_err!("transport failure"))
                } else {
                    Ok(IoResult::Bytes(BufferHandle::new_host(ByteBuffer::from(
                        vec![7; 4],
                    ))))
                },
            });
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(if self.control.ready.load(Ordering::SeqCst) {
            self.ready.lock().pop_front()
        } else {
            None
        })
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        self.waker.register(cx.waker());
        match self.poll()? {
            Some(completion) => Poll::Ready(Ok(completion)),
            None => Poll::Pending,
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        self.poll()?
            .ok_or_else(|| vortex_err!("test source is not ready"))
    }

    fn release(&self, owner: IoOwnerId) {
        self.ready
            .lock()
            .retain(|completion| completion.owner != owner);
    }

    fn clear(&self) {
        self.control.clears.fetch_add(1, Ordering::SeqCst);
        self.ready.lock().clear();
    }
}

fn file(control: &Arc<Control>) -> FileReads {
    FileReads::new(
        Arc::new(TestService(Arc::clone(control))),
        Arc::new(
            [
                (IoTarget::range(0, 4), ReadLifetime::ThroughRow(20)),
                (IoTarget::range(4, 4), ReadLifetime::ThroughRow(40)),
                (IoTarget::range(8, 4), ReadLifetime::File),
            ]
            .into_iter()
            .collect(),
        ),
    )
}

fn request(intent: IoIntent, id: u32, offset: u64) -> IoRequest {
    IoRequest {
        intent,
        request: IoRequestId(id),
        target: IoTarget::range(offset, 4),
    }
}

struct Hold {
    _claim: RowClaim,
}

impl IoConsumer for Hold {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Hold {
    fn state(&self) -> State {
        State::NeedsCompute
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(PlannerOutput::Done)
    }
}

#[test]
fn completed_reads_expire_at_the_earliest_unfinished_row() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let mut third = file.claim(0..60);
    let first = third.split_to(20);
    let second = third.split_to(40);
    let session = file.session();
    for (id, offset) in [0, 4, 8].into_iter().enumerate() {
        session.submit(
            IoOwnerId(0),
            vec![request(IoIntent::Fetch, u32::try_from(id)?, offset)],
        )?;
        session.wait()?.result?;
    }
    drop(second);
    assert_eq!(control.clears.load(Ordering::SeqCst), 0);
    drop(first);
    assert_eq!(control.clears.load(Ordering::SeqCst), 2);
    drop(third);
    assert_eq!(control.clears.load(Ordering::SeqCst), 3);
    // The engine may still hold the service and session after all the file's work has ended.
    assert!(file.state.lock().entries.is_empty());
    assert!(session.poll()?.is_none());
    Ok(())
}

#[test]
fn sequential_sessions_reuse_reads_until_the_region_finishes() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let work = file.claim(0..20);
    for id in 0..3 {
        let session = file.session();
        session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, id, 0)])?;
        assert_eq!(session.wait()?.request, IoRequestId(id));
        session.clear();
    }
    assert_eq!(control.reads.load(Ordering::SeqCst), 1);
    drop(work);
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    Ok(())
}

#[rstest]
fn prefetch_and_fetch_share_one_read(#[values(false, true)] prefetch: bool) -> VortexResult<()> {
    let control = Arc::new(Control::default());
    let file = file(&control);
    let _work = file.claim(0..20);
    let first = file.session();
    let second = file.session();
    let intent = if prefetch {
        IoIntent::Prefetch
    } else {
        IoIntent::Fetch
    };
    first.submit(IoOwnerId(0), vec![request(intent, 0, 0)])?;
    second.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
    assert_eq!(control.reads.load(Ordering::SeqCst), 1);
    assert!(first.poll()?.is_none());
    assert!(second.poll()?.is_none());
    // Cancelling the initiating session must not cancel the other session's shared read.
    first.clear();
    control.complete();
    second.wait()?.result?;
    assert!(first.poll()?.is_none());
    Ok(())
}

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn completion_wakes_every_worker_and_preserves_request_identity() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    let file = file(&control);
    let _work = file.claim(0..20);
    let workers: Vec<_> = (0..2)
        .map(|_| (file.session(), Arc::new(WakeCount::default())))
        .collect();
    for (index, (session, wake)) in workers.iter().enumerate() {
        session.submit(
            IoOwnerId(9),
            vec![request(IoIntent::Fetch, u32::try_from(index)?, 0)],
        )?;
        let waker = Waker::from(Arc::clone(wake));
        assert!(
            session
                .poll_completion(&mut Context::from_waker(&waker))
                .is_pending()
        );
        wake.0.store(0, Ordering::SeqCst);
    }
    control.complete();
    for (index, (session, wake)) in workers.iter().enumerate() {
        assert!(wake.0.load(Ordering::SeqCst) > 0);
        let completion = session.wait()?;
        assert_eq!(completion.owner, IoOwnerId(9));
        assert_eq!(completion.request, IoRequestId(u32::try_from(index)?));
        completion.result?;
    }
    assert_eq!(control.reads.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn read_errors_reach_every_subscriber() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.fail.store(true, Ordering::SeqCst);
    control.complete();
    let file = file(&control);
    let _work = file.claim(0..20);
    for _ in 0..2 {
        let session = file.session();
        session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
        let error = session.wait()?.result.expect_err("transport fails");
        assert!(error.to_string().contains("transport failure"));
    }
    assert_eq!(control.reads.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn unused_announcements_expire_with_their_region() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    let file = file(&control);
    let mut remaining = file.claim(0..40);
    let first = remaining.split_to(20);
    let session = file.session();
    session.submit(
        IoOwnerId(0),
        vec![
            request(IoIntent::Announce, 0, 0),
            request(IoIntent::Announce, 1, 0),
        ],
    )?;
    assert_eq!(control.announcements.load(Ordering::SeqCst), 2);
    assert_eq!(control.reads.load(Ordering::SeqCst), 0);
    drop(first);
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    Ok(())
}

struct Producer(RowClaim);

impl IoConsumer for Producer {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Producer {
    fn state(&self) -> State {
        if self.0.rows().is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let end = self.0.rows().start + 20;
        let claim = self.0.split_to(end);
        let rows = claim.rows().clone();
        Ok(PlannerOutput::Spawn(
            WorkScope {
                file_ordinal: 0,
                rows,
            },
            Box::new(Hold { _claim: claim }),
        ))
    }
}

#[test]
fn queued_children_keep_reads_live_across_run_handoffs() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let root = Box::new(Producer(file.claim(0..40)));
    let io = file.session();
    io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
    io.wait()?.result?;
    let mut producer = Run::new();
    producer.admit(
        root,
        WorkScope {
            file_ordinal: 0,
            rows: 0..40,
        },
        io,
    );
    let Progress::Spawn(first_scope, first) = producer.advance()? else {
        panic!("first child")
    };
    let Progress::Spawn(_, second) = producer.advance()? else {
        panic!("second child")
    };
    drop(producer);
    assert_eq!(control.clears.load(Ordering::SeqCst), 0);
    // The first child remains registered while in the engine's queue, then on another run.
    let mut worker = Run::new();
    worker.admit(first, first_scope, file.session());
    assert!(matches!(worker.advance()?, Progress::RootDone(_)));
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    drop(second);
    assert!(file.state.lock().entries.is_empty());
    Ok(())
}

#[test]
fn executions_do_not_share_identical_byte_ranges() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    for _ in 0..2 {
        let file = file(&control);
        let _work = file.claim(0..20);
        let session = file.session();
        session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
        session.wait()?.result?;
    }
    assert_eq!(control.reads.load(Ordering::SeqCst), 2);
    assert_eq!(control.clears.load(Ordering::SeqCst), 2);
    Ok(())
}

#[rstest]
fn cancellation_releases_pending_reads(#[values(false, true)] cancel: bool) -> VortexResult<()> {
    let control = Arc::new(Control::default());
    let file = file(&control);
    let root = Box::new(Hold {
        _claim: file.claim(0..20),
    });
    let session = file.session();
    session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
    let mut run = Run::new();
    let root = run.admit(
        root,
        WorkScope {
            file_ordinal: 0,
            rows: 0..20,
        },
        Arc::clone(&session),
    );
    if cancel {
        run.cancel(root);
    } else {
        drop(run);
    }
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    assert!(file.state.lock().entries.is_empty());
    assert!(session.poll()?.is_none());
    Ok(())
}

#[test]
fn releasing_one_owner_preserves_other_subscriptions() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    let file = file(&control);
    let _work = file.claim(0..20);
    let session = file.session();
    for owner in 0..2 {
        session.submit(IoOwnerId(owner), vec![request(IoIntent::Fetch, 0, 0)])?;
    }
    session.release(IoOwnerId(0));
    control.complete();
    let completion = session.wait()?;
    assert_eq!(completion.owner, IoOwnerId(1));
    completion.result?;
    assert!(session.poll()?.is_none());
    assert_eq!(control.reads.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn unknown_lifetimes_are_bounded_by_file_completion() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let mut second = file.claim(0..40);
    let first = second.split_to(20);
    let session = file.session();
    session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 12)])?;
    session.wait()?.result?;
    drop(first);
    assert_eq!(control.clears.load(Ordering::SeqCst), 0);
    drop(second);
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn splitting_transfers_rows_without_releasing_their_reads() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let mut remaining = file.claim(0..40);
    let session = file.session();
    session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
    session.wait()?.result?;

    let child = remaining.split_to(20);
    assert_eq!(child.rows(), &(0..20));
    assert_eq!(remaining.rows(), &(20..40));
    assert_eq!(control.clears.load(Ordering::SeqCst), 0);
    // Moving the claim between stages changes neither registration nor cached reads.
    let next_stage = child;
    drop(remaining);
    assert_eq!(control.clears.load(Ordering::SeqCst), 0);
    drop(next_stage);
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn restricting_a_claim_releases_skipped_rows() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let mut claim = file.claim(0..40);
    let session = file.session();
    for offset in [0, 8] {
        session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, offset)])?;
        session.wait()?.result?;
    }
    claim.restrict_to(20..35);
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    claim.restrict_to(35..35);
    assert_eq!(control.clears.load(Ordering::SeqCst), 2);
    assert!(file.state.lock().claims.is_empty());
    drop(claim);
    assert_eq!(control.clears.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn overlapping_consumers_keep_independent_claims() -> VortexResult<()> {
    let control = Arc::new(Control::default());
    control.complete();
    let file = file(&control);
    let mut first = file.claim(0..40);
    let second = first.share();
    let session = file.session();
    session.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0)])?;
    session.wait()?.result?;
    first.restrict_to(20..40);
    assert_eq!(control.clears.load(Ordering::SeqCst), 0);
    drop(second);
    assert_eq!(control.clears.load(Ordering::SeqCst), 1);
    assert_eq!(first.rows(), &(20..40));
    Ok(())
}

#[rstest]
#[case::empty_prefix(0, 1)]
#[case::two_ranges(20, 2)]
#[case::empty_suffix(40, 1)]
fn split_boundaries_register_only_nonempty_claims(#[case] end: u64, #[case] count: usize) {
    let control = Arc::new(Control::default());
    let file = file(&control);
    let mut remaining = file.claim(0..40);
    let child = remaining.split_to(end);
    assert_eq!(child.rows(), &(0..end));
    assert_eq!(remaining.rows(), &(end..40));
    assert_eq!(file.state.lock().claims.values().sum::<usize>(), count);
    drop(child);
    drop(remaining);
    assert!(file.state.lock().claims.is_empty());
}

#[rstest]
#[case::before_start(9)]
#[case::after_end(41)]
#[should_panic(expected = "split must lie within the claim")]
fn splits_outside_the_claim_are_rejected(#[case] end: u64) {
    let file = file(&Arc::new(Control::default()));
    let mut claim = file.claim(10..40);
    let _child = claim.split_to(end);
}
