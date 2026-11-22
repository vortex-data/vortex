// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Waker;

use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::assert_arrays_eq;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::request::IoRequest;

use super::*;
use crate::execution::planning::tests::scripted::Log;
use crate::execution::planning::tests::scripted::MorselStep;
use crate::execution::planning::tests::scripted::PlannerStep;
use crate::execution::planning::tests::scripted::RecordingIoSource;
use crate::execution::planning::tests::scripted::ScriptedPlanner;
use crate::execution::planning::tests::scripted::array_of;
use crate::execution::planning::tests::scripted::ctx;
use crate::execution::planning::tests::scripted::scope;

fn request(id: u32, offset: u64) -> IoRequest {
    IoRequest {
        intent: IoIntent::Fetch,
        request: IoRequestId(id),
        target: IoTarget::range(offset, 4),
    }
}

fn driver(source: &Arc<RecordingIoSource>) -> Driver {
    Driver::new(Arc::clone(source) as Arc<dyn IoSource>).with_step_limit(1_000)
}

fn source() -> Arc<RecordingIoSource> {
    let source = RecordingIoSource::default();
    source.canned_bytes(0, buffer![0u8, 1, 2, 3].into_byte_buffer());
    source.canned_bytes(4, buffer![4u8, 5, 6, 7].into_byte_buffer());
    Arc::new(source)
}

fn lifo_source() -> Arc<RecordingIoSource> {
    let source = RecordingIoSource::lifo();
    source.canned_bytes(0, buffer![0u8, 1, 2, 3].into_byte_buffer());
    source.canned_bytes(4, buffer![4u8, 5, 6, 7].into_byte_buffer());
    Arc::new(source)
}

fn one_batch_morsel(scope: WorkScope, rows: u64) -> PlannerStep {
    PlannerStep::Morsel(
        scope,
        vec![
            MorselStep::Compute(MorselOutput::Batch(array_of(rows))),
            MorselStep::Compute(MorselOutput::Done),
        ],
    )
}

#[test]
fn root_done_produces_nothing() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::boxed("root", vec![PlannerStep::Done], &log);
    let batches = driver(&source()).run(root)?;
    assert!(batches.is_empty());
    assert_eq!(log.events(), vec!["compute root"]);
    Ok(())
}

#[test]
fn one_morsel_one_batch() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![one_batch_morsel(scope(10..20), 3), PlannerStep::Done],
        &log,
    );
    let batches = driver(&source()).run(root)?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].scope, scope(10..20));
    assert_arrays_eq!(batches[0].array, array_of(3), &mut ctx());
    Ok(())
}

#[test]
fn two_morsels_emit_in_order() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            one_batch_morsel(scope(0..1), 1),
            one_batch_morsel(scope(1..3), 2),
            PlannerStep::Done,
        ],
        &log,
    );
    let batches = driver(&source()).run(root)?;
    let scopes: Vec<_> = batches.iter().map(|batch| batch.scope.clone()).collect();
    assert_eq!(scopes, vec![scope(0..1), scope(1..3)]);
    assert_arrays_eq!(batches[0].array, array_of(1), &mut ctx());
    assert_arrays_eq!(batches[1].array, array_of(2), &mut ctx());
    Ok(())
}

#[test]
fn dfs_finishes_child_subtree_before_parent_continues() -> VortexResult<()> {
    let log = Log::default();
    let child = PlannerStep::Planner(
        scope(0..5),
        vec![one_batch_morsel(scope(0..5), 1), PlannerStep::Done],
    );
    let root = ScriptedPlanner::boxed(
        "parent",
        vec![
            child,
            PlannerStep::Continue,
            one_batch_morsel(scope(5..10), 2),
            PlannerStep::Done,
        ],
        &log,
    );
    let batches = driver(&source()).run(root)?;
    let scopes: Vec<_> = batches.iter().map(|batch| batch.scope.clone()).collect();
    assert_eq!(scopes, vec![scope(0..5), scope(5..10)]);
    assert_eq!(
        log.events(),
        vec![
            "compute parent",
            "compute child",
            "compute morsel",
            "compute morsel",
            "compute child",
            "compute parent",
            "compute parent",
            "compute morsel",
            "compute morsel",
            "compute parent",
        ]
    );
    Ok(())
}

#[test]
fn needs_io_delivers_every_request_in_order() -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Io(vec![request(0, 4), request(1, 0)]),
            PlannerStep::Done,
        ],
        &log,
    );
    let batches = driver(&source).run(root)?;
    assert!(batches.is_empty());
    assert_eq!(
        source.performed(),
        vec![IoTarget::range(4, 4), IoTarget::range(0, 4)]
    );
    assert_eq!(
        log.events(),
        vec![
            "publish root",
            "deliver root 0 bytes",
            "deliver root 1 bytes",
            "compute root",
        ]
    );
    Ok(())
}

/// A fetch published while CPU work is left is performed once, and delivered when the planner
/// gets round to waiting for it.
#[test]
fn fetch_published_ahead_is_delivered_when_awaited() -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::NeedsIO(vec![request(0, 0)]),
            PlannerStep::Continue,
            PlannerStep::Await(vec![IoRequestId(0)]),
            PlannerStep::Done,
        ],
        &log,
    );
    driver(&source).run(root)?;
    assert_eq!(source.performed(), vec![IoTarget::range(0, 4)]);
    assert_eq!(
        log.events(),
        vec![
            "compute root",
            "compute root",
            "deliver root 0 bytes",
            "compute root",
        ]
    );
    Ok(())
}

#[test]
fn continue_requeues() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Continue,
            PlannerStep::Continue,
            PlannerStep::Done,
        ],
        &log,
    );
    driver(&source()).run(root)?;
    assert_eq!(
        log.events()
            .iter()
            .filter(|event| event.as_str() == "compute root")
            .count(),
        3
    );
    Ok(())
}

#[test]
fn io_failure_propagates_after_two_performs() -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    source.fail(IoTarget::range(4, 4), "disk on fire");
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Io(vec![request(0, 0), request(1, 4)]),
            PlannerStep::Done,
        ],
        &log,
    );
    let err = driver(&source).run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains("disk on fire")),
        "{err:?}"
    );
    assert_eq!(source.performed().len(), 2);
    Ok(())
}

#[test]
fn completions_out_of_order_within_one_batch() -> VortexResult<()> {
    let log = Log::default();
    let source = lifo_source();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Io(vec![request(0, 0), request(1, 4)]),
            PlannerStep::Done,
        ],
        &log,
    );
    driver(&source).run(root)?;
    assert_eq!(
        source.performed(),
        vec![IoTarget::range(0, 4), IoTarget::range(4, 4)],
        "submitted in request order"
    );
    assert_eq!(
        log.events(),
        vec![
            "publish root",
            "deliver root 1 bytes",
            "deliver root 0 bytes",
            "compute root",
        ]
    );
    Ok(())
}

#[test]
fn completions_out_of_order_across_parked_items() -> VortexResult<()> {
    let log = Log::default();
    let source = lifo_source();
    let child = |id: u32, offset: u64| {
        PlannerStep::Planner(
            scope(0..1),
            vec![
                PlannerStep::Io(vec![request(id, offset)]),
                PlannerStep::Done,
            ],
        )
    };
    let root = ScriptedPlanner::boxed(
        "root",
        vec![child(5, 0), child(6, 4), PlannerStep::Done],
        &log,
    );
    driver(&source).run(root)?;
    let deliveries: Vec<_> = log
        .events()
        .into_iter()
        .filter(|event| event.starts_with("deliver"))
        .collect();
    assert_eq!(
        deliveries,
        vec!["deliver child 6 bytes", "deliver child 5 bytes"],
        "both children were parked before either completed"
    );
    Ok(())
}

#[test]
fn lying_source_fails_before_delivery() -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    source.answer_with_size(IoTarget::range(0, 4), 4);
    let root = ScriptedPlanner::boxed(
        "root",
        vec![PlannerStep::Io(vec![request(0, 0)]), PlannerStep::Done],
        &log,
    );
    let err = driver(&source).run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains("answered")),
        "{err:?}"
    );
    assert_eq!(log.events(), vec!["publish root"]);
    Ok(())
}

#[test]
fn morsel_needs_io_midway() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Morsel(
                scope(0..4),
                vec![
                    MorselStep::Compute(MorselOutput::Batch(array_of(1))),
                    MorselStep::Io(vec![request(0, 0)]),
                    MorselStep::Compute(MorselOutput::Batch(array_of(2))),
                    MorselStep::Compute(MorselOutput::Done),
                ],
            ),
            PlannerStep::Done,
        ],
        &log,
    );
    let batches = driver(&source()).run(root)?;
    assert_eq!(batches.len(), 2);
    assert_arrays_eq!(batches[0].array, array_of(1), &mut ctx());
    assert_arrays_eq!(batches[1].array, array_of(2), &mut ctx());
    Ok(())
}

#[rstest]
#[case::empty_batch(
    vec![PlannerStep::Morsel(scope(0..1), vec![MorselStep::Compute(MorselOutput::Batch(array_of(0)))])],
    "empty batch"
)]
#[case::empty_io(vec![PlannerStep::Io(vec![])], "empty batch")]
#[case::republished_fetch(
    vec![
        PlannerStep::NeedsIO(vec![request(0, 0)]),
        PlannerStep::NeedsIO(vec![request(0, 0)]),
    ],
    "published while it is outstanding"
)]
#[case::waiting_for_nothing(vec![PlannerStep::Wait], "waits with no outstanding fetch")]
#[case::still_waiting_after_the_last_delivery(
    vec![PlannerStep::Io(vec![request(0, 0)]), PlannerStep::Wait],
    "waits with no outstanding fetch"
)]
fn protocol_errors(#[case] mut steps: Vec<PlannerStep>, #[case] message: &str) -> VortexResult<()> {
    let log = Log::default();
    steps.push(PlannerStep::Done);
    let root = ScriptedPlanner::boxed("root", steps, &log);
    let err = driver(&source()).run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains(message)),
        "{err:?}"
    );
    Ok(())
}

#[test]
fn step_limit_is_enforced() -> VortexResult<()> {
    let log = Log::default();
    let mut steps: Vec<_> = (0..100).map(|_| PlannerStep::Continue).collect();
    steps.push(PlannerStep::Done);
    let root = ScriptedPlanner::boxed("root", steps, &log);
    let driver = Driver::new(source() as Arc<dyn IoSource>).with_step_limit(10);
    let err = driver.run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref()
            .is_some_and(|m| m.contains("step limit of 10")),
        "{err:?}"
    );
    Ok(())
}

#[rstest]
#[case(IoIntent::Announce)]
#[case(IoIntent::Prefetch)]
fn optional_publication_does_not_park_or_receive_bytes(
    #[case] intent: IoIntent,
) -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    let mut optional = request(0, 0);
    optional.intent = intent;
    let root = ScriptedPlanner::boxed(
        "warm",
        vec![
            PlannerStep::NeedsIO(vec![optional]),
            one_batch_morsel(scope(0..1), 1),
            PlannerStep::Done,
        ],
        &log,
    );
    let batches = driver(&source).run(root)?;
    assert_eq!(batches.len(), 1);
    let submitted = source.submissions();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].1.intent, intent);
    assert!(source.performed().is_empty());
    assert!(
        !log.events()
            .iter()
            .any(|event| event.starts_with("deliver"))
    );
    Ok(())
}

#[test]
fn optional_requests_cannot_be_wait_dependencies() -> VortexResult<()> {
    let source = source();
    let mut optional = request(0, 0);
    optional.intent = IoIntent::Announce;
    let root = ScriptedPlanner::boxed(
        "root",
        vec![PlannerStep::Io(vec![optional])],
        &Log::default(),
    );
    let error = driver(&source)
        .run(root)
        .err()
        .map(|error| error.to_string());
    assert!(error.is_some_and(|error| error.contains("waits with no outstanding fetch")));
    assert!(source.performed().is_empty());
    Ok(())
}

#[test]
fn optional_registration_can_be_promoted_to_fetch() -> VortexResult<()> {
    let source = source();
    let mut optional = request(0, 0);
    optional.intent = IoIntent::Announce;
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::NeedsIO(vec![optional]),
            PlannerStep::Io(vec![request(0, 0)]),
            PlannerStep::Done,
        ],
        &Log::default(),
    );
    driver(&source).run(root)?;
    assert_eq!(source.submissions().len(), 2);
    assert_eq!(source.performed().len(), 1);
    Ok(())
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Run>();
};

fn ready_root(name: &'static str, log: &Log, rows: Range<u64>) -> Box<dyn Planner> {
    ScriptedPlanner::boxed(
        name,
        vec![one_batch_morsel(scope(rows), 1), PlannerStep::Done],
        log,
    )
}

fn reading_root(name: &'static str, log: &Log, rows: Range<u64>) -> Box<dyn Planner> {
    ScriptedPlanner::boxed(
        name,
        vec![
            PlannerStep::Io(vec![request(0, 0)]),
            one_batch_morsel(scope(rows), 2),
            PlannerStep::Done,
        ],
        log,
    )
}

/// Drains `run` until it waits or idles, describing what it handed back.
fn drain(run: &mut Run) -> VortexResult<Vec<String>> {
    let mut events = Vec::new();
    loop {
        events.push(match run.advance()? {
            Progress::Batch(batch) => format!("batch {:?} {:?}", batch.root, batch.scope.rows),
            Progress::RootDone(root) => format!("done {root:?}"),
            Progress::Spawn(scope, _) => format!("spawn {:?}", scope.rows),
            Progress::Waiting => {
                events.push("waiting".to_string());
                return Ok(events);
            }
            Progress::Idle => {
                events.push("idle".to_string());
                return Ok(events);
            }
        });
    }
}

/// Roots run side by side: a root with ready work finishes while another waits for its read, and
/// each root reads through, and finally clears, its own session.
#[test]
fn roots_finish_independently() -> VortexResult<()> {
    let log = Log::default();
    let (reading_io, ready_io) = (source(), source());
    let mut run = Run::new().with_step_limit(1_000);
    let reading = run.admit(
        reading_root("reading", &log, 0..2),
        scope(0..2),
        Arc::clone(&reading_io) as Arc<dyn IoSource>,
    );
    let ready = run.admit(
        ready_root("ready", &log, 2..3),
        scope(2..3),
        Arc::clone(&ready_io) as Arc<dyn IoSource>,
    );
    assert_eq!((reading, ready), (RootId(0), RootId(1)));
    assert_eq!(run.live_roots(), 2);

    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(1) 2..3", "done RootId(1)", "waiting"]
    );
    assert_eq!(run.live_roots(), 1);
    assert_eq!((reading_io.clears(), ready_io.clears()), (0, 1));
    assert!(ready_io.submissions().is_empty());

    run.complete(reading_io.wait()?)?;
    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(0) 0..2", "done RootId(0)", "idle"]
    );
    assert_eq!(run.live_roots(), 0);
    assert_eq!(reading_io.clears(), 1);
    Ok(())
}

#[test]
fn poll_completion_delivers_from_every_waiting_root() -> VortexResult<()> {
    let log = Log::default();
    let sources = [source(), source()];
    let mut run = Run::new().with_step_limit(1_000);
    for (index, io) in sources.iter().enumerate() {
        let rows = index as u64..index as u64 + 1;
        run.admit(
            reading_root("root", &log, rows.clone()),
            scope(rows),
            Arc::clone(io) as Arc<dyn IoSource>,
        );
    }
    assert_eq!(drain(&mut run)?, vec!["waiting"]);
    let mut cx = Context::from_waker(Waker::noop());
    assert!(matches!(run.poll_completion(&mut cx), Poll::Ready(Ok(()))));
    let mut events = drain(&mut run)?;
    events.sort();
    assert_eq!(
        events,
        vec![
            "batch RootId(0) 0..1",
            "batch RootId(1) 1..2",
            "done RootId(0)",
            "done RootId(1)",
            "idle",
        ]
    );
    Ok(())
}

#[test]
fn cancel_drops_a_roots_work_and_clears_its_session() -> VortexResult<()> {
    let log = Log::default();
    let (cancelled_io, kept_io) = (source(), source());
    let mut run = Run::new().with_step_limit(1_000);
    let cancelled = run.admit(
        reading_root("cancelled", &log, 0..2),
        scope(0..2),
        Arc::clone(&cancelled_io) as Arc<dyn IoSource>,
    );
    run.admit(
        reading_root("kept", &log, 2..4),
        scope(2..4),
        Arc::clone(&kept_io) as Arc<dyn IoSource>,
    );
    assert_eq!(drain(&mut run)?, vec!["waiting"]);

    run.cancel(cancelled);
    assert_eq!(run.live_roots(), 1);
    assert_eq!(cancelled_io.clears(), 1);
    run.complete(kept_io.wait()?)?;
    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(1) 2..4", "done RootId(1)", "idle"]
    );
    assert!(
        !log.events()
            .contains(&"deliver cancelled 0 bytes".to_string())
    );
    Ok(())
}

#[test]
fn dropping_a_run_clears_every_live_session() -> VortexResult<()> {
    let log = Log::default();
    let io = source();
    let mut run = Run::new();
    run.admit(
        reading_root("root", &log, 0..2),
        scope(0..2),
        Arc::clone(&io) as Arc<dyn IoSource>,
    );
    assert_eq!(drain(&mut run)?, vec!["waiting"]);
    drop(run);
    assert_eq!(io.clears(), 1);
    Ok(())
}

/// Waiting for M fetches costs the driver one delivery each, not a visit that re-lists the rest.
#[test]
fn a_waiting_item_is_not_revisited_per_delivery() -> VortexResult<()> {
    const FETCHES: u32 = 64;
    let log = Log::default();
    let io = RecordingIoSource::default();
    let batch: Vec<IoRequest> = (0..FETCHES)
        .map(|id| {
            io.canned_bytes(u64::from(id) * 4, buffer![0u8; 4].into_byte_buffer());
            request(id, u64::from(id) * 4)
        })
        .collect();
    let io = Arc::new(io);
    let root = ScriptedPlanner::boxed(
        "root",
        vec![PlannerStep::Io(batch), PlannerStep::Done],
        &log,
    );
    // Publish and the final compute: the deliveries in between cost no visit.
    Driver::new(Arc::clone(&io) as Arc<dyn IoSource>)
        .with_step_limit(2)
        .run(root)?;
    assert_eq!(io.performed().len(), FETCHES as usize);
    Ok(())
}

#[test]
fn waking_subtree_precedes_an_already_discovered_sibling() -> VortexResult<()> {
    let log = Log::default();
    let io = source();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Planner(
                scope(0..1),
                vec![
                    PlannerStep::Io(vec![request(0, 0)]),
                    one_batch_morsel(scope(0..1), 1),
                    PlannerStep::Done,
                ],
            ),
            PlannerStep::Morsel(
                scope(1..2),
                vec![
                    MorselStep::Compute(MorselOutput::Batch(array_of(1))),
                    MorselStep::Compute(MorselOutput::Batch(array_of(1))),
                    MorselStep::Compute(MorselOutput::Done),
                ],
            ),
            PlannerStep::Done,
        ],
        &log,
    );
    let mut run = Run::new().with_step_limit(1_000);
    run.admit(root, scope(0..2), Arc::clone(&io) as Arc<dyn IoSource>);
    let Progress::Batch(batch) = run.advance()? else {
        panic!("the later sibling should run while the first child waits");
    };
    assert_eq!(batch.scope, scope(1..2));

    run.complete(io.wait()?)?;
    // The waking child creates a grandchild after the sibling already exists. Its subtree
    // must still win, including over the sibling's continuation after its first batch.
    let Progress::Batch(batch) = run.advance()? else {
        panic!("the waking subtree should produce a batch first");
    };
    assert_eq!(batch.scope, scope(0..1));
    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(0) 1..2", "done RootId(0)", "idle"]
    );
    Ok(())
}

#[test]
fn runnable_roots_follow_admission_order() -> VortexResult<()> {
    let log = Log::default();
    let mut run = Run::new().with_step_limit(1_000);
    for rows in [0..1, 1..2] {
        let root = ScriptedPlanner::boxed(
            "root",
            vec![
                PlannerStep::Continue,
                one_batch_morsel(scope(rows.clone()), 1),
                PlannerStep::Done,
            ],
            &log,
        );
        run.admit(root, scope(rows), source());
    }
    assert_eq!(
        drain(&mut run)?,
        vec![
            "batch RootId(0) 0..1",
            "done RootId(0)",
            "batch RootId(1) 1..2",
            "done RootId(1)",
            "idle",
        ]
    );
    Ok(())
}

/// Makes the recording source's completions available to nonblocking polling.
struct PollingSource {
    inner: Arc<RecordingIoSource>,
    outstanding: AtomicUsize,
}

impl IoSource for PollingSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        self.outstanding.fetch_add(
            batch
                .iter()
                .filter(|request| request.intent == IoIntent::Fetch)
                .count(),
            Ordering::Relaxed,
        );
        self.inner.submit(owner, batch)
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        if self.outstanding.load(Ordering::Relaxed) == 0 {
            return Ok(None);
        }
        self.wait().map(Some)
    }

    fn poll_completion(&self, _cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        Poll::Ready(self.wait())
    }

    fn wait(&self) -> VortexResult<Completion> {
        let completion = self.inner.wait()?;
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        Ok(completion)
    }

    fn release(&self, owner: IoOwnerId) {
        self.inner.release(owner);
    }

    fn clear(&self) {
        self.outstanding.store(0, Ordering::Relaxed);
        self.inner.clear();
    }
}

#[test]
fn polls_io_while_a_later_branch_keeps_computing() -> VortexResult<()> {
    let log = Log::default();
    let io = Arc::new(PollingSource {
        inner: source(),
        outstanding: AtomicUsize::new(0),
    });
    let mut run = Run::new().with_step_limit(1_000);
    run.admit(reading_root("earlier", &log, 0..1), scope(0..1), io);
    let mut script: Vec<_> = (0..IO_POLL_INTERVAL * 2)
        .map(|_| PlannerStep::Continue)
        .collect();
    script.push(one_batch_morsel(scope(1..2), 1));
    script.push(PlannerStep::Done);
    run.admit(
        ScriptedPlanner::boxed("later", script, &log),
        scope(1..2),
        source(),
    );
    let Progress::Batch(batch) = run.advance()? else {
        panic!("ready IO should wake the earlier root before the later root finishes");
    };
    assert_eq!(batch.root, RootId(0));
    let later_visits = log
        .events()
        .iter()
        .filter(|event| event.as_str() == "compute later")
        .count();
    assert!(later_visits > 0 && later_visits <= IO_POLL_INTERVAL);
    Ok(())
}

/// Planner that spawns one independent root, then performs a fetch and emits a batch.
struct SpawningRoot {
    spawned: Option<(WorkScope, Box<dyn Planner>)>,
    inner: Box<dyn Planner>,
}

impl IoConsumer for SpawningRoot {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.inner.set_io_result(request, result);
    }
}

impl Planner for SpawningRoot {
    fn state(&self) -> State {
        if self.spawned.is_some() {
            State::NeedsCompute
        } else {
            self.inner.state()
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some((scope, child)) = self.spawned.take() {
            return Ok(PlannerOutput::Spawn(scope, child));
        }
        self.inner.compute()
    }
}

struct ControlledOrder {
    release_order: VecDeque<usize>,
    wakers: Vec<Waker>,
}

struct ControlledIoSession {
    index: usize,
    inner: RecordingIoSource,
    order: Arc<Mutex<ControlledOrder>>,
    clears: AtomicUsize,
}

impl ControlledIoSession {
    fn clear_count(&self) -> usize {
        self.clears.load(Ordering::Acquire)
    }
}

impl IoSource for ControlledIoSession {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        self.inner.submit(owner, batch)
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        let order = self.order.lock();
        if order.release_order.front() != Some(&self.index) {
            return Ok(None);
        }
        drop(order);
        self.inner.poll()
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        let mut order = self.order.lock();
        if order.release_order.front() != Some(&self.index) {
            order.wakers.push(cx.waker().clone());
            return Poll::Pending;
        }
        drop(order);
        Poll::Ready(self.inner.wait())
    }

    fn wait(&self) -> VortexResult<Completion> {
        self.inner.wait()
    }

    fn release(&self, owner: IoOwnerId) {
        self.inner.release(owner);
    }

    fn clear(&self) {
        self.clears.fetch_add(1, Ordering::AcqRel);
        self.inner.clear();
        let wakers = {
            let mut order = self.order.lock();
            if order.release_order.front() == Some(&self.index) {
                order.release_order.pop_front();
            }
            std::mem::take(&mut order.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }
}

struct ControlledIoService {
    order: Arc<Mutex<ControlledOrder>>,
    sessions: Mutex<Vec<Arc<ControlledIoSession>>>,
}

impl ControlledIoService {
    fn new(release_order: impl IntoIterator<Item = usize>) -> Arc<Self> {
        Arc::new(Self {
            order: Arc::new(Mutex::new(ControlledOrder {
                release_order: release_order.into_iter().collect(),
                wakers: Vec::new(),
            })),
            sessions: Mutex::new(Vec::new()),
        })
    }

    fn sessions(&self) -> Vec<Arc<ControlledIoSession>> {
        self.sessions.lock().clone()
    }
}

impl IoService for ControlledIoService {
    fn session(&self) -> Arc<dyn IoSource> {
        let mut sessions = self.sessions.lock();
        let index = sessions.len();
        let inner = RecordingIoSource::default();
        inner.canned_bytes(0, buffer![0u8, 1, 2, 3].into_byte_buffer());
        inner.canned_bytes(4, buffer![4u8, 5, 6, 7].into_byte_buffer());
        let session = Arc::new(ControlledIoSession {
            index,
            inner,
            order: Arc::clone(&self.order),
            clears: AtomicUsize::new(0),
        });
        sessions.push(Arc::clone(&session));
        session
    }
}

#[rstest]
#[case::parent_finishes_first([0, 1], [0..2, 2..4])]
#[case::child_finishes_first([1, 0], [2..4, 0..2])]
fn blocking_driver_isolates_spawned_root_sessions(
    #[case] first_run_order: [usize; 2],
    #[case] expected_scopes: [Range<u64>; 2],
) -> VortexResult<()> {
    let second_run_order = [first_run_order[0] + 2, first_run_order[1] + 2];
    let service = ControlledIoService::new(first_run_order.into_iter().chain(second_run_order));
    let driver =
        Driver::with_service(Arc::clone(&service) as Arc<dyn IoService>).with_step_limit(1_000);

    for run_idx in 0..2 {
        let log = Log::default();
        let child = reading_root("child", &log, 2..4);
        let root = Box::new(SpawningRoot {
            spawned: Some((scope(2..4), child)),
            inner: reading_root("parent", &log, 0..2),
        });

        let batches = driver.run(root)?;
        let actual_scopes: Vec<Range<u64>> = batches.iter().map(|b| b.scope.rows.clone()).collect();
        assert_eq!(actual_scopes, expected_scopes);

        let sessions = service.sessions();
        assert_eq!(sessions.len(), (run_idx + 1) * 2);
        for session in &sessions[run_idx * 2..(run_idx + 1) * 2] {
            assert_eq!(session.clear_count(), 1);
        }
    }
    Ok(())
}

#[test]
fn single_session_driver_rejects_spawned_independent_root() {
    let log = Log::default();
    let child = reading_root("child", &log, 2..4);
    let root = Box::new(SpawningRoot {
        spawned: Some((scope(2..4), child)),
        inner: reading_root("parent", &log, 0..2),
    });
    let io = source();
    let err = Driver::new(Arc::clone(&io) as Arc<dyn IoSource>)
        .with_step_limit(1_000)
        .run(root)
        .err()
        .expect("Driver::new must reject spawned independent roots");
    assert!(
        err.to_string().contains("Driver::with_service"),
        "unexpected error: {err}"
    );
}
