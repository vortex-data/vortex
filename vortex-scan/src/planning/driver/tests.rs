// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use super::*;
use crate::planning::tests::scripted::Log;
use crate::planning::tests::scripted::MorselStep;
use crate::planning::tests::scripted::Order;
use crate::planning::tests::scripted::PlannerStep;
use crate::planning::tests::scripted::RecordingSession;
use crate::planning::tests::scripted::ScriptedPlanner;
use crate::planning::tests::scripted::array_of;
use crate::planning::tests::scripted::ctx;
use crate::planning::tests::scripted::fetch;
use crate::planning::tests::scripted::key;
use crate::planning::tests::scripted::scope;

fn driver(session: &Arc<RecordingSession>) -> Driver {
    Driver::new(Arc::clone(session) as Arc<dyn SegmentIoSession>).with_step_limit(1_000)
}

/// A session that finishes one fetch per poll, oldest first.
fn source() -> Arc<RecordingSession> {
    Arc::new(RecordingSession::new(Order::Fifo, false))
}

/// A session that finishes one fetch per poll, newest first.
fn lifo_source() -> Arc<RecordingSession> {
    Arc::new(RecordingSession::new(Order::Lifo, false))
}

/// A session that finishes a fetch only once the test lets one arrive.
fn gated_source() -> Arc<RecordingSession> {
    Arc::new(RecordingSession::new(Order::Fifo, true))
}

fn with_intent(segment: u32, intent: IoIntent) -> IoRequest {
    IoRequest::new(key(segment), intent)
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
        vec![PlannerStep::Io(vec![fetch(1), fetch(0)]), PlannerStep::Done],
        &log,
    );
    let batches = driver(&source).run(root)?;
    assert!(batches.is_empty());
    assert_eq!(source.performed(), vec![key(1), key(0)]);
    assert_eq!(
        log.events(),
        vec![
            "publish root",
            "deliver root 1",
            "deliver root 0",
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
            PlannerStep::NeedsIO(vec![fetch(0)]),
            PlannerStep::Continue,
            PlannerStep::Await(vec![key(0)]),
            PlannerStep::Done,
        ],
        &log,
    );
    driver(&source).run(root)?;
    assert_eq!(source.performed(), vec![key(0)]);
    assert_eq!(
        log.events(),
        vec![
            "compute root",
            "compute root",
            "deliver root 0",
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
    source.fail(1, "disk on fire");
    let root = ScriptedPlanner::boxed(
        "root",
        vec![PlannerStep::Io(vec![fetch(0), fetch(1)]), PlannerStep::Done],
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
        vec![PlannerStep::Io(vec![fetch(0), fetch(1)]), PlannerStep::Done],
        &log,
    );
    driver(&source).run(root)?;
    assert_eq!(
        source.performed(),
        vec![key(0), key(1)],
        "submitted in request order"
    );
    assert_eq!(
        log.events(),
        vec![
            "publish root",
            "deliver root 1",
            "deliver root 0",
            "compute root",
        ]
    );
    Ok(())
}

#[test]
fn completions_out_of_order_across_parked_items() -> VortexResult<()> {
    let log = Log::default();
    let source = lifo_source();
    let child = |segment: u32| {
        PlannerStep::Planner(
            scope(0..1),
            vec![PlannerStep::Io(vec![fetch(segment)]), PlannerStep::Done],
        )
    };
    let root = ScriptedPlanner::boxed("root", vec![child(0), child(1), PlannerStep::Done], &log);
    driver(&source).run(root)?;
    let deliveries: Vec<_> = log
        .events()
        .into_iter()
        .filter(|event| event.starts_with("deliver"))
        .collect();
    assert_eq!(
        deliveries,
        vec!["deliver child 1", "deliver child 0"],
        "both children were parked before either completed"
    );
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
                    MorselStep::Io(vec![fetch(0)]),
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
        PlannerStep::NeedsIO(vec![fetch(0)]),
        PlannerStep::NeedsIO(vec![fetch(0)]),
    ],
    "published while it is outstanding"
)]
#[case::waiting_for_nothing(vec![PlannerStep::Wait], "waits with no outstanding fetch")]
#[case::still_waiting_after_the_last_delivery(
    vec![PlannerStep::Io(vec![fetch(0)]), PlannerStep::Wait],
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
    let driver = Driver::new(source() as Arc<dyn SegmentIoSession>).with_step_limit(10);
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
    let optional = with_intent(0, intent);
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
    let optional = with_intent(0, IoIntent::Announce);
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
    let optional = with_intent(0, IoIntent::Announce);
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::NeedsIO(vec![optional]),
            PlannerStep::Io(vec![fetch(0)]),
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
            PlannerStep::Io(vec![fetch(0)]),
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
/// each root reads through, and finally drops, its own session.
#[test]
fn roots_finish_independently() -> VortexResult<()> {
    let log = Log::default();
    let (reading_io, ready_io) = (gated_source(), gated_source());
    let mut run = Run::new().with_step_limit(1_000);
    let reading = run.admit(
        reading_root("reading", &log, 0..2),
        scope(0..2),
        Arc::clone(&reading_io) as Arc<dyn SegmentIoSession>,
    );
    let ready = run.admit(
        ready_root("ready", &log, 2..3),
        scope(2..3),
        Arc::clone(&ready_io) as Arc<dyn SegmentIoSession>,
    );
    assert_eq!((reading, ready), (RootId(0), RootId(1)));
    assert_eq!(run.live_roots(), 2);

    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(1) 2..3", "done RootId(1)", "waiting"]
    );
    assert_eq!(run.live_roots(), 1);
    assert_eq!(
        (Arc::strong_count(&reading_io), Arc::strong_count(&ready_io)),
        (2, 1)
    );
    assert!(ready_io.submissions().is_empty());

    reading_io.arrive();
    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(0) 0..2", "done RootId(0)", "idle"]
    );
    assert_eq!(run.live_roots(), 0);
    assert_eq!(Arc::strong_count(&reading_io), 1);
    Ok(())
}

#[test]
fn poll_completion_delivers_from_every_waiting_root() -> VortexResult<()> {
    let log = Log::default();
    let sources = [gated_source(), gated_source()];
    let mut run = Run::new().with_step_limit(1_000);
    for (index, io) in sources.iter().enumerate() {
        let rows = index as u64..index as u64 + 1;
        run.admit(
            reading_root("root", &log, rows.clone()),
            scope(rows),
            Arc::clone(io) as Arc<dyn SegmentIoSession>,
        );
    }
    assert_eq!(drain(&mut run)?, vec!["waiting"]);
    let mut cx = Context::from_waker(Waker::noop());
    assert!(run.poll_completion(&mut cx).is_pending());
    for io in &sources {
        io.arrive();
    }
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
fn cancel_drops_a_roots_work_and_its_session() -> VortexResult<()> {
    let log = Log::default();
    let (cancelled_io, kept_io) = (gated_source(), gated_source());
    let mut run = Run::new().with_step_limit(1_000);
    let cancelled = run.admit(
        reading_root("cancelled", &log, 0..2),
        scope(0..2),
        Arc::clone(&cancelled_io) as Arc<dyn SegmentIoSession>,
    );
    run.admit(
        reading_root("kept", &log, 2..4),
        scope(2..4),
        Arc::clone(&kept_io) as Arc<dyn SegmentIoSession>,
    );
    assert_eq!(drain(&mut run)?, vec!["waiting"]);

    run.cancel(cancelled);
    assert_eq!(run.live_roots(), 1);
    assert_eq!(Arc::strong_count(&cancelled_io), 1);
    kept_io.arrive();
    assert_eq!(
        drain(&mut run)?,
        vec!["batch RootId(1) 2..4", "done RootId(1)", "idle"]
    );
    assert!(!log.events().contains(&"deliver cancelled 0".to_string()));
    Ok(())
}

#[test]
fn dropping_a_run_drops_every_live_session() -> VortexResult<()> {
    let log = Log::default();
    let io = gated_source();
    let mut run = Run::new();
    run.admit(
        reading_root("root", &log, 0..2),
        scope(0..2),
        Arc::clone(&io) as Arc<dyn SegmentIoSession>,
    );
    assert_eq!(drain(&mut run)?, vec!["waiting"]);
    drop(run);
    assert_eq!(Arc::strong_count(&io), 1);
    Ok(())
}

/// Waiting for M fetches costs the driver one delivery each, not a visit that re-lists the rest.
#[test]
fn a_waiting_item_is_not_revisited_per_delivery() -> VortexResult<()> {
    const FETCHES: u32 = 64;
    let log = Log::default();
    let io = RecordingSession::new(Order::Fifo, false);
    let batch: Vec<IoRequest> = (0..FETCHES)
        .map(|segment| {
            io.canned_bytes(segment, ByteBuffer::from(vec![0u8; 4]));
            fetch(segment)
        })
        .collect();
    let io = Arc::new(io);
    let root = ScriptedPlanner::boxed(
        "root",
        vec![PlannerStep::Io(batch), PlannerStep::Done],
        &log,
    );
    // Publish and the final compute: the deliveries in between cost no visit.
    Driver::new(Arc::clone(&io) as Arc<dyn SegmentIoSession>)
        .with_step_limit(2)
        .run(root)?;
    assert_eq!(io.performed().len(), FETCHES as usize);
    Ok(())
}

#[test]
fn waking_subtree_precedes_an_already_discovered_sibling() -> VortexResult<()> {
    let log = Log::default();
    let io = gated_source();
    let root = ScriptedPlanner::boxed(
        "root",
        vec![
            PlannerStep::Planner(
                scope(0..1),
                vec![
                    PlannerStep::Io(vec![fetch(0)]),
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
    run.admit(
        root,
        scope(0..2),
        Arc::clone(&io) as Arc<dyn SegmentIoSession>,
    );
    let Progress::Batch(batch) = run.advance()? else {
        panic!("the later sibling should run while the first child waits");
    };
    assert_eq!(batch.scope, scope(1..2));

    io.arrive();
    run.poll_io()?;
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

#[test]
fn polls_io_while_a_later_branch_keeps_computing() -> VortexResult<()> {
    let log = Log::default();
    let io = source();
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
