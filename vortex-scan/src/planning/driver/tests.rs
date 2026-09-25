// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::assert_arrays_eq;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::*;
use vortex_io::request::IoTarget;
use crate::planning::tests::scripted::Log;
use crate::planning::tests::scripted::MorselStep;
use crate::planning::tests::scripted::PlannerStep;
use crate::planning::tests::scripted::RecordingIoSource;
use crate::planning::tests::scripted::ScriptedPlanner;
use crate::planning::tests::scripted::array_of;
use crate::planning::tests::scripted::ctx;
use crate::planning::tests::scripted::scope;

fn request(id: u32, offset: u64) -> IoRequest {
    IoRequest {
        intent: IoIntent::Fetch,
        request: IoRequestId(id),
        target: IoTarget::Range { offset, len: 4 },
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
    let root = ScriptedPlanner::pending("root", vec![PlannerStep::Done], &log);
    let batches = driver(&source()).run(root)?;
    assert!(batches.is_empty());
    assert_eq!(log.events(), vec!["start root", "compute root"]);
    Ok(())
}

#[test]
fn one_morsel_one_batch() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::pending(
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
    let root = ScriptedPlanner::pending(
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
fn fifo_across_levels() -> VortexResult<()> {
    let log = Log::default();
    let child = PlannerStep::Planner(
        scope(0..5),
        vec![one_batch_morsel(scope(0..5), 1), PlannerStep::Done],
    );
    let root = ScriptedPlanner::pending(
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
    Ok(())
}

#[test]
fn needs_io_delivers_every_request_in_order() -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    let root = ScriptedPlanner::pending(
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
        vec![
            IoTarget::Range { offset: 4, len: 4 },
            IoTarget::Range { offset: 0, len: 4 },
        ]
    );
    assert_eq!(
        log.events(),
        vec![
            "start root",
            "deliver root 0 bytes",
            "deliver root 1 bytes",
            "compute root",
        ]
    );
    Ok(())
}

#[test]
fn compute_needs_io_is_registered_once() -> VortexResult<()> {
    let log = Log::default();
    let source = source();
    let root = ScriptedPlanner::pending(
        "root",
        vec![
            PlannerStep::NeedsIO(vec![request(0, 0)]),
            PlannerStep::Io(vec![request(0, 0)]),
            PlannerStep::Done,
        ],
        &log,
    );
    driver(&source).run(root)?;
    assert_eq!(
        source.performed(),
        vec![IoTarget::Range { offset: 0, len: 4 }]
    );
    assert_eq!(
        log.events(),
        vec![
            "start root",
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
    let root = ScriptedPlanner::pending(
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
    source.fail(IoTarget::Range { offset: 4, len: 4 }, "disk on fire");
    let root = ScriptedPlanner::pending(
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
    let root = ScriptedPlanner::pending(
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
        vec![
            IoTarget::Range { offset: 0, len: 4 },
            IoTarget::Range { offset: 4, len: 4 },
        ],
        "submitted in request order"
    );
    assert_eq!(
        log.events(),
        vec![
            "start root",
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
    let root = ScriptedPlanner::pending(
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
    source.answer_with_size(IoTarget::Range { offset: 0, len: 4 }, 4);
    let root = ScriptedPlanner::pending(
        "root",
        vec![PlannerStep::Io(vec![request(0, 0)]), PlannerStep::Done],
        &log,
    );
    let err = driver(&source).run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains("answered")),
        "{err:?}"
    );
    assert_eq!(log.events(), vec!["start root"]);
    Ok(())
}

#[test]
fn start_failure_propagates() -> VortexResult<()> {
    let root = crate::planning::next::pending(|| vortex_error::vortex_bail!("cannot start root"));
    let err = driver(&source()).run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref()
            .is_some_and(|m| m.contains("cannot start root")),
        "{err:?}"
    );
    Ok(())
}

#[test]
fn morsel_needs_io_midway() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::pending(
        "root",
        vec![
            PlannerStep::Morsel(
                scope(0..4),
                vec![
                    MorselStep::Compute(MorselOutput::Batch(array_of(1))),
                    MorselStep::Compute(MorselOutput::NeedsIO(vec![request(0, 0)])),
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

#[test]
fn relisted_delivered_id_is_a_protocol_error() -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::pending(
        "root",
        vec![
            PlannerStep::Io(vec![request(0, 0)]),
            PlannerStep::Io(vec![request(0, 0)]),
            PlannerStep::Done,
        ],
        &log,
    );
    let err = driver(&source()).run(root).err().map(|e| e.to_string());
    assert!(
        err.as_deref()
            .is_some_and(|m| m.contains("delivered on the previous visit")),
        "{err:?}"
    );
    Ok(())
}

#[rstest]
#[case::empty_batch(
    PlannerStep::Morsel(scope(0..1), vec![MorselStep::Compute(MorselOutput::Batch(array_of(0)))]),
    "empty batch"
)]
#[case::empty_io(PlannerStep::Io(vec![]), "empty batch")]
fn protocol_errors(#[case] step: PlannerStep, #[case] message: &str) -> VortexResult<()> {
    let log = Log::default();
    let root = ScriptedPlanner::pending("root", vec![step, PlannerStep::Done], &log);
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
    let root = ScriptedPlanner::pending("root", steps, &log);
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
    let root = ScriptedPlanner::pending(
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
    assert!(!log.events().iter().any(|event| event.starts_with("deliver")));
    Ok(())
}

#[test]
fn optional_requests_cannot_be_wait_dependencies() -> VortexResult<()> {
    let source = source();
    let mut optional = request(0, 0);
    optional.intent = IoIntent::Announce;
    let root = ScriptedPlanner::pending(
        "root",
        vec![PlannerStep::Io(vec![optional])],
        &Log::default(),
    );
    let error = driver(&source).run(root).err().map(|error| error.to_string());
    assert!(error.is_some_and(|error| error.contains("only wait for Fetch")));
    assert!(source.submissions().is_empty());
    Ok(())
}

#[test]
fn optional_registration_can_be_promoted_to_fetch() -> VortexResult<()> {
    let source = source();
    let mut optional = request(0, 0);
    optional.intent = IoIntent::Announce;
    let root = ScriptedPlanner::pending(
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
