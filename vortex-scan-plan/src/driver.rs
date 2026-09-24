// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A single-worker blocking driver for the protocol.
//!
//! Work items are visited in FIFO order: one `state()` call plus at most one `compute()` or one
//! IO batch per visit, then the item is requeued at the back. Output order therefore follows
//! emission order, which the tests rely on.

use std::collections::VecDeque;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::io::IoBatch;
use crate::io::IoConsumer;
use crate::io::IoRequestId;
use crate::io::IoSource;
use crate::morsel::Morsel;
use crate::morsel::MorselOutput;
use crate::next::PendingPlanner;
use crate::planner::Planner;
use crate::planner::PlannerOutput;
use crate::planner::State;
use crate::planner::WorkScope;

/// One array produced by a morsel, with the scope the morsel was created for.
pub struct Batch {
    /// Scope of the morsel that produced the array.
    pub scope: WorkScope,
    /// The non-empty array.
    pub array: ArrayRef,
}

/// Drives a tree of planners and morsels to completion on the calling thread.
pub struct Driver {
    io: Arc<dyn IoSource>,
    step_limit: Option<usize>,
}

enum Item {
    Pending(Box<dyn PendingPlanner>),
    Live(Box<dyn Planner>),
    Morsel(Box<dyn Morsel>),
}

struct Work {
    scope: WorkScope,
    item: Item,
    /// Ids delivered on the previous visit; any of them re-listed by `state()` is a protocol error.
    delivered: Vec<IoRequestId>,
}

impl Driver {
    /// Creates a driver that performs every request through `io`.
    pub fn new(io: Arc<dyn IoSource>) -> Self {
        Self {
            io,
            step_limit: None,
        }
    }

    /// Fails with an error naming the limit if more than `steps` visits are needed. Tests use it
    /// to catch loops.
    pub fn with_step_limit(self, steps: usize) -> Self {
        Self {
            step_limit: Some(steps),
            ..self
        }
    }

    /// Runs `root` and everything it spawns, returning batches in emission order.
    pub fn run(&self, root: Box<dyn PendingPlanner>) -> VortexResult<Vec<Batch>> {
        let mut queue = VecDeque::new();
        queue.push_back(Work {
            scope: WorkScope {
                file_ordinal: 0,
                rows: 0..0,
            },
            item: Item::Pending(root),
            delivered: Vec::new(),
        });
        let mut output = Vec::new();
        let mut steps = 0usize;

        while let Some(work) = queue.pop_front() {
            steps += 1;
            if let Some(limit) = self.step_limit
                && steps > limit
            {
                vortex_bail!("driver exceeded its step limit of {limit}");
            }
            if let Some(next) = self.visit(work, &mut queue, &mut output)? {
                queue.push_back(next);
            }
        }
        Ok(output)
    }

    /// Visits one item and returns it if it should be requeued.
    fn visit(
        &self,
        work: Work,
        queue: &mut VecDeque<Work>,
        output: &mut Vec<Batch>,
    ) -> VortexResult<Option<Work>> {
        let Work {
            scope,
            item,
            delivered,
        } = work;
        match item {
            Item::Pending(pending) => Ok(Some(Work {
                scope,
                item: Item::Live(pending.start()?),
                delivered: Vec::new(),
            })),
            Item::Live(mut planner) => match planner.state() {
                State::Done => Ok(None),
                State::NeedsIO(batch) => {
                    let delivered = self.deliver(&mut *planner, batch, &delivered)?;
                    Ok(Some(Work {
                        scope,
                        item: Item::Live(planner),
                        delivered,
                    }))
                }
                State::NeedsCompute => {
                    let requeue = match planner.compute()? {
                        PlannerOutput::Done => false,
                        PlannerOutput::Continue | PlannerOutput::NeedsIO(_) => true,
                        PlannerOutput::Planner(scope, child) => {
                            queue.push_back(Work {
                                scope,
                                item: Item::Pending(child),
                                delivered: Vec::new(),
                            });
                            true
                        }
                        PlannerOutput::Morsel(scope, morsel) => {
                            queue.push_back(Work {
                                scope,
                                item: Item::Morsel(morsel),
                                delivered: Vec::new(),
                            });
                            true
                        }
                    };
                    Ok(requeue.then_some(Work {
                        scope,
                        item: Item::Live(planner),
                        delivered: Vec::new(),
                    }))
                }
            },
            Item::Morsel(mut morsel) => match morsel.state() {
                State::Done => Ok(None),
                State::NeedsIO(batch) => {
                    let delivered = self.deliver(&mut *morsel, batch, &delivered)?;
                    Ok(Some(Work {
                        scope,
                        item: Item::Morsel(morsel),
                        delivered,
                    }))
                }
                State::NeedsCompute => {
                    let requeue = match morsel.compute()? {
                        MorselOutput::Done => false,
                        MorselOutput::Continue | MorselOutput::NeedsIO(_) => true,
                        MorselOutput::Batch(array) => {
                            if array.is_empty() {
                                vortex_bail!(
                                    "morsel for scope {:?} returned an empty batch",
                                    scope
                                );
                            }
                            output.push(Batch {
                                scope: scope.clone(),
                                array,
                            });
                            true
                        }
                    };
                    Ok(requeue.then_some(Work {
                        scope,
                        item: Item::Morsel(morsel),
                        delivered: Vec::new(),
                    }))
                }
            },
        }
    }

    /// Performs every request in `batch` and delivers the results, returning the delivered ids.
    fn deliver(
        &self,
        consumer: &mut dyn IoConsumer,
        batch: IoBatch,
        previously_delivered: &[IoRequestId],
    ) -> VortexResult<Vec<IoRequestId>> {
        if batch.is_empty() {
            vortex_bail!("state() reported NeedsIO with an empty batch");
        }
        if let Some(stale) = batch
            .iter()
            .find(|request| previously_delivered.contains(&request.request))
        {
            vortex_bail!(
                "request {:?} was delivered on the previous visit but is listed again",
                stale.request
            );
        }
        let mut delivered = Vec::with_capacity(batch.len());
        for request in batch {
            let result = self.io.perform(&request.target)?;
            if !result.matches(&request.target) {
                vortex_bail!(
                    "IO source answered {:?} with {}",
                    request.target,
                    result.kind()
                );
            }
            consumer.set_io_result(request.request, result);
            delivered.push(request.request);
        }
        Ok(delivered)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::io::IoRequest;
    use crate::io::IoTarget;
    use crate::tests::scripted::Log;
    use crate::tests::scripted::MorselStep;
    use crate::tests::scripted::PlannerStep;
    use crate::tests::scripted::RecordingIoSource;
    use crate::tests::scripted::ScriptedPlanner;
    use crate::tests::scripted::array_of;
    use crate::tests::scripted::ctx;
    use crate::tests::scripted::scope;

    fn request(id: u32, offset: u64) -> IoRequest {
        IoRequest {
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
    fn compute_needs_io_is_performed_once_via_state() -> VortexResult<()> {
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
        let root = crate::next::pending(|| vortex_error::vortex_bail!("cannot start root"));
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
}
