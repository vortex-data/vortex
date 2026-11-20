// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sources that compile more of the plan at run time, over rows computed from data they read.

use std::ops::Range;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::synthetic::RowSource;
use super::synthetic::SyntheticPlan;
use super::*;
use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::operators::UNBOUNDED;
use crate::execution::operators::join_inlet;
use crate::plan::PlanRef;
use crate::test::SESSION;
use crate::test::Store;

fn index_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

/// What a [`DeferSource`] has done so far.
enum Phase {
    /// Joining its first child, whose values are rows of the second.
    Reading,
    /// Waiting for the compile of the second child it asked for to close this inlet.
    Waiting(usize),
    Done,
}

/// Reads its first child whole, then compiles its second child over the rows the first child's
/// values name, and emits them.
struct DeferSource {
    target: PlanRef,
    /// Compiles the target with a mask selecting nothing, whatever the values.
    select_none: bool,
    phase: Phase,
}

/// A plan of `rows`' rows of `target`, chosen by the sorted, distinct row indices `rows` holds.
fn defer(rows: PlanRef, target: PlanRef, select_none: bool) -> PlanRef {
    let children = vec![rows.clone(), target.clone()];
    SyntheticPlan::new(
        target.dtype().clone(),
        rows.row_count(),
        children,
        move |range, mask, compiler| {
            let Some(chain) = compiler.compile(&rows, range, mask)? else {
                return Ok(None);
            };
            let source = DeferSource {
                target: target.clone(),
                select_none,
                phase: Phase::Reading,
            };
            Ok(Some(compiler.join_with(vec![chain], |_| UNBOUNDED, source)))
        },
    )
    .into_plan()
}

impl Source for DeferSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        match self.phase {
            Phase::Reading => {
                if !cx.inlet(0).closed() {
                    return Ok(SourceReady::Blocked(SourceBlocked::Inlet(0)));
                }
                let values =
                    join_inlet(cx, 0, &index_dtype())?.execute::<PrimitiveArray>(cx.exec())?;
                let values = values.as_slice::<u64>();
                let (rows, mask) = match (values.first(), values.last()) {
                    (Some(&first), Some(&last)) if !self.select_none => {
                        let len = usize::try_from(last + 1 - first)?;
                        let mask = Mask::from_indices(
                            len,
                            values
                                .iter()
                                .map(|&row| usize::try_from(row - first))
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                        (first..last + 1, mask)
                    }
                    _ => (0..1, Mask::new_false(1)),
                };
                let inlet = cx.compile(&self.target, rows, mask, UNBOUNDED);
                self.phase = Phase::Waiting(inlet);
                Ok(SourceReady::Blocked(SourceBlocked::Inlet(inlet)))
            }
            Phase::Waiting(inlet) => Ok(if cx.inlet(inlet).closed() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Inlet(inlet))
            }),
            Phase::Done => Ok(SourceReady::Ready),
        }
    }
}

impl Operator for DeferSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Phase::Waiting(inlet) = self.phase else {
            return Ok(Step::End);
        };
        self.phase = Phase::Done;
        let array = join_inlet(cx, inlet, self.target.dtype())?;
        Ok(if array.is_empty() {
            Step::End
        } else {
            Step::Final(array)
        })
    }
}

/// Compiles its child from `compute`, which only `ready` may do.
struct CompilesInCompute(PlanRef);

impl Source for CompilesInCompute {}

impl Operator for CompilesInCompute {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        cx.compile(&self.0, 0..1, Mask::new_true(1), UNBOUNDED);
        Ok(Step::End)
    }
}

/// Compiles its child from `ready` but reports itself ready.
struct CompilesWithoutBlocking(PlanRef);

impl Source for CompilesWithoutBlocking {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        cx.compile(&self.0, 0..1, Mask::new_true(1), UNBOUNDED);
        Ok(SourceReady::Ready)
    }
}

impl Operator for CompilesWithoutBlocking {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(Step::End)
    }
}

/// A one-row plan compiled to `source`, which compiles a one-row target.
fn misbehaving<S: Source + 'static>(source: fn(PlanRef) -> S) -> PlanRef {
    let target = RowSource::plan(1, Vec::new(), None, false);
    SyntheticPlan::new(index_dtype(), 1, vec![target.clone()], move |_, _, _| {
        Ok(Some(Chain::new(source(target.clone()))))
    })
    .into_plan()
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// Reads the scan asked for before it next waited, as segment ids.
    Io(Vec<u32>),
    /// A read delivered for this segment.
    Delivered(u32),
}

struct Run {
    arrays: Vec<ArrayRef>,
    events: Vec<Event>,
    deferred: u64,
}

/// Runs `plan` over `rows`, completing reads only while the scan waits, oldest first or
/// newest first.
fn run(store: &Store, plan: &PlanRef, rows: Range<u64>, lifo: bool) -> VortexResult<Run> {
    let mask = Mask::new_true(usize::try_from(rows.end - rows.start)?);
    let mut scan = Scan::try_new(SESSION.clone(), plan.clone(), vec![Split { rows, mask }])?;
    let mut inflight: Vec<ReadRequest> = Vec::new();
    let mut batch = Vec::new();
    let mut arrays = Vec::new();
    let mut events = Vec::new();
    loop {
        match scan.step()? {
            Turn::Read(read) => {
                batch.push(*read.segment_id);
                inflight.push(read);
            }
            Turn::Output(_, array) => arrays.push(array),
            Turn::Waiting => {
                if !batch.is_empty() {
                    events.push(Event::Io(std::mem::take(&mut batch)));
                }
                let read = if lifo {
                    inflight.pop()
                } else {
                    (!inflight.is_empty()).then(|| inflight.remove(0))
                }
                .ok_or_else(|| vortex_err!("scan waits with no reads in flight"))?;
                events.push(Event::Delivered(*read.segment_id));
                scan.deliver(read.id, store.read(read.segment_id))?;
            }
            Turn::Done => break,
        }
    }
    assert_eq!(scan.live_ports(), 0, "a finished scan holds no port");
    assert_eq!(scan.awaited_reads(), 0, "a finished scan waits for no read");
    Ok(Run {
        arrays,
        events,
        deferred: scan.deferred(),
    })
}

/// The segment ids of each batch of reads, in the order the scan asked for them.
fn rounds(events: &[Event]) -> Vec<Vec<u32>> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Io(batch) => {
                let mut batch = batch.clone();
                batch.sort_unstable();
                Some(batch)
            }
            Event::Delivered(_) => None,
        })
        .collect()
}

fn assert_rows(expected: ArrayRef, arrays: Vec<ArrayRef>) -> VortexResult<()> {
    let actual = super::super::operators::join(expected.dtype(), arrays)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

/// Row indices 1, 4, 5, 6 and 13 of a 20-row target, chunked as segments 0 and 1.
fn indices(store: &Store) -> VortexResult<PlanRef> {
    let rows = Buffer::from_iter([1_u64, 4, 5, 6, 13]).into_array();
    store.chunked(&rows, &[2, 3])
}

/// A 20-row target holding ten times each row, chunked as three segments after `store`'s.
fn target(store: &Store) -> VortexResult<PlanRef> {
    store.chunked(&tens(0..20), &[5, 5, 10])
}

/// Ten times each of `rows`, as the target holds them.
fn tens(rows: impl IntoIterator<Item = i64>) -> ArrayRef {
    Buffer::from_iter(rows.into_iter().map(|row| row * 10)).into_array()
}

/// The target's segments are read only once the rows naming them have been delivered, and only
/// the chunks those rows fall in. Every delivery order gives the same rows.
#[rstest]
fn a_compiled_target_is_read_after_the_rows_naming_it(
    #[values(false, true)] lifo: bool,
) -> VortexResult<()> {
    let store = Store::default();
    let rows = indices(&store)?;
    let target = target(&store)?;
    let run = run(&store, &defer(rows, target, false), 0..5, lifo)?;
    assert_eq!(run.deferred, 1);
    // Rows 1, 4, 5, 6 and 13 fall in the target's chunks [0, 5), [5, 10) and [10, 20).
    assert_eq!(rounds(&run.events), [vec![0, 1], vec![2, 3, 4]]);
    let first_target_read = run
        .events
        .iter()
        .position(|event| matches!(event, Event::Io(batch) if batch.contains(&2)))
        .ok_or_else(|| vortex_err!("the target was never read"))?;
    let deliveries_before = run.events[..first_target_read]
        .iter()
        .filter(|event| matches!(event, Event::Delivered(_)))
        .count();
    assert_eq!(
        deliveries_before, 2,
        "the target is compiled once its rows are joined"
    );
    assert_rows(tens([1, 4, 5, 6, 13]), run.arrays)
}

/// A compile selecting no row gives a closed inlet, which wakes the source to finish.
#[test]
fn a_compile_selecting_nothing_gives_a_closed_inlet() -> VortexResult<()> {
    let store = Store::default();
    let rows = indices(&store)?;
    let target = target(&store)?;
    let run = run(&store, &defer(rows, target, true), 0..5, false)?;
    assert_eq!(run.deferred, 1);
    assert_eq!(
        rounds(&run.events),
        [vec![0, 1]],
        "nothing of the target is read"
    );
    assert!(run.arrays.is_empty());
    Ok(())
}

/// A compiled plan that itself compiles at run time takes one more round of reads.
#[rstest]
fn a_compiled_plan_can_compile_again(#[values(false, true)] lifo: bool) -> VortexResult<()> {
    let store = Store::default();
    // Rows 1 and 3 of the inner plan, whose rows are rows 2, 7, 11 and 15 of the target.
    let outer = store.flat(&Buffer::from_iter([1_u64, 3]).into_array())?;
    let inner = store.flat(&Buffer::from_iter([2_u64, 7, 11, 15]).into_array())?;
    let target = target(&store)?;
    let plan = defer(outer, defer(inner, target, false), false);
    let run = run(&store, &plan, 0..2, lifo)?;
    assert_eq!(run.deferred, 2);
    // Target rows 7 and 15 fall in its chunks [5, 10) and [10, 20).
    assert_eq!(rounds(&run.events), [vec![0], vec![1], vec![3, 4]]);
    assert_rows(tens([7, 15]), run.arrays)
}

/// A compile that fails at run time fails the scan's step.
#[test]
fn a_failed_compile_fails_the_step() -> VortexResult<()> {
    let store = Store::default();
    let rows = indices(&store)?;
    let target = SyntheticPlan::new(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        20,
        Vec::new(),
        |_, _, _| vortex_bail!("the target cannot compile"),
    )
    .into_plan();
    let Err(error) = run(&store, &defer(rows, target, false), 0..5, false) else {
        vortex_bail!("a failed compile must fail the scan");
    };
    assert!(
        error.to_string().contains("the target cannot compile"),
        "{error}"
    );
    Ok(())
}

/// Only a readiness check that then blocks may compile.
#[rstest]
#[case::from_compute(misbehaving(CompilesInCompute), "Only Source::ready may compile")]
#[case::without_blocking(misbehaving(CompilesWithoutBlocking), "must block")]
fn a_compile_outside_a_blocking_readiness_check_fails(
    #[case] plan: PlanRef,
    #[case] message: &str,
) -> VortexResult<()> {
    let Err(error) = run(&Store::default(), &plan, 0..1, false) else {
        vortex_bail!("a misplaced compile must fail the scan");
    };
    assert!(error.to_string().contains(message), "{error}");
    Ok(())
}

/// The segments a scan reports before it runs leave out those of compiles asked for at run time,
/// which depend on data it has not read.
#[test]
fn segments_leave_out_run_time_compiles() -> VortexResult<()> {
    let store = Store::default();
    let rows = indices(&store)?;
    let target = target(&store)?;
    let plan = defer(rows, target, false);
    let segments = Scan::segments(&SESSION, &plan, 0..5, &Mask::new_true(5))?;
    assert_eq!(
        segments.iter().map(|segment| **segment).collect::<Vec<_>>(),
        [0, 1]
    );
    let run = run(&store, &plan, 0..5, false)?;
    assert!(
        rounds(&run.events).concat().contains(&2),
        "the run still reads the target"
    );
    Ok(())
}
