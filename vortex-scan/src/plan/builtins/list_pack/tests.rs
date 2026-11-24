// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Fixtures use a 12-list domain with small offsets.
#![allow(clippy::cast_possible_truncation)]

use std::ops::Range;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::ListPackPlan;
use super::execute::ListIndexBuilder;
use crate::execution::list_offsets::list_structures;
use crate::execution::operators::join;
use crate::execution::pipeline::ReadRequest;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::execution::pipeline::Turn;
use crate::execution::pipeline::ValueCache;
use crate::plan::PlanRef;
use crate::test::SESSION;
use crate::test::Store;
use crate::test::struct_plan;

const LISTS: usize = 12;
/// Elements per element segment.
const ELEMENT_CHUNK: usize = 4;
/// Elements per window of a list's index.
const INDEX_WINDOW: u64 = 4;

/// Which lists of the domain are selected.
#[derive(Clone, Copy, Debug)]
enum Sel {
    All,
    Contiguous,
    EveryOther,
    FirstOnly,
    LastOnly,
    Empty,
}

impl Sel {
    fn mask(self, len: usize) -> Mask {
        match self {
            Sel::All => Mask::new_true(len),
            Sel::Contiguous => Mask::from_iter((0..len).map(|i| (3..9).contains(&i))),
            Sel::EveryOther => Mask::from_iter((0..len).map(|i| i % 2 == 1)),
            Sel::FirstOnly => Mask::from_indices(len, [0]),
            Sel::LastOnly => Mask::from_indices(len, [len - 1]),
            Sel::Empty => Mask::new_false(len),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Nulls {
    NonNullable,
    Some,
    All,
}

impl Nulls {
    /// Whether each list is valid, or `None` for non-nullable lists.
    fn valid(self, len: usize) -> Option<Vec<bool>> {
        match self {
            Nulls::NonNullable => None,
            Nulls::Some => Some((0..len).map(|i| i % 3 != 1).collect()),
            Nulls::All => Some(vec![false; len]),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Lengths {
    /// One to three elements each.
    Varied,
    /// Lists 4 to 7 are empty, the rest as `Varied`.
    EmptyInTheMiddle,
    AllEmpty,
}

impl Lengths {
    fn get(self, len: usize) -> Vec<u64> {
        (0..len)
            .map(|i| match self {
                Lengths::Varied => 1 + (i as u64 % 3),
                Lengths::EmptyInTheMiddle if (4..8).contains(&i) => 0,
                Lengths::EmptyInTheMiddle => 1 + (i as u64 % 3),
                Lengths::AllEmpty => 0,
            })
            .collect()
    }
}

/// A list column read from `store`, and what it holds.
struct Lists {
    plan: PlanRef,
    array: ArrayRef,
    /// The stored offsets, as `u64`.
    offsets: Vec<u64>,
    /// The element rows each element segment holds, by segment id.
    element_segments: Vec<(u32, Range<u64>)>,
}

impl Lists {
    /// The element segments overlapping the selected non-empty lists.
    fn element_segments_for(&self, mask: &Mask) -> Vec<u32> {
        self.element_segments
            .iter()
            .filter(|(_, rows)| {
                mask.iter().enumerate().any(|(list, selected)| {
                    let (start, end) = (self.offsets[list], self.offsets[list + 1]);
                    selected
                        && start < end
                        && start < rows.end
                        && rows.start < end
                })
            })
            .map(|(segment, _)| *segment)
            .collect()
    }

    fn is_element_segment(&self, segment: u32) -> bool {
        self.element_segments.iter().any(|(id, _)| *id == segment)
    }
}

fn offsets_array(offsets: &[u64], ptype: PType) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    Ok(PrimitiveArray::from_iter(offsets.iter().copied())
        .into_array()
        .cast(DType::Primitive(ptype, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array())
}

/// Writes a list column of `lengths`, whose offsets start at `base`, to `store`: its elements in
/// segments of [`ELEMENT_CHUNK`] first, then its offsets in two chunks, then its validity.
fn lists(
    store: &Store,
    lengths: &[u64],
    valid: Option<Vec<bool>>,
    ptype: PType,
    base: u64,
    elements: impl Fn(u64) -> VortexResult<(ArrayRef, PlanRef)>,
) -> VortexResult<Lists> {
    let mut offsets = vec![base];
    for length in lengths {
        offsets.push(offsets[offsets.len() - 1] + length);
    }
    let total = offsets[offsets.len() - 1];
    let first_segment = store.len() as u32;
    let (elements, elements_plan) = elements(total)?;
    let element_segments = (0..total.div_ceil(ELEMENT_CHUNK as u64))
        .map(|chunk| {
            let start = chunk * ELEMENT_CHUNK as u64;
            (
                first_segment + chunk as u32,
                start..total.min(start + ELEMENT_CHUNK as u64),
            )
        })
        .collect();
    let offsets_array = offsets_array(&offsets, ptype)?;
    let offsets_plan = store.chunked(&offsets_array, &[5, lengths.len() + 1 - 5])?;
    let (validity, validity_plan) = match &valid {
        Some(valid) => {
            let bools = BoolArray::from_iter(valid.iter().copied()).into_array();
            (Validity::Array(bools.clone()), Some(store.flat(&bools)?))
        }
        None => (Validity::NonNullable, None),
    };
    let nullability = validity.nullability();
    let array = ListArray::try_new(
        elements,
        offsets_array,
        validity,
        &mut SESSION.create_execution_ctx(),
    )?
    .into_array();
    let mut index = ListIndexBuilder::new(INDEX_WINDOW);
    index.push(&offsets);
    let index_plan = store.flat(&index.finish().into_array())?;
    let plan = ListPackPlan::try_new(
        nullability,
        lengths.len() as u64,
        elements_plan,
        offsets_plan,
        validity_plan,
    )?
    .with_index(index_plan)?
    .into_plan();
    Ok(Lists {
        plan,
        array,
        offsets,
        element_segments,
    })
}

/// `total` integer elements, ten times their row, in segments of [`ELEMENT_CHUNK`].
fn integers(store: &Store, total: u64) -> VortexResult<(ArrayRef, PlanRef)> {
    let array = PrimitiveArray::from_iter((0..total as i32).map(|row| row * 10)).into_array();
    let mut sizes = vec![ELEMENT_CHUNK; total as usize / ELEMENT_CHUNK];
    if !(total as usize).is_multiple_of(ELEMENT_CHUNK) {
        sizes.push(total as usize % ELEMENT_CHUNK);
    }
    let plan = store.chunked(&array, &sizes)?;
    Ok((array, plan))
}

struct Run {
    arrays: Vec<ArrayRef>,
    /// The segments of each batch of reads, sorted, in the order the scan asked for them.
    rounds: Vec<Vec<u32>>,
}

impl Run {
    fn segments(&self) -> Vec<u32> {
        let mut segments = self.rounds.concat();
        segments.sort_unstable();
        segments
    }
}

/// What is read of a list's offsets before the scan of it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadAhead {
    /// The offsets, whole.
    Offsets,
    /// The index of the offsets, for lists that have one.
    Index,
    Nothing,
}

/// Reads ahead for every list under `plan`, as a scan does ahead of its chunks.
fn read_ahead(store: &Store, plan: &PlanRef, ahead: ReadAhead) -> VortexResult<ValueCache> {
    let values = ValueCache::default();
    let max = match ahead {
        ReadAhead::Offsets => u64::MAX,
        ReadAhead::Index => 0,
        ReadAhead::Nothing => return Ok(values),
    };
    for list in list_structures(std::iter::once(plan))? {
        let Some(ahead) = list.ahead(max) else {
            continue;
        };
        let rows = usize::try_from(ahead.plan().row_count())?;
        let read = run_with(
            store,
            ahead.plan(),
            Mask::new_true(rows),
            false,
            ValueCache::default(),
        )?;
        values.insert(ahead.plan(), ahead.widen(&SESSION, read.arrays)?);
    }
    Ok(values)
}

/// Runs `plan` over all its rows under `mask`, completing reads only while the scan waits,
/// oldest first or newest first, after reading `ahead` of its lists' offsets.
fn run(
    store: &Store,
    plan: &PlanRef,
    mask: Mask,
    lifo: bool,
    ahead: ReadAhead,
) -> VortexResult<Run> {
    run_with(store, plan, mask, lifo, read_ahead(store, plan, ahead)?)
}

fn run_with(
    store: &Store,
    plan: &PlanRef,
    mask: Mask,
    lifo: bool,
    values: ValueCache,
) -> VortexResult<Run> {
    let rows = 0..plan.row_count();
    let mut scan = Scan::try_new(SESSION.clone(), plan.clone(), vec![Split { rows, mask }])?
        .with_value_cache(values);
    let mut inflight: Vec<ReadRequest> = Vec::new();
    let mut round = Vec::new();
    let mut rounds = Vec::new();
    let mut arrays = Vec::new();
    loop {
        match scan.step()? {
            Turn::Read(read) => {
                round.push(*read.segment_id);
                inflight.push(read);
            }
            Turn::Output(_, array) => arrays.push(array),
            Turn::Waiting => {
                if !round.is_empty() {
                    round.sort_unstable();
                    rounds.push(std::mem::take(&mut round));
                }
                let read = if lifo {
                    inflight.pop()
                } else {
                    (!inflight.is_empty()).then(|| inflight.remove(0))
                }
                .ok_or_else(|| vortex_err!("scan waits with no reads in flight"))?;
                scan.deliver(read.id, store.read(read.segment_id))?;
            }
            Turn::Done => break,
        }
    }
    assert_eq!(scan.live_ports(), 0, "a finished scan holds no port");
    Ok(Run { arrays, rounds })
}

fn assert_selected(expected: &ArrayRef, mask: &Mask, arrays: Vec<ArrayRef>) -> VortexResult<()> {
    let actual = join(expected.dtype(), arrays)?;
    let expected = expected.filter(mask.clone())?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// The selected lists come out whole and in order. With their offsets read ahead, only the
/// element segments that hold selected lists are read, in one round with the validity.
#[rstest]
fn lists_read_only_the_elements_they_select(
    #[values(
        Sel::All,
        Sel::Contiguous,
        Sel::EveryOther,
        Sel::FirstOnly,
        Sel::LastOnly,
        Sel::Empty
    )]
    sel: Sel,
    #[values(Nulls::NonNullable, Nulls::Some, Nulls::All)] nulls: Nulls,
    #[values(Lengths::Varied, Lengths::EmptyInTheMiddle, Lengths::AllEmpty)] lengths: Lengths,
    #[values(ReadAhead::Offsets, ReadAhead::Index, ReadAhead::Nothing)] ahead: ReadAhead,
) -> VortexResult<()> {
    let store = Store::default();
    let lists = lists(
        &store,
        &lengths.get(LISTS),
        nulls.valid(LISTS),
        PType::U32,
        3,
        |total| integers(&store, total),
    )?;
    let mask = sel.mask(LISTS);
    let run = run(&store, &lists.plan, mask.clone(), false, ahead)?;
    let elements_read: Vec<u32> = run
        .segments()
        .into_iter()
        .filter(|&segment| lists.is_element_segment(segment))
        .collect();
    let exact = lists.element_segments_for(&mask);
    match ahead {
        ReadAhead::Offsets => assert_eq!(elements_read, exact),
        ReadAhead::Index => {
            // A window of elements too many on either side of each run of selected lists, and a
            // window may reach into two segments.
            let runs = match mask.slices() {
                AllOr::All => 1,
                AllOr::None => 0,
                AllOr::Some(slices) => slices.len(),
            };
            let slack = 2 * runs * (INDEX_WINDOW as usize).div_ceil(ELEMENT_CHUNK).max(1) * 2;
            assert!(exact.iter().all(|segment| elements_read.contains(segment)));
            assert!(
                elements_read.len() <= exact.len() + slack,
                "{elements_read:?} for {exact:?}"
            );
        }
        ReadAhead::Nothing => {}
    }
    if ahead != ReadAhead::Nothing {
        assert!(run.rounds.len() <= 1, "{:?}", run.rounds);
    }
    assert_selected(&lists.array, &mask, run.arrays)
}

/// Lists come out in the type their offsets are stored in, whichever reads complete first.
#[rstest]
fn offsets_keep_their_stored_type(
    #[values(PType::U8, PType::U16, PType::U32, PType::U64, PType::I32, PType::I64)] ptype: PType,
    #[values(0, 3)] base: u64,
    #[values(Sel::All, Sel::EveryOther)] sel: Sel,
    #[values(false, true)] lifo: bool,
    #[values(ReadAhead::Offsets, ReadAhead::Index, ReadAhead::Nothing)] ahead: ReadAhead,
) -> VortexResult<()> {
    let store = Store::default();
    let lists = lists(
        &store,
        &Lengths::Varied.get(LISTS),
        Nulls::Some.valid(LISTS),
        ptype,
        base,
        |total| integers(&store, total),
    )?;
    let mask = sel.mask(LISTS);
    let run = run(&store, &lists.plan, mask.clone(), lifo, ahead)?;
    for array in &run.arrays {
        let list = array
            .as_opt::<List>()
            .ok_or_else(|| vortex_err!("a list column emits list arrays, got {array}"))?;
        assert_eq!(list.offsets().dtype().as_ptype(), ptype);
    }
    assert_selected(&lists.array, &mask, run.arrays)
}

/// With both levels' offsets read ahead, a list of lists reads the inner lists' validity and
/// only the elements of the selected lists, in one round.
#[rstest]
fn nested_lists_read_in_one_round(
    #[values(Sel::All, Sel::EveryOther, Sel::LastOnly)] sel: Sel,
    #[values(false, true)] lifo: bool,
    #[values(ReadAhead::Offsets, ReadAhead::Index, ReadAhead::Nothing)] ahead: ReadAhead,
) -> VortexResult<()> {
    let store = Store::default();
    let outer_lengths = Lengths::EmptyInTheMiddle.get(LISTS);
    let inner_count: u64 = outer_lengths.iter().sum();
    let inner_lengths = Lengths::Varied.get(inner_count as usize);
    let inner = lists(
        &store,
        &inner_lengths,
        Nulls::Some.valid(inner_count as usize),
        PType::I32,
        0,
        |total| integers(&store, total),
    )?;
    let outer = lists(&store, &outer_lengths, None, PType::U16, 0, |_| {
        Ok((inner.array.clone(), inner.plan.clone()))
    })?;
    let mask = sel.mask(LISTS);
    let run = run(&store, &outer.plan, mask.clone(), lifo, ahead)?;
    if ahead == ReadAhead::Index {
        assert_eq!(run.rounds.len(), 1, "{:?}", run.rounds);
    }
    if ahead == ReadAhead::Offsets {
        assert_eq!(run.rounds.len(), 1, "{:?}", run.rounds);
        // The inner lists the selected outer lists hold, and the element segments they span.
        let inner_selected = Mask::from_iter((0..inner_count as usize).map(|inner_list| {
            mask.iter().enumerate().any(|(list, selected)| {
                selected
                    && (outer.offsets[list]..outer.offsets[list + 1]).contains(&(inner_list as u64))
            })
        }));
        let elements_read: Vec<u32> = run
            .segments()
            .into_iter()
            .filter(|&segment| inner.is_element_segment(segment))
            .collect();
        assert_eq!(elements_read, inner.element_segments_for(&inner_selected));
    }
    assert_selected(&outer.array, &mask, run.arrays)
}

/// A list beside a column chunked elsewhere comes out aligned with it.
#[rstest]
fn a_list_in_a_struct_stays_aligned(
    #[values(Sel::All, Sel::Contiguous, Sel::EveryOther, Sel::LastOnly)] sel: Sel,
    #[values(false, true)] lifo: bool,
    #[values(ReadAhead::Offsets, ReadAhead::Index, ReadAhead::Nothing)] ahead: ReadAhead,
) -> VortexResult<()> {
    let store = Store::default();
    let lists = lists(
        &store,
        &Lengths::EmptyInTheMiddle.get(LISTS),
        Nulls::Some.valid(LISTS),
        PType::U32,
        3,
        |total| integers(&store, total),
    )?;
    let numbers = PrimitiveArray::from_iter(0..LISTS as i64).into_array();
    let expected =
        StructArray::from_fields(&[("lists", lists.array.clone()), ("numbers", numbers.clone())])?
            .into_array();
    let plan = struct_plan(
        LISTS as u64,
        expected.dtype().clone(),
        vec![lists.plan, store.chunked(&numbers, &[2, 7, 3])?],
    );
    let mask = sel.mask(LISTS);
    let run = run(&store, &plan, mask.clone(), lifo, ahead)?;
    assert_selected(&expected, &mask, run.arrays)
}

/// Offsets that decrease between selected lists fail the scan, even across lists the mask
/// leaves out.
#[rstest]
fn decreasing_offsets_fail_the_scan(
    #[values(ReadAhead::Offsets, ReadAhead::Nothing)] ahead: ReadAhead,
) -> VortexResult<()> {
    let store = Store::default();
    let (_, elements) = integers(&store, 8)?;
    let offsets = store.flat(&offsets_array(&[0, 4, 1, 5], PType::U32)?)?;
    let plan =
        ListPackPlan::try_new(Nullability::NonNullable, 3, elements, offsets, None)?.into_plan();
    let Err(error) = run(&store, &plan, Mask::from_indices(3, [0, 2]), false, ahead) else {
        return Err(vortex_err!("decreasing offsets must fail the scan"));
    };
    assert!(
        error.to_string().contains("List offsets decrease"),
        "{error}"
    );
    Ok(())
}
