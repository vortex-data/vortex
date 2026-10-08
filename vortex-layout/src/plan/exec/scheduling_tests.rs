// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of the graph's scheduling, driven through hand-built nodes over in-memory segments.
//!
//! Every source emits its own row indices, so whatever the tree does to them can be checked
//! against a model of the same tree built from plain arrays.

#![allow(clippy::cast_possible_truncation)]

use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::synthetic::Probe;
use super::synthetic::RowSource;
use super::synthetic::empty_segment;
use super::synthetic::row_dtype;
use super::*;
use crate::plan::ConcatPlan;
use crate::plan::FilterPlan;
use crate::plan::PackPlan;
use crate::test::SESSION;

#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// A batch returned by `compute`, as segment ids.
    Io(Vec<u32>),
    /// A result delivered for this segment.
    Delivered(u32),
    /// An array of this many rows reached the root.
    Piece(usize),
    /// A node yielded to the owner.
    Yield,
}

struct Run {
    arrays: Vec<ArrayRef>,
    events: Vec<Event>,
}

/// Drives a graph the way an owner does, answering reads with empty segments in the order
/// `pick` chooses, only while the graph waits.
fn run(
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    mut pick: impl FnMut(&[IoRequest]) -> usize,
) -> VortexResult<Run> {
    let mut graph =
        ExecGraph::try_new(SESSION.clone(), plan, rows, mask, 0, DecodeCache::default())?;
    let mut inflight: Vec<IoRequest> = Vec::new();
    let mut arrays = Vec::new();
    let mut events = Vec::new();
    loop {
        match graph.state() {
            ExecState::Done => break,
            ExecState::NeedsCompute => match graph.compute()? {
                ExecOutput::Piece(array) => {
                    assert!(!array.is_empty(), "the root never emits an empty array");
                    events.push(Event::Piece(array.len()));
                    arrays.push(array);
                }
                ExecOutput::NeedsIO(batch) => {
                    events.push(Event::Io(batch.iter().map(|r| *r.segment_id).collect()));
                    inflight.extend(batch);
                }
                ExecOutput::Yield => events.push(Event::Yield),
            },
            ExecState::Waiting => {
                if inflight.is_empty() {
                    return Err(vortex_err!("graph waits with no reads in flight"));
                }
                let request = inflight.remove(pick(&inflight));
                events.push(Event::Delivered(*request.segment_id));
                graph.set_io_result(request.id, empty_segment())?;
            }
        }
    }
    assert!(inflight.is_empty(), "graph finished with reads in flight");
    Ok(Run { arrays, events })
}

fn fifo(_: &[IoRequest]) -> usize {
    0
}

fn lifo(inflight: &[IoRequest]) -> usize {
    inflight.len() - 1
}

/// Delivers segments in the given order.
fn scripted(order: &[u32]) -> impl FnMut(&[IoRequest]) -> usize + '_ {
    let mut next = order.iter();
    move |inflight| {
        let segment = *next.next().expect("script covers every read");
        inflight
            .iter()
            .position(|r| *r.segment_id == segment)
            .expect("scripted segment is in flight")
    }
}

fn pieces(events: &[Event]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Piece(len) => Some(*len),
            _ => None,
        })
        .collect()
}

fn indices(values: impl IntoIterator<Item = u64>) -> ArrayRef {
    Buffer::from_iter(values).into_array()
}

/// Joins the arrays a run produced, in order, into one array.
fn join(dtype: &DType, arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(arrays, dtype.clone())?.into_array())
}

fn assert_rows(expected: ArrayRef, arrays: Vec<ArrayRef>) -> VortexResult<()> {
    let actual = join(expected.dtype(), arrays)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

fn segment(id: u32) -> Option<SegmentId> {
    Some(SegmentId::from(id))
}

fn fields(dtypes: impl IntoIterator<Item = DType>) -> StructFields {
    StructFields::from_iter(
        dtypes
            .into_iter()
            .enumerate()
            .map(|(i, dtype)| (format!("f{i}"), dtype)),
    )
}

fn struct_of(arrays: Vec<ArrayRef>, len: usize) -> VortexResult<ArrayRef> {
    let names = fields(arrays.iter().map(|array| array.dtype().clone()))
        .names()
        .clone();
    Ok(StructArray::try_new(names, arrays, len, Validity::NonNullable)?.into_array())
}

fn pack(children: Vec<PlanRef>) -> VortexResult<PlanRef> {
    let row_count = children[0].row_count();
    Ok(PackPlan::try_new(
        fields(children.iter().map(|child| child.dtype().clone())),
        Nullability::NonNullable,
        row_count,
        children,
        None,
    )?
    .into_plan())
}

fn concat(children: Vec<PlanRef>) -> VortexResult<PlanRef> {
    let dtype = children[0].dtype().clone();
    Ok(ConcatPlan::try_new(dtype, children)?.into_plan())
}

#[rstest]
#[case::one_piece(vec![], false)]
#[case::many_pieces(vec![3, 1, 4, 1, 5], false)]
#[case::yielding(vec![3, 1, 4, 1, 5], true)]
#[case::oversized_pieces(vec![100], true)]
fn source_pieces_come_out_in_order(
    #[case] cut: Vec<usize>,
    #[case] yielding: bool,
) -> VortexResult<()> {
    let plan = RowSource::plan(20, cut, None, yielding, false);
    let mask = Mask::from_iter((0..10).map(|i| i != 4));
    let run = run(&plan, 5..15, mask, fifo)?;
    assert_rows(indices((5..15).filter(|i| *i != 9)), run.arrays)
}

/// Chunks whose reads land in reverse order come out in chunk order, and nothing comes out
/// before the first chunk has landed.
#[rstest]
#[case::fifo(&[0, 1, 2, 3])]
#[case::lifo(&[3, 2, 1, 0])]
#[case::first_last(&[1, 2, 3, 0])]
fn concat_reorders_out_of_order_reads(#[case] order: &[u32]) -> VortexResult<()> {
    let chunks = (0..4)
        .map(|i| RowSource::plan(10, vec![4, 6], segment(i), false, false))
        .collect();
    let plan = concat(chunks)?;
    let run = run(&plan, 0..40, Mask::new_true(40), scripted(order))?;

    let first_piece = run
        .events
        .iter()
        .position(|e| matches!(e, Event::Piece(_)))
        .expect("something came out");
    let first_chunk_landed = run
        .events
        .iter()
        .position(|e| *e == Event::Delivered(0))
        .expect("chunk 0 landed");
    assert!(first_piece > first_chunk_landed);
    // Each chunk's own rows are local to the chunk.
    assert_rows(indices((0..4).flat_map(|_| 0..10)), run.arrays)
}

/// Fields cut at different places zip into structs at the boundaries they share, and the
/// structs come out in row order under any delivery order.
#[rstest]
fn pack_zips_misaligned_fields(#[values(true, false)] reverse: bool) -> VortexResult<()> {
    let a = concat(vec![
        RowSource::plan(7, vec![], segment(0), false, false),
        RowSource::plan(13, vec![5], segment(1), false, false),
    ])?;
    let b = RowSource::plan(20, vec![10], segment(2), false, false);
    let c = concat(vec![
        RowSource::plan(3, vec![], segment(3), false, false),
        RowSource::plan(17, vec![], segment(4), false, false),
    ])?;
    let plan = pack(vec![a, b, c])?;
    let pick: fn(&[IoRequest]) -> usize = if reverse { lifo } else { fifo };
    let run = run(&plan, 0..20, Mask::new_true(20), pick)?;

    let expected = struct_of(
        vec![
            indices((0..7).chain(0..13)),
            indices(0..20),
            indices((0..3).chain(0..17)),
        ],
        20,
    )?;
    assert_rows(expected, run.arrays)
}

/// A node runs only when its readiness rule holds, whatever order its inputs finish in.
#[rstest]
#[case::any(Ready::Any)]
#[case::closed_second(Ready::Closed(&[1]))]
#[case::all_closed(Ready::AllClosed)]
fn probe_runs_only_when_ready(#[case] ready: Ready) -> VortexResult<()> {
    let runs = Arc::new(Mutex::new(Vec::new()));
    let children = vec![
        RowSource::plan(10, vec![2, 2, 2], segment(0), false, false),
        RowSource::plan(10, vec![5], segment(1), false, false),
    ];
    let plan = Probe::plan(children, ready, Arc::clone(&runs));
    // The first child lands first, so under `Any` the node runs while the second is open.
    let run = run(&plan, 0..10, Mask::new_true(10), scripted(&[0, 1]))?;
    assert_rows(indices(0..10), run.arrays)?;

    let runs = runs.lock();
    assert!(!runs.is_empty());
    match ready {
        Ready::Any => assert!(runs[0] == [true, false], "ran as soon as port 0 closed"),
        Ready::Closed(_) => assert!(runs.iter().all(|ports| ports[1])),
        Ready::AllClosed => {
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0], [true, true]);
        }
    }
    Ok(())
}

/// A filter over a dense source keeps the selected rows as the pieces stream through, however
/// the source cuts them.
#[rstest]
#[case::one_piece(vec![])]
#[case::odd_cuts(vec![1, 3, 2, 7])]
fn filter_streams_dense_pieces(#[case] cut: Vec<usize>) -> VortexResult<()> {
    let plan = FilterPlan::new(RowSource::plan(20, cut, None, true, true)).into_plan();
    let mask = Mask::from_iter((0..14).map(|i| i % 3 == 0));
    let run = run(&plan, 3..17, mask.clone(), fifo)?;
    assert_rows(indices(3..17).filter(mask)?, run.arrays)
}

/// A source that yields between pieces is run once per piece, and its pieces reach the root in
/// order, one per run.
#[test]
fn yielding_source_is_run_once_per_piece() -> VortexResult<()> {
    let plan = RowSource::plan(12, vec![4, 4, 4], None, true, false);
    let run = run(&plan, 0..12, Mask::new_true(12), fifo)?;
    assert_eq!(pieces(&run.events), [4, 4, 4]);
    assert_rows(indices(0..12), run.arrays)
}

/// A tree of operators over sources, mirrored as plain arrays.
#[derive(Debug)]
enum Tree {
    /// A source of this many rows, cut into these pieces, read from this segment, yielding.
    Source {
        rows: u64,
        cut: Vec<usize>,
        segment: Option<u32>,
        yielding: bool,
    },
    Concat(Vec<Tree>),
    Pack(Vec<Tree>),
    /// A filter over a dense source.
    Filter(Box<Tree>),
}

impl Tree {
    fn rows(&self) -> u64 {
        match self {
            Tree::Source { rows, .. } => *rows,
            Tree::Concat(children) => children.iter().map(Tree::rows).sum(),
            Tree::Pack(children) => children[0].rows(),
            Tree::Filter(child) => child.rows(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Tree::Pack(children) => DType::Struct(
                fields(children.iter().map(Tree::dtype)),
                Nullability::NonNullable,
            ),
            _ => row_dtype(),
        }
    }

    fn plan(&self, dense: bool) -> VortexResult<PlanRef> {
        Ok(match self {
            Tree::Source {
                rows,
                cut,
                segment,
                yielding,
            } => RowSource::plan(
                *rows,
                cut.clone(),
                segment.map(SegmentId::from),
                *yielding,
                dense,
            ),
            Tree::Concat(children) => concat(
                children
                    .iter()
                    .map(|child| child.plan(false))
                    .collect::<VortexResult<_>>()?,
            )?,
            Tree::Pack(children) => pack(
                children
                    .iter()
                    .map(|child| child.plan(false))
                    .collect::<VortexResult<_>>()?,
            )?,
            Tree::Filter(child) => FilterPlan::new(child.plan(true)?).into_plan(),
        })
    }

    /// The selected rows of `rows`, as the tree's plan should produce them.
    fn expected(&self, rows: Range<u64>, mask: &Mask) -> VortexResult<ArrayRef> {
        match self {
            Tree::Source { .. } | Tree::Filter(_) => indices(rows).filter(mask.clone()),
            Tree::Concat(children) => {
                let mut parts = Vec::new();
                let mut offset = 0;
                for child in children {
                    let chunk = offset..offset + child.rows();
                    offset = chunk.end;
                    let local = rows.start.max(chunk.start)..rows.end.min(chunk.end);
                    if local.start >= local.end {
                        continue;
                    }
                    let mask = mask.slice(
                        (local.start - rows.start) as usize..(local.end - rows.start) as usize,
                    );
                    parts.push(
                        child
                            .expected(local.start - chunk.start..local.end - chunk.start, &mask)?,
                    );
                }
                join(&children[0].dtype(), parts)
            }
            Tree::Pack(children) => {
                let arrays = children
                    .iter()
                    .map(|child| child.expected(rows.clone(), mask))
                    .collect::<VortexResult<Vec<_>>>()?;
                struct_of(arrays, mask.true_count())
            }
        }
    }
}

/// A random tree. Sources get distinct segments so delivery order can be scripted.
fn random_tree(rng: &mut StdRng, depth: usize, rows: Option<u64>, next_segment: &mut u32) -> Tree {
    let kind = if depth == 0 {
        Kind::Source
    } else {
        [Kind::Source, Kind::Concat, Kind::Pack, Kind::Filter][rng.random_range(0..4)]
    };
    random_tree_of(rng, kind, depth, rows, next_segment)
}

#[derive(Clone, Copy)]
enum Kind {
    Source,
    Concat,
    Pack,
    Filter,
}

fn random_tree_of(
    rng: &mut StdRng,
    kind: Kind,
    depth: usize,
    rows: Option<u64>,
    next_segment: &mut u32,
) -> Tree {
    let rows = rows.unwrap_or_else(|| rng.random_range(1..24));
    match kind {
        Kind::Concat => {
            // Chunks summing to `rows`. Chunks of one column are leaves: a source, or a filter
            // over one, so every chunk has the column's dtype.
            let mut children = Vec::new();
            let mut left = rows;
            while left > 0 {
                let chunk = rng.random_range(1..=left);
                let leaf = if rng.random_bool(0.3) {
                    Kind::Filter
                } else {
                    Kind::Source
                };
                children.push(random_tree_of(rng, leaf, 0, Some(chunk), next_segment));
                left -= chunk;
            }
            Tree::Concat(children)
        }
        Kind::Pack => Tree::Pack(
            (0..rng.random_range(1..4))
                .map(|_| random_tree(rng, depth - 1, Some(rows), next_segment))
                .collect(),
        ),
        Kind::Filter => Tree::Filter(Box::new(random_tree_of(
            rng,
            Kind::Source,
            0,
            Some(rows),
            next_segment,
        ))),
        Kind::Source => {
            let segment = rng.random_bool(0.7).then(|| {
                *next_segment += 1;
                *next_segment - 1
            });
            let cut = (0..rng.random_range(0..4))
                .map(|_| rng.random_range(1..8))
                .collect();
            Tree::Source {
                rows,
                cut,
                segment,
                yielding: rng.random_bool(0.3),
            }
        }
    }
}

/// Random trees over random views, with reads completing in random order, come out as the
/// model says.
#[test]
fn random_trees_match_their_model() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..300 {
        let mut next_segment = 0;
        let tree = random_tree(&mut rng, 3, None, &mut next_segment);
        let total = tree.rows();
        let start = rng.random_range(0..=total);
        let end = rng.random_range(start..=total);
        let density = rng.random_range(0.0..=1.0);
        let mask = Mask::from_iter((start..end).map(|_| rng.random_bool(density)));

        let plan = tree.plan(false)?;
        let mut order = StdRng::seed_from_u64(rng.random());
        let run = run(&plan, start..end, mask.clone(), |inflight| {
            order.random_range(0..inflight.len())
        })
        .map_err(|e| vortex_err!("{tree:?} over {start}..{end}: {e}"))?;

        let expected = tree.expected(start..end, &mask)?;
        assert_eq!(expected.dtype(), &tree.dtype());
        assert_rows(expected, run.arrays)
            .map_err(|e| vortex_err!("{tree:?} over {start}..{end}: {e}"))?;
    }
    Ok(())
}

#[test]
fn input_take_slices_at_the_boundary() -> VortexResult<()> {
    let mut input = Input {
        spawned: true,
        ..Default::default()
    };
    input.push(PrimitiveArray::from_iter(0..4i32).into_array());
    input.push(PrimitiveArray::from_iter(4..10i32).into_array());
    assert_eq!(input.available(), 10);

    let first = input.take(6)?;
    assert_eq!(first.iter().map(|a| a.len()).collect::<Vec<_>>(), [4, 2]);
    assert_eq!(input.available(), 4);
    assert!(!input.finished());

    input.closed = true;
    let rest = input.take_all();
    assert_eq!(rest.iter().map(|a| a.len()).collect::<Vec<_>>(), [4]);
    assert!(input.finished());
    assert!(input.take(1).is_err());
    Ok(())
}
