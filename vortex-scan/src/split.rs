// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! How a scan is cut into parts and how its batches are cut, shared by every scan engine.
//!
//! A *part* is a scan's unit of work, which one host thread takes at a time. Each part pays a
//! fixed cost to start, such as planning, pruning against zone maps and setting up each conjunct,
//! and threads reading parts of the same column chunk contend for its segments. So a scan is cut
//! only where the chunks of the columns it reads end, into parts of up to [`MAX_PART_ROWS`]
//! rows. A chunk is cut inside only where a host asks for more parts than the chunks give, to keep
//! its threads busy. A part is measured by the rows of it that survive the scan's zone pruning,
//! so a selective filter's work is spread over the threads rather than left in the few parts
//! that hold it. [`split_points`] makes these cuts.
//!
//! A *batch* is a scan's unit of output. Within a part, the batches a scan returns are cut at the
//! chunk boundaries of their columns by [`batch_pieces`], so a column of a batch holds one
//! encoding, such as one dictionary, which a host can export without decoding it.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray as _;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::chunked::ChunkedArrayExt as _;
use vortex_array::arrays::struct_::StructArrayExt as _;
use vortex_array::expr::BoundExpression;
use vortex_array::optimizer::ArrayOptimizer as _;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::dynamic::DynamicExprUpdates;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::execution::filter_expr::bound_conjuncts;

/// The row grid a chunk is cut inside on: the writer's default row block, counted from the
/// chunk's start.
pub const ROW_BLOCK_SIZE: usize = 8192;

/// [`ROW_BLOCK_SIZE`] in the row domain.
const BLOCK: u64 = ROW_BLOCK_SIZE as u64;

/// The most surviving rows of a part that a host asks no more parts of.
pub const MAX_PART_ROWS: u64 = 128 * BLOCK;

/// The narrowest batch [`batch_pieces`] cuts: a span between two chunk boundaries of a column
/// narrower than this stays joined to its neighbour, as a split does in the layout reader.
pub const MIN_BATCH_ROWS: usize = ROW_BLOCK_SIZE;

/// The points at which a scan of `rows` is cut into parts: `rows.start`, each cut in order, then
/// `rows.end`. A scan with no rows has no parts.
///
/// Each part is weighed by its rows in `surviving`, the sorted, disjoint row ranges the scan's
/// zone pruning keeps, or by all its rows when that is `None`.
///
/// The rows are first cut at every one of the sorted chunk `boundaries` of the columns the scan
/// reads. Where fewer than `min_parts` of the spans between them weigh anything, the spans that
/// weigh the most per piece are cut inside, one more piece at a time, into pieces of equal weight
/// at multiples of [`ROW_BLOCK_SIZE`] from the span's start, until enough pieces weigh anything or
/// none can be cut further. The pieces are then joined in order into as many parts as the
/// larger of `min_parts` and the parts of [`MAX_PART_ROWS`] the weight needs, but no more than
/// the pieces that weigh anything, so that the heaviest part weighs as little as it can. A piece
/// that weighs nothing joins the part before it.
///
/// So a cut falls inside a chunk only where the chunks are too few for `min_parts`, and a scan
/// whose zones keep few rows spreads them over its parts.
///
/// The cuts depend only on the arguments, so two engines that agree on a scan's rows, boundaries
/// and surviving rows cut it into the same parts.
pub fn split_points(
    rows: Range<u64>,
    boundaries: &[u64],
    surviving: Option<&[Range<u64>]>,
    min_parts: usize,
) -> Vec<u64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let weight = |range: Range<u64>| match surviving {
        None => range.end - range.start,
        Some(surviving) => surviving_rows(surviving, range),
    };
    let total = weight(rows.clone());
    if total == 0 {
        return vec![rows.start, rows.end];
    }
    let inner = boundaries
        .iter()
        .copied()
        .filter(|&boundary| boundary > rows.start && boundary < rows.end);
    let edges: Vec<u64> = std::iter::once(rows.start)
        .chain(inner)
        .chain(std::iter::once(rows.end))
        .collect();
    let mut spans: Vec<Span> = edges
        .windows(2)
        .map(|pair| Span::new(pair[0]..pair[1], &weight))
        .collect();
    let floor = min_parts.max(1);
    loop {
        let weighed: usize = spans.iter().map(Span::weighed).sum();
        if weighed >= floor {
            break;
        }
        // One more piece for each missing part, to the spans with the most weight per piece.
        let mut heap: BinaryHeap<(PieceWeight, Reverse<usize>)> = spans
            .iter()
            .enumerate()
            .filter(|(_, span)| !span.full)
            .map(|(idx, span)| (span.piece_weight(), Reverse(idx)))
            .collect();
        if heap.is_empty() {
            break;
        }
        let mut cut = Vec::new();
        for _ in weighed..floor {
            let Some((_, Reverse(idx))) = heap.pop() else {
                break;
            };
            let span = &mut spans[idx];
            span.cuts += 1;
            if span.cuts < span.blocks() {
                heap.push((span.piece_weight(), Reverse(idx)));
            }
            cut.push(idx);
        }
        cut.sort_unstable();
        cut.dedup();
        for idx in cut {
            spans[idx].cut(&weight);
        }
    }
    let pieces: Vec<(Range<u64>, u64)> = spans.into_iter().flat_map(|span| span.pieces).collect();
    let weighed = pieces.iter().filter(|(_, weight)| *weight > 0).count();
    let capped = usize::try_from(total.div_ceil(MAX_PART_ROWS)).unwrap_or(usize::MAX);
    join(rows, &pieces, floor.max(capped).min(weighed))
}

/// A span between two chunk boundaries and the pieces it is cut into.
struct Span {
    range: Range<u64>,
    weight: u64,
    /// The pieces the span is cut into, at most.
    cuts: u64,
    pieces: Vec<(Range<u64>, u64)>,
    /// Whether cutting the span into more pieces would add no piece that weighs anything.
    full: bool,
}

impl Span {
    fn new(range: Range<u64>, weight: &impl Fn(Range<u64>) -> u64) -> Self {
        let span_weight = weight(range.clone());
        let mut span = Self {
            pieces: vec![(range.clone(), span_weight)],
            range,
            weight: span_weight,
            cuts: 1,
            full: false,
        };
        span.full = span_weight == 0 || span.blocks() <= 1;
        span
    }

    fn blocks(&self) -> u64 {
        (self.range.end - self.range.start).div_ceil(BLOCK)
    }

    /// The pieces of the span that weigh anything.
    fn weighed(&self) -> usize {
        self.pieces.iter().filter(|(_, weight)| *weight > 0).count()
    }

    /// The weight of each piece of the span were it cut once more.
    fn piece_weight(&self) -> PieceWeight {
        PieceWeight {
            weight: self.weight,
            pieces: self.cuts,
        }
    }

    /// Cuts the span into up to `cuts` pieces of about equal weight at multiples of
    /// [`ROW_BLOCK_SIZE`] from its start.
    fn cut(&mut self, weight: &impl Fn(Range<u64>) -> u64) {
        let range = self.range.clone();
        let blocks = self.blocks();
        let pieces = self.cuts.clamp(1, blocks);
        let mut start = range.start;
        self.pieces.clear();
        for piece in 1..pieces {
            let goal =
                u64::try_from(u128::from(self.weight) * u128::from(piece) / u128::from(pieces))
                    .vortex_expect("a share of the span's weight");
            // The first block end from the span's start whose prefix reaches the goal.
            let (mut lo, mut hi) = ((start - range.start).div_ceil(BLOCK).max(1), blocks);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if weight(range.start..range.start + mid * BLOCK) >= goal {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            let end = (range.start + lo * BLOCK).min(range.end);
            if end > start && end < range.end {
                self.pieces.push((start..end, weight(start..end)));
                start = end;
            }
        }
        self.pieces
            .push((start..range.end, weight(start..range.end)));
        let weighed = u64::try_from(self.weighed()).unwrap_or(u64::MAX);
        self.full = self.cuts >= blocks || weighed < self.cuts;
    }
}

/// The weight of each of a span's pieces, ordered by `weight / pieces` without rounding.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PieceWeight {
    weight: u64,
    pieces: u64,
}

impl Ord for PieceWeight {
    fn cmp(&self, other: &Self) -> Ordering {
        (u128::from(self.weight) * u128::from(other.pieces))
            .cmp(&(u128::from(other.weight) * u128::from(self.pieces)))
    }
}

impl PartialOrd for PieceWeight {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Joins the `pieces` of `rows`, in order, into `parts` parts whose heaviest part weighs as little
/// as it can, as the points between them. `parts` is at least one and at most the pieces that
/// weigh anything. A piece that weighs nothing joins the part before it.
fn join(rows: Range<u64>, pieces: &[(Range<u64>, u64)], parts: usize) -> Vec<u64> {
    let parts = parts.max(1);
    // The pieces that weigh anything from each piece on.
    let mut left = vec![0usize; pieces.len() + 1];
    for (idx, (_, weight)) in pieces.iter().enumerate().rev() {
        left[idx] = left[idx + 1] + usize::from(*weight > 0);
    }
    // The fewest parts no heavier than `limit` the pieces join into.
    let fewest = |limit: u64| {
        let (mut count, mut part) = (1usize, 0u64);
        for &(_, weight) in pieces {
            if weight > 0 && part > 0 && part + weight > limit {
                count += 1;
                part = 0;
            }
            part += weight;
        }
        count
    };
    let heaviest = pieces.iter().map(|(_, weight)| *weight).max().unwrap_or(0);
    let total = pieces.iter().map(|(_, weight)| *weight).sum::<u64>();
    let (mut lo, mut hi) = (heaviest, total.max(heaviest));
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if fewest(mid) <= parts {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let mut points = vec![rows.start];
    let (mut opened, mut part) = (1usize, 0u64);
    for (idx, (range, weight)) in pieces.iter().enumerate() {
        // A part closes where the next piece would make it too heavy, or where every piece left
        // that weighs anything must start a part of its own to make `parts`.
        if *weight > 0 && part > 0 && (part + weight > lo || left[idx] <= parts - opened) {
            points.push(range.start);
            opened += 1;
            part = 0;
        }
        part += weight;
    }
    points.push(rows.end);
    points
}

/// The rows of `range` in the sorted, disjoint `surviving` ranges.
fn surviving_rows(surviving: &[Range<u64>], range: Range<u64>) -> u64 {
    let first = surviving.partition_point(|kept| kept.end <= range.start);
    surviving[first..]
        .iter()
        .take_while(|kept| kept.start < range.end)
        .map(|kept| kept.end.min(range.end) - kept.start.max(range.start))
        .sum()
}

/// The units a scan of `rows` hands out, one per claim, in row order: the spans between the sorted
/// chunk `boundaries` of the columns the scan reads, leaving out the units of which no row is in `surviving`, the sorted, disjoint rows the
/// scan's zone pruning keeps (all rows when `None`).
///
/// A boundary that leaves a span narrower than [`MIN_BATCH_ROWS`] since the last one kept is
/// dropped, as is the last one kept where it leaves a span that narrow before `rows.end`, so a
/// column chunked finer than its neighbours does not cut the scan into slivers.
///
/// No unit cuts a chunk, so no two claims decode the same chunk. Unlike the layout reader, a
/// span wider than its default split is not cut further: a scan of a few narrow columns, whose
/// chunks are wide, runs fastest with a chunk per claim, and the files of a multi-file scan give
/// threads units enough.
pub fn units(
    rows: Range<u64>,
    boundaries: &[u64],
    surviving: Option<&[Range<u64>]>,
) -> Vec<Range<u64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let min = MIN_BATCH_ROWS as u64;
    let mut kept = vec![rows.start];
    for &boundary in boundaries
        .iter()
        .filter(|&&boundary| boundary > rows.start && boundary < rows.end)
    {
        if boundary - kept[kept.len() - 1] >= min {
            kept.push(boundary);
        }
    }
    if kept.len() > 1 && rows.end - kept[kept.len() - 1] < min {
        kept.pop();
    }
    kept.push(rows.end);
    kept.windows(2)
        .map(|pair| pair[0]..pair[1])
        .filter(|unit| {
            surviving.is_none_or(|surviving| surviving_rows(surviving, unit.clone()) > 0)
        })
        .collect()
}

/// The conjuncts of `filter` that hold no dynamic expression, such as the bound a top-N or a
/// join pushes down. The rows their zone pruning keeps when a scan is made hold for the whole
/// scan, so both engines measure the scan's parts by them.
pub fn static_conjuncts(filter: &BoundExpression) -> Vec<BoundExpression> {
    bound_conjuncts(filter)
        .into_iter()
        .filter(|conjunct| DynamicExprUpdates::new(conjunct).is_none())
        .collect()
}

/// The rows of `rows` that `kept`, a mask over them, keeps, as sorted, disjoint ranges.
pub fn kept_ranges(rows: &Range<u64>, kept: &Mask) -> Vec<Range<u64>> {
    let offset = |row: usize| rows.start + u64::try_from(row).vortex_expect("a row of the scan");
    match kept.slices() {
        AllOr::All => vec![rows.clone()],
        AllOr::None => Vec::new(),
        AllOr::Some(slices) => slices
            .iter()
            .map(|&(start, end)| offset(start)..offset(end))
            .collect(),
    }
}

/// The parts between consecutive `points`.
pub fn parts_between(points: &[u64]) -> Vec<Range<u64>> {
    points.windows(2).map(|pair| pair[0]..pair[1]).collect()
}

/// `parts` dealt out in order to at most `n` groups of contiguous parts, as even in their counts
/// as the parts allow. Never empty, so a scan with no parts is one empty group.
pub fn group_parts(parts: Vec<Range<u64>>, n: usize) -> Vec<Vec<Range<u64>>> {
    let groups = n.min(parts.len()).max(1);
    let (per, extra) = (parts.len() / groups, parts.len() % groups);
    let mut parts = parts.into_iter();
    (0..groups)
        .map(|group| {
            parts
                .by_ref()
                .take(per + usize::from(group < extra))
                .collect()
        })
        .collect()
}

/// The batches a scan returns for `array`: its chunks, if it is chunked, and each struct chunk
/// cut at the chunk boundaries of its fields.
///
/// A span narrower than [`MIN_BATCH_ROWS`] between boundaries stays joined to its neighbour, so a
/// column chunked finer than that can still hold several chunks in one batch. Every boundary left
/// inside a batch is therefore within [`MIN_BATCH_ROWS`] of the batch's start or end.
///
/// A struct is cut field by field, so a host sees each field's own encoding: a chunk that lies
/// wholly inside a piece passes through as it is, and a chunk cut by a piece is sliced with the
/// session's reduce rules, so a dictionary stays a dictionary.
pub fn batch_pieces(array: ArrayRef, session: &VortexSession) -> VortexResult<Vec<ArrayRef>> {
    let chunks = match array.as_opt::<Chunked>() {
        Some(chunked) => chunked.chunks(),
        None => vec![array],
    };
    let mut pieces = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let Some(fields) = chunk.as_opt::<Struct>() else {
            pieces.push(chunk);
            continue;
        };
        let bounds = fields
            .iter_unmasked_fields()
            .filter_map(|field| field.as_opt::<Chunked>())
            .flat_map(|field| field.chunk_offset_values().to_vec())
            .collect();
        let cuts = batch_cuts(bounds, chunk.len(), MIN_BATCH_ROWS);
        if cuts.is_empty() {
            pieces.push(chunk);
            continue;
        }
        let mut start = 0;
        for end in cuts.into_iter().chain([chunk.len()]) {
            pieces.push(slice_fields(&chunk, start..end, session)?);
            start = end;
        }
    }
    Ok(pieces)
}

/// `rows` of the struct `array`, sliced field by field: a chunked field keeps the chunks inside
/// `rows` whole, and a field or chunk cut by `rows` is sliced and reduced.
fn slice_fields(
    array: &ArrayRef,
    rows: Range<usize>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let slice = |array: &ArrayRef, rows: Range<usize>| -> VortexResult<ArrayRef> {
        if rows.start == 0 && rows.end == array.len() {
            return Ok(array.clone());
        }
        array.slice(rows)?.optimize_ctx(session)
    };
    let fields = array.as_::<Struct>();
    let sliced = fields
        .iter_unmasked_fields()
        .map(|field| {
            let Some(chunked) = field.as_opt::<Chunked>() else {
                return slice(field, rows.clone());
            };
            let offsets = chunked.chunk_offset_values();
            let mut kept = Vec::new();
            for (chunk, bounds) in chunked.chunks().iter().zip(offsets.windows(2)) {
                let (start, end) = (bounds[0].max(rows.start), bounds[1].min(rows.end));
                if start < end {
                    kept.push(slice(chunk, start - bounds[0]..end - bounds[0])?);
                }
            }
            Ok(match kept.len() {
                1 => kept.pop().vortex_expect("one chunk"),
                _ => ChunkedArray::try_new(kept, field.dtype().clone())?.into_array(),
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let validity = fields.struct_validity().slice(rows.clone())?;
    Ok(StructArray::try_new(fields.names().clone(), sliced, rows.len(), validity)?.into_array())
}

/// The rows at which to cut a batch of `len` rows at `bounds`, leaving no piece narrower than
/// `min` but a batch narrower than it.
fn batch_cuts(mut bounds: Vec<usize>, len: usize, min: usize) -> Vec<usize> {
    bounds.sort_unstable();
    bounds.dedup();
    let mut cuts: Vec<usize> = Vec::new();
    for bound in bounds {
        let last = cuts.last().copied().unwrap_or(0);
        if bound < len && bound - last >= min.max(1) {
            cuts.push(bound);
        }
    }
    if cuts.last().is_some_and(|&cut| len - cut < min) {
        cuts.pop();
    }
    cuts
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::*;

    const B: u64 = BLOCK;

    const M: u64 = MAX_PART_ROWS;

    #[rstest]
    #[case::empty(0..0, &[], None, 1, vec![])]
    #[case::one_chunk(0..3 * B, &[], None, 1, vec![0, 3 * B])]
    #[case::large_chunk_stays_whole(0..3 * M, &[], None, 1, vec![0, 3 * M])]
    #[case::joins_chunks_up_to_the_cap(
        0..4 * M, &[M / 2, M, 3 * M / 2, 2 * M, 3 * M], None, 1, vec![0, M, 2 * M, 3 * M, 4 * M]
    )]
    #[case::keeps_off_grid_boundaries(
        0..8 * B, &[3 * B - 100, 5 * B + 7], None, 3, vec![0, 3 * B - 100, 5 * B + 7, 8 * B]
    )]
    #[case::cuts_a_chunk_for_the_floor(0..4 * B, &[], None, 4, vec![0, B, 2 * B, 3 * B, 4 * B])]
    #[case::cuts_on_the_chunk_grid(
        0..2 * B + 100, &[100], None, 3, vec![0, 100, B + 100, 2 * B + 100]
    )]
    #[case::no_more_parts_than_blocks(0..2 * B + 1, &[], None, 8, vec![0, B, 2 * B, 2 * B + 1])]
    #[case::pruned_chunks_join_their_neighbour(
        0..4 * B, &[B, 2 * B, 3 * B], Some(&[B..2 * B, 3 * B..4 * B][..]), 2,
        vec![0, 3 * B, 4 * B]
    )]
    #[allow(clippy::single_range_in_vec_init, reason = "one surviving range")]
    #[case::cuts_by_surviving_rows(
        0..8 * B, &[], Some(&[6 * B..8 * B][..]), 2, vec![0, 7 * B, 8 * B]
    )]
    fn cuts(
        #[case] rows: Range<u64>,
        #[case] boundaries: &[u64],
        #[case] surviving: Option<&[Range<u64>]>,
        #[case] min_parts: usize,
        #[case] expected: Vec<u64>,
    ) {
        assert_eq!(
            split_points(rows, boundaries, surviving, min_parts),
            expected
        );
    }

    #[rstest]
    #[case::empty(0..0, &[], None, vec![])]
    #[case::one_unit_per_chunk(0..3 * B, &[B, 2 * B], None, vec![0..B, B..2 * B, 2 * B..3 * B])]
    #[case::slivers_join_the_unit_before(
        0..3 * B, &[100, B, B + 50, 2 * B], None, vec![0..B, B..2 * B, 2 * B..3 * B]
    )]
    #[case::a_short_tail_joins_the_last_unit(
        0..2 * B + 100, &[B, 2 * B], None, vec![0..B, B..2 * B + 100]
    )]
    #[case::wide_chunks_stay_whole(0..250_000, &[], None, vec![0..250_000])]
    #[allow(clippy::single_range_in_vec_init, reason = "one surviving range")]
    #[case::pruned_units_are_left_out(
        0..3 * B, &[B, 2 * B], Some(&[2 * B + 5..2 * B + 6][..]), vec![2 * B..3 * B]
    )]
    fn unit_cuts(
        #[case] rows: Range<u64>,
        #[case] boundaries: &[u64],
        #[case] surviving: Option<&[Range<u64>]>,
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(units(rows, boundaries, surviving), expected);
    }

    /// Without pruning, the units cover the rows in order, each non-empty and ending at a chunk
    /// boundary or the end of the rows.
    #[rstest]
    fn units_cover_the_rows(
        #[values(0..1, 0..B, 7..18 * B + 3, 0..1_000_000, 0..5_000_003)] rows: Range<u64>,
        #[values(1, 997, 3 * B + 5, 300_007)] chunk: u64,
    ) {
        let boundaries: Vec<u64> = (rows.start..rows.end)
            .step_by(usize::try_from(chunk).unwrap_or(1))
            .collect();
        let units = units(rows.clone(), &boundaries, None);
        assert_eq!(units.first().map(|unit| unit.start), Some(rows.start));
        assert_eq!(units.last().map(|unit| unit.end), Some(rows.end));
        assert!(units.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(units.iter().all(|unit| !unit.is_empty()
            && (unit.end == rows.end || boundaries.binary_search(&unit.end).is_ok())));
    }

    /// Every part of every scan is non-empty, the parts cover the rows in order, and they number
    /// at least the floor where the rows have the blocks for it.
    #[rstest]
    fn parts_cover_the_rows(
        #[values(0..1, 0..B, 7..18 * B + 3, 0..1_000_000, 0..5_000_000)] rows: Range<u64>,
        #[values(1, 3, 18, 64)] min_parts: usize,
    ) {
        let boundaries: Vec<u64> = (0..rows.end).step_by(15_841).collect();
        let points = split_points(rows.clone(), &boundaries, None, min_parts);
        assert_eq!(points.first(), Some(&rows.start));
        assert_eq!(points.last(), Some(&rows.end));
        assert!(points.windows(2).all(|pair| pair[0] < pair[1]));
        let blocks = usize::try_from((rows.end - rows.start).div_ceil(B)).unwrap_or(usize::MAX);
        assert!(points.len() > min_parts.min(blocks));
        let parts = points.windows(2).map(|pair| pair[1] - pair[0]);
        assert!(parts.clone().all(|rows| rows <= M + 15_841));
    }

    /// Where few rows survive, the parts hold equal surviving rows rather than equal rows.
    #[test]
    fn parts_share_the_surviving_rows() {
        let surviving = [10 * B..11 * B, 40 * B..48 * B, 90 * B..91 * B];
        let points = split_points(0..100 * B, &[50 * B], Some(&surviving), 5);
        let weights: Vec<u64> = points
            .windows(2)
            .map(|pair| surviving_rows(&surviving, pair[0]..pair[1]))
            .collect();
        assert_eq!(weights.len(), 5);
        let (least, most) = (weights.iter().min(), weights.iter().max());
        assert!(
            most.zip(least)
                .is_some_and(|(most, least)| most - least <= 2 * B),
            "{weights:?}"
        );
    }

    /// Chunks that already give the floor its parts are joined into exactly that many, not cut.
    #[test]
    fn chunks_that_meet_the_floor_are_not_cut() {
        let boundaries: Vec<u64> = (1..50)
            .map(|chunk| chunk * 3 * B - (chunk % 2) * B)
            .collect();
        let points = split_points(0..150 * B, &boundaries, None, 18);
        assert_eq!(points.len(), 19);
        assert!(points[1..18].iter().all(|point| boundaries.contains(point)));
    }

    /// Only the heaviest chunk is cut, into only as many pieces as the floor is short of.
    #[test]
    fn only_the_heaviest_chunk_is_cut() {
        let points = split_points(0..7 * B, &[5 * B, 6 * B], None, 4);
        assert_eq!(points, vec![0, 3 * B, 5 * B, 6 * B, 7 * B]);
    }

    #[rstest]
    #[case::fewer_parts(3, 4, vec![1, 1, 1])]
    #[case::even(4, 2, vec![2, 2])]
    #[case::uneven(5, 2, vec![3, 2])]
    #[case::none(0, 3, vec![0])]
    fn groups(#[case] parts: u64, #[case] n: usize, #[case] expected: Vec<usize>) {
        let parts = (0..parts).map(|part| part * 10..(part + 1) * 10).collect();
        let groups = group_parts(parts, n);
        assert_eq!(groups.iter().map(Vec::len).collect::<Vec<_>>(), expected);
    }

    #[rstest]
    #[case::none(vec![], 20_000, vec![])]
    #[case::wide_spans(vec![0, 9000, 20_000], 30_000, vec![9000, 20_000])]
    #[case::short_span_joins_the_next(vec![3000, 12_000], 30_000, vec![12_000])]
    #[case::short_tail_joins_the_last(vec![10_000, 25_000], 30_000, vec![10_000])]
    #[case::two_fields(vec![9000, 10_000, 20_000], 30_000, vec![9000, 20_000])]
    #[case::fine_chunks(vec![1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], 10_000, vec![])]
    fn batch_cuts_merge_short_spans(
        #[case] bounds: Vec<usize>,
        #[case] len: usize,
        #[case] expected: Vec<usize>,
    ) {
        assert_eq!(batch_cuts(bounds, len, MIN_BATCH_ROWS), expected);
    }

    fn chunked(lens: &[usize]) -> VortexResult<ArrayRef> {
        let chunks = lens
            .iter()
            .map(|&len| PrimitiveArray::from_iter((0..len).map(|_| 0u32)).into_array())
            .collect::<Vec<_>>();
        let dtype = chunks[0].dtype().clone();
        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }

    /// A struct is cut where its fields' chunks end, and a chunked batch into its chunks first.
    #[test]
    fn batches_are_cut_at_field_chunks() -> VortexResult<()> {
        let a = StructArray::from_fields(&[
            ("a", chunked(&[9000, 21_000])?),
            ("b", chunked(&[20_000, 10_000])?),
        ])?
        .into_array();
        let b = StructArray::from_fields(&[("a", chunked(&[5000])?), ("b", chunked(&[5000])?)])?
            .into_array();
        let dtype = a.dtype().clone();
        let batch = ChunkedArray::try_new(vec![a, b], dtype)?.into_array();
        let lens: Vec<usize> = batch_pieces(batch, &VortexSession::empty())?
            .iter()
            .map(|piece| piece.len())
            .collect();
        assert_eq!(lens, vec![9000, 11_000, 10_000, 5000]);
        Ok(())
    }

    /// A chunked field of `lens` chunks, each a dictionary over values of its own.
    fn dict_chunks(lens: &[usize]) -> VortexResult<ArrayRef> {
        let chunks = lens
            .iter()
            .enumerate()
            .map(|(chunk, &len)| {
                let values =
                    VarBinViewArray::from_iter_str([format!("{chunk}a"), format!("{chunk}b")])
                        .into_array();
                let codes =
                    Buffer::from_iter((0..len).map(|row| u8::from(row % 2 == 1))).into_array();
                Ok(DictArray::try_new(codes, values)?.into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let dtype = chunks[0].dtype().clone();
        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }

    /// Cutting a batch keeps each field's encoding: a whole chunk passes through, and a chunk cut
    /// in two is still a dictionary on either side, so a host can export it as one.
    #[test]
    fn cut_batches_keep_dictionaries() -> VortexResult<()> {
        let batch = StructArray::from_fields(&[
            ("a", dict_chunks(&[9000, 21_000])?),
            ("b", dict_chunks(&[20_000, 10_000])?),
        ])?
        .into_array();
        let pieces = batch_pieces(batch, &VortexSession::empty())?;
        assert_eq!(
            pieces.iter().map(|piece| piece.len()).collect::<Vec<_>>(),
            vec![9000, 11_000, 10_000]
        );
        for piece in &pieces {
            let fields = piece.as_opt::<Struct>().vortex_expect("a struct piece");
            for field in fields.iter_unmasked_fields() {
                assert!(
                    field.is::<Dict>(),
                    "a field of a piece is {}",
                    field.encoding_id()
                );
            }
        }
        Ok(())
    }
}
