// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! How a scan is cut into parts and how its batches are cut, shared by every scan engine.
//!
//! A *part* is a scan's unit of work, which one host thread takes at a time. It is sized to
//! amortize the fixed cost of starting work over a range of rows, such as planning, pruning
//! against zone maps and setting up each conjunct, while leaving enough parts to balance a scan
//! across threads. [`part_rows`] sizes a scan's parts from its shape and [`split_points`] cuts its
//! rows into them, snapping each cut to a chunk boundary of the columns the scan reads where one
//! is near, so a part rarely decodes a chunk that its neighbour decodes too.
//!
//! A *batch* is a scan's unit of output. Within a part, the batches a scan returns are cut at the
//! chunk boundaries of their columns by [`batch_pieces`], so a column of a batch holds one
//! encoding, such as one dictionary, which a host can export without decoding it.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Struct;
use vortex_array::arrays::chunked::ChunkedArrayExt as _;
use vortex_array::arrays::struct_::StructArrayExt as _;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::analysis::referenced_field_paths;

/// The row grid that parts are cut on where no chunk boundary is near: the writer's default row
/// block, so a cut off the boundaries still falls on the block every column's chunks align to.
pub const ROW_BLOCK_SIZE: usize = 8192;

/// [`ROW_BLOCK_SIZE`] in the row domain.
const BLOCK: u64 = ROW_BLOCK_SIZE as u64;

/// The rows of a part of a scan without a filter.
pub const MAX_ROWS_PER_CHUNK: usize = 8 * ROW_BLOCK_SIZE;

/// The rows of a part of a filtered scan, balancing the setup of its conjuncts against
/// parallelism.
pub const FILTERED_ROWS_PER_CHUNK: usize = 16 * ROW_BLOCK_SIZE;

/// The rows of a part of a filtered scan that reads one fixed-width field.
pub const NARROW_FILTERED_ROWS_PER_CHUNK: usize = 128 * ROW_BLOCK_SIZE;

/// The narrowest batch [`batch_pieces`] cuts: a span between two chunk boundaries of a column
/// narrower than this stays joined to its neighbour, as a split does in the layout reader.
pub const MIN_BATCH_ROWS: usize = ROW_BLOCK_SIZE;

/// The target rows of each part of a scan of `dtype` with `projection` and `filter`, both bound
/// against `dtype`.
///
/// A filtered scan's parts are larger, because each part sets up every conjunct. A filtered scan
/// that reads one fixed-width field is cheap enough per row that smaller parts mainly add setup,
/// so its parts are larger still. Wider and variable-width scans need more parts to avoid long
/// tails on a few threads.
pub fn part_rows(
    dtype: &DType,
    projection: &BoundExpression,
    filter: Option<&BoundExpression>,
) -> VortexResult<u64> {
    let Some(filter) = filter else {
        return Ok(MAX_ROWS_PER_CHUNK as u64);
    };
    let mut fields = referenced_field_paths(projection)?;
    fields.extend(referenced_field_paths(filter)?);
    let mut fields = fields.into_iter();
    let first = fields.next();
    let narrow = fields.next().is_none()
        && first
            .and_then(|field| field.resolve(dtype.clone()))
            .is_some_and(|dtype| matches!(dtype, DType::Primitive(..) | DType::Bool(..)));
    Ok(if narrow {
        NARROW_FILTERED_ROWS_PER_CHUNK as u64
    } else {
        FILTERED_ROWS_PER_CHUNK as u64
    })
}

/// The points at which a scan of `rows` is cut into parts of about `target` rows, and into at
/// least `min_parts` where its rows hold that many [`ROW_BLOCK_SIZE`] blocks: `rows.start`, each
/// cut in order, then `rows.end`.
///
/// The rows are cut into `k` parts of about equal rows, `k` being the larger of the parts of
/// `target` rows that they need and `min_parts`, but no more than the blocks they hold. Each cut
/// moves to the chunk boundary nearest its ideal row within a quarter of a part, from the sorted
/// `boundaries` of the columns the scan reads, or else to the nearest multiple of
/// [`ROW_BLOCK_SIZE`] after the cut before it. A scan with no rows has no parts.
///
/// The cuts depend only on the arguments, so two engines that agree on a scan's rows, boundaries
/// and target cut it into the same parts.
pub fn split_points(
    rows: Range<u64>,
    boundaries: &[u64],
    target: u64,
    min_parts: usize,
) -> Vec<u64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let len = rows.end - rows.start;
    let blocks = len.div_ceil(BLOCK);
    let parts = len
        .div_ceil(target.max(BLOCK))
        .max(u64::try_from(min_parts).unwrap_or(u64::MAX))
        .clamp(1, blocks);
    let reach = len / parts / 4;
    let mut points = Vec::with_capacity(usize::try_from(parts + 1).unwrap_or(usize::MAX));
    points.push(rows.start);
    for part in 1..parts {
        let prev = *points.last().vortex_expect("points start with rows.start");
        let offset = u64::try_from(u128::from(len) * u128::from(part) / u128::from(parts))
            .vortex_expect("a cut lies inside the rows");
        let ideal = rows.start + offset;
        let point = nearest_boundary(boundaries, ideal, reach)
            .filter(|&point| point > prev && point < rows.end)
            .or_else(|| grid_point(ideal, prev, rows.end));
        if let Some(point) = point {
            points.push(point);
        }
    }
    points.push(rows.end);
    points
}

/// The boundary nearest `ideal` that is at most `reach` rows from it, if any.
fn nearest_boundary(boundaries: &[u64], ideal: u64, reach: u64) -> Option<u64> {
    let at = boundaries.partition_point(|&boundary| boundary < ideal);
    let below = at.checked_sub(1).map(|at| boundaries[at]);
    let above = boundaries.get(at).copied();
    [below, above]
        .into_iter()
        .flatten()
        .filter(|&boundary| boundary.abs_diff(ideal) <= reach)
        .min_by_key(|&boundary| boundary.abs_diff(ideal))
}

/// The multiple of [`ROW_BLOCK_SIZE`] nearest `ideal`, or the first one after `prev` where that
/// is not after it, if it lies before `end`.
fn grid_point(ideal: u64, prev: u64, end: u64) -> Option<u64> {
    let nearest = (ideal + BLOCK / 2) / BLOCK * BLOCK;
    let point = if nearest > prev {
        nearest
    } else {
        (prev / BLOCK + 1) * BLOCK
    };
    (point < end).then_some(point)
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
pub fn batch_pieces(array: ArrayRef) -> Vec<ArrayRef> {
    let chunks = match array.as_opt::<Chunked>() {
        Some(chunked) => chunked.chunks(),
        None => vec![array],
    };
    let mut pieces = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let cuts = chunk
            .as_opt::<Struct>()
            .map(|array| {
                let bounds = array
                    .iter_unmasked_fields()
                    .filter_map(|field| field.as_opt::<Chunked>())
                    .flat_map(|field| field.chunk_offset_values().to_vec())
                    .collect();
                batch_cuts(bounds, chunk.len(), MIN_BATCH_ROWS)
            })
            .unwrap_or_default();
        let mut start = 0;
        for end in cuts.into_iter().chain([chunk.len()]) {
            pieces.push(if start == 0 && end == chunk.len() {
                chunk.clone()
            } else {
                chunk
                    .slice(start..end)
                    .vortex_expect("a cut lies inside its chunk")
            });
            start = end;
        }
    }
    pieces
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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_error::VortexResult;

    use super::*;

    const B: u64 = BLOCK;

    #[rstest]
    #[case::empty(0..0, &[], 4 * B, 1, vec![])]
    #[case::one_part(0..3 * B, &[], 4 * B, 1, vec![0, 3 * B])]
    #[case::on_the_grid(0..8 * B, &[], 2 * B, 1, vec![0, 2 * B, 4 * B, 6 * B, 8 * B])]
    #[case::snaps_to_a_near_boundary(
        0..8 * B, &[2 * B - 100, 4 * B - 100], 4 * B, 1, vec![0, 4 * B - 100, 8 * B]
    )]
    #[case::snaps_to_the_nearest_boundary(
        0..8 * B, &[4 * B - 900, 4 * B + 300], 4 * B, 1, vec![0, 4 * B + 300, 8 * B]
    )]
    #[case::ignores_a_far_boundary(0..8 * B, &[3 * B - 1], 4 * B, 1, vec![0, 4 * B, 8 * B])]
    #[case::at_least_min_parts(0..4 * B, &[], 16 * B, 4, vec![0, B, 2 * B, 3 * B, 4 * B])]
    #[case::no_more_parts_than_blocks(0..2 * B + 1, &[], 16 * B, 8, vec![0, B, 2 * B, 2 * B + 1])]
    #[case::grid_of_the_file(100..4 * B + 100, &[], 2 * B, 1, vec![100, 2 * B, 4 * B + 100])]
    fn cuts(
        #[case] rows: Range<u64>,
        #[case] boundaries: &[u64],
        #[case] target: u64,
        #[case] min_parts: usize,
        #[case] expected: Vec<u64>,
    ) {
        assert_eq!(split_points(rows, boundaries, target, min_parts), expected);
    }

    /// Every part of every scan is non-empty and the parts cover the rows in order.
    #[rstest]
    fn parts_cover_the_rows(
        #[values(0..1, 0..B, 7..18 * B + 3, 0..1_000_000)] rows: Range<u64>,
        #[values(1, 3, 18, 64)] min_parts: usize,
    ) {
        let boundaries: Vec<u64> = (0..rows.end).step_by(15_841).collect();
        let points = split_points(rows.clone(), &boundaries, 16 * B, min_parts);
        assert_eq!(points.first(), Some(&rows.start));
        assert_eq!(points.last(), Some(&rows.end));
        assert!(points.windows(2).all(|pair| pair[0] < pair[1]));
        let blocks = usize::try_from((rows.end - rows.start).div_ceil(B)).unwrap_or(usize::MAX);
        assert!(points.len() > min_parts.min(blocks));
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
        let lens: Vec<usize> = batch_pieces(batch)
            .iter()
            .map(|piece| piece.len())
            .collect();
        assert_eq!(lens, vec![9000, 11_000, 10_000, 5000]);
        Ok(())
    }
}
