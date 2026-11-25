// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cutting a prepared scan into parts, the units a host's threads take.
//!
//! A scan's parts lie on its chunk grid: each is one chunk of [`PreparedScan::chunk_rows`] rows,
//! so a host's threads take parts as large as the scan's chunks. A host with more threads than
//! a scan has parts cuts them further, on [`ROW_BLOCK_SIZE`] boundaries.

use std::ops::Range;
use std::sync::Arc;

use vortex_mask::Mask;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use super::ROW_BLOCK_SIZE;
use super::scan_chunks;
use crate::planning::planner::Planner;

/// The parts of `scan`'s rows: its chunks, in row order.
pub(crate) fn scan_parts(scan: &PreparedScan) -> Vec<Range<u64>> {
    scan_chunks(scan.rows().clone(), scan.chunk_rows)
}

/// `parts` cut into at least `n` pieces where they have the blocks for it. Each part gets its
/// share of the `n` pieces by rows, and is cut into that many runs of whole [`ROW_BLOCK_SIZE`]
/// blocks, as even as the blocks allow, the last run taking any partial block. Parts already at
/// least `n` are returned as they are.
pub(crate) fn cut_parts(parts: Vec<Range<u64>>, n: usize) -> Vec<Range<u64>> {
    if parts.len() >= n {
        return parts;
    }
    let rows: u128 = parts
        .iter()
        .map(|part| u128::from(part.end - part.start))
        .sum();
    parts
        .into_iter()
        .flat_map(|part| {
            let share = (u128::from(part.end - part.start) * n as u128).div_ceil(rows.max(1));
            cut_part(part, u64::try_from(share).unwrap_or(u64::MAX))
        })
        .collect()
}

/// `part` cut into `pieces` runs of whole blocks from its start, as even as its blocks allow.
fn cut_part(part: Range<u64>, pieces: u64) -> Vec<Range<u64>> {
    let block = ROW_BLOCK_SIZE as u64;
    let blocks = (part.end - part.start).div_ceil(block);
    let pieces = pieces.clamp(1, blocks.max(1));
    let (per, extra) = (blocks / pieces, blocks % pieces);
    let mut start = part.start;
    (0..pieces)
        .map(|piece| {
            let end = (start + (per + u64::from(piece < extra)) * block).min(part.end);
            let range = start..end;
            start = end;
            range
        })
        .collect()
}

/// The planner of one part of `scan`.
///
/// With a filter, the part first evaluates every pruning plan over its own rows, on the thread
/// that takes it, from the zone maps the file's cache holds. A static conjunct is so applied once
/// to each row, by the part that holds it. A dynamic conjunct, such as the bound a top-N pushes
/// down, is applied as it stands when the part is taken, so later parts skip more as it tightens.
pub(crate) fn part_planner(scan: &Arc<PreparedScan>, part: Range<u64>) -> Box<dyn Planner> {
    let chunks = {
        let scan = Arc::clone(scan);
        let part = part.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, part, selection, kept))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::for_part(
            Arc::clone(scan),
            part,
            Box::new(chunks),
        ))
    } else {
        chunks(None)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;

    use super::cut_parts;

    const B: u64 = 8192;

    #[rstest]
    #[case::enough_parts(vec![0..B, B..2 * B], 2, vec![0..B, B..2 * B])]
    #[case::one_part_in_blocks(vec![0..4 * B], 4, vec![0..B, B..2 * B, 2 * B..3 * B, 3 * B..4 * B])]
    #[case::uneven_blocks(vec![0..4 * B], 3, vec![0..2 * B, 2 * B..3 * B, 3 * B..4 * B])]
    #[case::sixteen_blocks_for_eighteen(vec![0..16 * B], 18, (0..16).map(|i| i * B..(i + 1) * B).collect())]
    #[case::fewer_parts_than_pieces(
        vec![0..16 * B, 16 * B..32 * B, 32 * B..48 * B],
        4,
        vec![0..8 * B, 8 * B..16 * B, 16 * B..24 * B, 24 * B..32 * B, 32 * B..40 * B, 40 * B..48 * B]
    )]
    #[case::partial_last_block(vec![0..2 * B + 5], 2, vec![0..2 * B, 2 * B..2 * B + 5])]
    #[case::too_few_rows(vec![0..B / 2], 4, vec![0..B / 2])]
    #[case::per_part(vec![0..2 * B, 2 * B..3 * B], 3, vec![0..B, B..2 * B, 2 * B..3 * B])]
    fn cuts_parts_on_blocks(
        #[case] parts: Vec<Range<u64>>,
        #[case] n: usize,
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(cut_parts(parts, n), expected);
    }
}
