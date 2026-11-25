// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work an opened scan's streams share: its parts, claimed a few column chunks at a time.
//!
//! A stream takes a part from the queue and claims its rows from the front, a few column chunks
//! at a time. A stream that finds the queue empty steals instead: it cuts the unclaimed rest of
//! the running part with the most surviving rows at the column chunk boundary nearest its middle
//! and runs the back half as a part of its own, which others may steal from in turn. Work that a
//! few parts hold, such as a selective filter's rows, is so shared out among the streams while it
//! runs, without counting the surviving rows up front.
//!
//! A part is only ever cut at a chunk boundary of the columns the scan reads, where every
//! dictionary chunk also ends, so no claim cuts a dictionary.

use std::cmp::Reverse;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use crate::stream::Claims;
use crate::stream::ExhaustedFlag;

/// The work of an opened scan, which every stream of it claims rows from.
pub(super) struct Work {
    state: Mutex<State>,
    /// The sorted chunk boundaries of the columns the scan reads.
    boundaries: Vec<u64>,
    /// The sorted rows the filter's static conjuncts keep, or `None` when they keep every row.
    surviving: Option<Vec<Range<u64>>>,
    /// The rows a claim covers at least, unless its part has fewer left.
    claim_rows: u64,
    /// The parts the queue started with.
    total: usize,
    /// The parts streams have taken from the queue.
    taken: AtomicUsize,
    /// Set once nothing is left to take or steal, or the scan's limit is spent.
    exhausted: ExhaustedFlag,
}

struct State {
    /// The parts no stream has taken yet, in row order.
    queued: VecDeque<Range<u64>>,
    /// The unclaimed rows of every part a stream has taken or stolen, by [`Owned`] index.
    running: Vec<Remainder>,
}

/// The unclaimed rows of a running part: from `cuts[0]` to its last cut, with the chunk boundaries
/// in between, where it may be claimed up to or stolen from.
struct Remainder {
    cuts: Vec<u64>,
}

impl Remainder {
    fn rows(&self) -> Range<u64> {
        match (self.cuts.first(), self.cuts.last()) {
            (Some(&start), Some(&end)) => start..end,
            _ => 0..0,
        }
    }
}

/// The running part a stream claims rows from: an index into [`State::running`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Owned(usize);

impl Work {
    /// The work of `parts`, cut only at `boundaries`, the sorted chunk boundaries of the columns
    /// the scan reads, and weighed for stealing by `surviving`, the sorted rows the static
    /// conjuncts keep. Each claim covers at least `claim_rows` rows where the part has them.
    pub(super) fn new(
        parts: Vec<Range<u64>>,
        boundaries: Vec<u64>,
        surviving: Option<Vec<Range<u64>>>,
        claim_rows: u64,
        exhausted: ExhaustedFlag,
    ) -> Self {
        let total = parts.len();
        if parts.is_empty() {
            exhausted.set();
        }
        Self {
            state: Mutex::new(State {
                queued: parts.into(),
                running: Vec::new(),
            }),
            boundaries,
            surviving,
            claim_rows: claim_rows.max(1),
            total,
            taken: AtomicUsize::new(0),
            exhausted,
        }
    }

    /// The next rows for the stream running `owned`, claimed from its front; once it has none
    /// left, from a part taken from the queue or, once that is empty, stolen from the back of
    /// another running part. `None` once nothing is left to claim.
    ///
    /// Every row of the scan is claimed exactly once, and the claims from one running part are in
    /// row order.
    pub(super) fn claim(&self, owned: &mut Option<Owned>) -> Option<Range<u64>> {
        let mut state = self.state.lock();
        let claimed = self.claim_locked(&mut state, owned);
        if state.queued.is_empty() && !state.running.iter().any(|rest| self.stealable(rest)) {
            self.exhausted.set();
        }
        claimed
    }

    fn claim_locked(&self, state: &mut State, owned: &mut Option<Owned>) -> Option<Range<u64>> {
        if let Some(rows) = owned.and_then(|Owned(index)| self.claim_front(state, index)) {
            return Some(rows);
        }
        let index = if let Some(part) = state.queued.pop_front() {
            self.taken.fetch_add(1, Ordering::Relaxed);
            state.running.push(self.remainder(part));
            state.running.len() - 1
        } else {
            let stolen = self.steal(state)?;
            state.running.push(stolen);
            state.running.len() - 1
        };
        *owned = Some(Owned(index));
        self.claim_front(state, index)
    }

    /// Claims rows from the front of the running part at `index`, up to the first cut at least
    /// [`claim_rows`](Self::claim_rows) on, or to its end.
    fn claim_front(&self, state: &mut State, index: usize) -> Option<Range<u64>> {
        let rest = &mut state.running[index];
        let start = *rest.cuts.first()?;
        if rest.cuts.len() < 2 {
            return None;
        }
        let at = rest.cuts[1..]
            .iter()
            .position(|&cut| cut - start >= self.claim_rows)
            .map_or(rest.cuts.len() - 1, |offset| offset + 1);
        let end = rest.cuts[at];
        rest.cuts.drain(..at);
        if rest.cuts.len() < 2 {
            rest.cuts.clear();
        }
        Some(start..end)
    }

    /// Cuts the back half off the stealable running part with the most surviving rows, at the
    /// chunk boundary nearest the middle of its surviving rows, the last of those as near, so a
    /// thief takes the surviving rows rather than pruned ones.
    fn steal(&self, state: &mut State) -> Option<Remainder> {
        let victim = state
            .running
            .iter_mut()
            .filter(|rest| self.stealable(rest))
            .max_by_key(|rest| self.weight(rest.rows()))?;
        let rows = victim.rows();
        let half = self.weight(rows.clone()) / 2;
        let inner = &victim.cuts[1..victim.cuts.len() - 1];
        let at = 1 + inner
            .iter()
            .enumerate()
            .min_by_key(|&(offset, &cut)| {
                (self.weight(rows.start..cut).abs_diff(half), Reverse(offset))
            })
            .map(|(offset, _)| offset)?;
        let back = victim.cuts.split_off(at);
        victim.cuts.push(back[0]);
        Some(Remainder { cuts: back })
    }

    /// Whether a stealing stream may cut `rest`: it has a boundary to cut at, more rows than one
    /// claim, and rows that survive.
    fn stealable(&self, rest: &Remainder) -> bool {
        let rows = rest.rows();
        rest.cuts.len() > 2 && rows.end - rows.start > self.claim_rows && self.weight(rows) > 0
    }

    /// The unclaimed rows of a part just taken, with the chunk boundaries inside it.
    fn remainder(&self, part: Range<u64>) -> Remainder {
        let first = self.boundaries.partition_point(|&cut| cut <= part.start);
        let last = self.boundaries.partition_point(|&cut| cut < part.end);
        let mut cuts = Vec::with_capacity(last.saturating_sub(first) + 2);
        cuts.push(part.start);
        cuts.extend_from_slice(&self.boundaries[first..last.max(first)]);
        cuts.push(part.end);
        Remainder { cuts }
    }

    /// The rows of `rows` that survive the static conjuncts.
    fn weight(&self, rows: Range<u64>) -> u64 {
        let Some(surviving) = &self.surviving else {
            return rows.end.saturating_sub(rows.start);
        };
        let first = surviving.partition_point(|kept| kept.end <= rows.start);
        surviving[first..]
            .iter()
            .take_while(|kept| kept.start < rows.end)
            .map(|kept| {
                kept.end
                    .min(rows.end)
                    .saturating_sub(kept.start.max(rows.start))
            })
            .sum()
    }

    /// Marks the scan exhausted, once its limit is spent.
    pub(super) fn exhaust(&self) {
        self.exhausted.set();
    }

    /// How many of the queue's parts streams have taken, out of how many in all.
    pub(super) fn claims(&self) -> Claims {
        Claims {
            claimed: self.taken.load(Ordering::Relaxed).min(self.total),
            total: self.total,
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;

    use super::Owned;
    use super::Work;
    use crate::stream::ExhaustedFlag;

    /// Claims every row of `work` with `streams` streams in turn, the later ones starting only once
    /// `delay` claims have been made, and returns each running part's claims in the order made.
    fn drain(work: &Work, streams: usize, delay: usize) -> Vec<(Owned, Range<u64>)> {
        let mut owned: Vec<Option<Owned>> = vec![None; streams];
        let mut done = vec![false; streams];
        let mut claims = Vec::new();
        while done.iter().any(|done| !done) {
            for stream in 0..streams {
                if done[stream] || (stream > 0 && claims.len() < delay) {
                    continue;
                }
                match work.claim(&mut owned[stream]) {
                    Some(rows) => {
                        claims.push((owned[stream].unwrap_or(Owned(usize::MAX)), rows));
                    }
                    None => done[stream] = true,
                }
            }
        }
        claims
    }

    #[rstest]
    #[case::one_stream(1, 0, None)]
    #[case::stealing_streams(4, 1, None)]
    #[case::late_thieves(3, 5, None)]
    #[case::concentrated_survivors(6, 1, Some(vec![900..1_000]))]
    fn stolen_remainders_claim_every_row_once_in_order(
        #[case] streams: usize,
        #[case] delay: usize,
        #[case] surviving: Option<Vec<Range<u64>>>,
    ) {
        let parts = vec![0..400, 400..1_000];
        let boundaries: Vec<u64> = (0..1_000).step_by(50).skip(1).collect();
        let work = Work::new(
            parts,
            boundaries.clone(),
            surviving,
            100,
            ExhaustedFlag::new(),
        );
        let claims = drain(&work, streams, delay);

        let mut rows: Vec<Range<u64>> = claims.iter().map(|(_, rows)| rows.clone()).collect();
        rows.sort_by_key(|rows| rows.start);
        let mut next = 0;
        for rows in &rows {
            assert_eq!(rows.start, next, "rows claimed twice or skipped: {rows:?}");
            assert!(rows.start < rows.end);
            assert!(boundaries.contains(&rows.end) || rows.end == 400 || rows.end == 1_000);
            next = rows.end;
        }
        assert_eq!(next, 1_000);

        for (owner, _) in &claims {
            let mine: Vec<&Range<u64>> = claims
                .iter()
                .filter(|(other, _)| other == owner)
                .map(|(_, rows)| rows)
                .collect();
            assert!(
                mine.windows(2).all(|pair| pair[0].end == pair[1].start),
                "a running part's claims are out of order: {mine:?}"
            );
        }
        assert!(work.exhausted.is_set());
    }

    #[test]
    fn thieves_split_running_parts() {
        let work = Work::new(
            vec![0..1_000],
            (0..1_000).step_by(50).skip(1).collect(),
            None,
            100,
            ExhaustedFlag::new(),
        );
        let (mut first, mut second) = (None, None);
        assert_eq!(work.claim(&mut first), Some(0..100));
        assert!(!work.exhausted.is_set());
        // The rest, 100..1000, is cut at its middle, and the thief takes the back half.
        assert_eq!(work.claim(&mut second), Some(550..650));
        assert_eq!(work.claim(&mut first), Some(100..200));
        assert_ne!(first, second);
    }

    #[test]
    fn thieves_follow_the_surviving_rows() {
        let work = Work::new(
            vec![0..1_000],
            (0..1_000).step_by(50).skip(1).collect(),
            Some(vec![800..1_000]),
            100,
            ExhaustedFlag::new(),
        );
        let (mut first, mut second) = (None, None);
        assert_eq!(work.claim(&mut first), Some(0..100));
        assert_eq!(work.claim(&mut second), Some(900..1_000));
        // Of the owner's rest, 100..900, only 800..900 survives: the next thief takes its back
        // half, and then the last of it, rather than the pruned rows in front.
        let (mut third, mut fourth) = (None, None);
        assert_eq!(work.claim(&mut third), Some(850..900));
        assert_eq!(work.claim(&mut fourth), Some(800..850));
    }

    #[test]
    fn parts_without_inner_boundaries_are_not_stolen() {
        let work = Work::new(vec![0..1_000], Vec::new(), None, 100, ExhaustedFlag::new());
        let (mut first, mut second) = (None, None);
        assert_eq!(work.claim(&mut first), Some(0..1_000));
        assert_eq!(work.claim(&mut second), None);
        assert!(work.exhausted.is_set());
    }
}
