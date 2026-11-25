// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work an opened scan's streams share: its units, claimed one at a time.
//!
//! The units are the scan's [`units`](crate::split::units), spans between the chunk boundaries of
//! the columns it reads that zone pruning left rows of. The first stream takes them all and claims one at a time from
//! the front. A stream with no unit left steals instead: it takes the back half of the unclaimed
//! units of the running stream with the most surviving rows, splitting them where the surviving
//! rows divide most evenly, and claims from the front of those in turn, which others may steal
//! from again. Work that a few units hold, such as a selective filter's rows, is so shared out
//! among the streams while it runs.
//!
//! # Exhaustion
//!
//! The scan is marked exhausted exactly once no unit is left unclaimed, and never before: even a
//! single unclaimed unit can be stolen. A host may rely on this to stop sending threads to a scan
//! for good once it reads the flag set, as DuckDB closes a file the first time a thread joining it
//! finds no work. Were exhaustion reported while a unit was left that could not be stolen, the
//! threads that would have stolen it would leave, and it would wait for its owner.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use crate::stream::Claims;
use crate::stream::ExhaustedFlag;

/// The work of an opened scan, which every stream of it claims units from.
pub(super) struct Work {
    state: Mutex<State>,
    /// The sorted rows the filter's static conjuncts keep, or `None` when they keep every row.
    surviving: Option<Vec<Range<u64>>>,
    /// Set once a stream found no unit left to take and had to steal, after which no stream
    /// claims ahead of its need.
    stealing: AtomicBool,
    /// The units in all.
    total: usize,
    /// The units streams have claimed.
    claimed: AtomicUsize,
    /// Set once no unit is left unclaimed, or the scan's limit is spent.
    exhausted: ExhaustedFlag,
}

struct State {
    /// The units no stream has taken yet, in row order, until the first stream takes them.
    queued: Option<VecDeque<Range<u64>>>,
    /// The unclaimed units each stream has taken or stolen, by [`Owned`] index, in row order.
    running: Vec<VecDeque<Range<u64>>>,
}

/// The units a stream claims from: an index into [`State::running`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Owned(usize);

impl Work {
    /// The work of `units`, in row order, weighed for stealing by `surviving`, the sorted rows the
    /// static conjuncts keep.
    pub(super) fn new(
        units: Vec<Range<u64>>,
        surviving: Option<Vec<Range<u64>>>,
        exhausted: ExhaustedFlag,
    ) -> Self {
        let total = units.len();
        if units.is_empty() {
            exhausted.set();
        }
        Self {
            state: Mutex::new(State {
                queued: Some(units.into()),
                running: Vec::new(),
            }),
            surviving,
            stealing: AtomicBool::new(false),
            total,
            claimed: AtomicUsize::new(0),
            exhausted,
        }
    }

    /// The next unit for the stream running `owned`, from its front; once it has none left, from
    /// the units no stream has taken or, once those are taken, stolen from the back of another
    /// stream's. `None` once no unit is left unclaimed.
    ///
    /// Every unit is claimed exactly once, and the claims from one [`Owned`] are in row order.
    pub(super) fn claim(&self, owned: &mut Option<Owned>) -> Option<Range<u64>> {
        let mut state = self.state.lock();
        let claimed = self.claim_locked(&mut state, owned);
        if claimed.is_some() {
            self.claimed.fetch_add(1, Ordering::Relaxed);
        }
        if state.queued.is_none() && state.running.iter().all(VecDeque::is_empty) {
            self.exhausted.set();
        }
        claimed
    }

    fn claim_locked(&self, state: &mut State, owned: &mut Option<Owned>) -> Option<Range<u64>> {
        if let Some(unit) = owned.and_then(|Owned(index)| state.running[index].pop_front()) {
            return Some(unit);
        }
        let units = match state.queued.take() {
            Some(units) => units,
            None => {
                self.stealing.store(true, Ordering::Relaxed);
                self.steal(state)?
            }
        };
        state.running.push(units);
        let index = state.running.len() - 1;
        *owned = Some(Owned(index));
        state.running[index].pop_front()
    }

    /// Takes the back half of the unclaimed units of the stream with the most surviving rows left,
    /// split where the surviving rows divide most evenly, the last such split where several do, so
    /// a thief takes surviving rows rather than pruned ones. A single unit is taken whole.
    fn steal(&self, state: &mut State) -> Option<VecDeque<Range<u64>>> {
        let victim = state
            .running
            .iter_mut()
            .filter(|units| !units.is_empty())
            .max_by_key(|units| (self.weight(units.iter()), units.len()))?;
        if victim.len() == 1 {
            return Some(std::mem::take(victim));
        }
        let half = self.weight(victim.iter()) / 2;
        let mut front = 0;
        let mut best = (u64::MAX, 0);
        for (at, unit) in victim.iter().enumerate().take(victim.len() - 1) {
            front += self.weight(std::iter::once(unit));
            let distance = front.abs_diff(half);
            if distance <= best.0 {
                best = (distance, at + 1);
            }
        }
        Some(victim.split_off(best.1))
    }

    /// Whether a stream has found no unit left to take and had to steal. From then on there are
    /// more streams than runs of units, so a stream claims only once it has run out, leaving the
    /// rest of its units to steal, rather than ahead to overlap reads.
    pub(super) fn stealing(&self) -> bool {
        self.stealing.load(Ordering::Relaxed)
    }

    /// The rows of `units` that survive the static conjuncts.
    fn weight<'a>(&self, units: impl Iterator<Item = &'a Range<u64>>) -> u64 {
        units
            .map(|unit| match &self.surviving {
                None => unit.end - unit.start,
                Some(surviving) => {
                    let first = surviving.partition_point(|kept| kept.end <= unit.start);
                    surviving[first..]
                        .iter()
                        .take_while(|kept| kept.start < unit.end)
                        .map(|kept| {
                            kept.end
                                .min(unit.end)
                                .saturating_sub(kept.start.max(unit.start))
                        })
                        .sum()
                }
            })
            .sum()
    }

    /// Marks the scan exhausted, once its limit is spent.
    pub(super) fn exhaust(&self) {
        self.exhausted.set();
    }

    /// How many of the units streams have claimed, out of how many in all.
    pub(super) fn claims(&self) -> Claims {
        Claims {
            claimed: self.claimed.load(Ordering::Relaxed).min(self.total),
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

    /// Claims every unit of `work` with `streams` streams in turn, the later ones starting only
    /// once `delay` claims have been made, and returns each claim with the [`Owned`] it came from,
    /// in the order made.
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
                    Some(unit) => {
                        assert!(!work.exhausted.is_set() || claims.len() + 1 == work.total);
                        claims.push((owned[stream].unwrap_or(Owned(usize::MAX)), unit));
                    }
                    None => done[stream] = true,
                }
            }
        }
        claims
    }

    /// Ten units of 100 rows, with the fourth left out as zone pruning would.
    fn units() -> Vec<Range<u64>> {
        (0..10u64)
            .filter(|&unit| unit != 3)
            .map(|unit| unit * 100..(unit + 1) * 100)
            .collect()
    }

    #[rstest]
    #[case::one_stream(1, 0, None)]
    #[case::stealing_streams(4, 1, None)]
    #[case::late_thieves(3, 5, None)]
    #[case::more_streams_than_units(16, 0, None)]
    #[case::concentrated_survivors(6, 1, Some(vec![850..1_000]))]
    fn every_unit_is_claimed_once_in_order(
        #[case] streams: usize,
        #[case] delay: usize,
        #[case] surviving: Option<Vec<Range<u64>>>,
    ) {
        let units = units();
        let work = Work::new(units.clone(), surviving, ExhaustedFlag::new());
        let claims = drain(&work, streams, delay);

        let mut claimed: Vec<Range<u64>> = claims.iter().map(|(_, unit)| unit.clone()).collect();
        claimed.sort_by_key(|unit| unit.start);
        assert_eq!(claimed, units);

        for (owner, _) in &claims {
            let mine: Vec<usize> = claims
                .iter()
                .filter(|(other, _)| other == owner)
                .map(|(_, unit)| units.iter().position(|u| u == unit).unwrap_or(usize::MAX))
                .collect();
            assert!(
                mine.windows(2).all(|pair| pair[0] + 1 == pair[1]),
                "an owner's claims are not consecutive units: {mine:?}"
            );
        }
        assert!(work.exhausted.is_set());
        assert_eq!(work.claims().claimed, units.len());
    }

    #[test]
    fn thieves_take_the_back_half() {
        let work = Work::new(units(), None, ExhaustedFlag::new());
        let (mut first, mut second) = (None, None);
        assert_eq!(work.claim(&mut first), Some(0..100));
        assert!(!work.stealing());
        // The rest, eight units, is split in half, and the thief claims the front of the back half.
        assert_eq!(work.claim(&mut second), Some(600..700));
        assert!(work.stealing());
        assert_eq!(work.claim(&mut first), Some(100..200));
        assert_eq!(work.claim(&mut first), Some(200..300));
        assert_eq!(work.claim(&mut first), Some(400..500));
        assert_ne!(first, second);
    }

    #[test]
    fn thieves_follow_the_surviving_rows() {
        let work = Work::new(units(), Some(vec![800..1_000]), ExhaustedFlag::new());
        let (mut first, mut second, mut third) = (None, None, None);
        assert_eq!(work.claim(&mut first), Some(0..100));
        // The rest's surviving rows are in its last two units, so the thief takes only the last.
        assert_eq!(work.claim(&mut second), Some(900..1_000));
        assert_eq!(work.claim(&mut third), Some(800..900));
        assert_eq!(work.claim(&mut first), Some(100..200));
    }

    #[test]
    fn a_last_unit_is_stolen_and_the_scan_exhausts_only_once_it_is_claimed() {
        let work = Work::new(vec![0..100, 100..200], None, ExhaustedFlag::new());
        let (mut owner, mut thief) = (None, None);
        assert_eq!(work.claim(&mut owner), Some(0..100));
        assert!(!work.exhausted.is_set());
        assert_eq!(work.claim(&mut thief), Some(100..200));
        assert!(work.exhausted.is_set());
        assert_eq!(work.claim(&mut owner), None);
        assert_eq!(work.claim(&mut thief), None);
    }

    #[test]
    fn no_units_open_exhausted() {
        let work = Work::new(Vec::new(), None, ExhaustedFlag::new());
        assert!(work.exhausted.is_set());
        assert_eq!(work.claim(&mut None), None);
    }
}
