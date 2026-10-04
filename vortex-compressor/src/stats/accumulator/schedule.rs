// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheduling the statistics of a tuple into loops over each block.
//!
//! Every statistic is defined in isolation as an [`IntAccumulator`]. A [`Schedule`] merges a flat
//! tuple of them into loops over each L1-sized block: all in one loop, each in its own loop, or
//! any grouping of consecutive statistics in between. The schedule is a constant, so changing it
//! changes neither the statistics nor the output, and every group decision folds at compile time.
//!
//! ```ignore
//! let stats = (MinMax::new(), Sum::new(), RunCount::new(), Sorted::new());
//! accumulate(values, &validity, Schedule::<_, FUSED>::new(stats));
//! accumulate(values, &validity, Schedule::<_, EACH>::new(stats));
//! // `MinMax` and `Sum` in one loop, `RunCount` and `Sorted` in another.
//! accumulate(values, &validity, Schedule::<_, { groups(&[2]) }>::new(stats));
//! ```

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStats;

/// Every statistic runs in one loop over each block.
pub const FUSED: u64 = 0;

/// Every statistic runs its own loop over each block.
pub const EACH: u64 = u64::MAX;

/// Returns the schedule whose groups start at each of `starts`, the positions of statistics in
/// the tuple. The first group starts at the first statistic.
pub const fn groups(starts: &[usize]) -> u64 {
    let mut breaks = 0;
    let mut i = 0;
    while i < starts.len() {
        breaks |= 1 << starts[i];
        i += 1;
    }
    breaks
}

/// Runs the statistics of a flat tuple `A` in the loops given by `BREAKS`.
///
/// Bit `i` of `BREAKS` set means statistic `i` starts a new loop; otherwise it joins the loop of
/// the statistic before it. Within a loop, each chunk passes to every statistic of the loop in
/// turn, which saves loads and loop overhead, and can share work between statistics, as long as
/// their state fits in registers together. A statistic alone in its loop keeps its own block
/// method. See [`FUSED`], [`EACH`] and [`groups`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Schedule<A, const BREAKS: u64>(pub A);

impl<A, const BREAKS: u64> Schedule<A, BREAKS> {
    /// Schedules the statistics of `stats`.
    pub fn new(stats: A) -> Self {
        Self(stats)
    }

    /// Returns the loop of statistic `i`.
    const fn group(i: usize) -> u32 {
        // Bits `1..=i` each start a new loop; statistic 0 always starts the first.
        let mask = ((1u64 << (i + 1)) - 1) & !1;
        (BREAKS & mask).count_ones()
    }

    /// Returns the number of statistics in loop `group`, of `count` statistics.
    const fn group_len(group: u32, count: usize) -> usize {
        let mut len = 0;
        let mut stat = 0;
        while stat < count {
            if Self::group(stat) == group {
                len += 1;
            }
            stat += 1;
        }
        len
    }
}

/// Runs loop `$g` of a schedule over a block of fully valid chunks.
macro_rules! scheduled_block {
    ($self:ident, $chunks:ident, $n:literal, $g:literal, [$($i:tt),+]) => {
        match Self::group_len($g, $n) {
            0 => {}
            // A statistic alone keeps its own block method, which may schedule its own loops.
            1 => {
                $(if Self::group($i) == $g {
                    $self.0.$i.block($chunks);
                })+
            }
            _ => {
                for chunk in $chunks {
                    $(if Self::group($i) == $g {
                        $self.0.$i.chunk(chunk);
                    })+
                }
                $(if Self::group($i) == $g {
                    $self.0.$i.end_block();
                })+
            }
        }
    };
}

/// Runs loop `$g` of a schedule over a block of filled chunks.
macro_rules! scheduled_filled_block {
    ($self:ident, $filled:ident, $valid:ident, $n:literal, $g:literal, [$($i:tt),+]) => {
        match Self::group_len($g, $n) {
            0 => {}
            1 => {
                $(if Self::group($i) == $g {
                    $self.0.$i.filled_block($filled, $valid);
                })+
            }
            _ => {
                for (chunk, &word) in $filled.iter().zip($valid) {
                    match word {
                        0 => {}
                        u64::MAX => {
                            $(if Self::group($i) == $g {
                                $self.0.$i.chunk(chunk);
                            })+
                        }
                        _ => {
                            $(if Self::group($i) == $g {
                                $self.0.$i.filled_chunk(chunk, word);
                            })+
                        }
                    }
                }
                $(if Self::group($i) == $g {
                    $self.0.$i.end_block();
                })+
            }
        }
    };
}

/// Implements [`IntAccumulator`] and [`ErasedAccumulator`] for a [`Schedule`] of a tuple of `$n`
/// statistics, which may be in at most `$n` loops. `$indices` lists `0..$n` once more, as one
/// token tree, for the per-loop expansion.
macro_rules! impl_schedule {
    ($n:literal; $indices:tt; $($name:ident $i:tt),+) => {
        impl<T: Copy, $($name: IntAccumulator<T>,)+ const BREAKS: u64> IntAccumulator<T>
            for Schedule<($($name,)+), BREAKS>
        {
            type Output = ($($name::Output,)+);

            const USES_FILL: bool = false $(|| $name::USES_FILL)+;

            #[inline(always)]
            fn start(&mut self, head: T) {
                self.0.start(head);
            }

            #[inline(always)]
            fn chunk(&mut self, values: &[T; CHUNK]) {
                self.0.chunk(values);
            }

            #[inline(always)]
            fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
                self.0.filled_chunk(filled, valid);
            }

            #[inline(always)]
            fn push(&mut self, value: T) {
                self.0.push(value);
            }

            #[inline(always)]
            fn block(&mut self, chunks: &[[T; CHUNK]]) {
                $(scheduled_block!(self, chunks, $n, $i, $indices);)+
            }

            #[inline(always)]
            fn filled_block(&mut self, filled: &[[T; CHUNK]], valid: &[u64]) {
                $(scheduled_filled_block!(self, filled, valid, $n, $i, $indices);)+
            }

            #[inline(always)]
            fn end_block(&mut self) {
                self.0.end_block();
            }

            #[inline]
            fn finish(self) -> Self::Output {
                self.0.finish()
            }
        }

        impl<T: Copy, $($name: ErasedAccumulator<T>,)+ const BREAKS: u64> ErasedAccumulator<T>
            for Schedule<($($name,)+), BREAKS>
        {
            #[inline]
            fn finish_into(self, stats: &mut IntStats) {
                self.0.finish_into(stats);
            }
        }
    };
}

impl_schedule!(1; [0]; A 0);
impl_schedule!(2; [0, 1]; A 0, B 1);
impl_schedule!(3; [0, 1, 2]; A 0, B 1, C 2);
impl_schedule!(4; [0, 1, 2, 3]; A 0, B 1, C 2, D 3);
impl_schedule!(5; [0, 1, 2, 3, 4]; A 0, B 1, C 2, D 3, E 4);
impl_schedule!(6; [0, 1, 2, 3, 4, 5]; A 0, B 1, C 2, D 3, E 4, F 5);
impl_schedule!(7; [0, 1, 2, 3, 4, 5, 6]; A 0, B 1, C 2, D 3, E 4, F 5, G 6);
impl_schedule!(8; [0, 1, 2, 3, 4, 5, 6, 7]; A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_assign_consecutive_loops() {
        type Planned = Schedule<(), { groups(&[3, 5, 6]) }>;
        let loops: Vec<u32> = (0..7).map(Planned::group).collect();
        assert_eq!(loops, [0, 0, 0, 1, 1, 2, 3]);
        assert_eq!(Planned::group_len(0, 7), 3);
        assert_eq!(Planned::group_len(3, 7), 1);
        assert_eq!(Planned::group_len(4, 7), 0);

        assert!((0..8).all(|i| Schedule::<(), FUSED>::group(i) == 0));
        assert!((0..8u32).all(|i| Schedule::<(), EACH>::group(i as usize) == i));
        // A break on the first statistic changes nothing.
        assert_eq!(groups(&[0, 2]) & !1, groups(&[2]));
    }
}
