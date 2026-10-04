// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Distinct values.

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_array::scalar::PValue;
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_map::HashMap;

use super::CHUNK;
use super::CHUNK_U32;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::IntValue;
use super::transitions;
use crate::stats::integer::DistinctInfo;

/// Value ranges up to this many values may be counted in a dense array instead of a hash map.
const DENSE_DISTINCT_MAX_RANGE: usize = 1 << 16;

/// Value ranges up to this many values are always counted in a dense array, regardless of the array
/// length. This covers every `u8` and `i8` array.
const DENSE_DISTINCT_ALWAYS_RANGE: usize = 1 << 8;

/// Where the occurrences of each distinct value are counted.
enum Counts<T> {
    /// Counts indexed by `value - min`, used when the value range is narrow. This avoids hashing
    /// entirely.
    Dense {
        /// The smallest countable value.
        min: T,
        /// `min` cast to `usize`. Casting preserves values modulo `2^usize::BITS`, so the wrapping
        /// difference of two cast values is the exact difference of any two values in the range.
        min_index: usize,
        /// The number of occurrences of `min + index`.
        counts: Vec<u32>,
    },
    /// Counts keyed by value.
    Hashed(HashMap<NativeValue<T>, u32, FxBuildHasher>),
}

/// The occurrences of each distinct valid value, and the number of runs.
///
/// Consecutive equal values are counted once per run, so the counts are only updated on value
/// changes. That tracks the runs for free, so the run count is part of the output: composing this
/// with [`RunCount`] would detect every value change twice.
pub struct Distinct<T> {
    /// The occurrences counted so far.
    counts: Counts<T>,
    /// The value of the current run.
    prev: T,
    /// Occurrences of `prev` in the current run not yet added to `counts`.
    pending: u32,
    /// The number of runs so far.
    runs: u32,
}

impl<T: IntegerPType> Distinct<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// Creates an accumulator for valid values within `min..=max`, choosing dense counting when
    /// the range is narrow relative to `len`, the number of values.
    pub fn new(min: T, max: T, len: usize) -> Self {
        let range_len = match (min.to_i128(), max.to_i128()) {
            (Some(min), Some(max)) => usize::try_from(max - min + 1).unwrap_or(usize::MAX),
            _ => usize::MAX,
        };

        // A dense counter is only worthwhile when it is not much larger than the array itself.
        let counts = if range_len <= DENSE_DISTINCT_ALWAYS_RANGE
            || (range_len <= DENSE_DISTINCT_MAX_RANGE && range_len <= len)
        {
            Counts::Dense {
                min,
                min_index: min.as_(),
                counts: vec![0; range_len],
            }
        } else {
            Counts::Hashed(HashMap::with_capacity_and_hasher(len / 2, FxBuildHasher))
        };

        Self {
            counts,
            prev: T::zero(),
            pending: 0,
            runs: 0,
        }
    }

    /// Creates an accumulator that needs no bounds, if `T` is narrow enough to always count densely
    /// over its whole domain.
    pub fn for_full_domain(len: usize) -> Option<Self> {
        (size_of::<T>() == 1).then(|| Self::new(T::min_value(), T::max_value(), len))
    }

    /// Adds the pending occurrences of `prev` to the counts.
    #[inline(always)]
    fn flush(&mut self) {
        let (value, count) = (self.prev, self.pending);
        match &mut self.counts {
            Counts::Dense {
                min_index, counts, ..
            } => counts[value.as_().wrapping_sub(*min_index)] += count,
            Counts::Hashed(map) => *map.entry(NativeValue(value)).or_insert(0) += count,
        }
        self.pending = 0;
    }
}

impl<T: IntegerPType> IntAccumulator<T> for Distinct<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// The distinct values and the run count.
    type Output = (DistinctInfo<T>, u32);

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.prev = head;
        self.runs = 1;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        if transitions(&self.prev, values) == 0 {
            self.pending += CHUNK_U32;
            return;
        }
        for &value in values {
            self.push(value);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        if value != self.prev {
            self.flush();
            self.prev = value;
            self.runs += 1;
        }
        self.pending += 1;
    }

    fn finish(mut self) -> (DistinctInfo<T>, u32) {
        self.flush();
        let distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher> = match self.counts {
            Counts::Dense { min, counts, .. } => {
                let min = min.to_i128().vortex_expect("integers fit in i128");
                counts
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .map(|(index, &count)| {
                        let value = <T as num_traits::NumCast>::from(min + index as i128)
                            .vortex_expect("values between min and max fit in the type");
                        (NativeValue(value), count)
                    })
                    .collect()
            }
            Counts::Hashed(map) => map,
        };
        (DistinctInfo::new(distinct_values), self.runs)
    }
}

/// The type-erased summary of the [`Distinct`] statistic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistinctSummary {
    /// The number of distinct valid values.
    pub distinct_count: u32,
    /// The most frequent valid value.
    pub most_frequent: PValue,
    /// The number of occurrences of `most_frequent`.
    pub top_frequency: u32,
    /// The number of runs of equal consecutive valid values.
    pub runs: u32,
}

/// The key of the [`Distinct`] statistic.
pub struct DistinctStat;

impl IntStat for DistinctStat {
    type Value = DistinctSummary;
}

impl<T: IntValue> ErasedAccumulator<T> for Distinct<T>
where
    NativeValue<T>: Eq + Hash,
{
    fn finish_into(self, stats: &mut IntStats) {
        let (info, runs) = self.finish();
        let (most_frequent, top_frequency) = info.most_frequent();
        stats.insert::<DistinctStat>(DistinctSummary {
            distinct_count: info.distinct_count(),
            most_frequent: most_frequent.to_pvalue(),
            top_frequency,
            runs,
        });
    }
}
