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
use super::IntAccumulator;
use super::Nulls;
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
    /// The value of the current run, if `pending` is not zero.
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

    /// Starts the first run at `value`, if no value was seen. Every run has at least one pending
    /// occurrence until the next run starts, so a zero `pending` means no value was seen.
    #[inline(always)]
    fn start_run(&mut self, value: T) {
        if self.pending == 0 {
            self.prev = value;
            self.runs = 1;
        }
    }

    /// Accumulates valid values, after the first run started.
    ///
    /// The run state is copied into a local for the loop: the counts are reached through `self`,
    /// so every count update could otherwise alias the run state and force it out of registers.
    #[inline(always)]
    fn scan(&mut self, values: impl Iterator<Item = T>) {
        let mut run = Run {
            prev: self.prev,
            pending: self.pending,
            runs: self.runs,
        };
        match &mut self.counts {
            Counts::Dense {
                min_index, counts, ..
            } => run.scan(
                values,
                &mut Dense {
                    min_index: *min_index,
                    counts,
                },
            ),
            Counts::Hashed(map) => run.scan(values, map),
        }
        self.prev = run.prev;
        self.pending = run.pending;
        self.runs = run.runs;
    }
}

/// The current run of a [`Distinct`], held apart from its counts.
struct Run<T> {
    /// The value of the current run.
    prev: T,
    /// Occurrences of `prev` in the current run not yet counted.
    pending: u32,
    /// The number of runs so far.
    runs: u32,
}

impl<T: Copy + PartialEq> Run<T> {
    /// Accumulates `values`, counting the occurrences of each run that ends.
    #[inline(always)]
    fn scan(&mut self, values: impl Iterator<Item = T>, counter: &mut impl Counter<T>) {
        for value in values {
            if value != self.prev {
                counter.add(self.prev, self.pending);
                self.prev = value;
                self.pending = 0;
                self.runs += 1;
            }
            self.pending += 1;
        }
    }
}

/// Counts occurrences of values.
trait Counter<T> {
    /// Adds `count` occurrences of `value`.
    fn add(&mut self, value: T, count: u32);
}

impl<T: IntegerPType> Counter<T> for Counts<T>
where
    NativeValue<T>: Eq + Hash,
{
    fn add(&mut self, value: T, count: u32) {
        match self {
            Self::Dense {
                min_index, counts, ..
            } => Dense {
                min_index: *min_index,
                counts,
            }
            .add(value, count),
            Self::Hashed(map) => map.add(value, count),
        }
    }
}

/// Dense counts, borrowed from [`Counts::Dense`].
struct Dense<'a> {
    /// See [`Counts::Dense`].
    min_index: usize,
    /// See [`Counts::Dense`].
    counts: &'a mut [u32],
}

impl<T: IntegerPType> Counter<T> for Dense<'_> {
    #[inline(always)]
    fn add(&mut self, value: T, count: u32) {
        self.counts[value.as_().wrapping_sub(self.min_index)] += count;
    }
}

impl<T: Copy> Counter<T> for HashMap<NativeValue<T>, u32, FxBuildHasher>
where
    NativeValue<T>: Eq + Hash,
{
    #[inline(always)]
    fn add(&mut self, value: T, count: u32) {
        *self.entry(NativeValue(value)).or_insert(0) += count;
    }
}

impl<T: IntegerPType> IntAccumulator<T> for Distinct<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// The distinct values and the run count.
    type Output = (DistinctInfo<T>, u32);

    // A filled null would count as an occurrence, so only valid values are accumulated.
    const NULLS: Nulls = Nulls::Skip;

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.start_run(values[0]);
        if transitions(&self.prev, values) == 0 {
            self.pending += CHUNK_U32;
            return;
        }
        self.scan(values.iter().copied());
    }

    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
        self.start_run(filled[valid.trailing_zeros() as usize]);
        let mut valid = valid;
        self.scan(std::iter::from_fn(|| {
            (valid != 0).then(|| {
                let index = valid.trailing_zeros() as usize;
                valid &= valid - 1;
                filled[index]
            })
        }));
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.start_run(value);
        self.scan(std::iter::once(value));
    }

    fn finish(mut self) -> (DistinctInfo<T>, u32) {
        if self.pending > 0 {
            self.counts.add(self.prev, self.pending);
        }
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

int_stat!(
    Distinct,
    DistinctStat: DistinctSummary,
    |(info, runs): (DistinctInfo<T>, u32)| {
        let (most_frequent, top_frequency) = info.most_frequent();
        DistinctSummary {
            distinct_count: info.distinct_count(),
            most_frequent: most_frequent.to_pvalue(),
            top_frequency,
            runs,
        }
    },
    where NativeValue<T>: Eq + Hash
);
