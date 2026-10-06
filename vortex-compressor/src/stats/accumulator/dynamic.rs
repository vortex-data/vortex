// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Statistics chosen at runtime.

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStats;
use super::Nulls;

/// An object-safe [`ErasedAccumulator`], for statistics chosen at runtime.
pub trait DynAccumulator<T: Copy> {
    /// See [`IntAccumulator::uses_fill`].
    fn uses_fill(&self) -> bool;

    /// See [`IntAccumulator::chunk`].
    fn chunk(&mut self, values: &[T; CHUNK]);

    /// See [`IntAccumulator::filled_chunk`].
    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64);

    /// See [`IntAccumulator::push`].
    fn push(&mut self, value: T);

    /// See [`IntAccumulator::block`].
    fn block(&mut self, chunks: &[[T; CHUNK]]);

    /// See [`IntAccumulator::filled_block`].
    fn filled_block(&mut self, filled: &[[T; CHUNK]], valid: &[u64]);

    /// See [`IntAccumulator::end_block`].
    fn end_block(&mut self);

    /// See [`ErasedAccumulator::finish_into`].
    fn finish_into(self: Box<Self>, stats: &mut IntStats);
}

impl<T: Copy, A: ErasedAccumulator<T>> DynAccumulator<T> for A {
    fn uses_fill(&self) -> bool {
        IntAccumulator::uses_fill(self)
    }

    fn chunk(&mut self, values: &[T; CHUNK]) {
        IntAccumulator::chunk(self, values);
    }

    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
        IntAccumulator::filled_chunk(self, filled, valid);
    }

    fn push(&mut self, value: T) {
        IntAccumulator::push(self, value);
    }

    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        IntAccumulator::block(self, chunks);
    }

    fn filled_block(&mut self, filled: &[[T; CHUNK]], valid: &[u64]) {
        IntAccumulator::filled_block(self, filled, valid);
    }

    fn end_block(&mut self) {
        IntAccumulator::end_block(self);
    }

    fn finish_into(self: Box<Self>, stats: &mut IntStats) {
        ErasedAccumulator::finish_into(*self, stats);
    }
}

/// Statistics chosen at runtime, which compose with statically typed ones.
///
/// Each statistic runs its own monomorphized loop over each L1-sized block, so the cost of
/// choosing it at runtime is one virtual call per block. In a fused loop, it costs one virtual
/// call per chunk.
///
/// ```ignore
/// let mut extra = DynStats::new();
/// if wants_sum {
///     extra.add(Sum::new());
/// }
/// let stats = compute(values, &validity, (Schedule::<_, FUSED>::new((MinMax::new(),)), extra));
/// ```
pub struct DynStats<T> {
    /// The statistics, in the order they were added.
    stats: Vec<Box<dyn DynAccumulator<T>>>,
}

impl<T: Copy> DynStats<T> {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self { stats: Vec::new() }
    }

    /// Adds a statistic.
    pub fn add(&mut self, stat: impl ErasedAccumulator<T> + 'static) -> &mut Self {
        self.stats.push(Box::new(stat));
        self
    }

    /// Returns the number of statistics.
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// Returns `true` if there are no statistics.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
}

impl<T: Copy> Default for DynStats<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> IntAccumulator<T> for DynStats<T> {
    /// The type-erased statistics.
    type Output = IntStats;

    // Decided at runtime by `uses_fill`.
    const NULLS: Nulls = Nulls::Skip;

    fn uses_fill(&self) -> bool {
        self.stats.iter().any(|stat| stat.uses_fill())
    }

    fn chunk(&mut self, values: &[T; CHUNK]) {
        for stat in &mut self.stats {
            stat.chunk(values);
        }
    }

    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
        for stat in &mut self.stats {
            stat.filled_chunk(filled, valid);
        }
    }

    fn push(&mut self, value: T) {
        for stat in &mut self.stats {
            stat.push(value);
        }
    }

    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        for stat in &mut self.stats {
            stat.block(chunks);
        }
    }

    fn filled_block(&mut self, filled: &[[T; CHUNK]], valid: &[u64]) {
        for stat in &mut self.stats {
            stat.filled_block(filled, valid);
        }
    }

    fn end_block(&mut self) {
        for stat in &mut self.stats {
            stat.end_block();
        }
    }

    fn finish(self) -> IntStats {
        let mut stats = IntStats::default();
        ErasedAccumulator::finish_into(self, &mut stats);
        stats
    }
}

impl<T: Copy> ErasedAccumulator<T> for DynStats<T> {
    fn finish_into(self, stats: &mut IntStats) {
        for stat in self.stats {
            stat.finish_into(stats);
        }
    }
}
