// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The number of runs.

use num_traits::PrimInt;

use super::CHUNK;
use super::IntAccumulator;
use super::Nulls;
use super::transitions;

/// The number of runs of equal consecutive valid values. Nulls do not break runs.
#[derive(Debug, Clone, Copy)]
pub struct RunCount<T> {
    /// The last value seen, if any.
    prev: Option<T>,
    /// The number of value changes so far.
    changes: u32,
}

impl<T: PrimInt> RunCount<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: None,
            changes: 0,
        }
    }
}

impl<T: PrimInt> Default for RunCount<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PrimInt> IntAccumulator<T> for RunCount<T> {
    type Output = u32;

    // A filled null repeats its neighbour, so it adds no runs.
    const NULLS: Nulls = Nulls::Fill;

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        let prev = self.prev.unwrap_or(values[0]);
        self.changes += transitions(&prev, values);
        self.prev = Some(values[CHUNK - 1]);
    }

    #[inline]
    fn finish(self) -> u32 {
        self.changes + u32::from(self.prev.is_some())
    }
}

int_stat!(RunCount, RunCountStat: u32, |runs| runs);
