// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Where the fuzz generators draw their decisions from.

use std::ops::RangeInclusive;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;

/// A stream of decisions for a fuzz generator.
///
/// libFuzzer targets draw from [`Unstructured`]. The ephyra drivers draw from a list of choices
/// that ephyra can shrink one decision at a time.
pub trait FuzzSource {
    /// A value in `range`; its start once the source is exhausted.
    fn int_in_range(&mut self, range: RangeInclusive<usize>) -> Result<usize>;

    /// An index below `len`; an error when `len` is zero.
    fn choose_index(&mut self, len: usize) -> Result<usize>;

    /// A boolean; `false` once the source is exhausted.
    fn bool(&mut self) -> Result<bool>;

    /// Decode a value with a byte-driven decoder from the next part of the source.
    fn bytes<R>(&mut self, decode: impl FnOnce(&mut Unstructured<'_>) -> Result<R>) -> Result<R>;
}

impl FuzzSource for Unstructured<'_> {
    fn int_in_range(&mut self, range: RangeInclusive<usize>) -> Result<usize> {
        Unstructured::int_in_range(self, range)
    }

    fn choose_index(&mut self, len: usize) -> Result<usize> {
        Unstructured::choose_index(self, len)
    }

    fn bool(&mut self) -> Result<bool> {
        bool::arbitrary(self)
    }

    fn bytes<R>(&mut self, decode: impl FnOnce(&mut Unstructured<'_>) -> Result<R>) -> Result<R> {
        decode(self)
    }
}
