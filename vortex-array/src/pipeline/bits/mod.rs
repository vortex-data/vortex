// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bit_sink;
mod chunked_iterator;
mod true_iterator;
mod vector;
mod view;
mod view_mut;

use arrow_buffer::BooleanBuffer;
pub use bit_sink::*;
pub use chunked_iterator::*;
pub use true_iterator::*;
pub use vector::*;
pub use view::*;
pub use view_mut::*;
use vortex_error::VortexResult;
use crate::pipeline::N_WORDS;

#[allow(clippy::len_without_is_empty)]
pub trait MaskSliceIterator {
    fn next_chunk(&mut self) -> Option<&[usize; N_WORDS]>;

    fn len(&self) -> usize;

    fn true_count(&self) -> usize;
}

/// Trait for writing bits in chunks of N (1024) bits at a time
pub trait BitSink {
    /// Get a mutable slice for writing the next chunk of N bits
    /// Returns a mutable reference to N_WORDS (16) usize values
    fn next_chunk(&mut self) -> Option<&mut [usize; N_WORDS]>;

    /// Commit exactly n bits from the current chunk (where n <= N)
    /// This finalizes the current chunk and prepares for the next one
    fn commit_n(&mut self, n: usize) -> VortexResult<()>;

    /// Finish writing and return the final BooleanBuffer
    fn finish(self) -> VortexResult<Option<BooleanBuffer>>;
}