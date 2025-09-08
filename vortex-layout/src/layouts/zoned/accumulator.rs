// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_error::VortexResult;

/// An accumulator that processes a stream of chunks. For each chunk, it accumulates into some
/// `Value` type which is yielded at the very end.
pub trait Accumulator {
    /// The value produced by accumulating chunks into the buffer.
    type Value;

    /// Push a new chunk into
    fn push_chunk(&mut self, chunk: &dyn Array) -> VortexResult<()>;

    /// Finish into a final `Value` element, or None if the chunk stream was not suitable for
    /// yielding the value type.
    fn finish(self) -> VortexResult<Option<Self::Value>>;
}
