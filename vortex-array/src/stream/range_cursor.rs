// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads row ranges, in increasing order, from one array stream.

use std::ops::Range;

use futures::TryStreamExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::stream::ArrayStream;
use crate::stream::SendableArrayStream;

/// Reads row ranges from an array stream, where each range starts at or after the end of the
/// range before it.
///
/// Rows between two ranges are skipped, and the part of an array that a range does not use is
/// kept for the next range. The cursor never reads the stream again from its start, so a caller
/// that reads the same rows of a stream many times reads its arrays only once.
pub struct ArrayRangeCursor {
    stream: SendableArrayStream,
    /// The part of the last array read from the stream that no range has used yet.
    current: Option<ArrayRef>,
    /// The row of the stream where `current` starts.
    position: u64,
}

impl ArrayRangeCursor {
    /// A cursor at the first row of `stream`.
    pub fn new(stream: SendableArrayStream) -> Self {
        Self {
            stream,
            current: None,
            position: 0,
        }
    }

    /// Whether the cursor can still read `range`: a cursor does not go back to rows before its
    /// position.
    pub fn can_read(&self, range: &Range<u64>) -> bool {
        range.start >= self.position
    }

    /// Reads the rows of `range`. The range must not be empty, and it must start at or after the
    /// end of the range read before it.
    pub async fn read(&mut self, range: Range<u64>) -> VortexResult<ArrayRef> {
        if range.start < self.position || range.start >= range.end {
            vortex_bail!(
                "Range {range:?} is empty or starts before the cursor's row {}",
                self.position
            );
        }

        let dtype = self.stream.dtype().clone();
        let mut chunks = Vec::new();
        while self.position < range.end {
            let chunk = match self.current.take() {
                Some(chunk) => chunk,
                None => self.stream.try_next().await?.ok_or_else(|| {
                    vortex_err!("Range {range:?} ends after the stream's last row")
                })?,
            };
            if chunk.is_empty() {
                continue;
            }

            let skip = usize::try_from(range.start.saturating_sub(self.position))
                .map_or(chunk.len(), |skip| skip.min(chunk.len()));
            self.position += skip as u64;
            if skip == chunk.len() {
                continue;
            }

            let remaining = chunk.slice(skip..chunk.len())?;
            let take = usize::try_from(range.end - self.position)
                .map_or(remaining.len(), |take| take.min(remaining.len()));
            chunks.push(remaining.slice(0..take)?);
            self.position += take as u64;
            if take < remaining.len() {
                self.current = Some(remaining.slice(take..remaining.len())?);
            }
        }

        if chunks.len() == 1 {
            Ok(chunks.remove(0))
        } else {
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use futures::stream;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ArrayRangeCursor;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;
    use crate::stream::ArrayStreamAdapter;

    fn cursor() -> ArrayRangeCursor {
        let chunks = [
            Buffer::from_iter(0u64..2).into_array(),
            Buffer::from_iter(2u64..5).into_array(),
            Buffer::from_iter(5u64..8).into_array(),
        ];
        let dtype = chunks[0].dtype().clone();
        let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));
        ArrayRangeCursor::new(Box::pin(stream))
    }

    #[test]
    fn reads_ordered_ranges_across_chunks() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let mut cursor = cursor();

        let first = block_on(cursor.read(1..4))?;
        assert_arrays_eq!(first, Buffer::from_iter(1u64..4).into_array(), &mut ctx);

        let second = block_on(cursor.read(5..8))?;
        assert_arrays_eq!(second, Buffer::from_iter(5u64..8).into_array(), &mut ctx);

        assert!(!cursor.can_read(&(7..8)));
        assert!(block_on(cursor.read(7..8)).is_err());
        Ok(())
    }

    #[test]
    fn rejects_ranges_past_the_stream() {
        let mut cursor = cursor();
        assert!(block_on(cursor.read(6..9)).is_err());
    }
}
