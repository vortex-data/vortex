// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Writes slices sequentially into the spare capacity of an empty [`BufferMut`].
///
/// This is useful when the caller knows the final output length up front and wants to avoid
/// repeatedly growing the initialized length while copying contiguous source slices.
///
/// All writes are bounds-checked against the declared output length; the check is a single
/// well-predicted branch per slice and is dominated by the copy itself. The unsafety is
/// contained to the copy into the bounds-checked spare-capacity range and the final
/// [`set_len`], which [`finish`] justifies from the writer's invariant that copies initialize
/// a contiguous prefix of the spare capacity.
///
/// [`set_len`]: BufferMut::set_len
/// [`finish`]: SpareBufferWriter::finish
#[must_use = "call `finish` to set the buffer length after writing"]
pub(crate) struct SpareBufferWriter<'a, T> {
    buffer: &'a mut BufferMut<T>,
    written: usize,
    output_len: usize,
}

impl<'a, T: Copy> SpareBufferWriter<'a, T> {
    /// Creates a writer for `output_len` values in `buffer`'s spare capacity.
    ///
    /// The target buffer must be empty and have capacity for at least `output_len` values.
    pub(crate) fn new(buffer: &'a mut BufferMut<T>, output_len: usize) -> VortexResult<Self> {
        vortex_ensure!(
            buffer.is_empty(),
            "slice copy buffer already has {} initialized values",
            buffer.len()
        );
        vortex_ensure!(
            output_len <= buffer.capacity(),
            "slice copy output length {output_len} exceeds buffer capacity {}",
            buffer.capacity()
        );

        Ok(Self {
            buffer,
            written: 0,
            output_len,
        })
    }

    /// Copies `source` into the next output slots.
    #[inline]
    pub(crate) fn copy_slice(&mut self, source: &[T]) -> VortexResult<()> {
        vortex_ensure!(
            source.len() <= self.output_len - self.written,
            "slice copy length {} exceeds remaining output length {}",
            source.len(),
            self.output_len - self.written
        );
        let end = self.written + source.len();
        let dst = &mut self.buffer.spare_capacity_mut()[self.written..end];
        // SAFETY: `MaybeUninit<T>` has the same layout as `T`, `dst` and `source` have equal
        // lengths, and `dst` lives in the exclusively borrowed buffer so the two cannot overlap.
        // This is `<[MaybeUninit<T>]>::write_copy_of_slice`, which is not yet stable.
        unsafe {
            ptr::copy_nonoverlapping(source.as_ptr(), dst.as_mut_ptr().cast::<T>(), source.len());
        }
        self.written = end;
        Ok(())
    }

    /// Sets the target buffer length after exactly `output_len` values have been written.
    pub(crate) fn finish(self) -> VortexResult<()> {
        vortex_ensure!(
            self.written == self.output_len,
            "slice copy length {} does not match declared output length {}",
            self.written,
            self.output_len
        );
        // SAFETY: `copy_slice` initializes the contiguous prefix `0..written` of the spare
        // capacity, and `new` checked that `output_len` does not exceed the capacity.
        unsafe {
            self.buffer.set_len(self.output_len);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;

    use super::SpareBufferWriter;

    #[test]
    fn writes_slices_into_spare_capacity() -> VortexResult<()> {
        let mut buffer = BufferMut::with_capacity(4);

        let mut writer = SpareBufferWriter::new(&mut buffer, 4)?;
        writer.copy_slice(&[1u16, 2])?;
        writer.copy_slice(&[3u16, 4])?;
        writer.finish()?;

        assert_eq!(buffer.as_slice(), &[1u16, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn rejects_overlong_copy() -> VortexResult<()> {
        let mut buffer = BufferMut::with_capacity(2);

        let mut writer = SpareBufferWriter::new(&mut buffer, 2)?;
        let error = writer.copy_slice(&[1u16, 2, 3]).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("exceeds remaining output length")
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_finish() -> VortexResult<()> {
        let mut buffer = BufferMut::with_capacity(2);

        let writer = SpareBufferWriter::<u16>::new(&mut buffer, 2)?;
        let error = writer.finish().unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match declared output length")
        );
        Ok(())
    }
}
