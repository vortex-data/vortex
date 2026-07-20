// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BufferMut;

/// Writes slices sequentially into the spare capacity of an empty [`BufferMut`].
///
/// This is useful when the caller knows the final output length up front and wants to avoid
/// repeatedly growing the initialized length while copying contiguous source slices.
pub struct SpareBufferWriter<'a, T> {
    buffer: &'a mut BufferMut<T>,
    next: *mut T,
    remaining: usize,
    output_len: usize,
}

impl<'a, T: Copy> SpareBufferWriter<'a, T> {
    /// Creates a writer for `output_len` values in `buffer`'s spare capacity.
    ///
    /// The target buffer must be empty and have capacity for at least `output_len` values.
    pub fn new(buffer: &'a mut BufferMut<T>, output_len: usize) -> VortexResult<Self> {
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

        let next = buffer.spare_capacity_mut().as_mut_ptr().cast();
        Ok(Self {
            buffer,
            next,
            remaining: output_len,
            output_len,
        })
    }

    /// Copies `source` into the next output slots.
    #[inline]
    pub fn copy_slice(&mut self, source: &[T]) -> VortexResult<()> {
        vortex_ensure!(
            source.len() <= self.remaining,
            "slice copy length {} exceeds remaining output length {}",
            source.len(),
            self.remaining
        );
        // SAFETY: the check above proves that `source` fits in the remaining output slots.
        unsafe { self.copy_slice_unchecked(source) };
        Ok(())
    }

    /// Copies `source` to the next output slots without checking the remaining output length.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `source.len()` does not exceed the remaining output length.
    #[inline]
    pub unsafe fn copy_slice_unchecked(&mut self, source: &[T]) {
        debug_assert!(source.len() <= self.remaining);
        // SAFETY: `next` points to the first unwritten slot, the caller guarantees the source fits
        // in the remaining capacity, and the mutable buffer borrow prevents reallocation.
        unsafe {
            ptr::copy_nonoverlapping(source.as_ptr(), self.next, source.len());
            self.next = self.next.add(source.len());
        }
        self.remaining -= source.len();
    }

    /// Sets the target buffer length after exactly `output_len` values have been written.
    pub fn finish(self) -> VortexResult<()> {
        vortex_ensure!(
            self.remaining == 0,
            "slice copy length {} does not match declared output length {}",
            self.output_len - self.remaining,
            self.output_len
        );
        // SAFETY: successful calls to the copy methods initialized exactly `output_len` slots.
        unsafe {
            self.buffer.set_len(self.output_len);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::SpareBufferWriter;
    use crate::BufferMut;

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
