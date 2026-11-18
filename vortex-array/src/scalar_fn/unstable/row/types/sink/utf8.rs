// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 output storage for row kernels.
//!
//! [`Utf8Sink`] owns one initialized view per output row and batch-wide byte buffers. Its rows are
//! initialized to empty strings, so the sink uses `()` as its write token. A [`Utf8Writer`]
//! consumes the exact row handle when it replaces that placeholder.

use std::sync::Arc;

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::OutputSink;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::FillDefault;
use crate::scalar_fn::unstable::row::ViewLen;
use crate::validity::Validity;

/// An owned UTF-8 output sink for row functions.
pub struct Utf8Sink {
    views: BufferMut<BinaryView>,
    buffers: Vec<ByteBufferMut>,
    allocator: BufferAllocatorRef,
}

/// A borrowed view of all UTF-8 output rows.
pub struct Utf8Rows<'a> {
    views: &'a mut [BinaryView],
    buffers: &'a mut Vec<ByteBufferMut>,
    allocator: &'a BufferAllocatorRef,
}

impl ViewLen for Utf8Rows<'_> {
    fn len(&self) -> usize {
        self.views.len()
    }
}

// Every view starts as the empty string, which is already the default placeholder.
impl FillDefault for Utf8Rows<'_> {
    fn fill_default(&mut self) {}
}

/// The handle used to write one UTF-8 output row.
pub struct Utf8Writer<'a> {
    view: &'a mut BinaryView,
    buffers: &'a mut Vec<ByteBufferMut>,
    allocator: &'a BufferAllocatorRef,
}

impl Utf8Writer<'_> {
    /// Write a string whose byte length is known to fit in a [`BinaryView`].
    ///
    /// # Panics
    ///
    /// Panics if the length or data buffer index exceeds [`u32::MAX`]. Use
    /// [`try_write`](Self::try_write) when an output can exceed these limits.
    pub fn write(self, value: impl AsRef<str>) {
        self.try_write(value)
            .vortex_expect("UTF-8 row must fit in a BinaryView");
    }

    /// Write a string into this row, returning an error if its byte length or data buffer index
    /// exceeds [`u32::MAX`].
    pub fn try_write(self, value: impl AsRef<str>) -> VortexResult<()> {
        let bytes = value.as_ref().as_bytes();
        if bytes.len() <= BinaryView::MAX_INLINED_SIZE {
            *self.view = BinaryView::make_view(bytes, 0, 0);
            return Ok(());
        }

        let length = view_length(bytes.len())?;
        let needs_buffer = self
            .buffers
            .last()
            .is_none_or(|buffer| buffer.len().saturating_add(bytes.len()) > MAX_BUFFER_LEN);
        let index = if needs_buffer {
            self.buffers.len()
        } else {
            self.buffers.len() - 1
        };
        let buffer_index = u32::try_from(index)
            .map_err(|_| vortex_err!("UTF-8 data buffer index must fit in u32, got {index}"))?;
        if needs_buffer {
            self.buffers.push(self.allocator.with_capacity(bytes.len()));
        }

        let buffer = self
            .buffers
            .last_mut()
            .vortex_expect("Utf8Sink creates a data buffer before writing");
        let offset =
            u32::try_from(buffer.len()).vortex_expect("Utf8Sink buffer offset must fit in u32");

        buffer.extend_from_slice(bytes);
        let prefix = *bytes
            .first_chunk()
            .vortex_expect("referenced strings have at least four bytes");
        *self.view = BinaryView::new_ref(length, prefix, buffer_index, offset);
        Ok(())
    }
}

fn view_length(len: usize) -> VortexResult<u32> {
    u32::try_from(len).map_err(|_| vortex_err!("UTF-8 row byte length must fit in u32, got {len}"))
}

// SAFETY: `with_capacity` initializes one distinct `BinaryView` for every row. `rows` retains the
// view slice and byte-buffer vector without changing the row mapping. `row_unchecked` lends one
// exact view slot and the executor cannot request another until the consuming `Utf8Writer` is
// dropped. Skipped rows retain initialized empty views. Errors and unwinds can drop every field,
// and `finish` only publishes initialized views referencing the sink-owned frozen buffers.
unsafe impl OutputSink for Utf8Sink {
    type Params = ();
    type Rows<'a> = Utf8Rows<'a>;
    type Row<'a> = Utf8Writer<'a>;
    type WriteToken = ();

    fn storage_dtype(_params: &Self::Params) -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    fn with_capacity(
        rows: usize,
        _params: &Self::Params,
        allocator: &BufferAllocatorRef,
    ) -> VortexResult<Self> {
        let mut views = allocator.with_capacity(rows);
        views.push_n(BinaryView::empty_view(), rows);

        Ok(Self {
            views,
            buffers: Vec::new(),
            allocator: allocator.clone(),
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        Utf8Rows {
            views: self.views.as_mut_slice(),
            buffers: &mut self.buffers,
            allocator: &self.allocator,
        }
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        let view = unsafe { rows.views.get_unchecked_mut(index) };

        Utf8Writer {
            view,
            buffers: rows.buffers,
            allocator: rows.allocator,
        }
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        let views = BufferHandle::new_host(self.views.freeze().into_byte_buffer());
        let buffers = self
            .buffers
            .into_iter()
            .map(|buffer| BufferHandle::new_host(buffer.freeze()))
            .collect::<Vec<_>>();

        Ok(VarBinViewArray::new_handle(
            views,
            Arc::from(buffers),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::LazyLock;

    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Utf8Sink;
    use crate::VortexSessionExecute as _;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar_fn::unstable::row::OutputSink;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn view_length_checks_the_u32_boundary() -> VortexResult<()> {
        let max = u32::MAX as usize;
        assert_eq!(super::view_length(max)?, u32::MAX);
        assert!(super::view_length(max + 1).is_err());
        Ok(())
    }

    #[test]
    fn sink_writes_owned_borrowed_and_cow_values() -> VortexResult<()> {
        let expected = ["short", "a referenced string", "owned", "borrowed cow"];
        let referenced = String::from("a referenced string");
        let mut sink = <Utf8Sink as OutputSink>::with_capacity(
            expected.len(),
            &(),
            BufferAllocatorRef::static_ref(),
        )?;

        {
            let mut rows = <Utf8Sink as OutputSink>::rows(&mut sink);

            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 0) }.write("short");
            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 1) }
                .try_write(referenced)?;
            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 2) }
                .write(Cow::Owned("owned".to_owned()));
            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 3) }
                .write(Cow::Borrowed("borrowed cow"));
        }

        // SAFETY: every row was initialized by the writes above.
        let array = unsafe { <Utf8Sink as OutputSink>::finish(sink) }?;
        let mut ctx = SESSION.create_execution_ctx();
        let array = array.execute::<VarBinViewArray>(&mut ctx)?;
        let actual = (0..array.len())
            .map(|index| String::from_utf8_lossy(&array.bytes_at(index)).into_owned())
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);

        Ok(())
    }

    #[test]
    fn sink_finishes_empty_and_skipped_rows() -> VortexResult<()> {
        let empty =
            <Utf8Sink as OutputSink>::with_capacity(0, &(), BufferAllocatorRef::static_ref())?;
        // SAFETY: a zero-row sink has no rows to initialize.
        let empty = unsafe { <Utf8Sink as OutputSink>::finish(empty) }?;
        assert!(empty.is_empty());
        assert_eq!(empty.dtype(), &DType::Utf8(Nullability::NonNullable));

        let mut skipped =
            <Utf8Sink as OutputSink>::with_capacity(2, &(), BufferAllocatorRef::static_ref())?;
        <Utf8Sink as OutputSink>::initialize_skipped_rows(&mut <Utf8Sink as OutputSink>::rows(
            &mut skipped,
        ));
        // SAFETY: the skipped-row initializer initialized every row.
        let skipped = unsafe { <Utf8Sink as OutputSink>::finish(skipped) }?;
        let mut ctx = SESSION.create_execution_ctx();
        let skipped = skipped.execute::<VarBinViewArray>(&mut ctx)?;

        assert_eq!(skipped.len(), 2);
        assert_eq!(skipped.dtype(), &DType::Utf8(Nullability::NonNullable));
        assert_eq!(skipped.bytes_at(0).as_slice(), b"");
        assert_eq!(skipped.bytes_at(1).as_slice(), b"");

        Ok(())
    }
}
