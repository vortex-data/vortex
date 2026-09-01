// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 output storage for row kernels.
//!
//! [`Utf8Sink`] owns one initialized view per output row and batch-wide byte buffers. Its rows are
//! initialized to empty strings, so the sink uses `()` as its write token. A [`Utf8Writer`]
//! consumes the exact row handle when it replaces that placeholder.

use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::OutputSink;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::ViewLen;
use crate::validity::Validity;

/// An owned UTF-8 output sink for row functions.
pub struct Utf8Sink {
    views: BufferMut<BinaryView>,
    buffers: Vec<ByteBufferMut>,
}

/// A borrowed view of all UTF-8 output rows.
pub struct Utf8Rows<'a> {
    views: &'a mut [BinaryView],
    buffers: &'a mut Vec<ByteBufferMut>,
}

impl ViewLen for Utf8Rows<'_> {
    fn len(&self) -> usize {
        self.views.len()
    }
}

/// The handle used to write one UTF-8 output row.
pub struct Utf8Writer<'a> {
    view: &'a mut BinaryView,
    buffers: &'a mut Vec<ByteBufferMut>,
}

impl Utf8Writer<'_> {
    /// Write a string into this row and consume the row handle.
    pub fn write(self, value: impl AsRef<str>) {
        let bytes = value.as_ref().as_bytes();
        if bytes.len() <= BinaryView::MAX_INLINED_SIZE {
            *self.view = BinaryView::make_view(bytes, 0, 0);
            return;
        }

        let needs_buffer = self
            .buffers
            .last()
            .is_none_or(|buffer| buffer.len().saturating_add(bytes.len()) > MAX_BUFFER_LEN);
        if needs_buffer {
            self.buffers.push(ByteBufferMut::with_capacity(bytes.len()));
        }

        let buffer_index = u32::try_from(self.buffers.len() - 1)
            .vortex_expect("Utf8Sink data buffer count must fit in u32");
        let buffer = self
            .buffers
            .last_mut()
            .vortex_expect("Utf8Sink creates a data buffer before writing");
        let offset =
            u32::try_from(buffer.len()).vortex_expect("Utf8Sink buffer offset must fit in u32");

        buffer.extend_from_slice(bytes);
        *self.view = BinaryView::make_view(bytes, buffer_index, offset);
    }
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

    fn skipped_rows_initializer() -> Option<fn(&mut Self::Rows<'_>)> {
        Some(|_| {})
    }

    fn storage_dtype(_params: &Self::Params) -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    fn with_capacity(rows: usize, _params: &Self::Params) -> VortexResult<Self> {
        let mut views = BufferMut::with_capacity(rows);
        views.push_n(BinaryView::empty_view(), rows);

        Ok(Self {
            views,
            buffers: Vec::new(),
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        Utf8Rows {
            views: self.views.as_mut_slice(),
            buffers: &mut self.buffers,
        }
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        let view = unsafe { rows.views.get_unchecked_mut(index) };

        Utf8Writer {
            view,
            buffers: rows.buffers,
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

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Utf8Sink;
    use crate::VortexSessionExecute as _;
    use crate::arrays::VarBinViewArray;
    use crate::scalar_fn::unstable::row::OutputSink;

    #[test]
    fn sink_writes_owned_borrowed_and_cow_values() -> VortexResult<()> {
        let expected = ["short", "a referenced string", "owned", "borrowed cow"];
        let referenced = String::from("a referenced string");
        let mut sink = <Utf8Sink as OutputSink>::with_capacity(expected.len(), &())?;

        {
            let mut rows = <Utf8Sink as OutputSink>::rows(&mut sink);

            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 0) }.write("short");
            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 1) }.write(referenced);
            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 2) }
                .write(Cow::Owned("owned".to_owned()));
            // SAFETY: each index is within the four-row sink and is written exactly once.
            unsafe { <Utf8Sink as OutputSink>::row_unchecked(&mut rows, 3) }
                .write(Cow::Borrowed("borrowed cow"));
        }

        // SAFETY: every row was initialized by the writes above.
        let array = unsafe { <Utf8Sink as OutputSink>::finish(sink) }?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let array = array.execute::<VarBinViewArray>(&mut ctx)?;
        let actual = (0..array.len())
            .map(|index| String::from_utf8_lossy(&array.bytes_at(index)).into_owned())
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);

        Ok(())
    }

    #[test]
    fn sink_finishes_empty_and_skipped_rows() -> VortexResult<()> {
        let empty = <Utf8Sink as OutputSink>::with_capacity(0, &())?;
        // SAFETY: a zero-row sink has no rows to initialize.
        let empty = unsafe { <Utf8Sink as OutputSink>::finish(empty) }?;
        assert!(empty.is_empty());

        let mut skipped = <Utf8Sink as OutputSink>::with_capacity(2, &())?;
        let initializer = <Utf8Sink as OutputSink>::skipped_rows_initializer()
            .expect("the UTF-8 sink initializes skipped rows");
        initializer(&mut <Utf8Sink as OutputSink>::rows(&mut skipped));
        // SAFETY: the skipped-row initializer initialized every row.
        let skipped = unsafe { <Utf8Sink as OutputSink>::finish(skipped) }?;
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let skipped = skipped.execute::<VarBinViewArray>(&mut ctx)?;

        assert_eq!(skipped.bytes_at(0).as_slice(), b"");
        assert_eq!(skipped.bytes_at(1).as_slice(), b"");

        Ok(())
    }
}
