// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Mutable variable-length binary vector.

use std::sync::Arc;

use vortex_buffer::{BufferMut, ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use crate::binaryview::BinaryViewType;
use crate::binaryview::vector::BinaryViewVector;
use crate::binaryview::view::{BinaryView, validate_views};
use crate::{VectorMutOps, VectorOps};

/// Mutable variable-length binary vector.
#[derive(Clone, Debug)]

/// A mutable vector of binary view data.
///
/// The immutable equivalent of this type is [`BinaryViewVector`].
pub struct BinaryViewVectorMut<T: BinaryViewType> {
    /// Views into the binary data.
    views: BufferMut<BinaryView>,
    /// Validity mask for the vector.
    validity: MaskMut,

    /// The completed buffers holding referenced binary data.
    buffers: Vec<ByteBuffer>,
    /// The current buffer being appended to, if any.
    open_buffer: Option<ByteBufferMut>,

    /// Marker trait for the [`BinaryViewType`].
    _marker: std::marker::PhantomData<T>,
}

impl<T: BinaryViewType> BinaryViewVectorMut<T> {
    /// Create a new [`BinaryViewVectorMut`] from its components, panicking if validation fails.
    ///
    /// # Errors
    ///
    /// This function will panic if any of the validation checks performed by [`try_new`][Self::try_new]
    /// fails.
    pub fn new(views: BufferMut<BinaryView>, buffers: Vec<ByteBuffer>, validity: MaskMut) -> Self {
        Self::try_new(views, buffers, validity)
            .vortex_expect("Failed to create `BinaryViewVectorMut`")
    }

    /// Tries to create a new [`BinaryViewVectorMut`] from its components.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the views.
    ///
    /// Returns an error if the views reference any data that is not a valid buffer
    pub fn try_new(
        views: BufferMut<BinaryView>,
        buffers: Vec<ByteBuffer>,
        validity: MaskMut,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            views.len() == validity.len(),
            "views buffer length {} != validity length {}",
            views.len(),
            validity.len()
        );

        validate_views(&views, &buffers, |index| validity.value(index), T::validate)?;

        Ok(Self {
            views,
            buffers,
            validity,
            open_buffer: None,
            _marker: std::marker::PhantomData,
        })
    }

    /// Creates a new [`BinaryViewVectorMut`] from the given bits and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the views.
    pub unsafe fn new_unchecked(
        views: BufferMut<BinaryView>,
        validity: MaskMut,
        buffers: Vec<ByteBuffer>,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::new(views, buffers, validity)
        } else {
            Self {
                views,
                buffers,
                validity,
                open_buffer: None,
                _marker: std::marker::PhantomData,
            }
        }
    }
}

impl<T: BinaryViewType> VectorMutOps for BinaryViewVectorMut<T> {
    type Immutable = BinaryViewVector<T>;

    fn len(&self) -> usize {
        self.views.len()
    }

    fn capacity(&self) -> usize {
        self.views.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        // Close any existing views into a new buffer
        if let Some(open) = self.open_buffer.take() {
            self.buffers.push(open.freeze());
        }

        // We build a lookup table to map BinaryView's from the `other` to have
        // valid buffer indices in the current array.
        let mut buf_index_lookup: Vec<u32> = Vec::with_capacity(other.buffers().len());
        let mut new_buffers = Vec::new();
        for buffer in other.buffers().iter() {
            let ptr = buffer.as_ptr().addr();
            let new_index: u32 = self
                .buffers
                .iter()
                .position(|b| b.as_ptr().addr() == ptr)
                .unwrap_or_else(|| self.buffers.len() + new_buffers.len())
                .try_into()
                .vortex_expect("buffer index must fit in u32");

            if new_index as usize == new_buffers.len() {
                // We need to append the buffer
                new_buffers.push(buffer.clone());
            }

            buf_index_lookup.push(new_index);
        }

        // rewrite the views using our lookup table
        let new_views_iter = rewrite_views(other.views().iter().copied(), &buf_index_lookup);

        self.buffers.extend(new_buffers);
        self.views.extend(new_views_iter);
        self.validity.append_mask(other.validity())
    }

    fn append_nulls(&mut self, n: usize) {
        self.views.push_n(BinaryView::empty_view(), n);
        self.validity.append_n(false, n);
    }

    fn freeze(mut self) -> Self::Immutable {
        // Freeze all components, close any in-progress views
        if let Some(open) = self.open_buffer.take() {
            self.buffers.push(open.freeze());
        }

        unsafe {
            BinaryViewVector::new_unchecked(
                self.views.freeze(),
                Arc::new(self.buffers.into()),
                self.validity.freeze(),
            )
        }
    }

    fn split_off(&mut self, _at: usize) -> Self {
        todo!()
    }

    fn unsplit(&mut self, _other: Self) {
        todo!()
    }
}

/// Create a new iterator that yields views rewritten with a new buffer index.
#[inline]
fn rewrite_views(
    views: impl Iterator<Item = BinaryView>,
    buf_index_lookup: &[u32],
) -> impl Iterator<Item = BinaryView> {
    views.map(|mut view| {
        if view.is_inlined() {
            return view;
        }
        let view = view.as_view_mut();
        let old_index = view.buffer_index;
        let new_index = *buf_index_lookup
            .get(old_index as usize)
            .unwrap_or(&old_index);
        view.buffer_index = new_index;
        BinaryView { _ref: *view }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::{ByteBuffer, buffer, buffer_mut};
    use vortex_mask::{Mask, MaskMut};

    use crate::binaryview::view::BinaryView;
    use crate::{StringVector, StringVectorMut, VectorMutOps, VectorOps};

    #[test]
    fn test_basic() {
        let strings_mut = StringVectorMut::new(
            buffer_mut![
                BinaryView::new_inlined(b"inlined1"),
                BinaryView::make_view(b"long string 1", 0, 0),
                BinaryView::new_inlined(b"inlined2"),
                BinaryView::make_view(b"long string 2", 0, 13),
                BinaryView::new_inlined(b"inlined3"),
                BinaryView::make_view(b"long string 3", 0, 26),
            ],
            vec![ByteBuffer::copy_from(
                "long string 1long string 2long string 3",
            )],
            MaskMut::new_true(6),
        );

        let strings = strings_mut.freeze();
        assert_eq!(strings.get(0), Some("inlined1"));
        assert_eq!(strings.get(1), Some("long string 1"));
        assert_eq!(strings.get(2), Some("inlined2"));
        assert_eq!(strings.get(3), Some("long string 2"));
        assert_eq!(strings.get(4), Some("inlined3"));
        assert_eq!(strings.get(5), Some("long string 3"));
    }

    #[test]
    fn test_extend_self_reference() {
        let buf0 = ByteBuffer::copy_from(
            b"a really very quite long string 1a really very quite long string 2",
        );
        let buf1 = ByteBuffer::copy_from(
            b"a really very quite long string 3a really very quite long string 4",
        );

        let mut strings_mut = StringVectorMut::new(
            buffer_mut![
                BinaryView::new_inlined(b"inlined0"),
                BinaryView::new_inlined(b"inlined1"),
                BinaryView::make_view(b"a really very quite long string 4", 1, 33),
                BinaryView::make_view(b"a really very quite long string 3", 1, 0),
                BinaryView::make_view(b"a really very quite long string 2", 0, 33),
                BinaryView::make_view(b"a really very quite long string 1", 0, 0),
            ],
            vec![buf0, buf1.clone()],
            MaskMut::new_true(6),
        );

        // The `StringVector` we extend from
        let strings = StringVector::new(
            buffer![BinaryView::make_view(
                b"a really very quite long string 2",
                0,
                33
            )],
            Arc::new(Box::new([buf1])),
            Mask::new_true(1),
        );

        strings_mut.extend_from_vector(&strings);

        let strings_finished = strings_mut.freeze();
        assert!(strings_finished.validity().all_true());

        assert_eq!(strings_finished.get(0).unwrap(), "inlined0");
        assert_eq!(strings_finished.get(1).unwrap(), "inlined1");
        assert_eq!(
            strings_finished.get(2).unwrap(),
            "a really very quite long string 4"
        );
        assert_eq!(
            strings_finished.get(3).unwrap(),
            "a really very quite long string 3"
        );
        assert_eq!(
            strings_finished.get(4).unwrap(),
            "a really very quite long string 2",
        );
        assert_eq!(
            strings_finished.get(5).unwrap(),
            "a really very quite long string 1"
        );
        assert_eq!(
            strings_finished.get(6).unwrap(),
            "a really very quite long string 4"
        );

        assert_eq!(strings_finished.buffers().len(), 2);
    }

    #[test]
    fn test_extend_nulls() {
        // Extend multiple times, with nulls.
        let mut mask1 = MaskMut::with_capacity(4);
        mask1.append_n(false, 2);
        mask1.append_n(true, 2);

        let mut strings_mut = StringVectorMut::new(
            buffer_mut![
                BinaryView::empty_view(),
                BinaryView::empty_view(),
                BinaryView::new_inlined(b"nonnull1"),
                BinaryView::new_inlined(b"nonnull2"),
            ],
            vec![ByteBuffer::empty()],
            mask1,
        );

        let strings = StringVector::new(
            buffer![
                BinaryView::new_inlined(b"extend1"),
                BinaryView::empty_view(),
                BinaryView::new_inlined(b"extend2"),
            ],
            Arc::new(Box::new([ByteBuffer::empty()])),
            Mask::from_iter([true, false, true]),
        );

        strings_mut.extend_from_vector(&strings);
        let strings_finished = strings_mut.freeze();

        assert_eq!(strings_finished.get(0), None);
        assert_eq!(strings_finished.get(1), None);
        assert_eq!(strings_finished.get(2), Some("nonnull1"));
        assert_eq!(strings_finished.get(3), Some("nonnull2"));
        assert_eq!(strings_finished.get(4), Some("extend1"));
        assert_eq!(strings_finished.get(5), None);
        assert_eq!(strings_finished.get(6), Some("extend2"));
    }
}
