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

// Default capacity for new string data buffers of 2MiB.
const BUFFER_CAPACITY: usize = 2 * 1024 * 1024;

/// A mutable vector of binary view data.
///
/// The immutable equivalent of this type is [`BinaryViewVector`].
#[derive(Clone, Debug)]
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

    /// Create a new empty [`BinaryViewVectorMut`], pre-allocated to hold the specified number
    /// of items. This does not reserve any memory for string data itself, only for the binary views
    /// and the validity bits.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(
            BufferMut::with_capacity(capacity),
            Vec::new(),
            MaskMut::with_capacity(capacity),
        )
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

    /// Append a repeated sequence of binary data to a vector.
    ///
    /// ```
    /// # use vortex_vector::binaryview::StringVectorMut;
    /// # use vortex_vector::VectorMutOps;
    /// let mut strings = StringVectorMut::with_capacity(4);
    /// strings.append_values("inlined", 2);
    /// strings.append_nulls(1);
    /// strings.append_values("large not inlined", 1);
    ///
    /// let strings = strings.freeze();
    ///
    /// assert_eq!(
    ///     [strings.get_ref(0), strings.get_ref(1), strings.get_ref(2), strings.get_ref(3)],
    ///     [Some("inlined"), Some("inlined"), None, Some("large not inlined")],
    /// );
    /// ```
    pub fn append_values(&mut self, value: &T::Slice, n: usize) {
        let bytes = value.as_ref();
        if bytes.len() <= BinaryView::MAX_INLINED_SIZE {
            self.views.push_n(BinaryView::new_inlined(bytes), n);
        } else {
            let buffer_index =
                u32::try_from(self.buffers.len()).vortex_expect("buffer count exceeds u32::MAX");

            let buf = self
                .open_buffer
                .get_or_insert_with(|| ByteBufferMut::with_capacity(BUFFER_CAPACITY));
            let offset = u32::try_from(buf.len()).vortex_expect("buffer length exceeds u32::MAX");
            buf.extend_from_slice(value.as_ref());

            self.views
                .push_n(BinaryView::make_view(bytes, buffer_index, offset), n);
        }

        self.validity.append_n(true, n);
    }

    /// Append a repeated sequence of binary data to a vector, from an owned buffer.
    ///
    /// The buffer will be used directly if possible, avoiding a copy.
    pub fn append_owned_values(&mut self, value: T::Scalar, n: usize) {
        let buffer: ByteBuffer = value.into();

        if buffer.len() <= BinaryView::MAX_INLINED_SIZE {
            self.views
                .push_n(BinaryView::new_inlined(buffer.as_ref()), n);
        } else {
            self.flush_open_buffer();

            let buffer_index = u32::try_from(self.buffers.len())
                .vortex_expect("buffer count exceeds u32::MAX")
                + 1;
            self.views
                .push_n(BinaryView::make_view(buffer.as_ref(), buffer_index, 0), n);
            self.buffers.push(buffer);
        }

        self.validity.append_n(true, n);
    }

    fn flush_open_buffer(&mut self) {
        if let Some(open) = self.open_buffer.take() {
            self.buffers.push(open.freeze());
        }
    }
}

impl<T: BinaryViewType> VectorMutOps for BinaryViewVectorMut<T> {
    type Immutable = BinaryViewVector<T>;

    fn len(&self) -> usize {
        self.views.len()
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        self.views.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
        self.validity.reserve(additional);
    }

    fn clear(&mut self) {
        self.views.clear();
        self.validity.clear();
        self.buffers.clear();
        self.open_buffer = None;
    }

    fn truncate(&mut self, len: usize) {
        self.views.truncate(len);
        self.validity.truncate(len);
    }

    fn extend_from_vector(&mut self, other: &BinaryViewVector<T>) {
        // Close any existing views into a new buffer
        self.flush_open_buffer();

        let offset =
            u32::try_from(self.buffers.len()).vortex_expect("buffer count exceeds u32::MAX");

        self.buffers.extend(other.buffers().iter().cloned());

        let new_views_iter = other.views().iter().copied().map(|mut v| {
            if v.is_inlined() {
                v
            } else {
                v.as_view_mut().buffer_index += offset;
                v
            }
        });
        self.views.extend(new_views_iter);

        self.validity.append_mask(other.validity())
    }

    fn append_nulls(&mut self, n: usize) {
        self.views.push_n(BinaryView::empty_view(), n);
        self.validity.append_n(false, n);
    }

    fn freeze(mut self) -> BinaryViewVector<T> {
        // Freeze all components, close any in-progress views
        self.flush_open_buffer();

        unsafe {
            BinaryViewVector::new_unchecked(
                self.views.freeze(),
                Arc::new(self.buffers.into_boxed_slice()),
                self.validity.freeze(),
            )
        }
    }

    fn split_off(&mut self, _at: usize) -> Self {
        todo!()
    }

    fn unsplit(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }

        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;
    use std::sync::Arc;

    use vortex_buffer::{ByteBuffer, buffer, buffer_mut};
    use vortex_mask::{Mask, MaskMut};

    use crate::binaryview::view::BinaryView;
    use crate::binaryview::{StringVector, StringVectorMut};
    use crate::{VectorMutOps, VectorOps};

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
        assert_eq!(strings.get_ref(0), Some("inlined1"));
        assert_eq!(strings.get_ref(1), Some("long string 1"));
        assert_eq!(strings.get_ref(2), Some("inlined2"));
        assert_eq!(strings.get_ref(3), Some("long string 2"));
        assert_eq!(strings.get_ref(4), Some("inlined3"));
        assert_eq!(strings.get_ref(5), Some("long string 3"));
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
            vec![buf0.clone(), buf1.clone()],
            MaskMut::new_true(6),
        );

        // The `StringVector` we extend from
        let strings = StringVector::new(
            buffer![BinaryView::make_view(
                b"a really very quite long string 2",
                0,
                33
            )],
            Arc::new(Box::new([buf1.clone()])),
            Mask::new_true(1),
        );

        strings_mut.extend_from_vector(&strings);

        let strings_finished = strings_mut.freeze();
        assert!(strings_finished.validity().all_true());

        assert_eq!(strings_finished.get_ref(0).unwrap(), "inlined0");
        assert_eq!(strings_finished.get_ref(1).unwrap(), "inlined1");
        assert_eq!(
            strings_finished.get_ref(2).unwrap(),
            "a really very quite long string 4"
        );
        assert_eq!(
            strings_finished.get_ref(3).unwrap(),
            "a really very quite long string 3"
        );
        assert_eq!(
            strings_finished.get_ref(4).unwrap(),
            "a really very quite long string 2",
        );
        assert_eq!(
            strings_finished.get_ref(5).unwrap(),
            "a really very quite long string 1"
        );
        assert_eq!(
            strings_finished.get_ref(6).unwrap(),
            "a really very quite long string 4"
        );

        assert_eq!(
            strings_finished.buffers().deref().as_ref(),
            &[buf0, buf1.clone(), buf1]
        );
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

        assert_eq!(strings_finished.get_ref(0), None);
        assert_eq!(strings_finished.get_ref(1), None);
        assert_eq!(strings_finished.get_ref(2), Some("nonnull1"));
        assert_eq!(strings_finished.get_ref(3), Some("nonnull2"));
        assert_eq!(strings_finished.get_ref(4), Some("extend1"));
        assert_eq!(strings_finished.get_ref(5), None);
        assert_eq!(strings_finished.get_ref(6), Some("extend2"));
    }
}
