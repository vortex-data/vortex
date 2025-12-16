// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length binary vector implementation.

use std::fmt::Debug;
use std::ops::BitAnd;
use std::ops::RangeBounds;
use std::sync::Arc;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::VectorOps;
use crate::binaryview::BinaryViewScalar;
use crate::binaryview::BinaryViewType;
use crate::binaryview::vector_mut::BinaryViewVectorMut;
use crate::binaryview::view::BinaryView;
use crate::binaryview::view::validate_views;

/// A variable-length binary vector.
///
/// This is the core vector for string and binary data.
#[derive(Debug, Clone)]
pub struct BinaryViewVector<T: BinaryViewType> {
    /// Views into the binary data.
    views: Buffer<BinaryView>,
    /// Buffers holding the referenced binary data.
    buffers: Arc<Box<[ByteBuffer]>>,
    /// Validity mask for the vector.
    validity: Mask,
    /// Marker trait for the [`BinaryViewType`].
    _marker: std::marker::PhantomData<T>,
}

impl<T: BinaryViewType> PartialEq for BinaryViewVector<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.views.len() != other.views.len() {
            return false;
        }
        // Validity patterns must match
        if self.validity != other.validity {
            return false;
        }
        // Compare all views, OR with !validity to ignore invalid positions
        self.views
            .iter()
            .zip(other.views.iter())
            .enumerate()
            .all(|(i, (self_view, other_view))| {
                // If invalid, treat as equal
                if !self.validity.value(i) {
                    return true;
                }
                // For valid elements, compare the actual byte content via the view
                let self_bytes: &[u8] = if self_view.is_inlined() {
                    self_view.as_inlined().value()
                } else {
                    let view_ref = self_view.as_view();
                    let buffer = &self.buffers[view_ref.buffer_index as usize];
                    &buffer[view_ref.as_range()]
                };

                let other_bytes: &[u8] = if other_view.is_inlined() {
                    other_view.as_inlined().value()
                } else {
                    let view_ref = other_view.as_view();
                    let buffer = &other.buffers[view_ref.buffer_index as usize];
                    &buffer[view_ref.as_range()]
                };

                self_bytes == other_bytes
            })
    }
}

impl<T: BinaryViewType> Eq for BinaryViewVector<T> {}

impl<T: BinaryViewType> BinaryViewVector<T> {
    /// Creates a new [`BinaryViewVector`] from the provided components.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate the consistency of the provided
    /// components.
    ///
    /// The caller must uphold all validation that would otherwise be validated by
    /// the [safe constructor](Self::try_new).
    pub unsafe fn new_unchecked(
        views: Buffer<BinaryView>,
        buffers: Arc<Box<[ByteBuffer]>>,
        validity: Mask,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::new(views, buffers, validity)
        } else {
            Self {
                views,
                validity,
                buffers,
                _marker: std::marker::PhantomData,
            }
        }
    }

    /// Create a new `BinaryViewVector` from its components, panicking if validation fails.
    ///
    /// # Errors
    ///
    /// This function will panic if any of the validation checks performed by
    /// [`try_new`](Self::try_new) fails.
    pub fn new(views: Buffer<BinaryView>, buffers: Arc<Box<[ByteBuffer]>>, validity: Mask) -> Self {
        Self::try_new(views, buffers, validity).vortex_expect("Failed to create `BinaryViewVector`")
    }

    /// Create a new [`BinaryViewVector`] from the provided components with validation.
    ///
    /// # Errors
    ///
    /// This function will return an error if any of the following validation checks fails:
    ///
    /// 1. The length of the `views` does not match the length of the provided `validity`
    /// 2. Any non-null `views` point to invalid `buffers` or buffer offset ranges
    /// 3. Any data stored inlined or in the `buffers` and referenced by the `views` does not
    ///    conform to the [validation constraints][BinaryViewType::validate] of this view type.
    pub fn try_new(
        views: Buffer<BinaryView>,
        buffers: Arc<Box<[ByteBuffer]>>,
        validity: Mask,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            views.len() == validity.len(),
            "views buffer length {} != validity length {}",
            views.len(),
            validity.len()
        );

        validate_views(
            &views,
            &*buffers,
            |index| validity.value(index),
            T::validate,
        )?;

        Ok(Self {
            views,
            buffers,
            validity,
            _marker: std::marker::PhantomData,
        })
    }

    /// Decomposes the vector into its constituent parts.
    pub fn into_parts(self) -> (Buffer<BinaryView>, Arc<Box<[ByteBuffer]>>, Mask) {
        (self.views, self.buffers, self.validity)
    }

    /// Get the `index` item from the vector as an owned `Scalar` type with zero-copy.
    ///
    /// This function will panic is `index` is out of range for the vector's length.
    pub fn get(&self, index: usize) -> Option<T::Scalar> {
        if !self.validity.value(index) {
            return None;
        }

        let view = &self.views[index];
        if view.is_inlined() {
            let view = view.as_inlined();

            // We find the occurrence of the inlined data in the views buffer.
            let buffer = self
                .views
                .clone()
                .into_byte_buffer()
                .aligned(Alignment::none())
                .slice_ref(&view.data[..view.size as usize]);

            // SAFETY: validation that the string data contained in this vector is performed
            //  at construction time, either in the constructor for safe construction, or by
            //  the caller (when using the unchecked constructor).
            Some(unsafe { T::scalar_from_buffer_unchecked(buffer) })
        } else {
            // Get a pointer into the buffer range
            let view_ref = view.as_view();
            let buffer = &self.buffers[view_ref.buffer_index as usize];

            let start = view_ref.offset as usize;
            let length = view_ref.size as usize;
            let buffer_slice = buffer.slice(start..start + length);

            // SAFETY: validation that the string data contained in this vector is performed
            //  at construction time, either in the constructor for safe construction, or by
            //  the caller (when using the unchecked constructor).
            Some(unsafe { T::scalar_from_buffer_unchecked(buffer_slice) })
        }
    }

    /// Get the `index` item from the vector as a native `Slice` type.
    ///
    /// This function will panic is `index` is out of range for the vector's length.
    pub fn get_ref(&self, index: usize) -> Option<&T::Slice> {
        if !self.validity.value(index) {
            return None;
        }

        let view = &self.views[index];
        if view.is_inlined() {
            let view = view.as_inlined();
            // SAFETY: validation that the string data contained in this vector is performed
            //  at construction time, either in the constructor for safe construction, or by
            //  the caller (when using the unchecked constructor).
            Some(unsafe { T::from_bytes_unchecked(&view.data[..view.size as usize]) })
        } else {
            // Get a pointer into the buffer range
            let view_ref = view.as_view();
            let buffer = &self.buffers[view_ref.buffer_index as usize];

            let start = view_ref.offset as usize;
            let length = view_ref.size as usize;

            // SAFETY: validation that the string data contained in this vector is performed
            //  at construction time, either in the constructor for safe construction, or by
            //  the caller (when using the unchecked constructor).
            Some(unsafe { T::from_bytes_unchecked(&buffer.as_bytes()[start..start + length]) })
        }
    }

    /// Buffers
    pub fn buffers(&self) -> &Arc<Box<[ByteBuffer]>> {
        &self.buffers
    }

    /// Views
    pub fn views(&self) -> &Buffer<BinaryView> {
        &self.views
    }
}

impl<T: BinaryViewType> VectorOps for BinaryViewVector<T> {
    type Mutable = BinaryViewVectorMut<T>;
    type Scalar = BinaryViewScalar<T>;

    fn len(&self) -> usize {
        self.views.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn mask_validity(&mut self, mask: &Mask) {
        self.validity = self.validity.bitand(mask);
    }

    fn scalar_at(&self, index: usize) -> BinaryViewScalar<T> {
        assert!(index < self.len());
        BinaryViewScalar::<T>::new(self.get(index))
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        BinaryViewVector {
            views: self.views.slice(range.clone()),
            buffers: self.buffers().clone(),
            validity: self.validity.slice(range),
            _marker: self._marker,
        }
    }

    fn clear(&mut self) {
        self.views.clear();
        self.validity = Mask::new_true(0);
        self.buffers = Arc::new(Box::new([]));
    }

    fn try_into_mut(self) -> Result<BinaryViewVectorMut<T>, Self> {
        let views_mut = match self.views.try_into_mut() {
            Ok(views_mut) => views_mut,
            Err(views) => {
                return Err(Self {
                    views,
                    validity: self.validity,
                    buffers: self.buffers,
                    _marker: std::marker::PhantomData,
                });
            }
        };

        let validity_mut = match self.validity.try_into_mut() {
            Ok(validity_mut) => validity_mut,
            Err(validity) => {
                return Err(Self {
                    views: views_mut.freeze(),
                    validity,
                    buffers: self.buffers,
                    _marker: std::marker::PhantomData,
                });
            }
        };

        let buffers_mut = match Arc::try_unwrap(self.buffers) {
            Ok(buffers) => buffers.into_vec(),
            Err(buffers) => {
                // Backup: collect a new Vec with clones of each buffer
                buffers.iter().cloned().collect()
            }
        };

        // SAFETY: the BinaryViewVector maintains the same invariants that are
        //  otherwise checked in the safe BinaryViewVectorMut constructor.
        unsafe {
            Ok(BinaryViewVectorMut::new_unchecked(
                views_mut,
                validity_mut,
                buffers_mut,
            ))
        }
    }

    fn into_mut(self) -> BinaryViewVectorMut<T> {
        let views_mut = self.views.into_mut();
        let validity_mut = self.validity.into_mut();

        // If someone else has a strong reference to the `Arc`, clone the underlying data (which is
        // just a **different** reference count increment).
        let buffers_mut = Arc::try_unwrap(self.buffers)
            .unwrap_or_else(|arc| (*arc).clone())
            .into_vec();

        // SAFETY: The BinaryViewVector maintains the exact same invariants as the immutable
        // version, so all invariants are still upheld.
        unsafe { BinaryViewVectorMut::new_unchecked(views_mut, validity_mut, buffers_mut) }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::VectorMutOps;
    use crate::VectorOps;
    use crate::binaryview::StringVector;
    use crate::binaryview::StringVectorMut;
    use crate::binaryview::view::BinaryView;

    #[test]
    #[should_panic(expected = "views buffer length 1 != validity length 100")]
    fn test_try_new_mismatch_validity_len() {
        StringVector::try_new(
            buffer![BinaryView::new_inlined(b"inlined")],
            Arc::new(Box::new([])),
            Mask::new_true(100),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(
        expected = "view at index 0 references invalid buffer: 100 out of bounds for BinaryViewVector with 0 buffers"
    )]
    fn test_try_new_invalid_buffer_offset() {
        StringVector::try_new(
            buffer![BinaryView::make_view(b"bad buffer ptr", 100, 0)],
            Arc::new(Box::new([])),
            Mask::new_true(1),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "start offset 4294967295 out of bounds for buffer 0 with size 19")]
    fn test_try_new_invalid_length() {
        StringVector::try_new(
            buffer![BinaryView::make_view(b"bad buffer ptr", 0, u32::MAX)],
            Arc::new(Box::new([ByteBuffer::copy_from(b"a very short buffer")])),
            Mask::new_true(1),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "view at index 0: inlined bytes failed utf-8 validation")]
    fn test_try_new_invalid_utf8_inlined() {
        StringVector::try_new(
            buffer![BinaryView::new_inlined(b"\x80")],
            Arc::new(Box::new([])),
            Mask::new_true(1),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "view at index 0: outlined bytes failed utf-8 validation")]
    fn test_try_new_invalid_utf8_outlined() {
        // 0xFF is never valid in UTF-8
        let sequence = b"\xff".repeat(13);
        StringVector::try_new(
            buffer![BinaryView::make_view(&sequence, 0, 0)],
            Arc::new(Box::new([ByteBuffer::copy_from(sequence)])),
            Mask::new_true(1),
        )
        .unwrap();
    }

    #[test]
    fn test_try_into_mut() {
        let mut shared_vec = StringVectorMut::with_capacity(5);
        shared_vec.append_nulls(2);
        shared_vec.append_values("an example value", 2);
        shared_vec.append_values("another example value", 1);

        let shared_vec = shared_vec.freeze();

        // Making a copy aliases the vector, preventing us from converting it back into mutable
        let shared_vec2 = shared_vec.clone();

        // The Err variant is returned, because the aliasing borrow from shared_vec2 is blocking us
        // from taking unique ownership of the memory.
        let shared_vec = shared_vec.try_into_mut().unwrap_err();

        // Dropping the aliasing borrow makes it possible to cast the unique reference to mut
        drop(shared_vec2);

        assert!(shared_vec.try_into_mut().is_ok());
    }

    #[test]
    fn test_binaryview_eq_identical_inlined() {
        // Test equality with inlined strings (<=12 bytes).
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_values("hello", 1);
        v1.append_values("world", 1);
        v1.append_values("test", 1);
        let v1 = v1.freeze();

        let mut v2 = StringVectorMut::with_capacity(3);
        v2.append_values("hello", 1);
        v2.append_values("world", 1);
        v2.append_values("test", 1);
        let v2 = v2.freeze();

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_identical_outlined() {
        // Test equality with outlined strings (>12 bytes).
        let mut v1 = StringVectorMut::with_capacity(2);
        v1.append_values("this is a longer string that won't be inlined", 1);
        v1.append_values("another long string for testing purposes", 1);
        let v1 = v1.freeze();

        let mut v2 = StringVectorMut::with_capacity(2);
        v2.append_values("this is a longer string that won't be inlined", 1);
        v2.append_values("another long string for testing purposes", 1);
        let v2 = v2.freeze();

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_different_length() {
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_values("a", 1);
        v1.append_values("b", 1);
        v1.append_values("c", 1);
        let v1 = v1.freeze();

        let mut v2 = StringVectorMut::with_capacity(2);
        v2.append_values("a", 1);
        v2.append_values("b", 1);
        let v2 = v2.freeze();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_different_validity() {
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_values("a", 1);
        v1.append_values("b", 1);
        v1.append_values("c", 1);
        let v1 = v1.freeze();

        let mut v2 = StringVectorMut::with_capacity(3);
        v2.append_values("a", 1);
        v2.append_nulls(1);
        v2.append_values("c", 1);
        let v2 = v2.freeze();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_different_values() {
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_values("hello", 1);
        v1.append_values("world", 1);
        v1.append_values("test", 1);
        let v1 = v1.freeze();

        let mut v2 = StringVectorMut::with_capacity(3);
        v2.append_values("hello", 1);
        v2.append_values("DIFFERENT", 1);
        v2.append_values("test", 1);
        let v2 = v2.freeze();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_ignores_invalid_positions_inlined() {
        // Two vectors with different values at invalid positions should be equal.
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_values("hello", 1);
        v1.append_values("value_a", 1); // This will be masked as invalid
        v1.append_values("test", 1);
        let mut v1 = v1.freeze();
        // Mask position 1 as invalid
        v1.mask_validity(&Mask::from_iter([true, false, true]));

        let mut v2 = StringVectorMut::with_capacity(3);
        v2.append_values("hello", 1);
        v2.append_values("value_b", 1); // Different value at invalid position
        v2.append_values("test", 1);
        let mut v2 = v2.freeze();
        v2.mask_validity(&Mask::from_iter([true, false, true]));

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_ignores_invalid_positions_outlined() {
        // Test with outlined strings at invalid positions.
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_values("this is a very long string that will be outlined", 1);
        v1.append_values("another long value that differs between vectors A", 1);
        v1.append_values("yet another long string for the test", 1);
        let mut v1 = v1.freeze();
        v1.mask_validity(&Mask::from_iter([true, false, true]));

        let mut v2 = StringVectorMut::with_capacity(3);
        v2.append_values("this is a very long string that will be outlined", 1);
        v2.append_values("different long value at the invalid position B", 1);
        v2.append_values("yet another long string for the test", 1);
        let mut v2 = v2.freeze();
        v2.mask_validity(&Mask::from_iter([true, false, true]));

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_empty() {
        let v1 = StringVectorMut::with_capacity(0).freeze();
        let v2 = StringVectorMut::with_capacity(0).freeze();

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_binaryview_eq_all_nulls() {
        let mut v1 = StringVectorMut::with_capacity(3);
        v1.append_nulls(3);
        let v1 = v1.freeze();

        let mut v2 = StringVectorMut::with_capacity(3);
        v2.append_nulls(3);
        let v2 = v2.freeze();

        assert_eq!(v1, v2);
    }
}
