// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length binary vector implementation.

use std::sync::Arc;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::VectorOps;
use crate::binaryview::BinaryViewType;
use crate::binaryview::vector_mut::BinaryViewVectorMut;
use crate::binaryview::view::{BinaryView, validate_views};

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

    /// Get the `index` item from the vector as a native `Slice` type.
    ///
    /// This function will panic is `index` is out of range for the vector's length.
    pub fn get(&self, index: usize) -> Option<&T::Slice> {
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

    fn len(&self) -> usize {
        self.views.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::{ByteBuffer, buffer};
    use vortex_mask::Mask;

    use crate::binaryview::view::BinaryView;
    use crate::binaryview::{StringVector, StringVectorMut};
    use crate::{VectorMutOps, VectorOps};

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
}
