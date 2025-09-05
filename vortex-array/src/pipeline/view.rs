// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::pipeline::N;
use crate::pipeline::bits::{BitView, BitViewMut};
use crate::pipeline::types::{Element, VType};

pub struct View<'a> {
    /// The physical type of the vector, which defines how the elements are stored.
    pub(super) vtype: VType,
    /// A pointer to the allocated elements buffer.
    /// Alignment is at least the size of the element type.
    /// The capacity of the elements buffer is N * `size_of::<T>()` where T is the element type.
    pub(super) elements: *const u8,
    /// The validity mask for the vector, indicating which elements in the buffer are valid.
    /// This value can be `None` if the expected DType is `NonNullable`.
    // TODO: support validity
    #[allow(dead_code)]
    pub(super) validity: Option<BitView<'a>>,
    // A selection mask over the elements and validity of the vector.
    pub(super) len: usize,

    /// Additional buffers of data used by the vector, such as string data.
    #[allow(dead_code)]
    pub(super) data: Vec<ByteBuffer>,

    /// Marker defining the lifetime of the contents of the vector.
    pub(super) _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> View<'a> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice<T>(&self) -> &'a [T]
    where
        T: Element,
    {
        debug_assert_eq!(self.vtype, T::vtype(), "Invalid type for canonical view");
        // SAFETY: We assume that the elements are of type T and that the view is valid.
        unsafe { std::slice::from_raw_parts(self.elements.cast(), self.len) }
    }

    /// Re-interpret cast the vector into a new type where the element has the same width.
    #[inline(always)]
    pub fn reinterpret_as<E: Element>(&mut self) {
        assert_eq!(
            self.vtype.byte_width(),
            size_of::<E>(),
            "Cannot reinterpret {} as {}",
            self.vtype,
            E::vtype()
        );
        self.vtype = E::vtype();
    }
}

pub struct ViewMut<'a> {
    /// The physical type of the vector, which defines how the elements are stored.
    pub(super) vtype: VType,
    /// A pointer to the allocated elements buffer.
    /// Alignment is at least the size of the element type.
    /// The capacity of the elements buffer is N * `size_of::<T>()` where T is the element type.
    // TODO(ngates): it would be nice to guarantee _wider_ alignment, ideally 128 bytes, so that
    //  we can use aligned load/store instructions for wide SIMD lanes.
    pub(super) elements: *mut u8,
    /// The validity mask for the vector, indicating which elements in the buffer are valid.
    /// This value can be `None` if the expected DType is `NonNullable`.
    pub(super) validity: Option<BitViewMut<'a>>,

    /// Additional buffers of data used by the vector, such as string data.
    // TODO(ngates): ideally these buffers are compressed somehow? E.g. using FSST?
    #[allow(dead_code)]
    pub(super) data: Vec<ByteBuffer>,

    /// Marker defining the lifetime of the contents of the vector.
    pub(super) _marker: std::marker::PhantomData<&'a mut ()>,

    pub len: usize,
}

impl<'a> ViewMut<'a> {
    pub fn new<E: Element>(elements: &'a mut [E], validity: Option<BitViewMut<'a>>) -> Self {
        assert_eq!(elements.len(), N);
        Self {
            vtype: E::vtype(),
            elements: elements.as_mut_ptr().cast(),
            validity,
            data: vec![],
            _marker: Default::default(),
            len: elements.len(),
        }
    }

    /// Re-interpret cast the vector into a new type where the element has the same width.
    #[inline(always)]
    pub fn reinterpret_as<E: Element>(&mut self) {
        assert_eq!(
            self.vtype.byte_width(),
            size_of::<E>(),
            "Cannot reinterpret {} as {}",
            self.vtype,
            E::vtype()
        );
        self.vtype = E::vtype();
    }

    /// Returns an immutable slice of the elements in the vector.
    #[inline(always)]
    pub fn as_slice<E: Element>(&self) -> &'a [E] {
        debug_assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        unsafe { std::slice::from_raw_parts(self.elements.cast::<E>(), self.len) }
    }

    /// Returns a mutable slice of the elements in the vector, allowing for modification.
    #[inline(always)]
    pub fn as_slice_mut<E: Element>(&mut self) -> &'a mut [E] {
        debug_assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        unsafe { std::slice::from_raw_parts_mut(self.elements.cast::<E>(), self.len) }
    }

    /// Access the validity mask of the vector.
    ///
    /// ## Panics
    ///
    /// Panics if the vector does not support validity, i.e. if the DType was non-nullable when
    /// it was created.
    pub fn validity(&mut self) -> &mut BitViewMut<'a> {
        self.validity
            .as_mut()
            .vortex_expect("Vector does not support validity")
    }

    pub fn add_buffer(&mut self, buffer: ByteBuffer) {
        self.data.push(buffer);
    }

    /// Flatten the view by bringing the selected elements of the mask to the beginning of
    /// the elements buffer.
    ///
    /// FIXME(ngates): also need to select validity bits.
    pub fn select_mask<E: Element + Display>(&mut self, mask: &BitView) {
        assert_eq!(
            self.vtype,
            E::vtype(),
            "ViewMut::flatten_mask: type mismatch"
        );

        match mask.true_count() {
            0 => {
                // If the mask has no true bits, we set the length to 0.
            }
            n if n  == self.len => {
                // If the mask has N true bits, we copy all elements.
            }
            n if n > 3 * N / 4 => {
                // High density: use iter_zeros to compact by removing gaps
                let slice = self.as_slice_mut::<E>();
                let mut write_idx = 0;
                let mut read_idx = 0;

                mask.iter_zeros(|zero_idx| {
                    // Copy elements from read_idx to zero_idx (exclusive) to write_idx
                    let count = zero_idx - read_idx;
                    unsafe {
                        // SAFETY: We assume that the elements are of type E and that the view is valid.
                        // Using memmove for potentially overlapping regions
                        std::ptr::copy(
                            slice.as_ptr().add(read_idx),
                            slice.as_mut_ptr().add(write_idx),
                            count,
                        );
                        write_idx += count;
                    }
                    read_idx = zero_idx + 1;
                });

                // Copy any remaining elements after the last zero
                unsafe {
                    std::ptr::copy(
                        slice.as_ptr().add(read_idx),
                        slice.as_mut_ptr().add(write_idx),
                        N - read_idx,
                    );
                }
            }
            _ => {
                let mut offset = 0;
                let slice = self.as_slice_mut::<E>();
                mask.iter_ones(|idx| {
                    unsafe {
                        // SAFETY: We assume that the elements are of type E and that the view is valid.
                        let value = *slice.get_unchecked(idx);
                        // TODO(joe): use ptr increment (not offset).
                        *slice.get_unchecked_mut(offset) = value;

                        offset += 1;
                    }
                });
            }
        }
    }
}
