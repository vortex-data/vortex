// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::pipeline::N;
use crate::pipeline::bits::{BitView, BitViewMut};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vec::Selection;

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

    // Indicates where the selected elements are positioned within the vector.
    pub(super) selection: Selection,

    /// Additional buffers of data used by the vector, such as string data.
    #[allow(dead_code)]
    pub(super) data: Vec<ByteBuffer>,

    /// Marker defining the lifetime of the contents of the vector.
    pub(super) _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> View<'a> {
    #[inline(always)]
    pub fn selection(&self) -> Selection {
        self.selection
    }

    pub fn as_array<T>(&self) -> &'a [T; N]
    where
        T: Element,
    {
        debug_assert_eq!(self.vtype, T::vtype(), "Invalid type for canonical view");
        // SAFETY: We assume that the elements are of type T and that the view is valid.
        unsafe { &*(self.elements.cast::<T>() as *const [T; N]) }
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

    /// The position of the selected values of this buffer.
    /// One of:
    /// * All - all N values are selected.
    /// * Prefix - the first n values are selected where i is the true count of the kernel mask.
    /// * Mask - the values are in the positions indicated by the kernel mask.
    pub(super) selection: Selection,

    /// Marker defining the lifetime of the contents of the vector.
    pub(super) _marker: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> ViewMut<'a> {
    pub fn new<E: Element>(elements: &'a mut [E], validity: Option<BitViewMut<'a>>) -> Self {
        assert_eq!(elements.len(), N);
        Self {
            vtype: E::vtype(),
            elements: elements.as_mut_ptr().cast(),
            validity,
            data: vec![],
            selection: Selection::Prefix,
            _marker: Default::default(),
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

    /// Returns an immutable array of the elements in the vector.
    #[inline(always)]
    pub fn as_array<E: Element>(&self) -> &'a [E; N] {
        debug_assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        unsafe { &*(self.elements.cast::<E>() as *const [E; N]) }
    }

    /// Returns a mutable array of the elements in the vector, allowing for modification.
    #[inline(always)]
    pub fn as_array_mut<E: Element>(&mut self) -> &'a mut [E; N] {
        debug_assert_eq!(self.vtype, E::vtype(), "Invalid type for canonical view");
        unsafe { &mut *(self.elements.cast::<E>() as *mut [E; N]) }
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

    #[inline(always)]
    pub fn selection(&self) -> Selection {
        self.selection
    }

    pub fn set_selection(&mut self, selection: Selection) {
        self.selection = selection;
    }

    /// Flatten the view by bringing the selected elements of the mask to the beginning of
    pub fn flatten<E: Element>(&mut self, selection: &BitView<'_>) {
        assert_eq!(
            self.vtype,
            E::vtype(),
            "ViewMut::flatten_mask: type mismatch"
        );

        if matches!(self.selection, Selection::Prefix) {
            // Nothing to do, all elements are already selected.
            return;
        }

        match selection.true_count() {
            0 | N => {
                // If the mask has no true bits or all true bits, we are already flattened.
            }
            n if n > 3 * N / 4 => {
                // High density: use iter_zeros to compact by removing gaps
                let slice = self.as_array_mut::<E>();
                let mut write_idx = 0;
                let mut read_idx = 0;

                selection.iter_zeros(|zero_idx| {
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
                let slice = self.as_array_mut::<E>();
                selection.iter_ones(|idx| {
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

        self.selection = Selection::Prefix
    }
}
