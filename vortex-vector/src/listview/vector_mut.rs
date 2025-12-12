// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVectorMut`].

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::MaskMut;

use super::ListViewScalar;
use super::ListViewVector;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorOps;
use crate::match_each_integer_pvector;
use crate::match_each_integer_pvector_mut;
use crate::primitive::PrimitiveVector;
use crate::primitive::PrimitiveVectorMut;
use crate::vector_ops::VectorMutOps;

/// A mutable vector of variable-width lists.
///
/// Each list is defined by 2 integers: an offset and a size (a "list view"), which point into a
/// child `elements` vector.
///
/// Note that the list views **do not** need to be sorted, nor do they have to be contiguous or
/// fully cover the `elements` vector. This means that multiple views can be pointing to the same
/// elements.
///
/// # Structure
///
/// - `elements`: The child vector of all list elements, stored as a [`Box<VectorMut>`].
/// - `offsets`: A [`PrimitiveVectorMut`] containing the starting offset of each list in the
///   `elements` vector.
/// - `sizes`: A [`PrimitiveVectorMut`] containing the size (number of elements) of each list.
/// - `validity`: A [`MaskMut`] indicating which lists are null.
#[derive(Debug, Clone)]
pub struct ListViewVectorMut {
    /// The mutable child vector of elements.
    pub(super) elements: Box<VectorMut>,

    /// Mutable offsets for each list into the elements array.
    ///
    /// Offsets are always integers, and always non-negative (even if the type is signed).
    pub(super) offsets: PrimitiveVectorMut,

    /// Mutable sizes (lengths) of each list.
    ///
    /// Sizes are always integers, and always non-negative (even if the type is signed).
    pub(super) sizes: PrimitiveVectorMut,

    /// The validity mask (where `true` represents a list is **not** null).
    ///
    /// Note that the `elements` vector will have its own internal validity, denoting if individual
    /// list elements are null.
    pub(super) validity: MaskMut,

    /// The length of the vector (which is the same as the length of the validity mask).
    ///
    /// This is stored here as a convenience, as the validity also tracks this information.
    pub(super) len: usize,
}

impl ListViewVectorMut {
    /// Creates a new [`ListViewVectorMut`] from its components.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `offsets` or `sizes` contain nulls values.
    /// - `offsets`, `sizes`, and `validity` do not all have the same length
    /// - The `sizes` integer width is not less than or equal to the `offsets` integer width (this
    ///   would cause overflow)
    /// - For any `i`, `offsets[i] + sizes[i]` causes an overflow or is greater than
    ///   `elements.len()` (even if the corresponding view is defined as null by the validity
    ///   array).
    pub fn new(
        elements: Box<VectorMut>,
        offsets: PrimitiveVectorMut,
        sizes: PrimitiveVectorMut,
        validity: MaskMut,
    ) -> Self {
        Self::try_new(elements, offsets, sizes, validity)
            .vortex_expect("Failed to create `ListViewVectorMut`")
    }

    /// Attempts to create a new [`ListViewVectorMut`] from its components.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - `offsets` or `sizes` contain nulls values.
    /// - `offsets`, `sizes`, and `validity` do not all have the same length
    /// - The `sizes` integer width is not less than or equal to the `offsets` integer width (this
    ///   would cause overflow)
    /// - For any `i`, `offsets[i] + sizes[i]` causes an overflow or is greater than
    ///   `elements.len()` (even if the corresponding view is defined as null by the validity
    ///   array).
    pub fn try_new(
        elements: Box<VectorMut>,
        offsets: PrimitiveVectorMut,
        sizes: PrimitiveVectorMut,
        validity: MaskMut,
    ) -> VortexResult<Self> {
        let len = validity.len();

        vortex_ensure!(
            offsets.len() == len,
            "Offsets length {} does not match validity length {len}",
            offsets.len(),
        );
        vortex_ensure!(
            sizes.len() == len,
            "Sizes length {} does not match validity length {len}",
            sizes.len(),
        );

        vortex_ensure!(
            offsets.validity().all_true(),
            "Offsets vector must not contain null values"
        );
        vortex_ensure!(
            sizes.validity().all_true(),
            "Sizes vector must not contain null values"
        );

        let offsets_width = offsets.ptype().byte_width();
        let sizes_width = sizes.ptype().byte_width();
        vortex_ensure!(
            sizes_width <= offsets_width,
            "Sizes integer width {sizes_width} must be \
                    <= offsets integer width {offsets_width} to prevent overflow",
        );

        // Check that each `offsets[i] + sizes[i] <= elements.len()`.
        validate_views_bound(elements.len() as u64, &offsets, &sizes)?;

        Ok(Self {
            elements,
            offsets,
            sizes,
            validity,
            len,
        })
    }

    /// Creates a new [`ListViewVectorMut`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - `offsets` and `sizes` must be non-nullable integer vectors.
    /// - `offsets`, `sizes`, and `validity` must have the same length.
    /// - Size integer width must be smaller than or equal to offset type (to prevent overflow).
    /// - For each `i`, `offsets[i] + sizes[i]` must not overflow and must be `<= elements.len()`
    ///   (even if the corresponding view is defined as null by the validity array).
    pub unsafe fn new_unchecked(
        elements: Box<VectorMut>,
        offsets: PrimitiveVectorMut,
        sizes: PrimitiveVectorMut,
        validity: MaskMut,
    ) -> Self {
        let len = validity.len();

        if cfg!(debug_assertions) {
            Self::new(elements, offsets, sizes, validity)
        } else {
            Self {
                elements,
                offsets,
                sizes,
                validity,
                len,
            }
        }
    }

    /// Creates a new [`ListViewVectorMut`] with the specified capacity.
    pub fn with_capacity(element_dtype: &DType, capacity: usize) -> Self {
        unsafe {
            Self::new_unchecked(
                Box::new(VectorMut::with_capacity(element_dtype, 0)),
                PrimitiveVectorMut::with_capacity(PType::U64, capacity),
                PrimitiveVectorMut::with_capacity(PType::U32, capacity),
                MaskMut::with_capacity(capacity),
            )
        }
    }

    /// Decomposes the [`ListViewVectorMut`] into its constituent parts (child elements, offsets,
    /// sizes, and validity).
    pub fn into_parts(
        self,
    ) -> (
        Box<VectorMut>,
        PrimitiveVectorMut,
        PrimitiveVectorMut,
        MaskMut,
    ) {
        (self.elements, self.offsets, self.sizes, self.validity)
    }

    /// Returns a reference to the elements vector.
    pub fn elements(&self) -> &VectorMut {
        &self.elements
    }

    /// Returns a reference to the offsets vector.
    pub fn offsets(&self) -> &PrimitiveVectorMut {
        &self.offsets
    }

    /// Returns a mutable handle to the offsets vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure that any offsets must be valid offsets within
    /// the elements.
    ///
    /// Caller must also ensure that offsets and sizes continue to be of same length.
    pub unsafe fn offsets_mut(&mut self) -> &mut PrimitiveVectorMut {
        &mut self.offsets
    }

    /// Returns a reference to the sizes vector.
    pub fn sizes(&self) -> &PrimitiveVectorMut {
        &self.sizes
    }

    /// Returns a mutable handle to the sizes vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure that any sizes, coupled with the corresponding offset,
    /// address valid ranges of elements.
    ///
    /// Caller must also ensure that offsets and sizes continue to be of same length.
    pub unsafe fn sizes_mut(&mut self) -> &mut PrimitiveVectorMut {
        &mut self.sizes
    }

    /// Returns a mutable handle to the validity mask of the vector.
    ///
    /// # Safety
    ///
    /// Callers must ensure modifying the length of the validity mask is only done
    /// with corresponding updates to length of the offsets and sizes.
    pub unsafe fn validity_mut(&mut self) -> &mut MaskMut {
        &mut self.validity
    }
}

impl VectorMutOps for ListViewVectorMut {
    type Immutable = ListViewVector;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        debug_assert!(
            self.offsets.capacity() <= self.sizes.capacity(),
            "the capacity of the sizes was somehow less than the offsets"
        );

        self.offsets.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.sizes.reserve(additional);
        self.elements.reserve(additional * 2); // Sane default TODO
        self.validity.reserve(additional);
    }

    fn clear(&mut self) {
        self.offsets.clear();
        self.sizes.clear();
        self.elements.clear();
        self.validity.clear();
        self.len = 0;
    }

    fn truncate(&mut self, len: usize) {
        self.offsets.truncate(len);
        self.sizes.truncate(len);
        self.validity.truncate(len);
        self.len = self.validity.len();
    }

    /// This will also panic if we try to extend the `ListViewVector` beyond the maximum offset
    /// representable by the type of the `offsets` primitive vector.
    fn extend_from_vector(&mut self, other: &ListViewVector) {
        // Extend the elements with the other's elements.
        let old_elements_len = self.elements.len() as u64;
        self.elements.extend_from_vector(&other.elements);
        let new_elements_len = self.elements.len() as u64;

        // Then extend the sizes with the other's sizes (these do not need any adjustment).
        self.sizes.extend_from_vector(&other.sizes);

        // We need this assertion to ensure that the casts below are infallible.
        assert!(
            new_elements_len < self.offsets.ptype().max_value_as_u64(),
            "the elements length {new_elements_len} is not representable by the offsets type {}",
            self.offsets.ptype()
        );

        // Finally, extend the offsets after adding the old `elements` length to each.
        adjust_and_extend_offsets(&mut self.offsets, &other.offsets, old_elements_len);

        self.validity.append_mask(&other.validity);
        self.len += other.len;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn append_nulls(&mut self, n: usize) {
        // To support easier copying to Arrow `List`s, we point the null views towards the ends of
        // the `elements` vector (with size 0) to hopefully keep offsets sorted if they were already
        // sorted.
        let elements_len = self.elements.len();

        debug_assert!(
            (elements_len as u64) < self.offsets.ptype().max_value_as_u64(),
            "the elements length {elements_len} is somehow not representable by the offsets type {}",
            self.offsets.ptype()
        );

        self.offsets.reserve(n);
        self.sizes.reserve(n);

        match_each_integer_pvector_mut!(&mut self.offsets, |offsets_vec| {
            for _ in 0..n {
                // SAFETY: We just reserved capacity for `n` elements above, and the cast must
                // succeed because the elements length must be representable by the offset type.
                #[allow(clippy::cast_possible_truncation)]
                unsafe {
                    offsets_vec.push_unchecked(elements_len as _)
                };
            }
        });

        match_each_integer_pvector_mut!(&mut self.sizes, |sizes_vec| {
            for _ in 0..n {
                // SAFETY: We just reserved capacity for `n` elements above, and `0` is
                // representable by all integer types.
                #[allow(clippy::cast_possible_truncation)]
                unsafe {
                    sizes_vec.push_unchecked(0 as _)
                };
            }
        });

        self.validity.append_n(false, n);
        self.len += n;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn append_zeros(&mut self, n: usize) {
        // To support easier copying to Arrow `List`s, we point the null views towards the ends of
        // the `elements` vector (with size 0) to hopefully keep offsets sorted if they were already
        // sorted.
        let elements_len = self.elements.len();

        debug_assert!(
            (elements_len as u64) < self.offsets.ptype().max_value_as_u64(),
            "the elements length {elements_len} is somehow not representable by the offsets type {}",
            self.offsets.ptype()
        );

        self.offsets.reserve(n);
        self.sizes.reserve(n);

        match_each_integer_pvector_mut!(&mut self.offsets, |offsets_vec| {
            for _ in 0..n {
                // SAFETY: We just reserved capacity for `n` elements above, and the cast must
                // succeed because the elements length must be representable by the offset type.
                #[allow(clippy::cast_possible_truncation)]
                unsafe {
                    offsets_vec.push_unchecked(elements_len as _)
                };
            }
        });

        match_each_integer_pvector_mut!(&mut self.sizes, |sizes_vec| {
            for _ in 0..n {
                // SAFETY: We just reserved capacity for `n` elements above, and `0` is
                // representable by all integer types.
                #[allow(clippy::cast_possible_truncation)]
                unsafe {
                    sizes_vec.push_unchecked(0 as _)
                };
            }
        });

        self.validity.append_n(true, n);
        self.len += n;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn append_scalars(&mut self, scalar: &ListViewScalar, n: usize) {
        if scalar.is_null() {
            self.append_nulls(n);
            return;
        }

        let offset = scalar
            .value()
            .offsets()
            .scalar_at(0)
            .to_usize()
            .vortex_expect("offset must be representable as usize");
        let size = scalar
            .value()
            .sizes()
            .scalar_at(0)
            .to_usize()
            .vortex_expect("size must be representable as usize");

        // Slice the elements vector to get the relevant elements for this list view.
        let elements = scalar.value().elements().slice(offset..offset + size);

        // Push the new elements onto our elements vector.
        let new_offset = self.elements.len();
        self.elements.extend_from_vector(&elements);

        match_each_integer_pvector_mut!(&mut self.offsets, |offsets_vec| {
            #[allow(clippy::cast_possible_truncation)]
            offsets_vec.append_values(new_offset as _, n)
        });

        match_each_integer_pvector_mut!(&mut self.sizes, |sizes_vec| {
            #[allow(clippy::cast_possible_truncation)]
            sizes_vec.append_values(size as _, n)
        });

        self.validity.append_n(true, n);
        self.len += n;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn freeze(self) -> ListViewVector {
        ListViewVector {
            offsets: self.offsets.freeze(),
            sizes: self.sizes.freeze(),
            elements: Arc::new(self.elements.freeze()),
            validity: self.validity.freeze(),
            len: self.len,
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

// TODO(connor): It would be better to separate everything inside the macros into its own function,
// but that would require adding another macro that sets a type `$type` to be used by the caller.
/// Checks that all views are `<= elements_len`.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from nested match_each_* macros"
)]
fn validate_views_bound(
    elements_len: u64,
    offsets: &PrimitiveVectorMut,
    sizes: &PrimitiveVectorMut,
) -> VortexResult<()> {
    let len = offsets.len();

    match_each_integer_pvector_mut!(&offsets, |offsets_vector| {
        match_each_integer_pvector_mut!(&sizes, |sizes_vector| {
            let offsets_slice = offsets_vector.as_ref();
            let sizes_slice = sizes_vector.as_ref();

            #[allow(clippy::unnecessary_cast)]
            for i in 0..len {
                let offset = offsets_slice[i] as u64;
                let size = sizes_slice[i] as u64;
                vortex_ensure!(offset + size <= elements_len);
            }
        });
    });

    Ok(())
}

// TODO(connor): It would be better to separate everything inside the macros into its own function,
// but that would require adding another macro that sets a type `$type` to be used by the caller.
/// Checks that all views are `<= elements_len`.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from nested match_each_* macros"
)]
fn adjust_and_extend_offsets(
    our_offsets: &mut PrimitiveVectorMut,
    other: &PrimitiveVector,
    old_elements_len: u64,
) {
    our_offsets.reserve(other.len());

    // Adjust each offset from `other` by adding the current elements length to each of the
    // incoming offsets.
    match_each_integer_pvector_mut!(our_offsets, |self_offsets| {
        match_each_integer_pvector!(other, |other_offsets| {
            let other_offsets_slice = other_offsets.as_ref();

            // Append each offset from `other`, adjusted by the elements_offset.
            for i in 0..other.len() {
                // All offset types are representable via a `u64` since we also ensure offsets
                // are always non-negative.
                #[allow(clippy::unnecessary_cast)]
                let adjusted_offset = other_offsets_slice[i] as u64 + old_elements_len;

                // SAFETY: We just reserved capacity for `other.len()` elements above, and we
                // also know the cast is fine because we verified above that the maximum
                // possible offset is representable by the offset type.
                #[allow(clippy::cast_possible_truncation)]
                unsafe {
                    self_offsets.push_unchecked(adjusted_offset as _);
                }
            }
        });
    });
}
