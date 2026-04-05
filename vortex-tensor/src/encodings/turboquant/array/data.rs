// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::encodings::turboquant::array::slots::Slot;
use crate::encodings::turboquant::vtable::TurboQuant;
use crate::utils::extension_element_ptype;
use crate::utils::extension_list_size;

/// TurboQuant array data.
///
/// TurboQuant is a lossy vector quantization encoding for [`Vector`](crate::vector::Vector)
/// extension arrays. It stores quantized coordinate codes and per-vector norms, along with shared
/// codebook centroids and SRHT rotation signs.
///
/// See the [module docs](crate::encodings::turboquant) for algorithmic details.
///
/// A degenerate TurboQuant array has zero rows and `bit_width == 0`, with all slots empty.
#[derive(Clone, Debug)]
pub struct TurboQuantData {
    /// The vector dimension `d`, cached from the `FixedSizeList` storage dtype's list size.
    ///
    /// Stored as a convenience field to avoid repeatedly extracting it from `dtype`.
    pub(crate) dimension: u32,

    /// The number of bits per coordinate (1-8), derived from `log2(centroids.len())`.
    ///
    /// This is 0 for degenerate empty arrays.
    pub(crate) bit_width: u8,
}

impl TurboQuantData {
    /// Build a TurboQuant array with validation.
    ///
    /// The `dimension` and `bit_width` are derived from the inputs:
    /// - `dimension` from the `dtype`'s `FixedSizeList` storage list size.
    /// - `bit_width` from `log2(centroids.len())` (0 for degenerate empty arrays).
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented
    /// in [`new_unchecked`](Self::new_unchecked).
    pub fn try_new(
        dtype: &DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> VortexResult<Self> {
        Self::validate(dtype, &codes, &norms, &centroids, &rotation_signs)?;

        // SAFETY: we validate that the inputs are valid above.
        Ok(unsafe { Self::new_unchecked(dtype, codes, norms, centroids, rotation_signs) })
    }

    /// Build a TurboQuant array without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// - `dtype` is a [`Vector`](crate::vector::Vector) extension type whose storage list size
    ///   is >= [`MIN_DIMENSION`](crate::encodings::turboquant::TurboQuant::MIN_DIMENSION).
    /// - `codes` is a non-nullable `FixedSizeListArray<u8>` with `list_size == padded_dim` and
    ///   `codes.len() == norms.len()`. Null vectors are represented by all-zero codes.
    /// - `norms` is a primitive array whose ptype matches the element type of the Vector's storage
    ///   dtype. The nullability must match `dtype.nullability()`. Norms carry the validity of the
    ///   entire array, since null vectors have null norms.
    /// - `centroids` is a non-nullable `PrimitiveArray<f32>` whose length is a power of 2 in
    ///   `[2, 256]` (i.e., `2^bit_width` for bit_width 1-8), or empty for degenerate arrays.
    /// - `rotation_signs` has `3 * padded_dim` elements, or is empty for degenerate arrays.
    /// - For degenerate (empty) arrays: all children must be empty.
    ///
    /// Violating these invariants may produce incorrect results during decompression.
    pub unsafe fn new_unchecked(
        dtype: &DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(dtype, &codes, &norms, &centroids, &rotation_signs)
            .vortex_expect("[Debug Assertion]: Invalid TurboQuantData parameters");

        let dimension = dtype
            .as_extension_opt()
            .and_then(|ext| extension_list_size(ext).ok())
            .vortex_expect("dtype must be a Vector extension type with FixedSizeList storage");

        let bit_width = if centroids.is_empty() {
            0
        } else {
            // Guaranteed to be 1-8 by validate().
            #[expect(clippy::cast_possible_truncation)]
            {
                centroids.len().trailing_zeros() as u8
            }
        };

        Self {
            dimension,
            bit_width,
        }
    }

    /// Validates the components that would be used to create a `TurboQuantData`.
    ///
    /// This function checks all the invariants required by [`new_unchecked`](Self::new_unchecked).
    pub fn validate(
        dtype: &DType,
        codes: &ArrayRef,
        norms: &ArrayRef,
        centroids: &ArrayRef,
        rotation_signs: &ArrayRef,
    ) -> VortexResult<()> {
        let ext = TurboQuant::validate_dtype(dtype)?;
        let dimension = extension_list_size(ext)?;
        let padded_dim = dimension.next_power_of_two();

        // Codes must be a non-nullable FixedSizeList<u8> with list_size == padded_dim.
        // Null vectors are represented by all-zero codes since validity lives in the norms array.
        let expected_codes_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            padded_dim,
            Nullability::NonNullable,
        );
        vortex_ensure_eq!(
            *codes.dtype(),
            expected_codes_dtype,
            "codes dtype does not match expected {expected_codes_dtype}",
        );

        let num_rows = codes.len();
        vortex_ensure_eq!(
            norms.len(),
            num_rows,
            "norms length must match codes length",
        );

        // Degenerate (empty) case: all children must be empty, and bit_width is 0.
        if num_rows == 0 {
            vortex_ensure!(
                centroids.is_empty(),
                "degenerate TurboQuant must have empty centroids, got length {}",
                centroids.len()
            );
            vortex_ensure!(
                rotation_signs.is_empty(),
                "degenerate TurboQuant must have empty rotation_signs, got length {}",
                rotation_signs.len()
            );
            return Ok(());
        }

        // Non-degenerate: derive and validate bit_width from centroids.
        let num_centroids = centroids.len();
        vortex_ensure!(
            num_centroids.is_power_of_two() && (2..=256).contains(&num_centroids),
            "centroids length must be a power of 2 in [2, 256], got {num_centroids}"
        );

        // Guaranteed to be 1-8 by the preceding power-of-2 and range checks.
        #[expect(clippy::cast_possible_truncation)]
        let bit_width = num_centroids.trailing_zeros() as u8;
        vortex_ensure!(
            (1..=8).contains(&bit_width),
            "derived bit_width must be 1-8, got {bit_width}"
        );

        // Norms dtype must match the element ptype of the Vector, with the parent's nullability.
        // Norms carry the validity of the entire TurboQuant array.
        let element_ptype = extension_element_ptype(ext)?;
        let expected_norms_dtype = DType::Primitive(element_ptype, dtype.nullability());
        vortex_ensure_eq!(
            *norms.dtype(),
            expected_norms_dtype,
            "norms dtype does not match expected {expected_norms_dtype}",
        );

        // Centroids are always f32 regardless of element type.
        let centroids_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        vortex_ensure_eq!(
            *centroids.dtype(),
            centroids_dtype,
            "centroids dtype must be non-nullable f32",
        );

        // Rotation signs count must be 3 * padded_dim.
        vortex_ensure_eq!(
            rotation_signs.len(),
            3 * padded_dim as usize,
            "rotation_signs length does not match expected 3 * {padded_dim}",
        );

        Ok(())
    }

    pub(crate) fn validate_against_outer(
        &self,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(
            slots.len(),
            Slot::COUNT,
            "TurboQuantArray expects {} slots, got {}",
            Slot::COUNT,
            slots.len()
        );

        let codes = slots[Slot::Codes as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray codes slot");
        let norms = slots[Slot::Norms as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray norms slot");
        let centroids = slots[Slot::Centroids as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray centroids slot");
        let rotation_signs = slots[Slot::RotationSigns as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray rotation_signs slot");

        Self::validate(dtype, codes, norms, centroids, rotation_signs)?;
        vortex_ensure_eq!(
            norms.len(),
            len,
            "TurboQuant norms length does not match outer length",
        );
        vortex_ensure_eq!(
            self.dimension,
            extension_list_size(TurboQuant::validate_dtype(dtype)?)?
        );

        let expected_bit_width = if centroids.is_empty() {
            0
        } else {
            u8::try_from(centroids.len().trailing_zeros())
                .vortex_expect("centroids bit_width fits in u8")
        };
        vortex_ensure_eq!(
            self.bit_width,
            expected_bit_width,
            "TurboQuant bit_width does not match centroids slot",
        );

        Ok(())
    }

    pub(crate) fn make_slots(
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> Vec<Option<ArrayRef>> {
        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Norms as usize] = Some(norms);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);
        slots
    }

    /// The vector dimension `d`, as stored in the [`Vector`](crate::vector::Vector) extension
    /// dtype's `FixedSizeList` storage.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// MSE bits per coordinate (1-8 for non-empty arrays, 0 for degenerate empty arrays).
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Padded dimension (next power of 2 >= [`dimension`](Self::dimension)).
    ///
    /// The SRHT rotation requires power-of-2 input, so non-power-of-2 dimensions are
    /// zero-padded to this value.
    pub fn padded_dim(&self) -> u32 {
        self.dimension.next_power_of_two()
    }
}

pub trait TurboQuantArrayExt: TypedArrayRef<TurboQuant> {
    fn dimension(&self) -> u32 {
        std::ops::Deref::deref(self).dimension()
    }

    fn bit_width(&self) -> u8 {
        std::ops::Deref::deref(self).bit_width()
    }

    fn padded_dim(&self) -> u32 {
        std::ops::Deref::deref(self).padded_dim()
    }

    fn codes(&self) -> &ArrayRef {
        self.as_ref().slots()[Slot::Codes as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray codes slot")
    }

    fn norms(&self) -> &ArrayRef {
        self.as_ref().slots()[Slot::Norms as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray norms slot")
    }

    fn centroids(&self) -> &ArrayRef {
        self.as_ref().slots()[Slot::Centroids as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray centroids slot")
    }

    fn rotation_signs(&self) -> &ArrayRef {
        self.as_ref().slots()[Slot::RotationSigns as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray rotation_signs slot")
    }
}

impl<T: TypedArrayRef<TurboQuant>> TurboQuantArrayExt for T {}
