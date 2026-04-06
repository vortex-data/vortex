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
    /// Build a `TurboQuantData` with validation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dimension` is less than [`MIN_DIMENSION`](TurboQuant::MIN_DIMENSION).
    /// - `bit_width` is greater than 8.
    pub fn try_new(dimension: u32, bit_width: u8) -> VortexResult<Self> {
        vortex_ensure!(
            dimension >= TurboQuant::MIN_DIMENSION,
            "TurboQuant requires dimension >= {}, got {dimension}",
            TurboQuant::MIN_DIMENSION
        );
        vortex_ensure!(
            bit_width <= 8,
            "bit_width is expected to be between 0 and 8, got {bit_width}"
        );
        Ok(Self {
            dimension,
            bit_width,
        })
    }

    /// Build a `TurboQuantData` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// - `dimension` is >= [`MIN_DIMENSION`](TurboQuant::MIN_DIMENSION).
    /// - `bit_width` is in the range `[0, 8]`.
    ///
    /// Violating these invariants may produce incorrect results during decompression.
    pub unsafe fn new_unchecked(dimension: u32, bit_width: u8) -> Self {
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
        let vector_metadata = TurboQuant::validate_dtype(dtype)?;
        let dimension = vector_metadata.dimensions();
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
        let element_ptype = vector_metadata.element_ptype();
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
