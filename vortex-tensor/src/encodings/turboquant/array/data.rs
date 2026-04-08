// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
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
/// extension arrays. It stores quantized coordinate codes for unit-norm vectors, along with shared
/// codebook centroids and the parameters of the current structured rotation.
///
/// Norms should be stored externally in the [`L2Denorm`](crate::scalar_fns::l2_denorm::L2Denorm)
/// `ScalarFnArray` wrapper.
///
/// See the [module docs](crate::encodings::turboquant) for algorithmic details.
///
/// Note that degenerate TurboQuant arrays have zero rows and `bit_width == 0`, with all slots
/// empty.
#[derive(Clone, Debug)]
pub struct TurboQuantData {
    /// The vector dimension `d`, cached from the `FixedSizeList` storage dtype's list size.
    ///
    /// Stored as a convenience field to avoid repeatedly extracting it from `dtype`.
    pub(crate) dimension: u32,

    /// The number of bits per coordinate (0-8), derived from `log2(centroids.len())`.
    ///
    /// This is 0 for degenerate empty arrays.
    pub(crate) bit_width: u8,

    /// The number of sign-diagonal + WHT rounds in the structured rotation.
    ///
    /// This is 0 for degenerate empty arrays.
    pub(crate) num_rounds: u8,
}

impl Display for TurboQuantData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dimension: {}, bit_width: {}, num_rounds: {}",
            self.dimension, self.bit_width, self.num_rounds
        )
    }
}

impl TurboQuantData {
    /// Build a `TurboQuantData` with validation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dimension` is less than [`MIN_DIMENSION`](TurboQuant::MIN_DIMENSION).
    /// - `bit_width` is greater than [`MAX_BIT_WIDTH`](TurboQuant::MAX_BIT_WIDTH).
    pub fn try_new(dimension: u32, bit_width: u8, num_rounds: u8) -> VortexResult<Self> {
        vortex_ensure!(
            dimension >= TurboQuant::MIN_DIMENSION,
            "TurboQuant requires dimension >= {}, got {dimension}",
            TurboQuant::MIN_DIMENSION
        );
        vortex_ensure!(
            bit_width <= TurboQuant::MAX_BIT_WIDTH,
            "bit_width is expected to be between 0 and {}, got {bit_width}",
            TurboQuant::MAX_BIT_WIDTH
        );

        Ok(Self {
            dimension,
            bit_width,
            num_rounds,
        })
    }

    /// Build a `TurboQuantData` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// - `dimension` is >= [`MIN_DIMENSION`](TurboQuant::MIN_DIMENSION).
    /// - `bit_width` is in the range `[0, MAX_BIT_WIDTH]`.
    /// - `num_rounds` is >= 1 (or 0 for degenerate empty arrays).
    ///
    /// Violating these invariants may produce incorrect results during decompression.
    pub unsafe fn new_unchecked(dimension: u32, bit_width: u8, num_rounds: u8) -> Self {
        Self {
            dimension,
            bit_width,
            num_rounds,
        }
    }

    /// Validates the components that would be used to create a `TurboQuantData`.
    ///
    /// This function checks all the invariants required by [`new_unchecked`](Self::new_unchecked).
    pub fn validate(
        dtype: &DType,
        codes: &ArrayRef,
        centroids: &ArrayRef,
        rotation_signs: &ArrayRef,
    ) -> VortexResult<()> {
        let vector_metadata = TurboQuant::validate_dtype(dtype)?;
        let dimension = vector_metadata.dimensions();
        let padded_dim = dimension.next_power_of_two();

        // TurboQuant arrays are always non-nullable. Nullability should be handled by the external
        // L2Denorm ScalarFnArray wrapper.
        vortex_ensure!(
            !dtype.is_nullable(),
            "TurboQuant dtype must be non-nullable, got {dtype}",
        );

        // Codes must be a non-nullable FixedSizeList<u8> with list_size == padded_dim.
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

        // Centroids are always f32 regardless of element type.
        let centroids_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        vortex_ensure_eq!(
            *centroids.dtype(),
            centroids_dtype,
            "centroids dtype must be non-nullable f32",
        );

        // Rotation signs must be a FixedSizeList<u8> with list_size == padded_dim. The FSL length
        // is the number of rotation rounds.
        let expected_signs_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            padded_dim,
            Nullability::NonNullable,
        );
        vortex_ensure_eq!(
            *rotation_signs.dtype(),
            expected_signs_dtype,
            "rotation_signs dtype does not match expected {expected_signs_dtype}",
        );
        // Degenerate (empty) case: all children must be empty, and bit_width is 0.
        let num_rows = codes.len();
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

        vortex_ensure!(
            !rotation_signs.is_empty(),
            "rotation_signs must have at least 1 round"
        );

        // Non-degenerate: derive and validate bit_width from centroids.
        let num_centroids = centroids.len();
        vortex_ensure!(
            num_centroids.is_power_of_two()
                && (2..=TurboQuant::MAX_CENTROIDS).contains(&num_centroids),
            "centroids length must be a power of 2 in [2, {}], got {num_centroids}",
            TurboQuant::MAX_CENTROIDS
        );

        #[expect(
            clippy::cast_possible_truncation,
            reason = "Guaranteed to be [1,8] by the preceding power-of-2 and range checks."
        )]
        let bit_width = num_centroids.trailing_zeros() as u8;
        vortex_ensure!(
            (1..=TurboQuant::MAX_BIT_WIDTH).contains(&bit_width),
            "derived bit_width must be 1-{}, got {bit_width}",
            TurboQuant::MAX_BIT_WIDTH
        );

        Ok(())
    }

    pub(crate) fn make_slots(
        codes: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> Vec<Option<ArrayRef>> {
        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);
        slots
    }

    /// The vector dimension `d`, as stored in the [`Vector`](crate::vector::Vector) extension
    /// dtype's `FixedSizeList` storage.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// MSE bits per coordinate (1-MAX_BIT_WIDTH for non-empty arrays, 0 for degenerate empty arrays).
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// The number of sign-diagonal + WHT rounds in the structured rotation.
    pub fn num_rounds(&self) -> u8 {
        self.num_rounds
    }

    /// Padded dimension (next power of 2 >= [`dimension`](Self::dimension)).
    ///
    /// The current Walsh-Hadamard-based structured rotation requires power-of-2 input, so
    /// non-power-of-2 dimensions are zero-padded to this value.
    pub fn padded_dim(&self) -> u32 {
        self.dimension.next_power_of_two()
    }
}

pub trait TurboQuantArrayExt: TypedArrayRef<TurboQuant> {
    fn codes(&self) -> &ArrayRef {
        self.as_ref().slots()[Slot::Codes as usize]
            .as_ref()
            .vortex_expect("TurboQuantArray codes slot")
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
