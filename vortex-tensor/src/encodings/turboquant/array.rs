// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant array definition: stores quantized coordinate codes, norms,
//! centroids (codebook), and rotation signs.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::utils::extension_element_ptype;
use crate::utils::extension_list_size;
use crate::vector::Vector;

/// Encoding marker type for TurboQuant.
#[derive(Clone, Debug)]
pub struct TurboQuant;

impl TurboQuant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant");
}

vtable!(TurboQuant, TurboQuant, TurboQuantData);

/// Serialized metadata for TurboQuant encoding: a single byte holding the `bit_width` (0-8).
///
/// All other fields (dimension, element type) are derived from the dtype and children.
/// A `bit_width` of 0 indicates a degenerate empty array.
#[derive(Clone, Debug)]
pub struct TurboQuantMetadata {
    /// MSE bits per coordinate (0 for degenerate empty arrays, 1-8 otherwise).
    pub bit_width: u8,
}

/// Slot positions for TurboQuantArray children.
#[repr(usize)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    Codes = 0,
    Norms = 1,
    Centroids = 2,
    RotationSigns = 3,
}

impl Slot {
    pub(crate) const COUNT: usize = 4;

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::Norms => "norms",
            Self::Centroids => "centroids",
            Self::RotationSigns => "rotation_signs",
        }
    }

    pub(crate) fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Codes,
            1 => Self::Norms,
            2 => Self::Centroids,
            3 => Self::RotationSigns,
            _ => vortex_error::vortex_panic!("invalid slot index {idx}"),
        }
    }
}

/// TurboQuant array data.
///
/// TurboQuant is a lossy vector quantization encoding for [`Vector`] extension arrays.
/// It stores quantized coordinate codes and per-vector norms, along with shared codebook
/// centroids and SRHT rotation signs. See the [module docs](super) for algorithmic details.
///
/// A degenerate TurboQuant array has zero rows and `bit_width == 0`, with all slots empty.
///
/// [`Vector`]: crate::vector::Vector
#[derive(Clone, Debug)]
pub struct TurboQuantData {
    /// The [`Vector`] extension dtype that this array encodes. The storage dtype within the
    /// extension determines the element type (f16, f32, or f64) and the list size (dimension).
    ///
    /// [`Vector`]: crate::vector::Vector
    pub(crate) dtype: DType,

    /// Child arrays stored as optional slots. See [`Slot`] for positions:
    ///
    /// - [`Codes`](Slot::Codes): `FixedSizeListArray<u8>` with `list_size == padded_dim`. Each row
    ///   holds one u8 centroid index per padded coordinate. The cascade compressor handles packing
    ///   to the actual `bit_width` on disk. The validity of the entire array is stored with this.
    ///
    /// - [`Norms`](Slot::Norms): Per-vector L2 norms, one per row. The dtype matches the element
    ///   type of the Vector (e.g., f64 norms for f64 vectors). Exact norms are stored during
    ///   compression, enabling O(1) L2 norm readthrough without decompression.
    ///
    /// - [`Centroids`](Slot::Centroids): `PrimitiveArray<f32>` codebook with `2^bit_width` entries
    ///   that is shared across all rows. We always store these as f32 regardless of the input
    ///   element type because quantization itself introduces far more error than f32 precision
    ///   loss, and f16 inputs can be upcast to f32 before quantization.
    ///
    /// - [`RotationSigns`](Slot::RotationSigns): `BitPackedArray` of `3 * padded_dim` 1-bit sign
    ///   values for the 3-round SRHT rotation, stored in inverse application order, and shared
    ///   across all rows.
    pub(crate) slots: Vec<Option<ArrayRef>>,

    /// The vector dimension `d`, cached from the `FixedSizeList` storage dtype's list size.
    /// Stored as a convenience field to avoid repeatedly extracting it from `dtype`.
    /// Non-power-of-2 dimensions are zero-padded to [`padded_dim`](Self::padded_dim) for the
    /// Walsh-Hadamard transform.
    pub(crate) dimension: u32,

    /// The number of bits per coordinate (1-8), derived from `log2(centroids.len())`.
    /// Zero for degenerate empty arrays.
    pub(crate) bit_width: u8,

    /// The stats for this array.
    pub(crate) stats_set: ArrayStats,
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
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> VortexResult<Self> {
        Self::validate(&dtype, &codes, &norms, &centroids, &rotation_signs)?;

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
    ///   is >= 3.
    /// - `codes` is a `FixedSizeListArray<u8>` with `list_size == padded_dim` and
    ///   `codes.len() == norms.len()`.
    /// - `norms` is a non-nullable primitive array whose ptype matches the element type of the
    ///   Vector's storage dtype.
    /// - `centroids` is a non-nullable `PrimitiveArray<f32>` whose length is a power of 2 in
    ///   `[2, 256]` (i.e., `2^bit_width` for bit_width 1-8), or empty for degenerate arrays.
    /// - `rotation_signs` has `3 * padded_dim` elements, or is empty for degenerate arrays.
    /// - For degenerate (empty) arrays: all children must be empty.
    ///
    /// Violating these invariants may produce incorrect results during decompression.
    pub unsafe fn new_unchecked(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&dtype, &codes, &norms, &centroids, &rotation_signs)
            .vortex_expect("[Debug Assertion]: Invalid TurboQuantData parameters");

        let dimension = dtype
            .as_extension_opt()
            .and_then(|ext| extension_list_size(ext).ok())
            .vortex_expect("dtype must be a Vector extension type with FixedSizeList storage");

        let bit_width = derive_bit_width(&centroids);

        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Norms as usize] = Some(norms);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);
        Self {
            dtype,
            slots,
            dimension,
            bit_width,
            stats_set: Default::default(),
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
        // Dtype must be a Vector extension type.
        let ext = dtype
            .as_extension_opt()
            .filter(|e| e.is::<Vector>())
            .ok_or_else(|| {
                vortex_error::vortex_err!(
                    "TurboQuant dtype must be a Vector extension type, got {dtype}"
                )
            })?;

        // Dimension is derived from the storage dtype's list size and must be >= 3.
        let dimension = extension_list_size(ext)?;
        vortex_ensure!(
            dimension >= 3,
            "TurboQuant requires dimension >= 3, got {dimension}"
        );

        let num_rows = norms.len();

        // Degenerate (empty) case: all children must be empty, bit_width is 0.
        if num_rows == 0 {
            vortex_ensure!(
                codes.is_empty(),
                "degenerate TurboQuant must have empty codes, got length {}",
                codes.len()
            );
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
        #[allow(clippy::cast_possible_truncation)]
        let bit_width = num_centroids.trailing_zeros() as u8;
        vortex_ensure!(
            (1..=8).contains(&bit_width),
            "derived bit_width must be 1-8, got {bit_width}"
        );

        // Norms dtype must match the element ptype of the Vector.
        let element_ptype = extension_element_ptype(ext)?;
        let expected_norms_dtype = DType::Primitive(element_ptype, Nullability::NonNullable);
        vortex_ensure!(
            *norms.dtype() == expected_norms_dtype,
            "norms dtype {} does not match expected {expected_norms_dtype} \
             (must match Vector element type)",
            norms.dtype()
        );

        // Centroids are always f32 regardless of element type.
        let f32_nn = DType::Primitive(PType::F32, Nullability::NonNullable);
        vortex_ensure!(
            *centroids.dtype() == f32_nn,
            "centroids dtype {} must be non-nullable f32",
            centroids.dtype()
        );

        // Row count consistency.
        vortex_ensure!(
            codes.len() == num_rows,
            "codes length {} does not match norms length {num_rows}",
            codes.len()
        );

        // Rotation signs count must be 3 * padded_dim.
        let padded_dim = dimension.next_power_of_two() as usize;
        vortex_ensure!(
            rotation_signs.len() == 3 * padded_dim,
            "rotation_signs length {} does not match expected 3 * {padded_dim} = {}",
            rotation_signs.len(),
            3 * padded_dim
        );

        Ok(())
    }

    /// The vector dimension `d`, as stored in the [`Vector`] extension dtype's
    /// `FixedSizeList` storage.
    ///
    /// [`Vector`]: crate::vector::Vector
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

    fn slot(&self, idx: usize) -> &ArrayRef {
        self.slots[idx]
            .as_ref()
            .vortex_expect("required slot is None")
    }

    /// The quantized codes child (`FixedSizeListArray<u8>`, one row per vector).
    pub fn codes(&self) -> &ArrayRef {
        self.slot(Slot::Codes as usize)
    }

    /// Per-vector L2 norms. The dtype matches the Vector's element type (f16, f32, or f64).
    pub fn norms(&self) -> &ArrayRef {
        self.slot(Slot::Norms as usize)
    }

    /// The codebook centroids (`PrimitiveArray<f32>`, length `2^bit_width`).
    ///
    /// Always f32 regardless of input element type: quantization noise dominates f32
    /// precision loss, and f16 inputs are upcast before quantization anyway.
    pub fn centroids(&self) -> &ArrayRef {
        self.slot(Slot::Centroids as usize)
    }

    /// The SRHT rotation signs (`BitPackedArray`, `3 * padded_dim` 1-bit values).
    ///
    /// Stored in inverse application order for efficient decode.
    pub fn rotation_signs(&self) -> &ArrayRef {
        self.slot(Slot::RotationSigns as usize)
    }
}

/// Derive `bit_width` from the centroids array length.
///
/// Returns 0 for empty centroids (degenerate array), otherwise `log2(centroids.len())`.
fn derive_bit_width(centroids: &ArrayRef) -> u8 {
    if centroids.is_empty() {
        0
    } else {
        // Guaranteed to be 0-8 by validate().
        #[allow(clippy::cast_possible_truncation)]
        {
            centroids.len().trailing_zeros() as u8
        }
    }
}
