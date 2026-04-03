// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant array definition: stores quantized coordinate codes, norms,
//! centroids (codebook), and rotation signs.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::vector::Vector;

/// Encoding marker type for TurboQuant.
#[derive(Clone, Debug)]
pub struct TurboQuant;

impl TurboQuant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant");
}

vtable!(TurboQuant, TurboQuant, TurboQuantData);

/// Protobuf metadata for TurboQuant encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantMetadata {
    /// Vector dimension d.
    #[prost(uint32, tag = "1")]
    pub dimension: u32,
    /// MSE bits per coordinate (1-8).
    #[prost(uint32, tag = "2")]
    pub bit_width: u32,
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

/// TurboQuant array.
///
/// Slots:
/// - 0: `codes` -- `FixedSizeListArray<u8>` (quantized indices, list_size=padded_dim).
/// - 1: `norms` -- `PrimitiveArray<f32>` (one per vector row).
/// - 2: `centroids` -- `PrimitiveArray<f32>` (codebook, length 2^bit_width).
/// - 3: `rotation_signs` -- `BitPackedArray` (3 * padded_dim, 1-bit u8 0/1, inverse order).
#[derive(Clone, Debug)]
pub struct TurboQuantData {
    pub(crate) dtype: DType,
    pub(crate) slots: Vec<Option<ArrayRef>>,
    pub(crate) dimension: u32,
    pub(crate) bit_width: u8,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantData {
    /// Build a TurboQuant array with validation.
    ///
    /// The `dtype` must be a [`Vector`] extension type. TurboQuant encodes the extension
    /// type directly, not its `FixedSizeList` storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented
    /// in [`new_unchecked`](Self::new_unchecked).
    ///
    /// [`Vector`]: crate::vector::Vector
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
        dimension: u32,
        bit_width: u8,
    ) -> VortexResult<Self> {
        Self::validate(
            &dtype,
            &codes,
            &norms,
            &centroids,
            &rotation_signs,
            dimension,
            bit_width,
        )?;

        // SAFETY: we validate that the inputs are valid above.
        Ok(unsafe {
            Self::new_unchecked(
                dtype,
                codes,
                norms,
                centroids,
                rotation_signs,
                dimension,
                bit_width,
            )
        })
    }

    /// Build a TurboQuant array without validation.
    ///
    /// * `dtype` must be a [`Vector`] extension type.
    /// * `codes` must be a `FixedSizeListArray<u8>` with `list_size == padded_dim`.
    /// * `norms` must be a `PrimitiveArray<f32>` with one element per row.
    /// * `centroids` must be a `PrimitiveArray<f32>` with `2^bit_width` elements.
    /// * `rotation_signs` must contain `3 * padded_dim` sign values.
    /// * `bit_width` must be 1-8.
    /// * `codes.len() == norms.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the inputs satisfy the invariants listed above. Violating them
    /// may produce incorrect results during decompression.
    ///
    /// [`Vector`]: crate::vector::Vector
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_unchecked(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
        dimension: u32,
        bit_width: u8,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(
            &dtype,
            &codes,
            &norms,
            &centroids,
            &rotation_signs,
            dimension,
            bit_width,
        )
        .vortex_expect("[Debug Assertion]: Invalid TurboQuantData parameters");

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
    #[allow(clippy::too_many_arguments)]
    pub fn validate(
        dtype: &DType,
        codes: &ArrayRef,
        norms: &ArrayRef,
        centroids: &ArrayRef,
        rotation_signs: &ArrayRef,
        dimension: u32,
        bit_width: u8,
    ) -> VortexResult<()> {
        vortex_ensure!(
            (1..=8).contains(&bit_width),
            "bit_width must be 1-8, got {bit_width}"
        );
        vortex_ensure!(
            dtype
                .as_extension_opt()
                .is_some_and(|ext| ext.is::<Vector>()),
            "TurboQuant dtype must be a Vector extension type, got {dtype}"
        );
        vortex_ensure!(
            dimension >= 3,
            "TurboQuant requires dimension >= 3, got {dimension}"
        );

        let num_rows = norms.len();
        vortex_ensure!(
            codes.len() == num_rows,
            "codes length {} does not match norms length {num_rows}",
            codes.len()
        );

        let expected_centroids = 1usize << bit_width;
        // Allow empty centroids for zero-row arrays.
        if num_rows > 0 {
            vortex_ensure!(
                centroids.len() == expected_centroids,
                "centroids length {} does not match expected 2^{bit_width} = {expected_centroids}",
                centroids.len()
            );
        }

        let padded_dim = dimension.next_power_of_two() as usize;
        // Allow empty rotation signs for zero-row arrays.
        if num_rows > 0 {
            vortex_ensure!(
                rotation_signs.len() == 3 * padded_dim,
                "rotation_signs length {} does not match expected 3 * {padded_dim} = {}",
                rotation_signs.len(),
                3 * padded_dim
            );
        }

        Ok(())
    }

    /// The vector dimension d.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// MSE bits per coordinate.
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Padded dimension (next power of 2 >= dimension).
    pub fn padded_dim(&self) -> u32 {
        self.dimension.next_power_of_two()
    }

    fn slot(&self, idx: usize) -> &ArrayRef {
        self.slots[idx]
            .as_ref()
            .vortex_expect("required slot is None")
    }

    /// The quantized codes child (FixedSizeListArray).
    pub fn codes(&self) -> &ArrayRef {
        self.slot(Slot::Codes as usize)
    }

    /// The norms child (`PrimitiveArray<f32>`).
    pub fn norms(&self) -> &ArrayRef {
        self.slot(Slot::Norms as usize)
    }

    /// The centroids (codebook) child (`PrimitiveArray<f32>`).
    pub fn centroids(&self) -> &ArrayRef {
        self.slot(Slot::Centroids as usize)
    }

    /// The MSE rotation signs child (BitPackedArray, length 3 * padded_dim).
    pub fn rotation_signs(&self) -> &ArrayRef {
        self.slot(Slot::RotationSigns as usize)
    }
}
