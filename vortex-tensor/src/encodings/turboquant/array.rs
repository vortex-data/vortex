// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant array definition: stores quantized coordinate codes, norms,
//! centroids (codebook), rotation signs, and optional QJL correction fields.

use vortex::array::ArrayRef;
use vortex::array::dtype::DType;
use vortex::array::stats::ArrayStats;
use vortex::array::vtable;
use vortex::array::vtable::ArrayId;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

/// Encoding marker type for TurboQuant.
#[derive(Clone, Debug)]
pub struct TurboQuant;

impl TurboQuant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant");
}

vtable!(TurboQuant);

/// Protobuf metadata for TurboQuant encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantMetadata {
    /// Vector dimension d.
    #[prost(uint32, tag = "1")]
    pub dimension: u32,
    /// MSE bits per coordinate (1-8).
    #[prost(uint32, tag = "2")]
    pub bit_width: u32,
    /// Whether QJL correction children are present.
    #[prost(bool, tag = "3")]
    pub has_qjl: bool,
}

/// Optional QJL (Quantized Johnson-Lindenstrauss) correction for unbiased
/// inner product estimation. When present, adds 3 additional children.
#[derive(Clone, Debug)]
pub struct QjlCorrection {
    /// Sign bits: `BoolArray`, length `num_rows * padded_dim`.
    pub(crate) signs: ArrayRef,
    /// Residual norms: `PrimitiveArray<f32>`, length `num_rows`.
    pub(crate) residual_norms: ArrayRef,
    /// QJL rotation signs: `BoolArray`, length `3 * padded_dim` (inverse order).
    pub(crate) rotation_signs: ArrayRef,
}

impl QjlCorrection {
    /// The QJL sign bits.
    pub fn signs(&self) -> &ArrayRef {
        &self.signs
    }

    /// The residual norms.
    pub fn residual_norms(&self) -> &ArrayRef {
        &self.residual_norms
    }

    /// The QJL rotation signs (BoolArray, inverse application order).
    pub fn rotation_signs(&self) -> &ArrayRef {
        &self.rotation_signs
    }
}

/// Slot positions for TurboQuantArray children.
#[repr(usize)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    Codes = 0,
    Norms = 1,
    Centroids = 2,
    RotationSigns = 3,
    QjlSigns = 4,
    QjlResidualNorms = 5,
    QjlRotationSigns = 6,
}

impl Slot {
    pub(crate) const COUNT: usize = 7;

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::Norms => "norms",
            Self::Centroids => "centroids",
            Self::RotationSigns => "rotation_signs",
            Self::QjlSigns => "qjl_signs",
            Self::QjlResidualNorms => "qjl_residual_norms",
            Self::QjlRotationSigns => "qjl_rotation_signs",
        }
    }

    pub(crate) fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Codes,
            1 => Self::Norms,
            2 => Self::Centroids,
            3 => Self::RotationSigns,
            4 => Self::QjlSigns,
            5 => Self::QjlResidualNorms,
            6 => Self::QjlRotationSigns,
            _ => vortex::error::vortex_panic!("invalid slot index {idx}"),
        }
    }
}

/// TurboQuant array.
///
/// Slots (always present):
/// - 0: `codes` — `FixedSizeListArray<u8>` (quantized indices, list_size=padded_dim)
/// - 1: `norms` — `PrimitiveArray<f32>` (one per vector row)
/// - 2: `centroids` — `PrimitiveArray<f32>` (codebook, length 2^bit_width)
/// - 3: `rotation_signs` — `BitPackedArray` (3 * padded_dim, 1-bit u8 0/1, inverse order)
///
/// Optional QJL slots (None when MSE-only):
/// - 4: `qjl_signs` — `FixedSizeListArray<u8>` (num_rows * padded_dim, 1-bit)
/// - 5: `qjl_residual_norms` — `PrimitiveArray<f32>` (one per row)
/// - 6: `qjl_rotation_signs` — `BitPackedArray` (3 * padded_dim, 1-bit, QJL rotation)
#[derive(Clone, Debug)]
pub struct TurboQuantArray {
    pub(crate) dtype: DType,
    pub(crate) slots: Vec<Option<ArrayRef>>,
    pub(crate) dimension: u32,
    pub(crate) bit_width: u8,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantArray {
    /// Build a TurboQuant array with MSE-only encoding (no QJL correction).
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_mse(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
        dimension: u32,
        bit_width: u8,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (1..=8).contains(&bit_width),
            "MSE bit_width must be 1-8, got {bit_width}"
        );
        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Norms as usize] = Some(norms);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);
        Ok(Self {
            dtype,
            slots,
            dimension,
            bit_width,
            stats_set: Default::default(),
        })
    }

    /// Build a TurboQuant array with QJL correction (MSE + QJL).
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_qjl(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
        qjl: QjlCorrection,
        dimension: u32,
        bit_width: u8,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (1..=8).contains(&bit_width),
            "MSE bit_width must be 1-8, got {bit_width}"
        );
        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Norms as usize] = Some(norms);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);
        slots[Slot::QjlSigns as usize] = Some(qjl.signs);
        slots[Slot::QjlResidualNorms as usize] = Some(qjl.residual_norms);
        slots[Slot::QjlRotationSigns as usize] = Some(qjl.rotation_signs);
        Ok(Self {
            dtype,
            slots,
            dimension,
            bit_width,
            stats_set: Default::default(),
        })
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

    /// Whether QJL correction is present.
    pub fn has_qjl(&self) -> bool {
        self.slots[Slot::QjlSigns as usize].is_some()
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

    /// The norms child (PrimitiveArray<f32>).
    pub fn norms(&self) -> &ArrayRef {
        self.slot(Slot::Norms as usize)
    }

    /// The centroids (codebook) child (PrimitiveArray<f32>).
    pub fn centroids(&self) -> &ArrayRef {
        self.slot(Slot::Centroids as usize)
    }

    /// The MSE rotation signs child (BitPackedArray, length 3 * padded_dim).
    pub fn rotation_signs(&self) -> &ArrayRef {
        self.slot(Slot::RotationSigns as usize)
    }

    /// The optional QJL correction fields, reconstructed from slots.
    pub fn qjl(&self) -> Option<QjlCorrection> {
        Some(QjlCorrection {
            signs: self.slots[Slot::QjlSigns as usize].clone()?,
            residual_norms: self.slots[Slot::QjlResidualNorms as usize].clone()?,
            rotation_signs: self.slots[Slot::QjlRotationSigns as usize].clone()?,
        })
    }
}
