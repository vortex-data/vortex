// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant array definition: stores quantized coordinate codes, norms,
//! centroids (codebook), rotation signs, and optional QJL correction fields.

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

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

/// TurboQuant array.
///
/// Core children (always present):
/// - 0: `codes` — `BitPackedArray` or `PrimitiveArray<u8>` (quantized indices)
/// - 1: `norms` — `PrimitiveArray<f32>` (one per vector row)
/// - 2: `centroids` — `PrimitiveArray<f32>` (codebook, length 2^bit_width)
/// - 3: `rotation_signs` — `BoolArray` (3 * padded_dim bits, inverse application order)
///
/// Optional QJL children (when `has_qjl` is true):
/// - 4: `qjl_signs` — `BoolArray` (num_rows * padded_dim bits)
/// - 5: `qjl_residual_norms` — `PrimitiveArray<f32>` (one per row)
/// - 6: `qjl_rotation_signs` — `BoolArray` (3 * padded_dim bits, QJL rotation, inverse order)
#[derive(Clone, Debug)]
pub struct TurboQuantArray {
    pub(crate) dtype: DType,
    pub(crate) codes: ArrayRef,
    pub(crate) norms: ArrayRef,
    pub(crate) centroids: ArrayRef,
    pub(crate) rotation_signs: ArrayRef,
    pub(crate) qjl: Option<QjlCorrection>,
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
        Ok(Self {
            dtype,
            codes,
            norms,
            centroids,
            rotation_signs,
            qjl: None,
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
        Ok(Self {
            dtype,
            codes,
            norms,
            centroids,
            rotation_signs,
            qjl: Some(qjl),
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
        self.qjl.is_some()
    }

    /// The quantized codes child.
    pub fn codes(&self) -> &ArrayRef {
        &self.codes
    }

    /// The norms child.
    pub fn norms(&self) -> &ArrayRef {
        &self.norms
    }

    /// The centroids (codebook) child.
    pub fn centroids(&self) -> &ArrayRef {
        &self.centroids
    }

    /// The MSE rotation signs child (BoolArray, length 3 * padded_dim).
    pub fn rotation_signs(&self) -> &ArrayRef {
        &self.rotation_signs
    }

    /// The optional QJL correction.
    pub fn qjl(&self) -> Option<&QjlCorrection> {
        self.qjl.as_ref()
    }
}
