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
/// inner product estimation. When present, adds 2 additional children.
///
/// The QJL correction reuses the MSE rotation matrix (stored in `rotation_signs`)
/// rather than maintaining a separate rotation. This halves the rotation sign
/// storage and avoids reconstructing a second `RotationMatrix` at decode time.
#[derive(Clone, Debug)]
pub struct QjlCorrection {
    /// Sign bits: `BitPackedArray` (1-bit), length `num_rows * padded_dim`.
    pub(crate) signs: ArrayRef,
    /// Residual norms: `PrimitiveArray<f32>`, length `num_rows`.
    pub(crate) residual_norms: ArrayRef,
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
}

/// TurboQuant array.
///
/// Core children (always present):
/// - 0: `codes` — `BitPackedArray` or `PrimitiveArray<u8>` (quantized indices)
/// - 1: `norms` — `PrimitiveArray<f32>` (one per vector row)
/// - 2: `centroids` — `PrimitiveArray<f32>` (codebook, length 2^bit_width)
/// - 3: `rotation_signs` — `BitPackedArray` (3 * padded_dim, 1-bit u8 0/1, inverse order)
///
/// Optional QJL children (when `has_qjl` is true):
/// - 4: `qjl_signs` — `BitPackedArray` (num_rows * padded_dim, 1-bit u8 0/1)
/// - 5: `qjl_residual_norms` — `PrimitiveArray<f32>` (one per row)
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
