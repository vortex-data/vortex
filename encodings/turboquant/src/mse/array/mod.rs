// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant MSE array definition: stores quantized coordinate codes, norms,
//! centroids (codebook), and rotation signs.

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::TurboQuantMSE;

vtable!(TurboQuantMSE);

/// Protobuf metadata for TurboQuant MSE encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantMSEMetadata {
    /// Vector dimension d.
    #[prost(uint32, tag = "1")]
    pub dimension: u32,
    /// Bits per coordinate (1-8).
    #[prost(uint32, tag = "2")]
    pub bit_width: u32,
    /// Padded dimension (next power of 2 >= dimension).
    #[prost(uint32, tag = "3")]
    pub padded_dim: u32,
    /// Deterministic seed for rotation matrix (kept for reproducibility).
    #[prost(uint64, tag = "4")]
    pub rotation_seed: u64,
}

/// TurboQuant MSE array.
///
/// Children:
/// - 0: `codes` — `BitPackedArray` or `PrimitiveArray<u8>` (quantized indices)
/// - 1: `norms` — `PrimitiveArray<f32>` (one per vector row)
/// - 2: `centroids` — `PrimitiveArray<f32>` (codebook, length 2^bit_width)
/// - 3: `rotation_signs` — `BoolArray` (3 * padded_dim bits, inverse application order)
#[derive(Clone, Debug)]
pub struct TurboQuantMSEArray {
    pub(crate) dtype: DType,
    pub(crate) codes: ArrayRef,
    pub(crate) norms: ArrayRef,
    pub(crate) centroids: ArrayRef,
    pub(crate) rotation_signs: ArrayRef,
    pub(crate) dimension: u32,
    pub(crate) bit_width: u8,
    pub(crate) padded_dim: u32,
    pub(crate) rotation_seed: u64,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantMSEArray {
    /// Build a new TurboQuantMSEArray.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
        dimension: u32,
        bit_width: u8,
        padded_dim: u32,
        rotation_seed: u64,
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
            dimension,
            bit_width,
            padded_dim,
            rotation_seed,
            stats_set: Default::default(),
        })
    }

    /// The vector dimension d.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Bits per coordinate.
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Padded dimension (next power of 2 >= dimension).
    pub fn padded_dim(&self) -> u32 {
        self.padded_dim
    }

    /// The rotation matrix seed.
    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// The bit-packed codes child.
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

    /// The rotation signs child (BoolArray, length 3 * padded_dim).
    pub fn rotation_signs(&self) -> &ArrayRef {
        &self.rotation_signs
    }
}
