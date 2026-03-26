// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant QJL array definition: wraps a TurboQuantMSEArray with 1-bit QJL
//! residual correction for unbiased inner product estimation.

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::TurboQuantQJL;

vtable!(TurboQuantQJL);

/// Protobuf metadata for TurboQuant QJL encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantQJLMetadata {
    /// Total bit width (2-9, including QJL bit; MSE child uses bit_width - 1).
    #[prost(uint32, tag = "1")]
    pub bit_width: u32,
    /// Padded dimension (next power of 2 >= dimension).
    #[prost(uint32, tag = "2")]
    pub padded_dim: u32,
    /// QJL rotation seed (for debugging/reproducibility).
    #[prost(uint64, tag = "3")]
    pub rotation_seed: u64,
}

/// TurboQuant QJL array.
///
/// Children:
/// - 0: `mse_inner` — `TurboQuantMSEArray` (at `bit_width - 1`)
/// - 1: `qjl_signs` — `BoolArray` (num_rows * padded_dim bits)
/// - 2: `residual_norms` — `PrimitiveArray<f32>` (one per row)
/// - 3: `rotation_signs` — `BoolArray` (3 * padded_dim bits, QJL rotation, inverse order)
#[derive(Clone, Debug)]
pub struct TurboQuantQJLArray {
    pub(crate) dtype: DType,
    pub(crate) mse_inner: ArrayRef,
    pub(crate) qjl_signs: ArrayRef,
    pub(crate) residual_norms: ArrayRef,
    pub(crate) rotation_signs: ArrayRef,
    pub(crate) bit_width: u8,
    pub(crate) padded_dim: u32,
    pub(crate) rotation_seed: u64,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantQJLArray {
    /// Build a new TurboQuantQJLArray.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        mse_inner: ArrayRef,
        qjl_signs: ArrayRef,
        residual_norms: ArrayRef,
        rotation_signs: ArrayRef,
        bit_width: u8,
        padded_dim: u32,
        rotation_seed: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (2..=9).contains(&bit_width),
            "QJL bit_width must be 2-9, got {bit_width}"
        );
        Ok(Self {
            dtype,
            mse_inner,
            qjl_signs,
            residual_norms,
            rotation_signs,
            bit_width,
            padded_dim,
            rotation_seed,
            stats_set: Default::default(),
        })
    }

    /// Total bit width (including QJL bit).
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Padded dimension.
    pub fn padded_dim(&self) -> u32 {
        self.padded_dim
    }

    /// QJL rotation seed.
    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// The inner MSE array child.
    pub fn mse_inner(&self) -> &ArrayRef {
        &self.mse_inner
    }

    /// The QJL sign bits child (BoolArray).
    pub fn qjl_signs(&self) -> &ArrayRef {
        &self.qjl_signs
    }

    /// The residual norms child.
    pub fn residual_norms(&self) -> &ArrayRef {
        &self.residual_norms
    }

    /// The QJL rotation signs child (BoolArray).
    pub fn rotation_signs(&self) -> &ArrayRef {
        &self.rotation_signs
    }
}
