// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant QJL array definition: wraps a TurboQuantMSEArray with 1-bit QJL
//! residual correction for unbiased inner product estimation.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_error::VortexResult;

use super::TurboQuantQJL;
use crate::TurboQuantMSEArray;

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
    pub(crate) mse_inner: Arc<TurboQuantMSEArray>,
    pub(crate) qjl_signs: ArrayRef,
    pub(crate) residual_norms: ArrayRef,
    pub(crate) rotation_signs: ArrayRef,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantQJLArray {
    /// Build a new TurboQuantQJLArray.
    pub fn try_new(
        dtype: DType,
        mse_inner: Arc<TurboQuantMSEArray>,
        qjl_signs: ArrayRef,
        residual_norms: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> VortexResult<Self> {
        Ok(Self {
            dtype,
            mse_inner,
            qjl_signs,
            residual_norms,
            rotation_signs,
            stats_set: Default::default(),
        })
    }

    /// Total bit width (including QJL bit).
    pub fn bit_width(&self) -> u8 {
        self.mse_inner.bit_width() + 1
    }

    /// Padded dimension.
    pub fn padded_dim(&self) -> u32 {
        self.mse_inner.padded_dim()
    }

    /// The inner MSE array child.
    pub fn mse_inner(&self) -> &TurboQuantMSEArray {
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
