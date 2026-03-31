// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 norm direct readthrough for TurboQuant.
//!
//! TurboQuant stores the exact original L2 norm of each vector in the `norms`
//! child. This enables O(1) per-vector norm lookup without any decompression.

use vortex_array::ArrayRef;

use crate::encodings::turboquant::array::TurboQuantArray;

/// Return the stored norms directly — no decompression needed.
///
/// The norms are computed before quantization, so they are exact (not affected
/// by the lossy encoding). The returned `ArrayRef` is a `PrimitiveArray<f32>`
/// with one element per vector row.
///
/// TODO: Wire into `vortex-tensor` L2Norm scalar function dispatch so that
/// `l2_norm(Extension(TurboQuant(...)))` short-circuits to this.
#[allow(dead_code)] // TODO: wire into vortex-tensor L2Norm dispatch
pub fn l2_norm_direct(array: &TurboQuantArray) -> &ArrayRef {
    array.norms()
}
