// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::DVector;
use vortex_dtype::i256;

/// An enum over all supported decimal mutable vector types.
#[derive(Clone, Debug)]
pub enum DecimalVector {
    /// A decimal vector with 8-bit integer representation.
    D8(DVector<i8>),
    /// A decimal vector with 16-bit integer representation.
    D16(DVector<i16>),
    /// A decimal vector with 32-bit integer representation.
    D32(DVector<i32>),
    /// A decimal vector with 64-bit integer representation.
    D64(DVector<i64>),
    /// A decimal vector with 128-bit integer representation.
    D128(DVector<i128>),
    /// A decimal vector with 256-bit integer representation.
    D256(DVector<i256>),
}
