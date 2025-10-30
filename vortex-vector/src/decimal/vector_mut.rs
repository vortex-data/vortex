// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::DVectorMut;
use vortex_dtype::i256;

/// An enum over all supported decimal mutable vector types.
#[derive(Clone, Debug)]
pub enum DecimalVectorMut {
    /// A decimal vector with 8-bit integer representation.
    D8(DVectorMut<i8>),
    /// A decimal vector with 16-bit integer representation.
    D16(DVectorMut<i16>),
    /// A decimal vector with 32-bit integer representation.
    D32(DVectorMut<i32>),
    /// A decimal vector with 64-bit integer representation.
    D64(DVectorMut<i64>),
    /// A decimal vector with 128-bit integer representation.
    D128(DVectorMut<i128>),
    /// A decimal vector with 256-bit integer representation.
    D256(DVectorMut<i256>),
}
