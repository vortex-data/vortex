// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::DVector;
use vortex_dtype::i256;

/// An enum over all supported decimal mutable vector types.
#[derive(Clone, Debug)]
pub enum DecimalVector {
    D8(DVector<i8>),
    D16(DVector<i16>),
    D32(DVector<i32>),
    D64(DVector<i64>),
    D128(DVector<i128>),
    D256(DVector<i256>),
}
