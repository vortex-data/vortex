// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::DVectorMut;
use vortex_dtype::i256;

/// An enum over all supported decimal mutable vector types.
#[derive(Clone, Debug)]
pub enum DecimalVectorMut {
    D8(DVectorMut<i8>),
    D16(DVectorMut<i16>),
    D32(DVectorMut<i32>),
    D64(DVectorMut<i64>),
    D128(DVectorMut<i128>),
    D256(DVectorMut<i256>),
}
