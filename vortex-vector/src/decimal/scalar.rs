// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{ScalarOps, VectorMut};
use vortex_dtype::i256;

/// Represents a decimal scalar value.
pub enum DScalar {
    I8(Option<i8>),
    I16(Option<i16>),
    I32(Option<i32>),
    I64(Option<i64>),
    I128(Option<i128>),
    I256(Option<i256>),
}

impl ScalarOps for DScalar {
    fn is_valid(&self) -> bool {
        match self {
            DScalar::I8(v) => v.is_some(),
            DScalar::I16(v) => v.is_some(),
            DScalar::I32(v) => v.is_some(),
            DScalar::I64(v) => v.is_some(),
            DScalar::I128(v) => v.is_some(),
            DScalar::I256(v) => v.is_some(),
        }
    }

    fn repeat(&self, _n: usize) -> VectorMut {
        todo!()
    }
}
