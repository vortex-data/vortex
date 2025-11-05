// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{ScalarOps, VectorMut};
use vortex_dtype::half::f16;

/// Represents a primitive scalar value.
pub enum PScalar {
    I8(Option<i8>),
    I16(Option<i16>),
    I32(Option<i32>),
    I64(Option<i64>),
    U8(Option<u8>),
    U16(Option<u16>),
    U32(Option<u32>),
    U64(Option<u64>),
    F16(Option<f16>),
    F32(Option<f32>),
    F64(Option<f64>),
}

impl ScalarOps for PScalar {
    fn is_valid(&self) -> bool {
        match self {
            PScalar::I8(v) => v.is_some(),
            PScalar::I16(v) => v.is_some(),
            PScalar::I32(v) => v.is_some(),
            PScalar::I64(v) => v.is_some(),
            PScalar::U8(v) => v.is_some(),
            PScalar::U16(v) => v.is_some(),
            PScalar::U32(v) => v.is_some(),
            PScalar::U64(v) => v.is_some(),
            PScalar::F16(v) => v.is_some(),
            PScalar::F32(v) => v.is_some(),
            PScalar::F64(v) => v.is_some(),
        }
    }

    fn repeat(&self, n: usize) -> VectorMut {
        todo!()
    }
}
