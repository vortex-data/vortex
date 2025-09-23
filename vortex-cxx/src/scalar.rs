// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;

use crate::dtype::DType;

pub(crate) struct Scalar {
    pub(crate) inner: vortex::scalar::Scalar,
}

macro_rules! primitive_scalar_new {
    ($name:ident, $type:ty) => {
        pub(crate) fn $name(value: $type) -> Box<Scalar> {
            Box::new(Scalar {
                inner: vortex::scalar::Scalar::from(value),
            })
        }
    };
}

primitive_scalar_new!(bool_scalar_new, bool); // bool is not primitive but reuse the macro here
primitive_scalar_new!(i8_scalar_new, i8);
primitive_scalar_new!(i16_scalar_new, i16);
primitive_scalar_new!(i32_scalar_new, i32);
primitive_scalar_new!(i64_scalar_new, i64);
primitive_scalar_new!(u8_scalar_new, u8);
primitive_scalar_new!(u16_scalar_new, u16);
primitive_scalar_new!(u32_scalar_new, u32);
primitive_scalar_new!(u64_scalar_new, u64);
primitive_scalar_new!(f32_scalar_new, f32);
primitive_scalar_new!(f64_scalar_new, f64);

pub(crate) fn string_scalar_new(value: &str) -> Box<Scalar> {
    Box::new(Scalar {
        inner: vortex::scalar::Scalar::from(value),
    })
}

pub(crate) fn binary_scalar_new(value: &[u8]) -> Box<Scalar> {
    Box::new(Scalar {
        inner: vortex::scalar::Scalar::from(value),
    })
}

impl Scalar {
    pub(crate) fn cast_scalar(&self, dtype: &DType) -> Result<Box<Scalar>> {
        Ok(Box::new(Scalar {
            inner: self.inner.cast(&dtype.inner)?,
        }))
    }
}
