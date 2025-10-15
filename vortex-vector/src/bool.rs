// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_mask::Mask;

use crate::ops::VectorOps;
use crate::{BoolVectorMut, Vector};

/// An immutable vector of boolean values.
pub struct BoolVector {
    pub(super) dtype: DType,
    pub(super) bits: BitBuffer,
    pub(super) validity: Mask,
}

impl From<BoolVector> for Vector {
    fn from(v: BoolVector) -> Self {
        Self::Bool(v)
    }
}

impl VectorOps for BoolVector {
    type Mutable = BoolVectorMut;

    fn len(&self) -> usize {
        self.bits.len()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        let bits = match self.bits.try_into_mut() {
            Ok(bits) => bits,
            Err(bits) => {
                return Err(BoolVector {
                    dtype: self.dtype,
                    bits,
                    validity: self.validity,
                });
            }
        };

        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(BoolVector {
                    dtype: self.dtype,
                    bits: bits.freeze(),
                    validity,
                });
            }
        };

        Ok(BoolVectorMut {
            dtype: self.dtype,
            bits,
            validity,
        })
    }
}
