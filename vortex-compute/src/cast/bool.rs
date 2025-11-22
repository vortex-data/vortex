// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cast::Cast;
use num_traits::{One, Zero};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_vector::bool::BoolVector;
use vortex_vector::null::NullVector;
use vortex_vector::primitive::PVector;
use vortex_vector::{Vector, VectorOps};

impl Cast for BoolVector {
    fn cast(&self, dtype: &DType) -> VortexResult<Vector> {
        match dtype {
            DType::Null if self.validity().all_false() => {
                // Can cast an all-null BoolVector to NullVector.
                Ok(NullVector::new(self.len()).into())
            }
            DType::Bool(n) if n.is_nullable() || self.validity().all_true() => {
                // If the target dtype is nullable, or if the source BoolVector has no nulls,
                // we can cast directly to BoolVector.
                Ok(self.clone().into())
            }
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |T| {
                    Ok(PVector::<T>::new(
                        Buffer::<T>::from_trusted_len_iter(
                            self.bits()
                                .iter()
                                .map(|b| if b { T::one() } else { T::zero() }),
                        ),
                        self.validity().clone(),
                    )
                    .into())
                })
            }
            DType::Extension(ext_dtype) => self.cast(ext_dtype.storage_dtype()),
            _ => {
                vortex_bail!("Cannot cast BoolVector to type {}", dtype);
            }
        }
    }
}
