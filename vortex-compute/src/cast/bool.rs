// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::One;
use num_traits::Zero;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;
use vortex_vector::primitive::PScalar;
use vortex_vector::primitive::PVector;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl Cast for BoolVector {
    type Output = Vector;

    /// Casts to Bool (identity) or Primitive (as 0/1).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            DType::Bool(n) if n.is_nullable() || self.validity().all_true() => {
                Ok(self.clone().into())
            }
            DType::Primitive(ptype, n) if n.is_nullable() || self.validity().all_true() => {
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
            _ => {
                vortex_bail!("Cannot cast BoolVector to {}", target_dtype);
            }
        }
    }
}

impl Cast for BoolScalar {
    type Output = Scalar;

    /// Casts to Bool (identity) or Primitive (as 0/1).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }
        match target_dtype {
            DType::Bool(n) if n.is_nullable() || self.is_valid() => Ok(self.clone().into()),
            DType::Primitive(ptype, n) if n.is_nullable() || self.is_valid() => {
                match_each_native_ptype!(ptype, |T| {
                    let value = self.value().map(|b| if b { T::one() } else { T::zero() });
                    Ok(PScalar::<T>::new(value).into())
                })
            }
            _ => {
                vortex_bail!("Cannot cast BoolScalar to {}", target_dtype);
            }
        }
    }
}
