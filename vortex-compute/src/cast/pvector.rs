// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_vector::null::NullVector;
use vortex_vector::primitive::{PVector, PrimitiveVector};
use vortex_vector::{Vector, VectorOps, match_each_pvector};

use crate::cast::Cast;

impl Cast for PrimitiveVector {
    fn cast(&self, dtype: &DType) -> VortexResult<Vector> {
        match_each_pvector!(self, |v| { Cast::cast(v, dtype) })
    }
}

impl<T: NativePType> Cast for PVector<T> {
    fn cast(&self, dtype: &DType) -> VortexResult<Vector> {
        match dtype {
            // Can cast an all-null PVector to NullVector.
            DType::Null if self.validity().all_false() => Ok(NullVector::new(self.len()).into()),
            // We're already the correct PType, and we have compatible nullability.
            DType::Primitive(target_ptype, n)
                if *target_ptype == T::PTYPE && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            // We're not the correct PType, but we do have compatible nullability.
            DType::Primitive(target_ptype, n) if n.is_nullable() || self.validity().all_true() => {
                vortex_bail!(
                    "Casting from PType {} to PType {} not yet implemented",
                    T::PTYPE,
                    target_ptype
                );
            }
            DType::Extension(ext_dtype) => self.cast(ext_dtype.storage_dtype()),
            _ => {
                vortex_bail!("Cannot cast {:?} to dtype {}", self, dtype);
            }
        }
    }
}
