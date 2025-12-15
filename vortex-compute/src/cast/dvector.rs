// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::DecimalType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::decimal::DScalar;
use vortex_vector::decimal::DVector;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl<D: NativeDecimalType> Cast for DVector<D> {
    type Output = Vector;

    /// Casts to Decimal with potentially different precision and native type.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        let DType::Decimal(ddt, n) = target_dtype else {
            vortex_bail!("Cannot cast DVector to {}", target_dtype);
        };

        // Check nullability compatibility
        if !n.is_nullable() && !self.validity().all_true() {
            vortex_bail!(
                "Cannot cast nullable DVector to non-nullable {}",
                target_dtype
            );
        }

        // Scale changes require multiplication/division by powers of 10
        if ddt.scale() != self.scale() {
            vortex_bail!(
                "Casting DVector with scale {} to scale {} not yet implemented",
                self.scale(),
                ddt.scale()
            );
        }

        // If the precision is the same, it's an identity cast
        if ddt.precision() == self.precision() {
            return Ok(self.clone().into());
        }

        // If the precision is wider, we may need to upcast the underlying type
        if ddt.precision() > self.precision() {
            // Need to upcast to a wider type
            let target_type = DecimalType::smallest_decimal_value_type(ddt);
            match_each_decimal_value_type!(target_type, |T| {
                return upcast_dvector::<D, T>(self, ddt.precision());
            })
        }

        // TODO(ngates): we need to rebuild the vector as that will validate all values
        //  fit into the precision / scale.
        vortex_bail!(
            "Downcasting DVector from precision {} to {} not yet implemented",
            self.precision(),
            ddt.precision()
        );
    }
}

/// Upcast a DVector<D> to DVector<T> where T is wider than D.
fn upcast_dvector<D: NativeDecimalType, T: NativeDecimalType>(
    source: &DVector<D>,
    target_precision: u8,
) -> VortexResult<Vector> {
    let target_ps = PrecisionScale::<T>::try_new(target_precision, source.scale())?;

    // Upcast each element using BigCast. This should never fail since T is wider than D.
    let elements: Buffer<T> = source
        .elements()
        .iter()
        .map(|&v| T::from(v).vortex_expect("upcast should never fail"))
        .collect();

    let validity = source.validity().clone();

    // SAFETY: We've upcasted from a narrower type, so all values fit.
    Ok(unsafe { DVector::new_unchecked(target_ps, elements, validity) }.into())
}

impl<D: NativeDecimalType> Cast for DScalar<D> {
    type Output = Scalar;

    /// Casts to Decimal (identity with same precision/scale and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same precision, scale, and compatible nullability.
            DType::Decimal(ddt, n)
                if ddt.precision() == self.precision()
                    && ddt.scale() == self.scale()
                    && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            // TODO(connor): cast to different precision/scale
            DType::Decimal(..) => {
                vortex_bail!(
                    "Casting DScalar to {} with different precision/scale not yet implemented",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast DScalar to {}", target_dtype);
            }
        }
    }
}
