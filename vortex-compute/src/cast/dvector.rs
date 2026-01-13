// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::DecimalType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::i256;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
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
            // TODO(connor): cast to different scale
            DType::Decimal(ddt, n)
                if ddt.scale() == self.scale() && (n.is_nullable() || self.is_valid()) =>
            {
                let p = ddt.precision();
                if p <= <i8 as NativeDecimalType>::MAX_PRECISION {
                    DScalar::maybe_new(
                        PrecisionScale::<i8>::new(ddt.precision(), ddt.scale()),
                        self.value().and_then(|v| v.to_i8()),
                    )
                    .map(|ds| ds.into())
                    .ok_or_else(|| vortex_err!("Couldn't cast DScalar ({self:?} to {ddt:?}"))
                } else if p <= <i16 as NativeDecimalType>::MAX_PRECISION {
                    DScalar::maybe_new(
                        PrecisionScale::<i16>::new(ddt.precision(), ddt.scale()),
                        self.value().and_then(|v| v.to_i16()),
                    )
                    .map(|ds| ds.into())
                    .ok_or_else(|| vortex_err!("Couldn't cast DScalar ({self:?} to {ddt:?}"))
                } else if p <= <i32 as NativeDecimalType>::MAX_PRECISION {
                    DScalar::maybe_new(
                        PrecisionScale::<i32>::new(ddt.precision(), ddt.scale()),
                        self.value().and_then(|v| v.to_i32()),
                    )
                    .map(|ds| ds.into())
                    .ok_or_else(|| vortex_err!("Couldn't cast DScalar ({self:?} to {ddt:?}"))
                } else if p <= <i64 as NativeDecimalType>::MAX_PRECISION {
                    DScalar::maybe_new(
                        PrecisionScale::<i64>::new(ddt.precision(), ddt.scale()),
                        self.value().and_then(|v| v.to_i64()),
                    )
                    .map(|ds| ds.into())
                    .ok_or_else(|| vortex_err!("Couldn't cast DScalar ({self:?} to {ddt:?}"))
                } else if p <= <i128 as NativeDecimalType>::MAX_PRECISION {
                    DScalar::maybe_new(
                        PrecisionScale::<i128>::new(ddt.precision(), ddt.scale()),
                        self.value().and_then(|v| v.to_i128()),
                    )
                    .map(|ds| ds.into())
                    .ok_or_else(|| vortex_err!("Couldn't cast DScalar ({self:?} to {ddt:?}"))
                } else if p <= <i256 as NativeDecimalType>::MAX_PRECISION {
                    DScalar::maybe_new(
                        PrecisionScale::<i256>::new(ddt.precision(), ddt.scale()),
                        self.value().and_then(|v| v.to_i256()),
                    )
                    .map(|ds| ds.into())
                    .ok_or_else(|| vortex_err!("Couldn't cast DScalar ({self:?} to {ddt:?}"))
                } else {
                    vortex_bail!(
                        "Target precision {p} is out of range for supported decimal values"
                    )
                }
            }
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::DType;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::DecimalTypeDowncast;
    use vortex_dtype::NativeDecimalType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PrecisionScale;
    use vortex_dtype::i256;
    use vortex_error::VortexResult;
    use vortex_vector::ScalarOps;
    use vortex_vector::decimal::DScalar;

    use crate::cast::Cast;

    #[rstest]
    #[case(2, 0, 42i8)]
    #[case(2, 1, 99i8)]
    #[case(2, -1, 10i8)]
    fn cast_dscalar_identity(
        #[case] precision: u8,
        #[case] scale: i8,
        #[case] value: i8,
    ) -> VortexResult<()> {
        let ps = PrecisionScale::<i8>::new(precision, scale);
        let scalar = DScalar::maybe_new(ps, Some(value)).unwrap();
        let target = DType::Decimal(
            DecimalDType::new(precision, scale),
            Nullability::NonNullable,
        );
        let result = scalar.cast(&target)?;
        let ds = result.into_decimal().into_i8();
        assert_eq!(ds.value(), Some(value));
        assert_eq!(ds.precision(), precision);
        assert_eq!(ds.scale(), scale);
        Ok(())
    }

    #[test]
    fn cast_dscalar_null_to_nullable() -> VortexResult<()> {
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, None).unwrap();
        let target = DType::Decimal(DecimalDType::new(2, 0), Nullability::Nullable);
        let result = scalar.cast(&target)?;
        assert!(!result.as_decimal().is_valid());
        Ok(())
    }

    #[test]
    fn cast_dscalar_null_to_non_nullable_fails() {
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, None).unwrap();
        let target = DType::Decimal(DecimalDType::new(2, 0), Nullability::NonNullable);
        assert!(scalar.cast(&target).is_err());
    }

    #[rstest]
    #[case(2, 4, 42i8)] // i8 -> i16 (precision 2 -> 4)
    #[case(2, 9, 99i8)] // i8 -> i32 (precision 2 -> 9)
    #[case(2, 18, 10i8)] // i8 -> i64 (precision 2 -> 18)
    #[case(2, 38, 55i8)] // i8 -> i128 (precision 2 -> 38)
    fn cast_dscalar_upcast_precision(
        #[case] src_precision: u8,
        #[case] target_precision: u8,
        #[case] value: i8,
    ) -> VortexResult<()> {
        let scale = 0i8;
        let ps = PrecisionScale::<i8>::new(src_precision, scale);
        let scalar = DScalar::maybe_new(ps, Some(value)).unwrap();
        let target = DType::Decimal(
            DecimalDType::new(target_precision, scale),
            Nullability::NonNullable,
        );
        let result = scalar.cast(&target)?;
        let ds = result.as_decimal();
        assert!(ds.is_valid());
        assert_eq!(ds.precision(), target_precision);
        assert_eq!(ds.scale(), scale);
        Ok(())
    }

    #[test]
    fn cast_dscalar_i8_to_i16() -> VortexResult<()> {
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, Some(42i8)).unwrap();
        // Precision 4 requires i16
        let target = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let result = scalar.cast(&target)?;
        let ds = result.into_decimal().into_i16();
        assert_eq!(ds.value(), Some(42i16));
        assert_eq!(ds.precision(), 4);
        Ok(())
    }

    #[test]
    fn cast_dscalar_i8_to_i32() -> VortexResult<()> {
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, Some(99i8)).unwrap();
        // Precision 9 requires i32
        let target = DType::Decimal(DecimalDType::new(9, 0), Nullability::NonNullable);
        let result = scalar.cast(&target)?;
        let ds = result.into_decimal().into_i32();
        assert_eq!(ds.value(), Some(99i32));
        assert_eq!(ds.precision(), 9);
        Ok(())
    }

    #[test]
    fn cast_dscalar_i16_to_i64() -> VortexResult<()> {
        let ps = PrecisionScale::<i16>::new(4, 2);
        let scalar = DScalar::maybe_new(ps, Some(1234i16)).unwrap();
        // Precision 18 requires i64
        let target = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
        let result = scalar.cast(&target)?;
        let ds = result.into_decimal().into_i64();
        assert_eq!(ds.value(), Some(1234i64));
        assert_eq!(ds.precision(), 18);
        assert_eq!(ds.scale(), 2);
        Ok(())
    }

    #[test]
    fn cast_dscalar_i32_to_i128() -> VortexResult<()> {
        let ps = PrecisionScale::<i32>::new(9, 0);
        let scalar = DScalar::maybe_new(ps, Some(123456789i32)).unwrap();
        // Precision 38 requires i128
        let target = DType::Decimal(DecimalDType::new(38, 0), Nullability::NonNullable);
        let result = scalar.cast(&target)?;
        let ds = result.into_decimal().into_i128();
        assert_eq!(ds.value(), Some(123456789i128));
        assert_eq!(ds.precision(), 38);
        Ok(())
    }

    #[test]
    fn cast_dscalar_different_scale_fails() {
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, Some(42i8)).unwrap();
        let target = DType::Decimal(DecimalDType::new(2, 1), Nullability::NonNullable);
        assert!(scalar.cast(&target).is_err());
    }

    #[test]
    fn cast_dscalar_to_non_decimal_fails() {
        use vortex_dtype::PType;
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, Some(42i8)).unwrap();
        let target = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert!(scalar.cast(&target).is_err());
    }

    #[test]
    fn cast_dscalar_downcast_precision_within_same_type() -> VortexResult<()> {
        // Downcast within the same native type (i8 precision 2 -> precision 1)
        // should work as long as the value fits
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, Some(9i8)).unwrap(); // value 9 fits in precision 1
        let target = DType::Decimal(DecimalDType::new(1, 0), Nullability::NonNullable);
        let result = scalar.cast(&target)?;
        let ds = result.into_decimal().into_i8();
        assert_eq!(ds.value(), Some(9i8));
        assert_eq!(ds.precision(), 1);
        Ok(())
    }

    #[test]
    fn cast_dscalar_downcast_value_too_large_fails() {
        // Value 42 doesn't fit in precision 1 (max 9)
        let ps = PrecisionScale::<i8>::new(2, 0);
        let scalar = DScalar::maybe_new(ps, Some(42i8)).unwrap();
        let target = DType::Decimal(DecimalDType::new(1, 0), Nullability::NonNullable);
        assert!(scalar.cast(&target).is_err());
    }

    #[rstest]
    #[case(<i8 as NativeDecimalType>::MAX_PRECISION)]
    #[case(<i16 as NativeDecimalType>::MAX_PRECISION)]
    #[case(<i32 as NativeDecimalType>::MAX_PRECISION)]
    #[case(<i64 as NativeDecimalType>::MAX_PRECISION)]
    #[case(<i128 as NativeDecimalType>::MAX_PRECISION)]
    #[case(<i256 as NativeDecimalType>::MAX_PRECISION)]
    fn cast_dscalar_to_max_precision_boundary(#[case] target_precision: u8) -> VortexResult<()> {
        let ps = PrecisionScale::<i8>::new(1, 0);
        let scalar = DScalar::maybe_new(ps, Some(1i8)).unwrap();
        let target = DType::Decimal(
            DecimalDType::new(target_precision, 0),
            Nullability::NonNullable,
        );
        let result = scalar.cast(&target)?;
        assert_eq!(result.as_decimal().precision(), target_precision);
        Ok(())
    }
}
