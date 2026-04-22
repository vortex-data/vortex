// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A test extension type layering a refinement on top of another refinement.
//!
//! [`EvenDivisibleInt`] refines [`DivisibleInt`] with the additional requirement that the value
//! is even. Its storage `DType` is therefore `DType::Extension(DivisibleInt)`, and its validation
//! chain transitively inherits `DivisibleInt`'s divisibility check via [`ExtRefinedSource`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::divisible_int::DivisibleInt;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtRefinedSource;
use crate::dtype::extension::RefinementVTable;
use crate::extension::EmptyMetadata;

/// Refinement of [`DivisibleInt`] requiring the stored value to additionally be even.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EvenDivisibleInt;

impl RefinementVTable for EvenDivisibleInt {
    type Source = ExtRefinedSource<DivisibleInt>;
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new("test.even_divisible_int")
    }

    fn refine_scalar(_metadata: &Self::Metadata, source_value: u64) -> VortexResult<u64> {
        if source_value.is_multiple_of(2) {
            Ok(source_value)
        } else {
            vortex_bail!("{} is not even", source_value)
        }
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _bytes: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::super::divisible_int::DivisibleInt;
    use super::super::divisible_int::Divisor;
    use super::EvenDivisibleInt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::EmptyMetadata;
    use crate::scalar::PValue;
    use crate::scalar::ScalarValue;

    fn even_dtype(divisor: u64) -> VortexResult<ExtDType<EvenDivisibleInt>> {
        let inner = ExtDType::<DivisibleInt>::try_new(
            Divisor(divisor),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?;
        ExtDType::<EvenDivisibleInt>::try_new(EmptyMetadata, DType::Extension(inner.erased()))
    }

    #[test]
    fn accepts_valid_ext_over_ext_storage() -> VortexResult<()> {
        let _dtype = even_dtype(4)?;
        Ok(())
    }

    #[test]
    fn rejects_non_extension_storage() {
        let built = ExtDType::<EvenDivisibleInt>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::U64, Nullability::NonNullable),
        );
        assert!(built.is_err(), "must reject non-extension storage");
    }

    #[test]
    fn rejects_mismatched_inner_extension() -> VortexResult<()> {
        use crate::extension::uuid::Uuid;
        let uuid = ExtDType::<Uuid>::try_new(
            Default::default(),
            DType::FixedSizeList(
                std::sync::Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
                16,
                Nullability::NonNullable,
            ),
        )?;

        let built =
            ExtDType::<EvenDivisibleInt>::try_new(EmptyMetadata, DType::Extension(uuid.erased()));
        assert!(
            built.is_err(),
            "must reject non-DivisibleInt inner extension"
        );
        Ok(())
    }

    #[test]
    fn unpack_accepts_even_divisible_value() -> VortexResult<()> {
        let dtype = even_dtype(3)?;
        // 12 is divisible by 3 and even, so both the inner DivisibleInt predicate and the outer
        // EvenDivisibleInt predicate succeed.
        let storage = ScalarValue::Primitive(PValue::U64(12));
        let value = EvenDivisibleInt::unpack_native(&dtype, &storage)?;
        assert_eq!(value, 12);
        Ok(())
    }

    #[test]
    fn unpack_rejects_odd_divisible_value() -> VortexResult<()> {
        let dtype = even_dtype(3)?;
        // 9 is divisible by 3 (inner predicate succeeds) but odd (outer predicate fails).
        let storage = ScalarValue::Primitive(PValue::U64(9));
        assert!(EvenDivisibleInt::unpack_native(&dtype, &storage).is_err());
        Ok(())
    }

    #[test]
    fn unpack_rejects_not_divisible_value() -> VortexResult<()> {
        let dtype = even_dtype(3)?;
        // 8 is even but not divisible by 3. The inner `DivisibleInt` predicate fires before we
        // ever reach the outer even-ness check, proving that refinements compose.
        let storage = ScalarValue::Primitive(PValue::U64(8));
        assert!(EvenDivisibleInt::unpack_native(&dtype, &storage).is_err());
        Ok(())
    }
}
