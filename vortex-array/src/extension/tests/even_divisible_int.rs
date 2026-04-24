// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A test extension type layering a refinement on top of another refinement.
//!
//! [`EvenDivisibleInt`] refines [`DivisibleInt`] with the additional requirement that the
//! value is even. Its storage `DType` is therefore `DType::Extension(DivisibleInt)`, and its
//! validation chain transitively inherits `DivisibleInt`'s divisibility check: when the
//! outer `ExtDType` is constructed, the inner `DivisibleInt` extension already ran its own
//! `validate_dtype`.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::divisible_int::DivisibleInt;
use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::EmptyMetadata;
use crate::scalar::ScalarValue;

/// Refinement of [`DivisibleInt`] requiring the stored value to additionally be even.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EvenDivisibleInt;

impl ExtVTable for EvenDivisibleInt {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new("test.even_divisible_int")
    }

    fn is_refinement(&self) -> bool {
        true
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        let DType::Extension(inner) = ext_dtype.storage_dtype() else {
            vortex_bail!(
                "`EvenDivisibleInt` requires extension storage, got {}",
                ext_dtype.storage_dtype(),
            );
        };
        vortex_ensure!(
            inner.is::<DivisibleInt>(),
            "`EvenDivisibleInt` requires `DivisibleInt` storage, got {}",
            inner.id(),
        );
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        // Compose with `DivisibleInt::unpack_native`: the inner refinement's divisibility
        // check runs first; only values that pass reach the even-ness check here.
        let DType::Extension(inner) = ext_dtype.storage_dtype() else {
            vortex_bail!("unreachable: validate_dtype rejects non-extension storage");
        };
        let inner_typed = inner.as_typed::<DivisibleInt>().ok_or_else(|| {
            vortex_err!("unreachable: validate_dtype rejects non-`DivisibleInt` inner extension")
        })?;
        let n = DivisibleInt::unpack_native(inner_typed, storage_value)?;
        vortex_ensure!(n.is_multiple_of(2), "{n} is not even");
        Ok(n)
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
        // 12 is divisible by 3 and even, so both the inner DivisibleInt predicate and the
        // outer EvenDivisibleInt predicate succeed.
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
        // 8 is even but not divisible by 3. The inner `DivisibleInt` predicate fires before
        // we ever reach the outer even-ness check, proving that refinements compose via
        // nested `unpack_native` calls.
        let storage = ScalarValue::Primitive(PValue::U64(8));
        assert!(EvenDivisibleInt::unpack_native(&dtype, &storage).is_err());
        Ok(())
    }

    #[test]
    fn is_refinement_is_true() {
        assert!(EvenDivisibleInt.is_refinement());
    }
}
