// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A test extension type representing unsigned integers divisible by a given divisor.
//!
//! `DivisibleInt` is a refinement of `Primitive(U64)`: every valid value is a `u64`, with the
//! additional invariant that it is divisible by the metadata-provided [`Divisor`]. Its
//! `ExtVTable::is_refinement` returns `true` so that generic scalar-fn dispatch can peel it
//! to its storage dtype automatically.

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// The divisor stored as extension metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Divisor(pub u64);

impl fmt::Display for Divisor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "divisible by {}", self.0)
    }
}

/// Refinement type for unsigned integers that must be divisible by the metadata divisor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DivisibleInt;

impl ExtVTable for DivisibleInt {
    type Metadata = Divisor;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new("test.divisible_int")
    }

    fn is_refinement(&self) -> bool {
        true
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        match ext_dtype.storage_dtype() {
            DType::Primitive(PType::U64, _) => Ok(()),
            other => vortex_bail!("`DivisibleInt` requires `U64` storage, got {other}"),
        }
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let ScalarValue::Primitive(pv) = storage_value else {
            vortex_bail!("`DivisibleInt` expected a primitive scalar, got {storage_value:?}");
        };
        let n = pv.cast::<u64>()?;
        let divisor = ext_dtype.metadata().0;
        vortex_ensure!(
            n.is_multiple_of(divisor),
            "{n} is not divisible by {divisor}",
        );
        Ok(n)
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.0.to_le_bytes().to_vec())
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_ensure!(data.len() == 8, "divisible int metadata must be 8 bytes");
        let bytes: [u8; 8] = data
            .try_into()
            .map_err(|_| vortex_error::vortex_err!("divisible int metadata must be 8 bytes"))?;
        let n = u64::from_le_bytes(bytes);
        vortex_ensure!(n > 0, "divisor must be greater than 0");
        Ok(Divisor(n))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::DivisibleInt;
    use super::Divisor;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;

    #[test]
    fn metadata_roundtrip() -> VortexResult<()> {
        let vtable = DivisibleInt;
        let divisor = Divisor(42);

        let bytes = vtable.serialize_metadata(&divisor)?;
        let decoded = vtable.deserialize_metadata(&bytes)?;

        assert_eq!(decoded, divisor);
        Ok(())
    }

    #[test]
    fn rejects_zero_divisor() {
        let bytes = 0u64.to_le_bytes();
        assert!(DivisibleInt.deserialize_metadata(&bytes).is_err());
    }

    #[test]
    fn rejects_wrong_storage_dtype() {
        let divisor = Divisor(10);

        assert!(
            ExtDType::<DivisibleInt>::try_new(
                divisor,
                DType::Primitive(PType::I32, Nullability::NonNullable)
            )
            .is_err()
        );
        assert!(
            ExtDType::<DivisibleInt>::try_new(divisor, DType::Utf8(Nullability::NonNullable))
                .is_err()
        );
        assert!(
            ExtDType::<DivisibleInt>::try_new(
                divisor,
                DType::Primitive(PType::U64, Nullability::NonNullable)
            )
            .is_ok()
        );
    }

    #[test]
    fn is_refinement_is_true() {
        assert!(DivisibleInt.is_refinement());
    }
}
