// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A test extension type representing unsigned integers divisible by a given divisor.

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

/// Extension type for unsigned integers that must be divisible by the metadata divisor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DivisibleInt;

impl ExtVTable for DivisibleInt {
    type Metadata = Divisor;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new_ref("test.divisible_int")
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

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            matches!(ext_dtype.storage_dtype(), DType::Primitive(PType::U64, _)),
            "divisible int storage dtype must be u64"
        );
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let value = storage_value.as_primitive().cast::<u64>()?;
        let metadata = ext_dtype.metadata();
        if value % metadata.0 != 0 {
            vortex_bail!("{} is not divisible by {}", value, metadata.0);
        }
        Ok(value)
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
        let vtable = DivisibleInt;
        let bytes = 0u64.to_le_bytes();
        assert!(vtable.deserialize_metadata(&bytes).is_err());
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
}
