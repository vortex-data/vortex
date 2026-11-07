// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A test extension type representing unsigned integers divisible by a given divisor.

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::dtype::DType;
use crate::dtype::Nullability;
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

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test.divisible_int")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.0.to_le_bytes().to_vec())
    }

    fn deserialize_metadata(
        &self,
        data: &[u8],
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        vortex_ensure_eq!(data.len(), 8, "divisible int metadata has the wrong length");
        let bytes: [u8; 8] = data
            .try_into()
            .map_err(|_| vortex_error::vortex_err!("divisible int metadata must be 8 bytes"))?;
        let n = u64::from_le_bytes(bytes);
        vortex_ensure!(n > 0, "divisor must be greater than 0");
        Ok(Divisor(n))
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Primitive(PType::U64, Nullability::NonNullable)
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
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::DivisibleInt;
    use super::Divisor;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDTypePlugin;
    use crate::dtype::extension::ExtVTable;

    #[test]
    fn read_accepts_u64_storage() -> VortexResult<()> {
        let metadata = DivisibleInt.serialize_metadata(&Divisor(10))?;
        let storage_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);

        DivisibleInt.deserialize(&metadata, storage_dtype)?;

        Ok(())
    }

    #[rstest]
    #[case::wrong_primitive(DType::Primitive(PType::I32, Nullability::NonNullable))]
    #[case::non_primitive(DType::Utf8(Nullability::NonNullable))]
    fn read_rejects_wrong_storage_dtype(#[case] storage_dtype: DType) -> VortexResult<()> {
        let metadata = DivisibleInt.serialize_metadata(&Divisor(10))?;

        assert!(DivisibleInt.deserialize(&metadata, storage_dtype).is_err());

        Ok(())
    }
}
