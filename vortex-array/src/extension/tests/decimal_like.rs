// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A fixed-scale decimal fixture with a signed 64-bit coefficient.

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::dtype::Claim;
use crate::dtype::Claims;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// An extension dtype whose metadata fixes the decimal scale for every coefficient.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct DecimalLike;

impl ExtVTable for DecimalLike {
    type Metadata = i8;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = i64;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.decimal_like");
        *ID
    }

    fn serialize_metadata(&self, scale: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(scale.to_le_bytes().to_vec())
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
        let scale = bytes
            .try_into()
            .map_err(|_| vortex_err!("decimal-like scale must be one byte"))?;
        Ok(i8::from_le_bytes(scale))
    }

    fn metadata_from_storage(
        &self,
        scale: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(scale)
    }

    fn storage_dtype(_scale: &Self::Metadata) -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn claims(_scale: &Self::Metadata) -> Claims {
        // A dtype fixes the scale, so coefficient order is decimal order.
        Claims::none().with(Claim::Ord)
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        storage_value.as_primitive().cast::<i64>()
    }
}
