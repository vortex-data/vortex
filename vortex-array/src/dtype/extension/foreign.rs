// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ForeignExtMetadata(pub Vec<u8>);

impl Display for ForeignExtMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0.len())
    }
}

/// Placeholder extension dtype used when deserializing an unknown extension ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ForeignExtDType {
    id: ExtId,
}

impl ForeignExtDType {
    pub fn new(id: ExtId) -> Self {
        Self { id }
    }

    pub fn from_parts(
        id: ExtId,
        metadata: Vec<u8>,
        storage_dtype: DType,
    ) -> VortexResult<ExtDTypeRef> {
        Ok(
            ExtDType::try_with_vtable(Self::new(id), ForeignExtMetadata(metadata), storage_dtype)?
                .erased(),
        )
    }
}

impl ExtVTable for ForeignExtDType {
    type Metadata = ForeignExtMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        self.id
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.0.clone())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ForeignExtMetadata(metadata.to_vec()))
    }

    fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}
