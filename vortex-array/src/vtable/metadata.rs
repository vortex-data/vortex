use std::fmt::Formatter;

use vortex_error::VortexResult;

use crate::encoding::Encoding;
use crate::{ArrayRef, DeserializeMetadata};

pub trait MetadataVTable<Array> {
    fn validate_metadata(&self, metadata: Option<&[u8]>) -> VortexResult<()>;

    fn display_metadata(&self, array: &Array, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl<E: Encoding> MetadataVTable<ArrayRef> for E {
    fn validate_metadata(&self, metadata: Option<&[u8]>) -> VortexResult<()> {
        E::Metadata::deserialize(metadata).map(|_| ())
    }

    fn display_metadata(&self, array: &ArrayRef, f: &mut Formatter<'_>) -> std::fmt::Result {
        <E::Metadata as DeserializeMetadata>::format(array.metadata_bytes(), f)
    }
}
