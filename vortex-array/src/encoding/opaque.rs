use std::any::Any;
use std::fmt::{Debug, Display, Formatter};

use arrow_array::ArrayRef;
use vortex_error::{vortex_bail, vortex_panic, VortexResult};
use vortex_mask::Mask;

use crate::encoding::EncodingId;
use crate::visitor::ArrayVisitor;
use crate::vtable::{
    CanonicalVTable, ComputeVTable, EncodingVTable, MetadataVTable, StatisticsVTable,
    ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use crate::{ArrayData, Canonical};

/// An encoding of an array that we cannot interpret.
///
/// Vortex allows for pluggable encodings. This can lead to issues when one process produces a file
/// using a custom encoding, and then another process without knowledge of the encoding attempts
/// to read it.
///
/// `OpaqueEncoding` allows deserializing these arrays. Many common operations will fail, but it
/// allows deserialization and introspection in a type-erased manner on the children and metadata.
///
/// We hold the original 16-bit encoding ID for producing helpful error messages.
#[derive(Debug, Clone, Copy)]
pub struct OpaqueEncoding(pub u16);

impl VariantsVTable<ArrayData> for OpaqueEncoding {}

impl MetadataVTable<ArrayData> for OpaqueEncoding {
    fn validate_metadata(&self, _metadata: Option<&[u8]>) -> VortexResult<()> {
        Ok(())
    }

    fn display_metadata(&self, _array: &ArrayData, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpaqueMetadata")
    }
}

impl EncodingVTable for OpaqueEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new("vortex.opaque", self.0)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CanonicalVTable<ArrayData> for OpaqueEncoding {
    fn into_canonical(&self, _array: ArrayData) -> VortexResult<Canonical> {
        vortex_bail!(
            "OpaqueEncoding: into_canonical cannot be called for opaque array ({})",
            self.0
        )
    }
}

impl ComputeVTable for OpaqueEncoding {}

impl StatisticsVTable<ArrayData> for OpaqueEncoding {}

impl ValidateVTable<ArrayData> for OpaqueEncoding {}

impl ValidityVTable<ArrayData> for OpaqueEncoding {
    fn is_valid(&self, _array: &ArrayData, _index: usize) -> VortexResult<bool> {
        vortex_panic!(
            "OpaqueEncoding: is_valid cannot be called for opaque array ({})",
            self.0
        )
    }

    fn logical_validity(&self, _array: &ArrayData) -> VortexResult<Mask> {
        vortex_panic!(
            "OpaqueEncoding: logical_validity cannot be called for opaque array ({})",
            self.0
        )
    }
}

impl VisitorVTable<ArrayData> for OpaqueEncoding {
    fn accept(&self, _array: &ArrayData, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        vortex_bail!(
            "OpaqueEncoding: into_canonical cannot be called for opaque array ({})",
            self.0
        )
    }
}
