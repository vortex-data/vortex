use std::any::Any;
use std::fmt::{Debug, Formatter};

use vortex_error::{vortex_bail, vortex_panic, VortexResult};
use vortex_mask::Mask;

use crate::encoding::EncodingId;
use crate::visitor::ArrayVisitor;
use crate::vtable::{
    CanonicalVTable, ComputeVTable, EncodingVTable, MetadataVTable, StatisticsVTable,
    ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use crate::{Array, Canonical};

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

impl VariantsVTable<Array> for OpaqueEncoding {}

impl MetadataVTable<Array> for OpaqueEncoding {
    fn validate_metadata(&self, _metadata: Option<&[u8]>) -> VortexResult<()> {
        Ok(())
    }

    fn display_metadata(&self, _array: &Array, f: &mut Formatter<'_>) -> std::fmt::Result {
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

impl CanonicalVTable<Array> for OpaqueEncoding {
    fn into_canonical(&self, _array: Array) -> VortexResult<Canonical> {
        vortex_bail!(
            "OpaqueEncoding: into_canonical cannot be called for opaque array ({})",
            self.0
        )
    }
}

impl ComputeVTable for OpaqueEncoding {}

impl StatisticsVTable<Array> for OpaqueEncoding {}

impl ValidateVTable<Array> for OpaqueEncoding {}

impl ValidityVTable<Array> for OpaqueEncoding {
    fn is_valid(&self, _array: &Array, _index: usize) -> VortexResult<bool> {
        vortex_panic!(
            "OpaqueEncoding: is_valid cannot be called for opaque array ({})",
            self.0
        )
    }

    fn logical_validity(&self, _array: &Array) -> VortexResult<Mask> {
        vortex_panic!(
            "OpaqueEncoding: logical_validity cannot be called for opaque array ({})",
            self.0
        )
    }
}

impl VisitorVTable<Array> for OpaqueEncoding {
    fn accept(&self, _array: &Array, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        vortex_bail!(
            "OpaqueEncoding: into_canonical cannot be called for opaque array ({})",
            self.0
        )
    }
}
