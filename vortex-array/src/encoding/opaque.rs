use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use vortex_error::{vortex_bail, vortex_panic, VortexResult};

use crate::compute::ComputeVTable;
use crate::encoding::{EncodingId, EncodingVTable};
use crate::stats::StatisticsVTable;
use crate::validity::{LogicalValidity, ValidityVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    ArrayData, ArrayMetadata, ArrayTrait, Canonical, IntoCanonicalVTable, MetadataVTable,
    TrySerializeArrayMetadata,
};

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

impl EncodingVTable for OpaqueEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new("vortex.opaque", self.0)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_dyn(
        &self,
        _array: &ArrayData,
        _f: &mut dyn for<'b> FnMut(&'b (dyn ArrayTrait + 'b)) -> VortexResult<()>,
    ) -> VortexResult<()> {
        vortex_bail!(
            "OpaqueEncoding: with_dyn cannot be called for opaque array ({})",
            self.0
        )
    }
}

impl IntoCanonicalVTable for OpaqueEncoding {
    fn into_canonical(&self, _array: ArrayData) -> VortexResult<Canonical> {
        vortex_bail!(
            "OpaqueEncoding: into_canonical cannot be called for opaque array ({})",
            self.0
        )
    }
}

impl ComputeVTable for OpaqueEncoding {}

impl MetadataVTable for OpaqueEncoding {
    fn load_metadata(&self, _metadata: Option<&[u8]>) -> VortexResult<Arc<dyn ArrayMetadata>> {
        Ok(Arc::new(OpaqueMetadata))
    }
}

impl StatisticsVTable<ArrayData> for OpaqueEncoding {}

impl ValidityVTable<ArrayData> for OpaqueEncoding {
    fn is_valid(&self, _array: &ArrayData, _index: usize) -> bool {
        vortex_panic!(
            "OpaqueEncoding: is_valid cannot be called for opaque array ({})",
            self.0
        )
    }

    fn logical_validity(&self, _array: &ArrayData) -> LogicalValidity {
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

#[derive(Debug)]
pub struct OpaqueMetadata;

impl TrySerializeArrayMetadata for OpaqueMetadata {
    fn try_serialize_metadata(&self) -> VortexResult<Arc<[u8]>> {
        vortex_bail!("OpaqueMetadata cannot be serialized")
    }
}

impl Display for OpaqueMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpaqueMetadata")
    }
}

impl ArrayMetadata for OpaqueMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}
