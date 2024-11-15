use std::fmt::Debug;

use vortex_error::{vortex_bail, VortexResult};

use crate::encoding::{ArrayEncoding, EncodingId};
use crate::{ArrayData, ArrayTrait, Canonical};

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

impl ArrayEncoding for OpaqueEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new("vortex.opaque", self.0)
    }

    fn canonicalize(&self, _array: ArrayData) -> VortexResult<Canonical> {
        vortex_bail!(
            "OpaqueArray: canonicalize cannot be called for opaque array ({})",
            self.0
        );
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
