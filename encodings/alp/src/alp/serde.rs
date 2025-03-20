use serde::{Deserialize, Serialize};
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    EncodingId, SerdeMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};

use super::ALPEncoding;
use crate::{ALPArray, Exponents};

impl EncodingVTable for ALPEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.alp")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = SerdeMetadata::<ALPMetadata>::deserialize(parts.metadata())?;

        let encoded_ptype = match &dtype {
            DType::Primitive(PType::F32, n) => DType::Primitive(PType::I32, *n),
            DType::Primitive(PType::F64, n) => DType::Primitive(PType::I64, *n),
            d => vortex_panic!(MismatchedTypes: "f32 or f64", d),
        };
        let encoded = parts.child(0).decode(ctx, encoded_ptype, len)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = parts.child(1).decode(ctx, p.indices_dtype(), p.len())?;
                let values = parts.child(2).decode(ctx, dtype, p.len())?;
                Ok::<_, VortexError>(Patches::new(len, p.offset(), indices, values))
            })
            .transpose()?;

        Ok(ALPArray::try_new(encoded, metadata.exponents, patches)?.into_array())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPMetadata {
    pub(crate) exponents: Exponents,
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<SerdeMetadata<ALPMetadata>> for ALPArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded());
        if let Some(patches) = self.patches() {
            visitor.visit_patches(patches);
        }
    }

    fn _metadata(&self) -> SerdeMetadata<ALPMetadata> {
        SerdeMetadata(ALPMetadata {
            exponents: self.exponents(),
            patches: self
                .patches()
                .map(|p| p.to_metadata(self.len(), self.dtype()))
                .transpose()
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}
