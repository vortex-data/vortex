use serde::{Deserialize, Serialize};
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef, ArrayVisitorImpl, Canonical,
    DeserializeMetadata, Encoding, EncodingId, SerdeMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{
    VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic,
};

use super::{ALPEncoding, alp_encode, alp_encode_with_exponents};
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

    fn encode(&self, input: &Canonical, like: Option<&dyn Array>) -> VortexResult<ArrayRef> {
        let Canonical::Primitive(parray) = input else {
            vortex_bail!("Expected a primitive input")
        };

        let like_alp = like
            .map(|like| {
                like.as_opt::<<Self as Encoding>::Array>().ok_or_else(|| {
                    vortex_err!(
                        "Expected {} encoded array but got {}",
                        self.id(),
                        like.vtable().id()
                    )
                })
            })
            .transpose()?;
        let exponents = like_alp.map(|a| a.exponents());

        let alp = match exponents {
            Some(e) => alp_encode_with_exponents(parray, e)?,
            None => alp_encode(parray)?,
        };

        Ok(alp.into_array())
    }

    fn from_children(
        &self,
        existing: ArrayRef,
        new_children: Vec<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        if existing.nchildren() != new_children.len() {
            vortex_bail!("Children length doesn't match")
        }

        let mut children = new_children.into_iter();
        let existing = existing.as_::<<Self as Encoding>::Array>();

        let encoded = children.next().vortex_expect("");
        let patches = if children.len() == 2 {
            let existing_patches = existing.patches().vortex_expect("Must have patches");
            let patches_indices = children.next().vortex_expect("");
            let patches_values = children.next().vortex_expect("");
            Some(Patches::new(
                existing_patches.array_len(),
                existing_patches.offset(),
                patches_indices,
                patches_values,
            ))
        } else {
            None
        };

        let valid = ALPArray::try_new(encoded, existing.exponents(), patches)?;

        Ok(valid.into_array())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPMetadata {
    exponents: Exponents,
    patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<SerdeMetadata<ALPMetadata>> for ALPArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
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

#[cfg(test)]
mod tests {
    use vortex_array::SerdeMetadata;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::Exponents;
    use crate::alp::serde::ALPMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alp_metadata() {
        check_metadata(
            "alp.metadata",
            SerdeMetadata(ALPMetadata {
                patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
                exponents: Exponents {
                    e: u8::MAX,
                    f: u8::MAX,
                },
            }),
        );
    }
}
