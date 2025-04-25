use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef, ArrayVisitorImpl, Canonical,
    DeserializeMetadata, Encoding, EncodingId, ProstMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_panic};

use super::{ALPEncoding, alp_encode};
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
        let metadata = ProstMetadata::<ALPMetadata>::deserialize(parts.metadata())?;

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

        Ok(ALPArray::try_new(
            encoded,
            Exponents {
                e: u8::try_from(metadata.exp_e).vortex_expect("Exponent e overflow"),
                f: u8::try_from(metadata.exp_f).vortex_expect("Exponent f overflow"),
            },
            patches,
        )?
        .into_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let parray = input.clone().into_primitive()?;

        let like_alp = like
            .map(|like| {
                like.as_opt::<<Self as Encoding>::Array>().ok_or_else(|| {
                    vortex_err!(
                        "Expected {} encoded array but got {}",
                        self.id(),
                        like.encoding()
                    )
                })
            })
            .transpose()?;
        let exponents = like_alp.map(|a| a.exponents());
        let alp = alp_encode(&parray, exponents)?;

        Ok(Some(alp.into_array()))
    }
}

#[derive(Clone, prost::Message)]
pub struct ALPMetadata {
    #[prost(uint32, tag = "1")]
    exp_e: u32,
    #[prost(uint32, tag = "2")]
    exp_f: u32,
    #[prost(message, optional, tag = "3")]
    patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<ProstMetadata<ALPMetadata>> for ALPArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded());
        if let Some(patches) = self.patches() {
            visitor.visit_patches(patches);
        }
    }

    fn _metadata(&self) -> ProstMetadata<ALPMetadata> {
        let exponents = self.exponents();
        ProstMetadata(ALPMetadata {
            exp_e: exponents.e as u32,
            exp_f: exponents.f as u32,
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
    use vortex_array::ProstMetadata;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::alp::serde::ALPMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alp_metadata() {
        check_metadata(
            "alp.metadata",
            ProstMetadata(ALPMetadata {
                patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
                exp_e: u32::MAX,
                exp_f: u32::MAX,
            }),
        );
    }
}
