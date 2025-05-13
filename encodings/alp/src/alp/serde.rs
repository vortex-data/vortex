use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef, Canonical,
    DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_panic};

use super::{ALPEncoding, alp_encode};
use crate::{ALPArray, ALPVTable, Exponents};

#[derive(Clone, prost::Message)]
pub struct ALPMetadata {
    #[prost(uint32, tag = "1")]
    exp_e: u32,
    #[prost(uint32, tag = "2")]
    exp_f: u32,
    #[prost(message, optional, tag = "3")]
    patches: Option<PatchesMetadata>,
}

impl SerdeVTable<ALPVTable> for ALPVTable {
    type Metadata = ProstMetadata<ALPMetadata>;

    fn metadata(array: &ALPArray) -> VortexResult<Option<Self::Metadata>> {
        let exponents = array.exponents();
        Ok(Some(ProstMetadata(ALPMetadata {
            exp_e: exponents.e as u32,
            exp_f: exponents.f as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
                .transpose()?,
        })))
    }

    fn build(
        _encoding: &ALPEncoding,
        dtype: DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<ALPArray> {
        let encoded_ptype = match &dtype {
            DType::Primitive(PType::F32, n) => DType::Primitive(PType::I32, *n),
            DType::Primitive(PType::F64, n) => DType::Primitive(PType::I64, *n),
            d => vortex_panic!(MismatchedTypes: "f32 or f64", d),
        };
        let encoded = children[0].decode(ctx, encoded_ptype, len)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children[1].decode(ctx, p.indices_dtype(), p.len())?;
                let values = children[2].decode(ctx, dtype, p.len())?;
                Ok::<_, VortexError>(Patches::new(len, p.offset(), indices, values))
            })
            .transpose()?;

        ALPArray::try_new(
            encoded,
            Exponents {
                e: u8::try_from(metadata.exp_e).vortex_expect("Exponent e overflow"),
                f: u8::try_from(metadata.exp_f).vortex_expect("Exponent f overflow"),
            },
            patches,
        )
    }
}

impl EncodeVTable<ALPVTable> for ALPVTable {
    fn encode(
        encoding: &ALPEncoding,
        canonical: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ALPArray>> {
        let parray = canonical.clone().into_primitive()?;

        let like_alp = like
            .map(|like| {
                like.as_opt::<Self>().ok_or_else(|| {
                    vortex_err!(
                        "Expected {} encoded array but got {}",
                        encoding.id(),
                        like.encoding_id()
                    )
                })
            })
            .transpose()?;
        let exponents = like_alp.map(|a| a.exponents());
        let alp = alp_encode(&parray, exponents)?;

        Ok(Some(alp))
    }
}

impl VisitorVTable<ALPVTable> for ALPVTable {
    fn visit_buffers(_array: &ALPArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ALPArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded());
        if let Some(patches) = array.patches() {
            visitor.visit_patches(patches);
        }
    }

    fn with_children(array: &ALPArray, children: &[ArrayRef]) -> VortexResult<ALPArray> {
        let encoded = children[0].clone();

        let patches = array.patches().map(|existing| {
            let indices = children[1].clone();
            let values = children[2].clone();
            Patches::new(existing.array_len(), existing.offset(), indices, values)
        });

        ALPArray::try_new(encoded, array.exponents(), patches)
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
