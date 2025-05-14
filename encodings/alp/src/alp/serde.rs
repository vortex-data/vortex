use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};

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
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ALPArray> {
        let encoded_ptype = match &dtype {
            DType::Primitive(PType::F32, n) => DType::Primitive(PType::I32, *n),
            DType::Primitive(PType::F64, n) => DType::Primitive(PType::I64, *n),
            d => vortex_panic!(MismatchedTypes: "f32 or f64", d),
        };
        let encoded = children.get(0, &encoded_ptype, len)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(1, &p.indices_dtype(), p.len())?;
                let values = children.get(2, dtype, p.len())?;
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
        _encoding: &ALPEncoding,
        canonical: &Canonical,
        like: Option<&ALPArray>,
    ) -> VortexResult<Option<ALPArray>> {
        let parray = canonical.clone().into_primitive()?;
        let exponents = like.map(|a| a.exponents());
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
