use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef, ArrayVisitorImpl, Canonical,
    DeserializeMetadata, Encoding, EncodingId, ProstMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use super::{ALPRDEncoding, RDEncoder};
use crate::ALPRDArray;

#[derive(Clone, prost::Message)]
pub struct ALPRDMetadata {
    #[prost(uint32, tag = "1")]
    right_bit_width: u32,
    #[prost(uint32, tag = "2")]
    dict_len: u32,
    #[prost(uint32, repeated, tag = "3")]
    dict: Vec<u32>,
    #[prost(enumeration = "PType", tag = "4")]
    left_parts_ptype: i32,
    #[prost(message, tag = "5")]
    patches: Option<PatchesMetadata>,
}

impl EncodingVTable for ALPRDEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.alprd")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = ProstMetadata::<ALPRDMetadata>::deserialize(parts.metadata())?;

        if parts.nchildren() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                parts.nchildren()
            );
        }

        let left_parts_dtype = DType::Primitive(metadata.left_parts_ptype(), dtype.nullability());
        let left_parts = parts.child(0).decode(ctx, left_parts_dtype.clone(), len)?;
        let left_parts_dictionary = Buffer::from_iter(
            metadata.dict.as_slice()[0..metadata.dict_len as usize]
                .iter()
                .map(|&i| u16::try_from(i).vortex_expect("Dictionary index out of range")),
        );

        let right_parts_dtype = match &dtype {
            DType::Primitive(PType::F32, _) => {
                DType::Primitive(PType::U32, Nullability::NonNullable)
            }
            DType::Primitive(PType::F64, _) => {
                DType::Primitive(PType::U64, Nullability::NonNullable)
            }
            _ => vortex_bail!("Expected f32 or f64 dtype, got {:?}", dtype),
        };
        let right_parts = parts.child(1).decode(ctx, right_parts_dtype, len)?;

        let left_parts_patches = metadata
            .patches
            .map(|p| {
                let indices = parts.child(2).decode(ctx, p.indices_dtype(), p.len())?;
                let values = parts.child(3).decode(ctx, left_parts_dtype, p.len())?;
                Ok::<_, VortexError>(Patches::new(len, p.offset(), indices, values))
            })
            .transpose()?;

        Ok(ALPRDArray::try_new(
            dtype,
            left_parts,
            left_parts_dictionary,
            right_parts,
            u8::try_from(metadata.right_bit_width).vortex_expect("Bit width out of range"),
            left_parts_patches,
        )?
        .into_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let parray = input.clone().into_primitive()?;

        let like_alprd = like
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

        let alprd_array = match like_alprd {
            None => {
                let encoder = match parray.ptype() {
                    PType::F32 => RDEncoder::new(parray.as_slice::<f32>()),
                    PType::F64 => RDEncoder::new(parray.as_slice::<f64>()),
                    ptype => vortex_bail!("cannot ALPRD compress ptype {ptype}"),
                };
                encoder.encode(&parray)
            }
            Some(like) => {
                let encoder = RDEncoder::from_parts(
                    like.right_bit_width(),
                    like.left_parts_dictionary().to_vec(),
                );
                encoder.encode(&parray)
            }
        };

        Ok(Some(alprd_array.into_array()))
    }
}

impl ArrayVisitorImpl<ProstMetadata<ALPRDMetadata>> for ALPRDArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("left_parts", self.left_parts());
        visitor.visit_child("right_parts", self.right_parts());
        if let Some(patches) = self.left_parts_patches() {
            visitor.visit_patches(patches);
        }
    }

    fn _metadata(&self) -> ProstMetadata<ALPRDMetadata> {
        let dict = self
            .left_parts_dictionary()
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>();

        ProstMetadata(ALPRDMetadata {
            right_bit_width: self.right_bit_width() as u32,
            dict_len: self.left_parts_dictionary().len() as u32,
            dict,
            left_parts_ptype: PType::try_from(self.left_parts().dtype())
                .vortex_expect("Must be a valid PType") as i32,
            patches: self
                .left_parts_patches()
                .map(|p| p.to_metadata(self.len(), self.left_parts().dtype()))
                .transpose()
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::alp_rd::serde::ALPRDMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alprd_metadata() {
        check_metadata(
            "alprd.metadata",
            ProstMetadata(ALPRDMetadata {
                right_bit_width: u32::MAX,
                patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
                dict: Vec::new(),
                left_parts_ptype: PType::U64 as i32,
                dict_len: 8,
            }),
        );
    }
}
