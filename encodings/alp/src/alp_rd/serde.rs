use serde::{Deserialize, Serialize};
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef, ArrayVisitorImpl, Canonical,
    DeserializeMetadata, Encoding, EncodingId, SerdeMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use super::{ALPRDEncoding, RDEncoder};
use crate::ALPRDArray;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPRDMetadata {
    right_bit_width: u8,
    dict_len: u8,
    dict: [u16; 8],
    left_parts_ptype: PType,
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
        let metadata = SerdeMetadata::<ALPRDMetadata>::deserialize(parts.metadata())?;

        if parts.nchildren() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                parts.nchildren()
            );
        }

        let left_parts_dtype = DType::Primitive(metadata.left_parts_ptype, dtype.nullability());
        let left_parts = parts.child(0).decode(ctx, left_parts_dtype.clone(), len)?;
        let left_parts_dictionary =
            Buffer::copy_from(&metadata.dict.as_slice()[0..metadata.dict_len as usize]);

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
            metadata.right_bit_width,
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

impl ArrayVisitorImpl<SerdeMetadata<ALPRDMetadata>> for ALPRDArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("left_parts", self.left_parts());
        visitor.visit_child("right_parts", self.right_parts());
        if let Some(patches) = self.left_parts_patches() {
            visitor.visit_patches(patches);
        }
    }

    fn _metadata(&self) -> SerdeMetadata<ALPRDMetadata> {
        let mut dict = [0u16; 8];
        dict[0..self.left_parts_dictionary().len()].copy_from_slice(self.left_parts_dictionary());

        SerdeMetadata(ALPRDMetadata {
            right_bit_width: self.right_bit_width(),
            dict_len: self.left_parts_dictionary().len() as u8,
            dict,
            left_parts_ptype: PType::try_from(self.left_parts().dtype())
                .vortex_expect("Must be a valid PType"),
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
    use vortex_array::SerdeMetadata;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::alp_rd::serde::ALPRDMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alprd_metadata() {
        check_metadata(
            "alprd.metadata",
            SerdeMetadata(ALPRDMetadata {
                right_bit_width: u8::MAX,
                patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
                dict: [0u16; 8],
                left_parts_ptype: PType::U64,
                dict_len: 8,
            }),
        );
    }
}
