use serde::{Deserialize, Serialize};
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, DeserializeMetadata,
    SerdeMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexError, VortexExpect, VortexResult};

use crate::{ALPRDArray, ALPRDEncoding};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPRDMetadata {
    right_bit_width: u8,
    dict_len: u8,
    dict: [u16; 8],
    left_parts_ptype: PType,
    patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<SerdeMetadata<ALPRDMetadata>> for ALPRDArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
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

impl SerdeVTable<&ALPRDArray> for ALPRDEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
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
}
#[cfg(test)]
mod test {
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
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
