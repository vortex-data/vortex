use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef,
    DeserializeMetadata, RkyvMetadata,
};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{SparseArray, SparseEncoding};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct SparseMetadata {
    patches: PatchesMetadata,
}

impl ArrayVisitorImpl<RkyvMetadata<SparseMetadata>> for SparseArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let fill_value_buffer = self.fill_value.value().to_flexbytes().into_inner();
        visitor.visit_buffer(&fill_value_buffer);
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_patches(self.patches())
    }

    fn _metadata(&self) -> RkyvMetadata<SparseMetadata> {
        RkyvMetadata(SparseMetadata {
            patches: self
                .patches()
                .to_metadata(self.len(), self.dtype())
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}

impl SerdeVTable<&SparseArray> for SparseEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 2 {
            vortex_bail!(
                "Expected 2 children for sparse encoding, found {}",
                parts.nchildren()
            )
        }
        let metadata = RkyvMetadata::<SparseMetadata>::deserialize(parts.metadata())?;
        assert_eq!(
            metadata.patches.offset(),
            0,
            "Patches must start at offset 0"
        );

        let patch_indices = parts.child(0).decode(
            ctx,
            metadata.patches.indices_dtype(),
            metadata.patches.len(),
        )?;
        let patch_values = parts
            .child(1)
            .decode(ctx, dtype.clone(), metadata.patches.len())?;

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let fill_value = Scalar::new(dtype, ScalarValue::from_flexbytes(&parts.buffer(0)?)?);

        Ok(SparseArray::try_new(patch_indices, patch_values, len, fill_value)?.into_array())
    }
}
