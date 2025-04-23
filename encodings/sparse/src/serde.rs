use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef,
    ArrayVisitorImpl, Canonical, DeserializeMetadata, Encoding, EncodingId, ProstMetadata,
};
use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{SparseArray, SparseEncoding};

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct SparseMetadata {
    #[prost(message, required, tag = "1")]
    patches: PatchesMetadata,
}

impl EncodingVTable for SparseEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.sparse")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 2 {
            vortex_bail!(
                "Expected 2 children for sparse encoding, found {}",
                parts.nchildren()
            )
        }
        let metadata = ProstMetadata::<SparseMetadata>::deserialize(parts.metadata())?;
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

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let like = like
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

        // Try and cast the "like" fill value into the array's type. This is useful for cases where we narrow the arrays type.
        let fill_value = like.and_then(|arr| arr.fill_scalar().cast(input.as_ref().dtype()).ok());

        Ok(Some(SparseArray::encode(input.as_ref(), fill_value)?))
    }
}

impl ArrayVisitorImpl<ProstMetadata<SparseMetadata>> for SparseArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let fill_value_buffer = self
            .fill_value
            .value()
            .to_flexbytes::<ByteBufferMut>()
            .freeze();
        visitor.visit_buffer(&fill_value_buffer);
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_patches(self.patches())
    }

    fn _metadata(&self) -> ProstMetadata<SparseMetadata> {
        ProstMetadata(SparseMetadata {
            patches: self
                .patches()
                .to_metadata(self.len(), self.dtype())
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}
