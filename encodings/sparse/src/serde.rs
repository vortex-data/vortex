use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{SparseArray, SparseEncoding, SparseVTable};

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct SparseMetadata {
    #[prost(message, required, tag = "1")]
    patches: PatchesMetadata,
}

impl SerdeVTable<SparseVTable> for SparseVTable {
    type Metadata = ProstMetadata<SparseMetadata>;

    fn metadata(array: &SparseArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(SparseMetadata {
            patches: array.patches().to_metadata(array.len(), array.dtype())?,
        })))
    }

    fn build(
        _encoding: &SparseEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<SparseArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for sparse encoding, found {}",
                children.len()
            )
        }
        assert_eq!(
            metadata.patches.offset(),
            0,
            "Patches must start at offset 0"
        );

        let patch_indices =
            children.get(0, &metadata.patches.indices_dtype(), metadata.patches.len())?;
        let patch_values = children.get(1, dtype, metadata.patches.len())?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let fill_value = Scalar::new(dtype.clone(), ScalarValue::from_protobytes(&buffers[0])?);

        SparseArray::try_new(patch_indices, patch_values, len, fill_value)
    }
}

impl EncodeVTable<SparseVTable> for SparseVTable {
    fn encode(
        _encoding: &SparseEncoding,
        input: &Canonical,
        like: Option<&SparseArray>,
    ) -> VortexResult<Option<SparseArray>> {
        // Try and cast the "like" fill value into the array's type. This is useful for cases where we narrow the arrays type.
        let fill_value = like.and_then(|arr| arr.fill_scalar().cast(input.as_ref().dtype()).ok());

        // TODO(ngates): encode should only handle arrays that _can_ be made sparse.
        Ok(SparseArray::encode(input.as_ref(), fill_value)?
            .as_opt::<SparseVTable>()
            .cloned())
    }
}

impl VisitorVTable<SparseVTable> for SparseVTable {
    fn visit_buffers(array: &SparseArray, visitor: &mut dyn ArrayBufferVisitor) {
        let fill_value_buffer = array
            .fill_value
            .value()
            .to_protobytes::<ByteBufferMut>()
            .freeze();
        visitor.visit_buffer(&fill_value_buffer);
    }

    fn visit_children(array: &SparseArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_patches(array.patches())
    }
}
