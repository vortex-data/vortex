use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail};

use super::{FixedSizeListArray, FixedSizeListVTable};
use crate::arrays::FixedSizeListEncoding;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

#[derive(Clone, prost::Message)]
pub struct FixedSizeListMetadata {
    #[prost(uint64, tag = "1")]
    len: u64,
    #[prost(uint32, tag = "2")]
    list_size: u32,
}

impl SerdeVTable<FixedSizeListVTable> for FixedSizeListVTable {
    type Metadata = ProstMetadata<FixedSizeListMetadata>;

    fn metadata(array: &FixedSizeListArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(FixedSizeListMetadata {
            len: array.len() as u64,
            list_size: array.list_size(),
        })))
    }

    fn build(
        encoding: &FixedSizeListEncoding,
        dtype: &DType,
        len: usize,
        metadata: &FixedSizeListMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FixedSizeListArray> {
        unimplemented!("TODO(connor)[FixedSizeList")
    }
}

impl VisitorVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn visit_buffers(_array: &FixedSizeListArray, _visitor: &mut dyn ArrayBufferVisitor) {
        unimplemented!("TODO(connor)[FixedSizeList")
    }

    fn visit_children(array: &FixedSizeListArray, visitor: &mut dyn ArrayChildVisitor) {
        unimplemented!("TODO(connor)[FixedSizeList")
    }
}
