use serde::{Deserialize, Serialize};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    EncodingId, SerdeMetadata,
};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult};

use crate::{RunEndArray, RunEndEncoding};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEndMetadata {
    ends_ptype: PType,
    num_runs: usize,
    offset: usize,
}

impl EncodingVTable for RunEndEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.runend")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = SerdeMetadata::<RunEndMetadata>::deserialize(parts.metadata())?;

        let ends_dtype = DType::Primitive(metadata.ends_ptype, Nullability::NonNullable);
        let ends = parts.child(0).decode(ctx, ends_dtype, metadata.num_runs)?;

        let values = parts.child(1).decode(ctx, dtype, metadata.num_runs)?;

        Ok(RunEndArray::with_offset_and_length(ends, values, metadata.offset, len)?.into_array())
    }
}

impl ArrayVisitorImpl<SerdeMetadata<RunEndMetadata>> for RunEndArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", self.ends());
        visitor.visit_child("values", self.values());
    }

    fn _metadata(&self) -> SerdeMetadata<RunEndMetadata> {
        SerdeMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(self.ends().dtype()).vortex_expect("Must be a valid PType"),
            num_runs: self.ends().len(),
            offset: self.offset(),
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::SerdeMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            SerdeMetadata(RunEndMetadata {
                offset: usize::MAX,
                ends_ptype: PType::U64,
                num_runs: usize::MAX,
            }),
        );
    }
}
