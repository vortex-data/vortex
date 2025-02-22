use serde::{Deserialize, Serialize};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, DeserializeMetadata,
    EmptyMetadata, SerdeMetadata,
};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEndMetadata {
    ends_ptype: PType,
    num_runs: usize,
    offset: usize,
}

impl ArrayVisitorImpl<SerdeMetadata<RunEndMetadata>> for RunEndArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", self.ends());
        visitor.visit_child("values", self.values());
    }

    fn _metadata(&self) -> EmptyMetadata {
        SerdeMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(self.ends().dtype()).expect("Must be a valid PType"),
            num_runs: self.num_runs(),
            offset: self.offset(),
        })
    }
}

impl SerdeVTable<&RunEndArray> for RunEndEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = SerdeMetadata::<RunEndMetadata>::deserialize(parts.metadata())?;

        let ends_dtype = DType::Primitive(metadata.ends_ptype, Nullability::NonNullable);
        let ends = parts.child(0).decode(ctx, ends_dtype, metadata.num_runs)?;

        let values = parts.child(1).decode(ctx, dtype, len)?;

        Ok(RunEndArray::with_offset_and_length(ends, values, metadata.offset, len)?.into_array())
    }
}
