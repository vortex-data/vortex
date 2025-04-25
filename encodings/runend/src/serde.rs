use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, Canonical,
    DeserializeMetadata, EncodingId, ProstMetadata,
};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult};

use crate::compress::runend_encode;
use crate::{RunEndArray, RunEndEncoding};

#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    num_runs: u64,
    #[prost(uint64, tag = "3")]
    offset: u64,
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
        let metadata = ProstMetadata::<RunEndMetadata>::deserialize(parts.metadata())?;

        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = parts.child(0).decode(ctx, ends_dtype, runs)?;

        let values = parts.child(1).decode(ctx, dtype, runs)?;

        Ok(RunEndArray::with_offset_and_length(
            ends,
            values,
            usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize"),
            len,
        )?
        .into_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let parray = input.clone().into_primitive()?;

        let (ends, values) = runend_encode(&parray)?;

        Ok(Some(
            RunEndArray::try_new(ends.to_array(), values)?.to_array(),
        ))
    }
}

impl ArrayVisitorImpl<ProstMetadata<RunEndMetadata>> for RunEndArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", self.ends());
        visitor.visit_child("values", self.values());
    }

    fn _metadata(&self) -> ProstMetadata<RunEndMetadata> {
        ProstMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(self.ends().dtype()).vortex_expect("Must be a valid PType")
                as i32,
            num_runs: self.ends().len() as u64,
            offset: self.offset() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            ProstMetadata(RunEndMetadata {
                ends_ptype: PType::U64 as i32,
                num_runs: u64::MAX,
                offset: u64::MAX,
            }),
        );
    }
}
