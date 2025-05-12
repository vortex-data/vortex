use vortex_array::serde::ArrayParts;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, Canonical,
    DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::compress::runend_encode;
use crate::{RunEndArray, RunEndEncoding, RunEndVTable};

#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    num_runs: u64,
    #[prost(uint64, tag = "3")]
    offset: u64,
}

impl SerdeVTable<RunEndVTable> for RunEndVTable {
    type Metadata = ProstMetadata<RunEndMetadata>;

    fn metadata(array: &RunEndArray) -> Option<Self::Metadata> {
        Some(ProstMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(array.ends().dtype()).vortex_expect("Must be a valid PType")
                as i32,
            num_runs: array.ends().len() as u64,
            offset: array.offset() as u64,
        }))
    }

    fn decode(
        _encoding: &RunEndEncoding,
        dtype: DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<RunEndArray> {
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children[0].decode(ctx, ends_dtype, runs)?;

        let values = children[1].decode(ctx, dtype, runs)?;

        RunEndArray::with_offset_and_length(
            ends,
            values,
            usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize"),
            len,
        )
    }
}

impl EncodeVTable<RunEndVTable> for RunEndVTable {
    fn encode(
        _encoding: &RunEndEncoding,
        canonical: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<RunEndArray>> {
        let parray = canonical.clone().into_primitive()?;
        let (ends, values) = runend_encode(&parray)?;
        Ok(Some(RunEndArray::try_new(ends.to_array(), values)?))
    }
}

impl VisitorVTable<RunEndVTable> for RunEndVTable {
    fn visit_buffers(_array: &RunEndArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &RunEndArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", array.ends());
        visitor.visit_child("values", array.values());
    }

    fn with_children(array: &RunEndArray, children: &[ArrayRef]) -> VortexResult<RunEndArray> {
        if children.len() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 children, got {}", children.len());
        }
        if array.offset() != 0 {
            vortex_bail!(InvalidArgument: "RunEndArray with offset cannot have children replaced");
        }
        RunEndArray::try_new(children[0].clone(), children[1].clone())
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
