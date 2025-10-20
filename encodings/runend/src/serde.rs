// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult};

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
    fn build(
        _encoding: &RunEndEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<<RunEndVTable as VTable>::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<RunEndArray> {
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children.get(0, &ends_dtype, runs)?;

        let values = children.get(1, dtype, runs)?;

        RunEndArray::try_new_offset_length(
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
        _like: Option<&RunEndArray>,
    ) -> VortexResult<Option<RunEndArray>> {
        let parray = canonical.clone().into_primitive();
        let (ends, values) = runend_encode(&parray);
        // SAFETY: runend_decode implementation must return valid RunEndArray
        //  components.
        unsafe {
            Ok(Some(RunEndArray::new_unchecked(
                ends.to_array(),
                values,
                0,
                parray.len(),
            )))
        }
    }
}

impl VisitorVTable<RunEndVTable> for RunEndVTable {
    fn metadata(array: &RunEndArray) -> <RunEndVTable as VTable>::Metadata {
        ProstMetadata(RunEndMetadata {
            ends_ptype: array.ends().dtype().as_ptype() as i32,
            num_runs: array.ends().len() as u64,
            offset: array.offset() as u64,
        })
    }

    fn visit_buffers(_array: &RunEndArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &RunEndArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", array.ends());
        visitor.visit_child("values", array.values());
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
