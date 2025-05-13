use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{DType, PType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};

use super::PrimitiveEncoding;
use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, EmptyMetadata};

impl SerdeVTable<PrimitiveVTable> for PrimitiveVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &PrimitiveArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &PrimitiveEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PrimitiveArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let ptype = PType::try_from(dtype)?;

        if !buffer.is_aligned(Alignment::new(ptype.byte_width())) {
            vortex_bail!(
                "Buffer is not aligned to {}-byte boundary",
                ptype.byte_width()
            );
        }
        if buffer.len() != ptype.byte_width() * len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for {}, {}",
                buffer.len(),
                ptype.byte_width() * len,
                ptype.byte_width(),
                len,
            );
        }

        match_each_native_ptype!(ptype, |$P| {
            let buffer = Buffer::<$P>::from_byte_buffer(buffer);
            Ok(PrimitiveArray::new(buffer, validity))
        })
    }
}

impl VisitorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn visit_buffers(array: &PrimitiveArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.byte_buffer());
    }

    fn visit_children(array: &PrimitiveArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
    }

    fn with_children(
        array: &PrimitiveArray,
        children: &[ArrayRef],
    ) -> VortexResult<PrimitiveArray> {
        let validity = if array.validity().is_array() {
            Validity::Array(children[0].clone())
        } else {
            array.validity().clone()
        };

        Ok(PrimitiveArray::from_byte_buffer(
            array.byte_buffer().clone(),
            array.ptype(),
            validity,
        ))
    }
}
