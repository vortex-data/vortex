use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use super::{BinaryView, VarBinViewVTable};
use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, VisitorVTable};
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, EmptyMetadata};

impl SerdeVTable<VarBinViewVTable> for VarBinViewVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &VarBinViewArray) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn decode(
        _encoding: &VarBinViewEncoding,
        dtype: DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<VarBinViewArray> {
        if buffers.is_empty() {
            vortex_bail!("Expected at least 1 buffer, got {}", buffers.len());
        }
        let views = Buffer::<BinaryView>::from_byte_buffer(buffers[0].clone());

        if views.len() != len {
            vortex_bail!("Expected {} views, got {}", len, views.len());
        }

        let validity = if children.len() == 0 {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children[0].decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 children, got {}", children.len());
        };

        let buffers: Vec<ByteBuffer> = buffers[1..].to_vec();
        VarBinViewArray::try_new(views, buffers, dtype, validity)
    }
}

impl VisitorVTable<VarBinViewVTable> for VarBinViewVTable {
    fn visit_buffers(array: &VarBinViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in array.buffers() {
            visitor.visit_buffer(buffer);
        }
        visitor.visit_buffer(&array.views().clone().into_byte_buffer());
    }

    fn visit_children(array: &VarBinViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }

    fn with_children(
        array: &VarBinViewArray,
        children: &[ArrayRef],
    ) -> VortexResult<VarBinViewArray> {
        let mut this = array.clone();

        if let Validity::Array(array) = &mut this.validity {
            *array = children[0].clone();
        }

        Ok(this)
    }
}
