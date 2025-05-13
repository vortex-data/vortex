use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::StructEncoding;
use crate::arrays::{StructArray, StructVTable};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, EmptyMetadata};

impl SerdeVTable<StructVTable> for StructVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &StructArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &StructEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<StructArray> {
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let validity = if children.len() == struct_dtype.nfields() {
            Validity::from(*nullability)
        } else if children.len() == struct_dtype.nfields() + 1 {
            // Validity is the first child if it exists.
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let children = (0..children.len())
            .map(|i| {
                let child_dtype = struct_dtype
                    .field_by_index(i)
                    .vortex_expect("no out of bounds");
                children.get(i, &child_dtype, len)
            })
            .try_collect()?;

        StructArray::try_new_with_dtype(children, struct_dtype.clone(), len, validity)
    }
}

impl VisitorVTable<StructVTable> for StructVTable {
    fn visit_buffers(_array: &StructArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &StructArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
        for (idx, name) in array.names().iter().enumerate() {
            visitor.visit_child(name.as_ref(), &array.fields()[idx]);
        }
    }

    fn with_children(array: &StructArray, children: &[ArrayRef]) -> VortexResult<StructArray> {
        let validity = if array.validity().is_array() {
            Validity::Array(children[0].clone())
        } else {
            array.validity().clone()
        };

        let fields_idx = if validity.is_array() { 1_usize } else { 0 };
        let fields = children[fields_idx..].to_vec();

        StructArray::try_new_with_dtype(fields, array.struct_dtype().clone(), array.len(), validity)
    }
}
