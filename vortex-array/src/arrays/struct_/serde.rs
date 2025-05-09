use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::StructEncoding;
use crate::arrays::StructArray;
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, EmptyMetadata, EncodingId,
};

impl EncodingVTable for StructEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.struct")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let validity = if parts.nchildren() == struct_dtype.nfields() {
            Validity::from(nullability)
        } else if parts.nchildren() == struct_dtype.nfields() + 1 {
            // Validity is the first child if it exists.
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                parts.nchildren()
            );
        };

        let children = (0..parts.nchildren())
            .map(|i| {
                let child_parts = parts.child(i);
                let child_dtype = struct_dtype
                    .field_by_index(i)
                    .vortex_expect("no out of bounds");
                child_parts.decode(ctx, child_dtype, len)
            })
            .try_collect()?;

        Ok(StructArray::try_new_with_dtype(children, struct_dtype, len, validity)?.into_array())
    }
}

impl ArrayVisitorImpl for StructArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
        for (idx, name) in self.names().iter().enumerate() {
            visitor.visit_child(name.as_ref(), &self.fields()[idx]);
        }
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
