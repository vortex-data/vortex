use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::arrays::{StructArray, StructEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::variants::StructArrayTrait;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for StructArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
        for (idx, name) in self.names().iter().enumerate() {
            let child = self
                .maybe_null_field_by_idx(idx)
                .vortex_expect("no out of bounds");
            visitor.visit_child(name.as_ref(), &child);
        }
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&StructArray> for StructEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let struct_dtype = dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("Expected struct dtype, found {:?}", dtype))?;

        let validity = if parts.nchildren() == struct_dtype.nfields() {
            Validity::from(dtype.nullability())
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

        Ok(
            StructArray::try_new(struct_dtype.names().clone(), children, len, validity)?
                .into_array(),
        )
    }
}
