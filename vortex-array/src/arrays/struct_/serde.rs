use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};

use crate::arrays::{StructArray, StructEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayRef, ContextRef};

impl SerdeVTable<&StructArray> for StructEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
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
                child_parts.decode(ctx, child_dtype.clone(), len)
            })
            .try_collect()?;

        Ok(
            StructArray::try_new(struct_dtype.names().clone(), children, len, validity)?
                .into_array(),
        )
    }
}
