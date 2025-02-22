use serde::{Deserialize, Serialize};
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{Array, ArrayRef, ContextRef, DeserializeMetadata, SerdeMetadata};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexResult};

use crate::{DictArray, DictEncoding};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictMetadata {
    codes_ptype: PType,
    values_len: u32,
}

impl SerdeVTable<&DictArray> for DictEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                parts.nchildren()
            )
        }
        let metadata = SerdeMetadata::<DictMetadata>::deserialize(parts.metadata())?;

        let codes_dtype = DType::Primitive(metadata.codes_ptype, dtype.nullability());
        let codes = parts.child(0).decode(ctx, codes_dtype, len)?;

        let values = parts
            .child(1)
            .decode(ctx, dtype, metadata.values_len as usize)?;

        Ok(DictArray::try_new(codes, values)?.into_array())
    }
}
