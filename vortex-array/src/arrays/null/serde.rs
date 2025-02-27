use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{NullArray, NullEncoding};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayContext, ArrayRef, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for NullArray {
    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&NullArray> for NullEncoding {
    fn decode(
        &self,
        _parts: &ArrayParts,
        _ctx: &ArrayContext,
        _dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(len).into_array())
    }
}
