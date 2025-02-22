use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayRef, ContextRef};

impl SerdeVTable<&ConstantArray> for ConstantEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let sv = ScalarValue::from_flexbytes(&parts.buffers()?[0])?;
        let scalar = Scalar::new(dtype, sv);
        Ok(ConstantArray::new(scalar, len).into_array())
    }
}
