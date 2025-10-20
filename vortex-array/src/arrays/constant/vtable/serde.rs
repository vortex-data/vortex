// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantEncoding, ConstantVTable};
use crate::serde::ArrayChildren;
use crate::vtable::{SerdeVTable, VTable};

impl SerdeVTable<ConstantVTable> for ConstantVTable {
    fn build(
        _encoding: &ConstantEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &<ConstantVTable as VTable>::Metadata,
        buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let sv = ScalarValue::from_protobytes(&buffers[0])?;
        let scalar = Scalar::new(dtype.clone(), sv);
        Ok(ConstantArray::new(scalar, len))
    }
}
