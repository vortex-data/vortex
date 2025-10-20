// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::extension::{ExtensionArray, ExtensionEncoding, ExtensionVTable};
use crate::serde::ArrayChildren;
use crate::vtable::{SerdeVTable, VTable};

impl SerdeVTable<ExtensionVTable> for ExtensionVTable {
    fn build(
        _encoding: &ExtensionEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &<ExtensionVTable as VTable>::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ExtensionArray> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Not an extension DType");
        };
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }
        let storage = children.get(0, ext_dtype.storage_dtype(), len)?;
        Ok(ExtensionArray::new(ext_dtype.clone(), storage))
    }
}
