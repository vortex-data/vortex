// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::EmptyMetadata;
use crate::arrays::{MaskedArray, MaskedEncoding, MaskedVTable};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;

impl SerdeVTable<MaskedVTable> for MaskedVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &MaskedArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &MaskedEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<MaskedArray> {
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffer, got {}", buffers.len());
        }

        let child = children.get(0, dtype, len)?;

        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "`MaskedArray::build` expects 1 or 2 children, got {}",
                children.len()
            );
        };

        MaskedArray::try_new(child, validity)
    }
}
