// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::BoolArray;
use crate::ProstMetadata;
use crate::arrays::BoolVTable;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, VTable};

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl SerdeVTable<BoolVTable> for BoolVTable {
    type Metadata = ProstMetadata<BoolMetadata>;

    fn metadata(array: &BoolArray) -> VortexResult<Option<Self::Metadata>> {
        let bit_offset = array.bit_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {bit_offset}");
        Ok(Some(ProstMetadata(BoolMetadata {
            offset: u32::try_from(bit_offset).vortex_expect("checked"),
        })))
    }

    fn build(
        _encoding: &<BoolVTable as VTable>::Encoding,
        dtype: &DType,
        len: usize,
        metadata: &BoolMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<BoolArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        BoolArray::try_new(buffers[0].clone(), metadata.offset as usize, len, validity)
    }
}
