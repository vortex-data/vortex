// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_vector::binaryview::BinaryView;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::arrays::varbinview::VarBinViewArray;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

vtable!(VarBinView);

#[derive(Debug)]
pub struct VarBinViewVTable;

impl VTable for VarBinViewVTable {
    type Array = VarBinViewArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.varbinview")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        VarBinViewVTable.as_vtable()
    }

    fn metadata(_array: &VarBinViewArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<VarBinViewArray> {
        if buffers.is_empty() {
            vortex_bail!("Expected at least 1 buffer, got {}", buffers.len());
        }
        let mut buffers: Vec<ByteBuffer> = buffers
            .iter()
            .map(|b| b.clone().try_to_bytes())
            .collect::<VortexResult<Vec<_>>>()?;
        let views = buffers.pop().vortex_expect("buffers non-empty");

        let views = Buffer::<BinaryView>::from_byte_buffer(views);

        if views.len() != len {
            vortex_bail!("Expected {} views, got {}", len, views.len());
        }

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 children, got {}", children.len());
        };

        VarBinViewArray::try_new(views, Arc::from(buffers), dtype.clone(), validity)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        match children.len() {
            0 => {}
            1 => {
                let [validity]: [ArrayRef; 1] = children
                    .try_into()
                    .map_err(|_| vortex_err!("Failed to convert children to array"))?;
                array.validity = Validity::Array(validity);
            }
            _ => vortex_bail!(
                "VarBinViewArray expects 0 or 1 children (validity?), got {}",
                children.len()
            ),
        }
        Ok(())
    }
}
