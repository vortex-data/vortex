// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_vector::Vector;
use vortex_vector::binaryview::{BinaryVector, BinaryView, StringVector};

use crate::arrays::varbinview::VarBinViewArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{
    ArrayId, ArrayVTable, ArrayVTableExt, NotSupported, VTable, ValidityVTableFromValidityHelper,
};
use crate::{EmptyMetadata, vtable};

mod array;
mod canonical;
mod operations;
mod operator;
mod validity;
mod visitor;

vtable!(VarBinView);

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
    type OperatorVTable = Self;

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
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<VarBinViewArray> {
        if buffers.is_empty() {
            vortex_bail!("Expected at least 1 buffer, got {}", buffers.len());
        }
        let mut buffers: Vec<ByteBuffer> = buffers.to_vec();
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

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        Ok(match array.dtype() {
            DType::Utf8(_) => unsafe {
                StringVector::new_unchecked(
                    array.views().clone(),
                    Arc::new(array.buffers().to_vec().into_boxed_slice()),
                    array.validity_mask(),
                )
            }
            .into(),
            DType::Binary(_) => unsafe {
                BinaryVector::new_unchecked(
                    array.views().clone(),
                    Arc::new(array.buffers().to_vec().into_boxed_slice()),
                    array.validity_mask(),
                )
            }
            .into(),
            _ => unreachable!("VarBinViewArray must have Binary or Utf8 dtype"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct VarBinViewVTable;
