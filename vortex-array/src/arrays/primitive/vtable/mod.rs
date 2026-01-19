// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityHelper;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod operations;
pub mod rules;
mod validity;
mod visitor;

pub use rules::PrimitiveMaskedValidityRule;
use vortex_buffer::Alignment;

use crate::arrays::primitive::vtable::rules::RULES;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;

vtable!(Primitive);

impl VTable for PrimitiveVTable {
    type Array = PrimitiveArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.primitive")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        PrimitiveVTable.as_vtable()
    }

    fn metadata(_array: &PrimitiveArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<PrimitiveArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let ptype = PType::try_from(dtype)?;

        if buffer.len() != ptype.byte_width() * len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for {}, {}",
                buffer.len(),
                ptype.byte_width() * len,
                ptype.byte_width(),
                len,
            );
        }

        // For host buffers, we eagerly check alignment on construction.
        // TODO(aduffy): check for device buffers. CUDA buffers are generally 256-byte aligned,
        //  but not sure about other devices.
        if let Some(host_buf) = buffer.as_host_opt() {
            vortex_ensure!(
                host_buf.is_aligned(Alignment::new(ptype.byte_width())),
                "PrimitiveArray::build: Buffer must be aligned to {}",
                ptype.byte_width()
            );
        }

        // SAFETY: checked ahead of time
        unsafe {
            Ok(PrimitiveArray::new_unchecked_from_handle(
                buffer, ptype, validity,
            ))
        }
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() <= 1,
            "PrimitiveArray can have at most 1 child (validity), got {}",
            children.len()
        );

        array.validity = if children.is_empty() {
            Validity::from(array.dtype().nullability())
        } else {
            Validity::Array(children.into_iter().next().vortex_expect("checked"))
        };

        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let result = match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().slice_typed::<T>(range.clone()),
                T::PTYPE,
                array.validity().slice(range),
            )
            .into_array()
        });
        Ok(Some(result))
    }
}

#[derive(Debug)]
pub struct PrimitiveVTable;
