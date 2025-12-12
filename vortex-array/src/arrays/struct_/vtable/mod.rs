// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_vector::struct_::StructVector;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::arrays::struct_::StructArray;
use crate::kernel::BindCtx;
use crate::kernel::KernelRef;
use crate::kernel::kernel;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;

vtable!(Struct);

impl VTable for StructVTable {
    type Array = StructArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.struct")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        StructVTable.as_vtable()
    }

    fn metadata(_array: &StructArray) -> VortexResult<Self::Metadata> {
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
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<StructArray> {
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (Validity::from(*nullability), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            // Validity is the first child if it exists.
            let validity = children.get(0, &Validity::DTYPE, len)?;
            (Validity::Array(validity), 1_usize)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let children: Vec<_> = (0..struct_dtype.nfields())
            .map(|i| {
                let child_dtype = struct_dtype
                    .field_by_index(i)
                    .vortex_expect("no out of bounds");
                children.get(non_data_children + i, &child_dtype, len)
            })
            .try_collect()?;

        StructArray::try_new_with_dtype(children, struct_dtype.clone(), len, validity)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        let DType::Struct(struct_dtype, _nullability) = &array.dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", array.dtype)
        };

        // First child is validity (if present), followed by fields
        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (array.validity.clone(), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            (Validity::Array(children[0].clone()), 1_usize)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let fields: Arc<[ArrayRef]> = children.into_iter().skip(non_data_children).collect();
        vortex_ensure!(
            fields.len() == struct_dtype.nfields(),
            "Expected {} field children, found {}",
            struct_dtype.nfields(),
            fields.len()
        );

        array.fields = fields;
        array.validity = validity;
        Ok(())
    }

    fn bind_kernel(array: &Self::Array, ctx: &mut BindCtx) -> VortexResult<KernelRef> {
        let fields: Box<[_]> = array
            .fields()
            .iter()
            .map(|field| field.bind_kernel(ctx))
            .try_collect()?;
        let validity_mask = array.validity_mask();

        Ok(kernel(move || {
            // SAFETY: we know that all field lengths match the struct array length, and the validity
            let fields = fields.into_iter().map(|k| k.execute()).try_collect()?;
            Ok(unsafe { StructVector::new_unchecked(Arc::new(fields), validity_mask) }.into())
        }))
    }
}

#[derive(Debug)]
pub struct StructVTable;
