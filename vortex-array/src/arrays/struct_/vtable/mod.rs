// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::vtable::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityHelper;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod operations;
mod rules;
mod validity;
mod visitor;

use crate::vtable::ArrayId;

vtable!(Struct);

impl VTable for StructVTable {
    type Array = StructArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
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

    fn canonicalize(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        Ok(Canonical::Struct(array.clone()))
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let fields: Vec<_> = array
            .unmasked_fields()
            .iter()
            .map(|field| field.slice(range.clone()))
            .try_collect()?;

        // SAFETY: Slicing preserves all StructArray invariants
        Ok(Some(
            unsafe {
                StructArray::new_unchecked(
                    fields,
                    array.struct_fields().clone(),
                    range.len(),
                    array.validity().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}

#[derive(Debug)]
pub struct StructVTable;

impl StructVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.struct");
}
