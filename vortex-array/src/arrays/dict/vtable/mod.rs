// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use super::DictArray;
use super::DictMetadata;
use super::take_canonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::DeserializeMetadata;
use crate::IntoArray;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::ConstantArray;
use crate::arrays::vtable::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

mod array;
mod canonical;
mod encode;
mod operations;
mod rules;
mod validity;
mod visitor;

vtable!(Dict);

#[derive(Debug)]
pub struct DictVTable;

impl VTable for DictVTable {
    type Array = DictArray;

    type Metadata = ProstMetadata<DictMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.dict")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DictVTable.as_vtable()
    }

    fn metadata(array: &DictArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(array.codes().dtype())? as i32,
            values_len: u32::try_from(array.values().len()).map_err(|_| {
                vortex_err!(
                    "Dictionary values size {} overflowed u32",
                    array.values().len()
                )
            })?,
            is_nullable_codes: Some(array.codes().dtype().is_nullable()),
            all_values_referenced: Some(array.all_values_referenced),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(buffer)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DictArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;
        let all_values_referenced = metadata.all_values_referenced.unwrap_or(false);

        // SAFETY: We've validated the metadata and children.
        Ok(unsafe {
            DictArray::new_unchecked(codes, values).set_all_values_referenced(all_values_referenced)
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "DictArray expects exactly 2 children (codes, values), got {}",
            children.len()
        );
        let [codes, values]: [ArrayRef; 2] = children
            .try_into()
            .map_err(|_| vortex_err!("Failed to convert children to array"))?;
        array.codes = codes;
        array.values = values;
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(canonical) = execute_fast_path(array, ctx)? {
            return Ok(canonical);
        }

        let values = array.values().clone().execute::<Canonical>(ctx)?;
        let codes = array
            .codes()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_primitive();

        // TODO(ngates): if indices are sorted and unique (strict-sorted), then we should delegate to
        //  the filter function since they're typically optimised for this case.
        // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
        //  such that canonicalize does less work.

        let canonical = take_canonical(values, &codes);

        let result_dtype = array
            .dtype()
            .union_nullability(array.codes().dtype().nullability());
        vortex_ensure!(
            canonical.as_ref().dtype() == &result_dtype,
            "Dict result dtype mismatch: expected {:?}, got {:?}",
            result_dtype,
            canonical.as_ref().dtype()
        );
        vortex_ensure!(
            canonical.as_ref().len() == array.len(),
            "Dict result length mismatch: expected {}, got {}",
            array.len(),
            canonical.as_ref().len()
        );

        Ok(canonical)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Check for fast-path execution conditions.
pub(super) fn execute_fast_path(
    array: &DictArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Canonical>> {
    // Empty array - nothing to do
    if array.is_empty() {
        let result_dtype = array
            .dtype()
            .union_nullability(array.codes().dtype().nullability());
        return Ok(Some(Canonical::empty(&result_dtype)));
    }

    // All codes are null - result is all nulls
    if array.codes.all_invalid() {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), array.codes.len())
                .into_array()
                .execute(ctx)?,
        ));
    }

    Ok(None)
}
