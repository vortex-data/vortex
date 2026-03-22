// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use kernel::PARENT_KERNELS;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use super::DictArray;
use super::DictMetadata;
use super::take_canonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::DeserializeMetadata;
use crate::DynArray;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::ConstantArray;
use crate::arrays::dict::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionStep;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
mod kernel;
mod operations;
mod validity;

vtable!(Dict);

#[derive(Debug)]
pub struct Dict;

impl Dict {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.dict");
}

impl VTable for Dict {
    type Array = DictArray;

    type Metadata = ProstMetadata<DictMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &Dict
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &DictArray) -> usize {
        array.codes.len()
    }

    fn dtype(array: &DictArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DictArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DictArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.codes.array_hash(state, precision);
        array.values.array_hash(state, precision);
    }

    fn array_eq(array: &DictArray, other: &DictArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.codes.array_eq(&other.codes, precision)
            && array.values.array_eq(&other.values, precision)
    }

    fn nbuffers(_array: &DictArray) -> usize {
        0
    }

    fn buffer(_array: &DictArray, idx: usize) -> BufferHandle {
        vortex_panic!("DictArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &DictArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &DictArray) -> usize {
        2
    }

    fn child(array: &DictArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.codes().clone(),
            1 => array.values().clone(),
            _ => vortex_panic!("DictArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &DictArray, idx: usize) -> String {
        match idx {
            0 => "codes".to_string(),
            1 => "values".to_string(),
            _ => vortex_panic!("DictArray child_name index {idx} out of bounds"),
        }
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

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        if let Some(canonical) = execute_fast_path(array, ctx)? {
            return Ok(ExecutionStep::Done(canonical));
        }

        // TODO(joe): if the values are constant return a constant
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

        Ok(ExecutionStep::Done(
            take_canonical(values, &codes, ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

/// Check for fast-path execution conditions.
pub(super) fn execute_fast_path(
    array: &DictArray,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    // Empty array - nothing to do
    if array.is_empty() {
        let result_dtype = array
            .dtype()
            .union_nullability(array.codes().dtype().nullability());
        return Ok(Some(Canonical::empty(&result_dtype).into_array()));
    }

    // All codes are null - result is all nulls
    if array.codes.all_invalid()? {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), array.codes.len())
                .into_array(),
        ));
    }

    Ok(None)
}
