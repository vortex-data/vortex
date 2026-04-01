// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::arrays::ListArray;
use crate::arrays::list::array::NUM_SLOTS;
use crate::arrays::list::array::SLOT_NAMES;
use crate::arrays::list::array::VALIDITY_SLOT;
use crate::arrays::list::compute::PARENT_KERNELS;
use crate::arrays::list::compute::rules::PARENT_RULES;
use crate::arrays::listview::list_view_from_list;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::metadata::DeserializeMetadata;
use crate::metadata::SerializeMetadata;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
mod operations;
mod validity;
vtable!(List);

#[derive(Clone, prost::Message)]
pub struct ListMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
}

impl VTable for List {
    type Array = ListArray;

    type Metadata = ProstMetadata<ListMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &Self::Array) -> &Self {
        &List
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ListArray) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &ListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ListArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.elements().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &ListArray, other: &ListArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &ListArray) -> usize {
        0
    }

    fn buffer(_array: &ListArray, idx: usize) -> BufferHandle {
        vortex_panic!("ListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ListArray, idx: usize) -> Option<String> {
        vortex_panic!("ListArray buffer_name index {idx} out of bounds")
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn metadata(array: &ListArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ListMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(SerializeMetadata::serialize(metadata)))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<ListMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ListArray> {
        let validity = if children.len() == 2 {
            Validity::from(dtype.nullability())
        } else if children.len() == 3 {
            let validity = children.get(2, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 2 or 3 children, got {}", children.len());
        };

        let DType::List(element_dtype, _) = &dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.0.elements_len)?,
        )?;

        let offsets = children.get(
            1,
            &DType::Primitive(metadata.0.offset_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        ListArray::try_new(elements, offsets, validity)
    }

    fn slots(array: &ListArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ListArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut ListArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ListArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.validity = match &slots[VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.dtype.nullability()),
        };
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            list_view_from_list(ListArray::clone(&array), ctx)?.into_array(),
        ))
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct List;

impl List {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.list");
}
