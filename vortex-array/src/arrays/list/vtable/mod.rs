// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::list::ListData;
use crate::arrays::list::array::NUM_SLOTS;
use crate::arrays::list::array::SLOT_NAMES;
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
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
mod operations;
mod validity;
vtable!(List, List, ListData);

#[derive(Clone, prost::Message)]
pub struct ListMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
}

impl VTable for List {
    type ArrayData = ListData;

    type Metadata = ProstMetadata<ListMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &ListData) -> &Self {
        &List
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ListData) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &ListData) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &ListData, state: &mut H, precision: Precision) {
        array.elements().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &ListData, other: &ListData, precision: Precision) -> bool {
        array.elements().array_eq(other.elements(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.validity().array_eq(&other.validity(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ListArray buffer_name index {idx} out of bounds")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<ListData> {
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

        ListData::try_new(elements, offsets, validity)
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ListArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            list_view_from_list(array, ctx)?.into_array(),
        ))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
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
