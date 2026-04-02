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
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::ListViewArray;
use crate::arrays::listview::array::NUM_SLOTS;
use crate::arrays::listview::array::SLOT_NAMES;
use crate::arrays::listview::array::VALIDITY_SLOT;
use crate::arrays::listview::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
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
vtable!(ListView);

#[derive(Clone, Debug)]
pub struct ListView;

impl ListView {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.listview");
}

#[derive(Clone, prost::Message)]
pub struct ListViewMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    size_ptype: i32,
}

impl VTable for ListView {
    type Array = ListViewArray;

    type Metadata = ProstMetadata<ListViewMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &Self::Array) -> &Self {
        &ListView
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ListViewArray) -> usize {
        debug_assert_eq!(array.offsets().len(), array.sizes().len());
        array.offsets().len()
    }

    fn dtype(array: &ListViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ListViewArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.elements().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.sizes().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &ListViewArray, other: &ListViewArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.sizes().array_eq(other.sizes(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &ListViewArray) -> usize {
        0
    }

    fn buffer(_array: &ListViewArray, idx: usize) -> BufferHandle {
        vortex_panic!("ListViewArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ListViewArray, idx: usize) -> Option<String> {
        vortex_panic!("ListViewArray buffer_name index {idx} out of bounds")
    }

    fn metadata(array: &ListViewArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ListViewMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
            size_ptype: PType::try_from(array.sizes().dtype())? as i32,
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
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ListViewArray> {
        vortex_ensure!(
            buffers.is_empty(),
            "`ListViewArray::build` expects no buffers"
        );

        let DType::List(element_dtype, _) = dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };

        let validity = if children.len() == 3 {
            Validity::from(dtype.nullability())
        } else if children.len() == 4 {
            let validity = children.get(3, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "`ListViewArray::build` expects 3 or 4 children, got {}",
                children.len()
            );
        };

        // Get elements with the correct length from metadata.
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.0.elements_len)?,
        )?;

        // Get offsets with proper type from metadata.
        let offsets = children.get(
            1,
            &DType::Primitive(metadata.0.offset_ptype(), Nullability::NonNullable),
            len,
        )?;

        // Get sizes with proper type from metadata.
        let sizes = children.get(
            2,
            &DType::Primitive(metadata.0.size_ptype(), Nullability::NonNullable),
            len,
        )?;

        ListViewArray::try_new(elements, offsets, sizes, validity)
    }

    fn slots(array: &ListViewArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ListViewArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut ListViewArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ListViewArray expects exactly {} slots, got {}",
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

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
