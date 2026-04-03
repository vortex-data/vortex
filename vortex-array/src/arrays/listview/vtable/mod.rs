// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::listview::ListViewData;
use crate::arrays::listview::array::NUM_SLOTS;
use crate::arrays::listview::array::SLOT_NAMES;
use crate::arrays::listview::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
mod operations;
mod validity;
vtable!(ListView, ListView, ListViewData);

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
    type ArrayData = ListViewData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn array_hash<H: std::hash::Hasher>(array: &ListViewData, state: &mut H, precision: Precision) {
        array.elements().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.sizes().array_hash(state, precision);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &ListViewData, other: &ListViewData, precision: Precision) -> bool {
        array.elements().array_eq(other.elements(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.sizes().array_eq(other.sizes(), precision)
            && array.validity().array_eq(&other.validity(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ListViewArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ListViewArray buffer_name index {idx} out of bounds")
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(ProstMetadata(ListViewMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
            size_ptype: PType::try_from(array.sizes().dtype())? as i32,
        })
        .serialize()))
    }

    fn validate(&self, data: &ListViewData, dtype: &DType, len: usize) -> VortexResult<()> {
        vortex_ensure!(
            data.len() == len,
            "ListViewArray length {} does not match outer length {}",
            data.len(),
            len
        );

        let actual_dtype = data.dtype();
        vortex_ensure!(
            &actual_dtype == dtype,
            "ListViewArray dtype {} does not match outer dtype {}",
            actual_dtype,
            dtype
        );

        Ok(())
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,        metadata: &[u8],

        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ListViewData> {
        let metadata = ProstMetadata::<ListViewMetadata>::deserialize(metadata)?;
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
            usize::try_from(metadata.elements_len)?,
        )?;

        // Get offsets with proper type from metadata.
        let offsets = children.get(
            1,
            &DType::Primitive(metadata.offset_ptype(), Nullability::NonNullable),
            len,
        )?;

        // Get sizes with proper type from metadata.
        let sizes = children.get(
            2,
            &DType::Primitive(metadata.size_ptype(), Nullability::NonNullable),
            len,
        )?;

        ListViewData::try_new(elements, offsets, sizes, validity)
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
            "ListViewArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
