// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewData;
use crate::arrays::listview::array::NUM_SLOTS;
use crate::arrays::listview::array::SLOT_NAMES;
use crate::arrays::listview::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
mod operations;
mod validity;
/// A [`ListView`]-encoded Vortex array.
pub type ListViewArray = Array<ListView>;

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

impl ArrayHash for ListViewData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.is_zero_copy_to_list().hash(state);
    }
}

impl ArrayEq for ListViewData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.is_zero_copy_to_list() == other.is_zero_copy_to_list()
    }
}

impl VTable for ListView {
    type ArrayData = ListViewData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            ListViewMetadata {
                elements_len: array.elements().len() as u64,
                offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
                size_ptype: PType::try_from(array.sizes().dtype())? as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn validate(
        &self,
        _data: &ListViewData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ListViewArray expected {NUM_SLOTS} slots, found {}",
            slots.len()
        );
        let elements = slots[crate::arrays::listview::array::ELEMENTS_SLOT]
            .as_ref()
            .vortex_expect("ListViewArray elements slot");
        let offsets = slots[crate::arrays::listview::array::OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("ListViewArray offsets slot");
        let sizes = slots[crate::arrays::listview::array::SIZES_SLOT]
            .as_ref()
            .vortex_expect("ListViewArray sizes slot");
        vortex_ensure!(
            offsets.len() == len && sizes.len() == len,
            "ListViewArray length {} does not match outer length {}",
            offsets.len(),
            len
        );

        let actual_dtype = DType::List(Arc::new(elements.dtype().clone()), dtype.nullability());
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
        len: usize,
        metadata: &[u8],

        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        let metadata = ListViewMetadata::decode(metadata)?;
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

        ListViewData::validate(&elements, &offsets, &sizes, &validity)?;
        let data = ListViewData::try_new()?;
        let slots = ListViewData::make_slots(&elements, &offsets, &sizes, &validity, len);
        Ok(crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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
